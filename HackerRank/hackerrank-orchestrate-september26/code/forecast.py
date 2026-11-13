"""Deterministic, recommendation-free daily financial forecast simulation.

The module consumes typed records from :mod:`data_loading`; it does not parse CSVs,
interpret evidence, choose payment plans, or create output files.  It answers one
question only: whether explicitly supplied cash flows keep a balance at or above the
profile's minimum through a configurable horizon.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from enum import Enum
from typing import Iterable, Mapping, Sequence

from data_loading import CashState, ClassifiedEvent, ExchangeRate, FinancialEvent, FinancialProfile, RequestContext


class ForecastInputError(ValueError):
    """Raised when a forecast would require an unsupported or invented assumption."""


class MissingExchangeRateError(ForecastInputError):
    """Raised when a foreign-currency flow has no exact supplied rate for its cash date."""


class CashFlowSource(str, Enum):
    """The origin of a simulated cash-flow entry."""

    FUTURE_SETTLED_EVENT = "future_settled_event"
    FUTURE_SCHEDULED_EVENT = "future_scheduled_event"
    PENDING_DEBIT_RESERVATION = "pending_debit_reservation"
    RECURRING_CALLER_SCHEDULE = "recurring_caller_schedule"
    CANDIDATE_PAYMENT = "candidate_payment"


@dataclass(frozen=True)
class CandidatePayment:
    """A debit introduced by the caller for safety simulation, not a recommendation."""

    payment_id: str
    payment_date: date
    amount: Decimal
    currency: str | None = None
    description: str = ""


@dataclass(frozen=True)
class RecurringCashFlow:
    """An explicitly supplied recurrence schedule.

    ``occurrence_count=None`` deliberately means repeat through the forecast horizon.
    This is valid only because the caller has supplied both the start date and interval;
    the engine never derives either value from historical records.
    """

    flow_id: str
    first_occurrence_date: date
    amount: Decimal
    currency: str
    direction: str
    frequency_days: int
    occurrence_count: int | None = None
    source_event_id: str | None = None


@dataclass(frozen=True)
class CashFlowEntry:
    """One converted income or debit used in a particular daily record."""

    entry_id: str
    cash_date: date
    amount: Decimal
    currency: str
    home_currency_amount: Decimal
    direction: str
    source: CashFlowSource
    description: str = ""
    source_event_id: str | None = None


@dataclass(frozen=True)
class DailyForecastRecord:
    """One inclusive simulation day, with all values in the user's home currency."""

    forecast_date: date
    opening_balance: Decimal
    total_income: Decimal
    total_expenses: Decimal
    candidate_payments: Decimal
    closing_balance: Decimal
    minimum_required_balance: Decimal
    is_safe: bool
    cash_flow_entries: tuple[CashFlowEntry, ...]


@dataclass(frozen=True)
class ForecastResult:
    """A complete simulation result, without any recommendation or plan ranking."""

    safe: bool
    minimum_projected_balance: Decimal
    minimum_balance_date: date
    ending_balance: Decimal
    daily_records: tuple[DailyForecastRecord, ...]
    excluded_historical_event_ids: tuple[str, ...]
    ignored_pending_credit_event_ids: tuple[str, ...]
    unprojected_recurring_event_ids: tuple[str, ...]


class ForecastEngine:
    """Simulate known future cash events against one financial profile."""

    def __init__(self, exchange_rates: Mapping[tuple[date, str, str], ExchangeRate]) -> None:
        self._exchange_rates = dict(exchange_rates)

    @classmethod
    def from_repository(cls, repository: object) -> "ForecastEngine":
        """Create an engine from a ``DataRepository`` without coupling to CSV parsing."""
        rates = getattr(repository, "exchange_rates_by_key")
        return cls(rates)

    def forecast_request_context(
        self,
        context: RequestContext,
        candidate_payments: Sequence[CandidatePayment] = (),
        recurring_cash_flows: Sequence[RecurringCashFlow] = (),
        horizon_days: int = 90,
        start_date: date | None = None,
    ) -> ForecastResult:
        """Forecast a loaded request context; defaults to its request date.

        Existing detected recurring records are deliberately *not* extrapolated.  Pass
        explicit ``recurring_cash_flows`` when a schedule is known independently.
        """
        classified_events = context.financial_state.all_events
        return self.forecast(
            profile=context.financial_profile,
            events=tuple(item.event for item in classified_events),
            detected_recurring_events=classified_events,
            start_date=start_date or context.request.request_date,
            candidate_payments=candidate_payments,
            recurring_cash_flows=recurring_cash_flows,
            horizon_days=horizon_days,
        )

    def forecast(
        self,
        profile: FinancialProfile,
        events: Sequence[FinancialEvent],
        start_date: date,
        candidate_payments: Sequence[CandidatePayment] = (),
        recurring_cash_flows: Sequence[RecurringCashFlow] = (),
        horizon_days: int = 90,
        detected_recurring_events: Sequence[ClassifiedEvent] = (),
    ) -> ForecastResult:
        """Run an inclusive daily cash-flow simulation.

        A 90-day horizon creates 90 daily records: ``start_date`` through
        ``start_date + 89 days``. Historical settled records are retained only in the
        result's audit list and are not applied again to the current available balance.
        Pending debits reserve cash on their supplied cash date; pending credits are
        ignored as required by the challenge statement.
        """
        if horizon_days <= 0:
            raise ForecastInputError("horizon_days must be positive")
        end_date = start_date + timedelta(days=horizon_days - 1)
        flows_by_date: defaultdict[date, list[CashFlowEntry]] = defaultdict(list)
        historical: list[str] = []
        ignored_pending_credits: list[str] = []

        for event in events:
            cash_date = self._cash_date(event)
            if cash_date < start_date:
                historical.append(event.event_id)
                continue
            if cash_date > end_date:
                continue
            source = self._event_source(event)
            if source is None:
                if event.cash_state is CashState.PENDING and event.direction == "credit":
                    ignored_pending_credits.append(event.event_id)
                continue
            flows_by_date[cash_date].append(self._event_entry(event, cash_date, profile.home_currency, source))

        self._add_candidate_payments(flows_by_date, candidate_payments, profile.home_currency, start_date, end_date)
        self._add_recurring_cash_flows(flows_by_date, recurring_cash_flows, profile.home_currency, start_date, end_date)
        unprojected = tuple(
            item.event.event_id
            for item in detected_recurring_events
            if item.is_recurring and item.event.event_id not in {flow.source_event_id for flow in recurring_cash_flows}
        )
        return self._simulate(profile, start_date, horizon_days, flows_by_date, historical, ignored_pending_credits, unprojected)

    @staticmethod
    def _cash_date(event: FinancialEvent) -> date:
        """Use settlement date when supplied; event date is the explicit fallback."""
        return event.settlement_date or event.event_date

    @staticmethod
    def _event_source(event: FinancialEvent) -> CashFlowSource | None:
        if event.cash_state in {CashState.INACTIVE, CashState.NON_CASH}:
            return None
        if event.cash_state is CashState.PENDING:
            return CashFlowSource.PENDING_DEBIT_RESERVATION if event.direction == "debit" else None
        if event.cash_state is CashState.SCHEDULED:
            return CashFlowSource.FUTURE_SCHEDULED_EVENT
        if event.cash_state is CashState.SETTLED:
            return CashFlowSource.FUTURE_SETTLED_EVENT
        return None

    def _event_entry(self, event: FinancialEvent, cash_date: date, home_currency: str, source: CashFlowSource) -> CashFlowEntry:
        if event.amount is None:
            raise ForecastInputError(f"{event.event_id} has no numeric amount; image interpretation is outside this module")
        return CashFlowEntry(
            entry_id=event.event_id, cash_date=cash_date, amount=event.amount, currency=event.currency,
            home_currency_amount=self._to_home_currency(event.amount, event.currency, home_currency, cash_date),
            direction=event.direction, source=source, description=event.description, source_event_id=event.event_id,
        )

    def _add_candidate_payments(self, flows_by_date: defaultdict[date, list[CashFlowEntry]], payments: Iterable[CandidatePayment], home_currency: str, start_date: date, end_date: date) -> None:
        seen: set[str] = set()
        for payment in payments:
            if payment.payment_id in seen:
                raise ForecastInputError(f"Duplicate candidate payment ID: {payment.payment_id}")
            seen.add(payment.payment_id)
            if payment.amount < Decimal("0"):
                raise ForecastInputError(f"Candidate payment {payment.payment_id} has a negative amount")
            if not start_date <= payment.payment_date <= end_date:
                raise ForecastInputError(f"Candidate payment {payment.payment_id} falls outside the forecast horizon")
            currency = payment.currency or home_currency
            flows_by_date[payment.payment_date].append(CashFlowEntry(
                entry_id=payment.payment_id, cash_date=payment.payment_date, amount=payment.amount, currency=currency,
                home_currency_amount=self._to_home_currency(payment.amount, currency, home_currency, payment.payment_date),
                direction="debit", source=CashFlowSource.CANDIDATE_PAYMENT, description=payment.description,
            ))

    def _add_recurring_cash_flows(self, flows_by_date: defaultdict[date, list[CashFlowEntry]], schedules: Iterable[RecurringCashFlow], home_currency: str, start_date: date, end_date: date) -> None:
        seen: set[str] = set()
        for schedule in schedules:
            if schedule.flow_id in seen:
                raise ForecastInputError(f"Duplicate recurring flow ID: {schedule.flow_id}")
            seen.add(schedule.flow_id)
            if schedule.amount < Decimal("0") or schedule.frequency_days <= 0:
                raise ForecastInputError(f"Recurring flow {schedule.flow_id} has an invalid amount or interval")
            if schedule.direction not in {"credit", "debit"}:
                raise ForecastInputError(f"Recurring flow {schedule.flow_id} must be a credit or debit")
            occurrence_date = schedule.first_occurrence_date
            occurrence = 0
            while occurrence_date <= end_date and (schedule.occurrence_count is None or occurrence < schedule.occurrence_count):
                if occurrence_date >= start_date:
                    flows_by_date[occurrence_date].append(CashFlowEntry(
                        entry_id=f"{schedule.flow_id}:{occurrence + 1}", cash_date=occurrence_date, amount=schedule.amount,
                        currency=schedule.currency, home_currency_amount=self._to_home_currency(schedule.amount, schedule.currency, home_currency, occurrence_date),
                        direction=schedule.direction, source=CashFlowSource.RECURRING_CALLER_SCHEDULE, source_event_id=schedule.source_event_id,
                    ))
                occurrence_date += timedelta(days=schedule.frequency_days)
                occurrence += 1

    def _to_home_currency(self, amount: Decimal, from_currency: str, home_currency: str, cash_date: date) -> Decimal:
        if from_currency == home_currency:
            return amount
        rate = self._exchange_rates.get((cash_date, from_currency, home_currency))
        if rate is None:
            raise MissingExchangeRateError(f"Missing rate for {from_currency}->{home_currency} on {cash_date.isoformat()}")
        return amount * rate.rate

    @staticmethod
    def _simulate(profile: FinancialProfile, start_date: date, horizon_days: int, flows_by_date: Mapping[date, Sequence[CashFlowEntry]], historical: list[str], ignored_pending_credits: list[str], unprojected: tuple[str, ...]) -> ForecastResult:
        balance = profile.current_available_balance
        minimum_balance = balance
        minimum_date = start_date
        records: list[DailyForecastRecord] = []
        for offset in range(horizon_days):
            current_date = start_date + timedelta(days=offset)
            entries = tuple(flows_by_date.get(current_date, ()))
            income = sum((entry.home_currency_amount for entry in entries if entry.direction == "credit"), Decimal("0"))
            payments = sum((entry.home_currency_amount for entry in entries if entry.source is CashFlowSource.CANDIDATE_PAYMENT), Decimal("0"))
            expenses = sum((entry.home_currency_amount for entry in entries if entry.direction == "debit" and entry.source is not CashFlowSource.CANDIDATE_PAYMENT), Decimal("0"))
            opening = balance
            balance = opening + income - expenses - payments
            if balance < minimum_balance:
                minimum_balance, minimum_date = balance, current_date
            records.append(DailyForecastRecord(
                forecast_date=current_date, opening_balance=opening, total_income=income, total_expenses=expenses,
                candidate_payments=payments, closing_balance=balance, minimum_required_balance=profile.minimum_balance_to_keep,
                is_safe=balance >= profile.minimum_balance_to_keep, cash_flow_entries=entries,
            ))
        return ForecastResult(
            safe=all(record.is_safe for record in records), minimum_projected_balance=minimum_balance,
            minimum_balance_date=minimum_date, ending_balance=balance, daily_records=tuple(records),
            excluded_historical_event_ids=tuple(historical), ignored_pending_credit_event_ids=tuple(ignored_pending_credits),
            unprojected_recurring_event_ids=unprojected,
        )
