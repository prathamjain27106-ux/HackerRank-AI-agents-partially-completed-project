"""Deterministic calculation of today's maximum safe payment.

This module delegates all cash-flow simulation to ``ForecastEngine``.  It neither
changes forecast semantics nor recommends a payment method or spending change.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Sequence

from data_loading import RequestContext
from forecast import CandidatePayment, ForecastEngine, ForecastInputError, ForecastResult, RecurringCashFlow


class AffordabilityInputError(ValueError):
    """Raised when an amount-safe-to-pay input conflicts with the dataset contract."""


@dataclass(frozen=True)
class SafePaymentResult:
    """The deterministic maximum payment and its verified forecast evidence."""

    amount_safe_to_pay: Decimal
    baseline_forecast: ForecastResult
    payment_forecast: ForecastResult


def calculate_amount_safe_to_pay(
    engine: ForecastEngine,
    request_context: RequestContext,
    requested_amount: Decimal,
    forecast_start_date: date | None = None,
    payment_currency: str | None = None,
    horizon_days: int = 90,
    recurring_cash_flows: Sequence[RecurringCashFlow] = (),
) -> SafePaymentResult:
    """Return the largest home-currency payment safe on the forecast start date.

    The challenge contract states that request and output amounts use the user's home
    currency.  Therefore a non-home ``payment_currency`` is rejected rather than
    applying an inverse FX conversion with arbitrary Decimal rounding.  The selected
    amount is always verified by passing it as a ``CandidatePayment`` to the supplied
    engine.

    With a start-day debit and fixed forecast inputs, each daily closing balance is
    linear and non-increasing in the candidate amount.  The exact headroom is thus the
    smallest baseline closing-balance surplus above the required minimum; no monetary
    iteration or arbitrary rounding is needed.
    """
    if requested_amount < Decimal("0"):
        raise AffordabilityInputError("requested_amount must not be negative")
    profile = request_context.financial_profile
    currency = payment_currency or profile.home_currency
    if currency != profile.home_currency:
        raise AffordabilityInputError(
            "amount_safe_to_pay must be calculated in the user's home currency"
        )
    start_date = forecast_start_date or request_context.request.request_date
    try:
        baseline = engine.forecast_request_context(
            request_context,
            recurring_cash_flows=recurring_cash_flows,
            horizon_days=horizon_days,
            start_date=start_date,
        )
    except ForecastInputError:
        raise

    if baseline.safe:
        headroom = min(
            record.closing_balance - record.minimum_required_balance
            for record in baseline.daily_records
        )
        amount = min(requested_amount, max(Decimal("0"), headroom))
    else:
        amount = Decimal("0")

    payment = CandidatePayment(
        payment_id=f"amount-safe-to-pay:{request_context.request.request_id}",
        payment_date=start_date,
        amount=amount,
        currency=currency,
        description="Deterministic amount_safe_to_pay safety check",
    )
    payment_forecast = engine.forecast_request_context(
        request_context,
        candidate_payments=(payment,),
        recurring_cash_flows=recurring_cash_flows,
        horizon_days=horizon_days,
        start_date=start_date,
    )
    if amount > Decimal("0") and not payment_forecast.safe:
        raise RuntimeError("Calculated amount_safe_to_pay failed its forecast verification")
    return SafePaymentResult(amount, baseline, payment_forecast)
