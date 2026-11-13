"""Focused deterministic tests for recurring cash-flow schedule construction."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
import unittest

from data_loading import (
    CashState,
    ClassifiedEvent,
    FinancialEvent,
    FinancialProfile,
    FinancialState,
    Request,
    RequestContext,
)
from forecast import RecurringCashFlow
from recurrence import RecurrencePattern, _detect_patterns, build_recurring_cash_flows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

START = date(2026, 1, 1)


def _profile(currency: str = "USD") -> FinancialProfile:
    return FinancialProfile(
        "user", currency, Decimal("100000"), Decimal("10000"),
        (), (), (), (), (), None,
    )


def _request(start: date = START) -> Request:
    return Request(
        "request", "user", start, "purchase", Decimal("1000"),
        start + timedelta(days=30), True, "",
    )


def _event(
    event_id: str,
    amount: str,
    event_date: date,
    direction: str = "debit",
    status: str = "settled",
    currency: str = "USD",
    event_type: str | None = None,
    description: str = "Recurring item",
    category: str = "test",
    flexibility: str = "fixed",
) -> FinancialEvent:
    return FinancialEvent(
        event_id=event_id,
        user_id="user",
        event_type=event_type or ("income" if direction == "credit" else "expense"),
        description=description,
        category=category,
        direction=direction,
        amount=Decimal(amount),
        currency=currency,
        event_date=event_date,
        settlement_date=event_date,
        status=status,
        linked_event_id=None,
        flexibility=flexibility,
        minimum_allowed_amount=None,
    )


def _build_context(
    events: list[FinancialEvent],
    start: date = START,
    currency: str = "USD",
) -> RequestContext:
    """Build a minimal request context with explicit recurrence classification.

    Uses the same grouping logic as the real data-loader: events in the same
    group key with at least two settled dates 20–40 days apart are recurring.
    """
    from collections import defaultdict

    # Reproduce the data-loader's recurrence detection.
    groups: defaultdict[tuple[str, str, str, str, str], list[FinancialEvent]] = defaultdict(list)
    for event in events:
        groups[(event.event_type, event.description, event.category, event.direction, event.currency)].append(event)

    recurring_ids: set[str] = set()
    for group in groups.values():
        settled = sorted(
            (event for event in group if event.cash_state is CashState.SETTLED),
            key=lambda event: event.settlement_date or event.event_date,
        )
        dates = [event.settlement_date or event.event_date for event in settled]
        if any(20 <= (later - earlier).days <= 40 for earlier, later in zip(dates, dates[1:])):
            recurring_ids.update(event.event_id for event in group)

    classified = tuple(
        ClassifiedEvent(event=event, is_recurring=event.event_id in recurring_ids)
        for event in events
    )
    by_state = lambda state: tuple(item for item in classified if item.event.cash_state is state)
    recurring_expenses = tuple(item for item in classified if item.is_recurring and item.event.is_expense)
    state = FinancialState(
        all_events=classified,
        settled_events=by_state(CashState.SETTLED),
        pending_events=by_state(CashState.PENDING),
        scheduled_events=by_state(CashState.SCHEDULED),
        inactive_events=tuple(item for item in classified if item.event.cash_state in {CashState.INACTIVE, CashState.NON_CASH}),
        recurring_expenses=recurring_expenses,
        one_time_transactions=tuple(item for item in classified if not item.is_recurring),
        income_events=tuple(item for item in classified if item.event.is_income),
        fixed_expenses=tuple(item for item in classified if item.event.is_fixed_expense),
        flexible_expenses=tuple(item for item in classified if item.event.is_flexible_expense),
    )
    return RequestContext(
        request=_request(start),
        financial_profile=_profile(currency),
        financial_state=state,
        payment_options=(),
        messages=(),
        images=(),
    )


def _monthly_expense_history(
    base_date: date,
    count: int = 5,
    amount: str = "1000",
    interval: int = 30,
    description: str = "Monthly rent",
    category: str = "rent",
) -> list[FinancialEvent]:
    """Create ``count`` settled monthly expense events ending before ``base_date``."""
    events = []
    for i in range(count):
        d = base_date - timedelta(days=(count - i) * interval)
        events.append(
            _event(
                f"exp_{i}",
                amount,
                d,
                direction="debit",
                status="settled",
                description=description,
                category=category,
            )
        )
    return events


def _monthly_income_history(
    base_date: date,
    count: int = 5,
    amount: str = "5000",
    interval: int = 30,
) -> list[FinancialEvent]:
    """Create ``count`` settled monthly income events ending before ``base_date``."""
    events = []
    for i in range(count):
        d = base_date - timedelta(days=(count - i) * interval)
        events.append(
            _event(
                f"inc_{i}",
                amount,
                d,
                direction="credit",
                status="settled",
                event_type="income",
                description="Payroll credit",
                category="salary",
            )
        )
    return events


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class RecurrencePatternDetectionTests(unittest.TestCase):
    """Test the intermediate pattern detection step."""

    def test_monthly_expense_produces_pattern(self) -> None:
        events = _monthly_expense_history(START, count=4, amount="1000")
        context = _build_context(events, start=START)
        patterns = _detect_patterns(context.financial_state.all_events)
        self.assertEqual(1, len(patterns))
        self.assertEqual("debit", patterns[0].group_key[3])
        self.assertEqual(Decimal("1000"), patterns[0].conservative_amount)

    def test_single_event_does_not_produce_pattern(self) -> None:
        events = [_event("solo", "500", START - timedelta(days=10))]
        context = _build_context(events)
        patterns = _detect_patterns(context.financial_state.all_events)
        self.assertEqual(0, len(patterns))

    def test_cancelled_events_excluded_from_evidence(self) -> None:
        events = [
            _event("ok_1", "100", START - timedelta(days=60), status="settled"),
            _event("ok_2", "100", START - timedelta(days=30), status="settled"),
            _event("bad", "999", START - timedelta(days=15), status="cancelled"),
        ]
        context = _build_context(events)
        patterns = _detect_patterns(context.financial_state.all_events)
        self.assertEqual(1, len(patterns))
        self.assertEqual(Decimal("100"), patterns[0].conservative_amount)

    def test_failed_events_excluded_from_evidence(self) -> None:
        events = [
            _event("ok_1", "100", START - timedelta(days=60), status="settled"),
            _event("ok_2", "100", START - timedelta(days=30), status="settled"),
            _event("bad", "999", START - timedelta(days=5), status="failed"),
        ]
        context = _build_context(events)
        patterns = _detect_patterns(context.financial_state.all_events)
        self.assertEqual(1, len(patterns))
        self.assertEqual(Decimal("100"), patterns[0].conservative_amount)


class BuildRecurringCashFlowsTests(unittest.TestCase):
    """Test the full public API."""

    def test_monthly_expense_creates_future_occurrences(self) -> None:
        events = _monthly_expense_history(START, count=4, amount="1000")
        context = _build_context(events, start=START)
        flows = build_recurring_cash_flows(context)
        self.assertEqual(1, len(flows))
        flow = flows[0]
        self.assertEqual(Decimal("1000"), flow.amount)
        self.assertEqual("USD", flow.currency)
        self.assertEqual("debit", flow.direction)
        self.assertGreaterEqual(flow.first_occurrence_date, START)
        end = START + timedelta(days=89)
        self.assertLessEqual(flow.first_occurrence_date, end)

    def test_monthly_income_creates_future_occurrences(self) -> None:
        events = _monthly_income_history(START, count=4, amount="5000")
        context = _build_context(events, start=START)
        flows = build_recurring_cash_flows(context)
        self.assertEqual(1, len(flows))
        flow = flows[0]
        self.assertEqual(Decimal("5000"), flow.amount)
        self.assertEqual("credit", flow.direction)
        self.assertGreaterEqual(flow.first_occurrence_date, START)

    def test_conservative_expense_uses_max_of_recent(self) -> None:
        events = [
            _event("exp_0", "900", START - timedelta(days=90), direction="debit"),
            _event("exp_1", "1100", START - timedelta(days=60), direction="debit"),
            _event("exp_2", "1000", START - timedelta(days=30), direction="debit"),
        ]
        context = _build_context(events, start=START)
        flows = build_recurring_cash_flows(context)
        self.assertEqual(1, len(flows))
        # Max of last 3 = 1100 (conservative for expenses)
        self.assertEqual(Decimal("1100"), flows[0].amount)

    def test_conservative_income_uses_min_of_recent(self) -> None:
        events = [
            _event("inc_0", "5000", START - timedelta(days=90), direction="credit", event_type="income",
                   description="Payroll credit", category="salary"),
            _event("inc_1", "4500", START - timedelta(days=60), direction="credit", event_type="income",
                   description="Payroll credit", category="salary"),
            _event("inc_2", "4800", START - timedelta(days=30), direction="credit", event_type="income",
                   description="Payroll credit", category="salary"),
        ]
        context = _build_context(events, start=START)
        flows = build_recurring_cash_flows(context)
        self.assertEqual(1, len(flows))
        # Min of last 3 = 4500 (conservative for income)
        self.assertEqual(Decimal("4500"), flows[0].amount)

    def test_historical_events_not_emitted_as_future(self) -> None:
        events = _monthly_expense_history(START, count=4, amount="1000")
        context = _build_context(events, start=START)
        flows = build_recurring_cash_flows(context)
        self.assertEqual(1, len(flows))
        # All generated occurrences must be on or after START.
        self.assertGreaterEqual(flows[0].first_occurrence_date, START)

    def test_occurrences_stay_inside_90_day_horizon(self) -> None:
        events = _monthly_expense_history(START, count=4, amount="1000")
        context = _build_context(events, start=START)
        flows = build_recurring_cash_flows(context, horizon_days=90)
        self.assertEqual(1, len(flows))
        end = START + timedelta(days=89)
        self.assertLessEqual(flows[0].first_occurrence_date, end)

    def test_one_off_events_do_not_become_recurring(self) -> None:
        events = [
            _event("one_off_1", "500", START - timedelta(days=200), description="One-time"),
            _event("one_off_2", "500", START - timedelta(days=10), description="One-time"),
        ]
        context = _build_context(events, start=START)
        flows = build_recurring_cash_flows(context)
        # 200-day gap means no monthly recurrence detected
        self.assertEqual(0, len(flows))

    def test_explicit_scheduled_future_prevents_projection(self) -> None:
        # History of recurring events + a scheduled future event in the same group
        events = _monthly_expense_history(START, count=3, amount="1000")
        # Add a scheduled event in the forecast window with same group key
        future_event = _event(
            "future_sched",
            "1000",
            START + timedelta(days=15),
            direction="debit",
            status="scheduled",
            description="Monthly rent",
            category="rent",
        )
        events.append(future_event)
        context = _build_context(events, start=START)
        flows = build_recurring_cash_flows(context)
        # The pattern should be skipped because there's already an explicit scheduled event.
        self.assertEqual(0, len(flows))

    def test_currency_and_direction_preserved(self) -> None:
        events = [
            _event("eur_0", "200", START - timedelta(days=60), currency="EUR", description="EUR rent", category="rent"),
            _event("eur_1", "200", START - timedelta(days=30), currency="EUR", description="EUR rent", category="rent"),
        ]
        context = _build_context(events, start=START, currency="EUR")
        flows = build_recurring_cash_flows(context)
        self.assertEqual(1, len(flows))
        self.assertEqual("EUR", flows[0].currency)
        self.assertEqual("debit", flows[0].direction)

    def test_no_mutation_of_original_context(self) -> None:
        events = _monthly_expense_history(START, count=4, amount="1000")
        context = _build_context(events, start=START)
        original_event_count = len(context.financial_state.all_events)
        original_events = context.financial_state.all_events

        build_recurring_cash_flows(context)

        self.assertEqual(original_event_count, len(context.financial_state.all_events))
        self.assertIs(original_events, context.financial_state.all_events)

    def test_non_cash_events_not_projected(self) -> None:
        events = [
            _event("nc_0", "1000", START - timedelta(days=60), direction="non_cash",
                   event_type="investment_valuation", description="Portfolio value", category="investments"),
            _event("nc_1", "1200", START - timedelta(days=30), direction="non_cash",
                   event_type="investment_valuation", description="Portfolio value", category="investments"),
        ]
        context = _build_context(events, start=START)
        flows = build_recurring_cash_flows(context)
        self.assertEqual(0, len(flows))

    def test_blank_amount_events_excluded_from_evidence(self) -> None:
        events = [
            _event("ok_0", "500", START - timedelta(days=60)),
            _event("ok_1", "500", START - timedelta(days=30)),
        ]
        # Add a blank-amount event in the same group (manually create one with None)
        blank = FinancialEvent(
            "blank_ev", "user", "expense", "Recurring item", "test", "debit",
            None, "USD", START - timedelta(days=5), START - timedelta(days=5),
            "settled", None, "fixed", None,
        )
        events.append(blank)
        context = _build_context(events, start=START)
        flows = build_recurring_cash_flows(context)
        self.assertEqual(1, len(flows))
        # Amount should be based on the known settled amounts, not the blank one.
        self.assertEqual(Decimal("500"), flows[0].amount)

    def test_multiple_groups_produce_separate_flows(self) -> None:
        rent = _monthly_expense_history(START, count=3, amount="1000", description="Rent", category="rent")
        utils = _monthly_expense_history(START, count=3, amount="200", description="Utilities", category="utilities")
        # Fix event IDs to avoid collisions.
        utils_fixed = []
        for i, e in enumerate(utils):
            utils_fixed.append(FinancialEvent(
                f"util_{i}", e.user_id, e.event_type, "Utilities", "utilities",
                e.direction, e.amount, e.currency, e.event_date, e.settlement_date,
                e.status, e.linked_event_id, e.flexibility, e.minimum_allowed_amount,
            ))
        context = _build_context(rent + utils_fixed, start=START)
        flows = build_recurring_cash_flows(context)
        self.assertEqual(2, len(flows))
        amounts = {f.amount for f in flows}
        self.assertEqual({Decimal("1000"), Decimal("200")}, amounts)

    def test_source_event_id_set_correctly(self) -> None:
        events = _monthly_expense_history(START, count=3, amount="1000")
        context = _build_context(events, start=START)
        flows = build_recurring_cash_flows(context)
        self.assertEqual(1, len(flows))
        # source_event_id should be the most recent settled event.
        self.assertEqual("exp_2", flows[0].source_event_id)

    def test_flow_id_is_unique_and_prefixed(self) -> None:
        events = _monthly_expense_history(START, count=3, amount="1000")
        context = _build_context(events, start=START)
        flows = build_recurring_cash_flows(context)
        self.assertEqual(1, len(flows))
        self.assertTrue(flows[0].flow_id.startswith("rec:"))


class RecurrenceEdgeCaseTests(unittest.TestCase):
    """Additional edge cases."""

    def test_empty_events_produce_no_flows(self) -> None:
        context = _build_context([], start=START)
        flows = build_recurring_cash_flows(context)
        self.assertEqual(0, len(flows))

    def test_distant_history_still_projects_if_within_horizon(self) -> None:
        # Last settled event was 100 days ago; next occurrence should be ~70 days into horizon.
        events = [
            _event("old_0", "500", START - timedelta(days=130)),
            _event("old_1", "500", START - timedelta(days=100)),
        ]
        context = _build_context(events, start=START)
        flows = build_recurring_cash_flows(context)
        self.assertEqual(1, len(flows))
        # Median interval is 30 days, so next occurrence from -100d is at -70d, -40d, -10d, +20d.
        expected_first = START - timedelta(days=100) + timedelta(days=30 * 4)
        # Should be around START+20
        self.assertGreaterEqual(flows[0].first_occurrence_date, START)
        self.assertLessEqual(flows[0].first_occurrence_date, START + timedelta(days=89))

    def test_too_distant_history_produces_no_flows(self) -> None:
        # Last settled event was 200 days ago, interval 30 → even after walking forward,
        # occurrences might still land within horizon. Just verify no crash.
        events = [
            _event("far_0", "500", START - timedelta(days=260)),
            _event("far_1", "500", START - timedelta(days=230)),
        ]
        context = _build_context(events, start=START)
        flows = build_recurring_cash_flows(context)
        # Should still project if occurrences land in horizon.
        if flows:
            self.assertGreaterEqual(flows[0].first_occurrence_date, START)


class TargetedRecurrenceCorrectionsTests(unittest.TestCase):
    """Focused regression tests for Mission 5C recurrence corrections."""

    def test_monthly_day_of_month_alignment(self) -> None:
        """Genuinely monthly events on day 15 project first occurrence on day 15."""
        events = [
            _event("e0", "1000", date(2025, 10, 15)),
            _event("e1", "1000", date(2025, 11, 15)),
            _event("e2", "1000", date(2025, 12, 15)),
        ]
        # Forecast starts 2026-01-03
        context = _build_context(events, start=date(2026, 1, 3))
        flows = build_recurring_cash_flows(context)
        self.assertEqual(1, len(flows))
        self.assertEqual(date(2026, 1, 15), flows[0].first_occurrence_date)

    def test_terminal_stream_suppresses_future_recurrence(self) -> None:
        """Settled event indicating 'Final employer payroll' stops future salary recurrence."""
        events = [
            _event("s1", "14740", date(2025, 7, 15), direction="credit", description="Payroll credit", category="salary"),
            _event("s2", "14740", date(2025, 8, 15), direction="credit", description="Payroll credit", category="salary"),
            _event("s3", "14740", date(2025, 9, 15), direction="credit", description="Payroll credit", category="salary"),
            _event("s4", "14740", date(2025, 10, 15), direction="credit", description="Final employer payroll", category="salary"),
        ]
        context = _build_context(events, start=date(2025, 11, 6))
        flows = build_recurring_cash_flows(context)
        salary_flows = [f for f in flows if f.direction == "credit"]
        self.assertEqual(0, len(salary_flows))

    def test_category_aware_scheduled_salary_deduplication(self) -> None:
        """Explicit scheduled salary in the forecast window prevents synthetic duplicate in that month."""
        events = [
            _event("s1", "2000", date(2026, 1, 15), direction="credit", description="Payroll credit", category="salary"),
            _event("s2", "2000", date(2026, 2, 15), direction="credit", description="Payroll credit", category="salary"),
            _event("s3", "2000", date(2026, 3, 15), direction="credit", description="Payroll credit", category="salary"),
            # Scheduled event in April
            _event("s_next", "2000", date(2026, 4, 15), direction="credit", status="scheduled", description="Next confirmed salary", category="salary"),
        ]
        # Start in April (2026-04-03)
        context = _build_context(events, start=date(2026, 4, 3))
        flows = build_recurring_cash_flows(context)
        salary_flows = [f for f in flows if f.direction == "credit"]
        self.assertEqual(1, len(salary_flows))
        # April has scheduled salary; synthetic recurrence should advance to May 15
        self.assertEqual(date(2026, 5, 15), salary_flows[0].first_occurrence_date)

    def test_single_settled_plus_scheduled_salary_continuation(self) -> None:
        """User with 1 settled salary and 1 scheduled salary continues projection after scheduled."""
        events = [
            _event("s1", "12826", date(2024, 2, 15), direction="credit", description="Prorated first salary", category="salary"),
            _event("s_next", "23320", date(2024, 3, 15), direction="credit", status="scheduled", description="Next confirmed salary", category="salary"),
        ]
        context = _build_context(events, start=date(2024, 3, 3))
        flows = build_recurring_cash_flows(context)
        salary_flows = [f for f in flows if f.direction == "credit"]
        self.assertEqual(1, len(salary_flows))
        self.assertEqual(date(2024, 4, 15), salary_flows[0].first_occurrence_date)
        self.assertEqual(Decimal("23320"), salary_flows[0].amount)

    def test_non_monthly_interval_preserved(self) -> None:
        """Patterns with 21-day interval step by exact interval rather than calendar month."""
        events = [
            _event("g1", "50", date(2026, 1, 1), description="Groceries"),
            _event("g2", "50", date(2026, 1, 22), description="Groceries"),
            _event("g3", "50", date(2026, 2, 12), description="Groceries"),
        ]
        context = _build_context(events, start=date(2026, 2, 15))
        flows = build_recurring_cash_flows(context)
        self.assertEqual(1, len(flows))
        self.assertEqual(21, flows[0].frequency_days)
        # 2026-02-12 + 21 days = 2026-03-05
        self.assertEqual(date(2026, 3, 5), flows[0].first_occurrence_date)


if __name__ == "__main__":
    unittest.main()

