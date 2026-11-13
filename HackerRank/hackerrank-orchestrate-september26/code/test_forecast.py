"""Unit tests for deterministic, recommendation-free forecast behavior."""

from datetime import date
from dataclasses import replace
from decimal import Decimal
import unittest

from data_loading import ExchangeRate, FinancialEvent, FinancialProfile
from forecast import CandidatePayment, ForecastEngine, RecurringCashFlow


START = date(2026, 1, 1)


def profile(balance: str = "100", minimum: str = "20", currency: str = "USD") -> FinancialProfile:
    return FinancialProfile("user", currency, Decimal(balance), Decimal(minimum), (), (), (), (), (), None)


def event(event_id: str, amount: str, event_date: date, direction: str, status: str = "scheduled", currency: str = "USD", event_type: str | None = None) -> FinancialEvent:
    return FinancialEvent(
        event_id, "user", event_type or ("income" if direction == "credit" else "expense"), event_id, "test", direction,
        Decimal(amount), currency, event_date, event_date, status, None, "fixed", None,
    )


class ForecastTests(unittest.TestCase):
    def test_simple_income_and_expense_case(self) -> None:
        result = ForecastEngine({}).forecast(profile(), [event("income", "50", START, "credit"), event("expense", "30", START, "debit")], START, horizon_days=2)
        self.assertTrue(result.safe)
        self.assertEqual(Decimal("120"), result.ending_balance)
        self.assertEqual(Decimal("50"), result.daily_records[0].total_income)
        self.assertEqual(Decimal("30"), result.daily_records[0].total_expenses)

    def test_minimum_balance_violation(self) -> None:
        result = ForecastEngine({}).forecast(profile(), [event("expense", "81", START, "debit")], START)
        self.assertFalse(result.safe)
        self.assertEqual(Decimal("19"), result.minimum_projected_balance)
        self.assertEqual(START, result.minimum_balance_date)

    def test_safe_forecast(self) -> None:
        result = ForecastEngine({}).forecast(profile(), [event("expense", "80", START, "debit")], START)
        self.assertTrue(result.safe)
        self.assertEqual(Decimal("20"), result.ending_balance)

    def test_candidate_payment_can_cause_violation(self) -> None:
        result = ForecastEngine({}).forecast(profile(), [], START, [CandidatePayment("payment", START, Decimal("81"))])
        self.assertFalse(result.safe)
        self.assertEqual(Decimal("81"), result.daily_records[0].candidate_payments)

    def test_explicit_recurring_event(self) -> None:
        schedule = RecurringCashFlow("rent", START, Decimal("10"), "USD", "debit", 30, occurrence_count=3)
        result = ForecastEngine({}).forecast(profile(), [], START, recurring_cash_flows=[schedule], horizon_days=61)
        self.assertEqual(Decimal("70"), result.ending_balance)
        self.assertEqual(Decimal("10"), result.daily_records[30].total_expenses)

    def test_multiple_events_on_same_day(self) -> None:
        result = ForecastEngine({}).forecast(profile(), [event("income", "20", START, "credit"), event("expense-one", "5", START, "debit"), event("expense-two", "6", START, "debit")], START)
        self.assertEqual(Decimal("20"), result.daily_records[0].total_income)
        self.assertEqual(Decimal("11"), result.daily_records[0].total_expenses)
        self.assertEqual(Decimal("109"), result.ending_balance)

    def test_pending_debit_is_reserved_but_pending_credit_is_ignored(self) -> None:
        pending_debit = event("pending-debit", "10", START, "debit", status="pending")
        pending_credit = event("pending-credit", "50", START, "credit", status="pending")
        result = ForecastEngine({}).forecast(profile(), [pending_debit, pending_credit], START)
        self.assertEqual(Decimal("90"), result.ending_balance)
        self.assertEqual(("pending-credit",), result.ignored_pending_credit_event_ids)

    def test_decimal_precision(self) -> None:
        result = ForecastEngine({}).forecast(profile(balance="0", minimum="0"), [event("income", "0.30", START, "credit"), event("first", "0.10", START, "debit"), event("second", "0.20", START, "debit")], START)
        self.assertEqual(Decimal("0.00"), result.ending_balance)

    def test_currency_conversion(self) -> None:
        rate = ExchangeRate(START, "EUR", "USD", Decimal("1.25"))
        result = ForecastEngine({(START, "EUR", "USD"): rate}).forecast(profile(), [event("foreign", "10", START, "debit", currency="EUR")], START)
        self.assertEqual(Decimal("87.50"), result.ending_balance)

    def test_settlement_date_controls_timing_and_fx_conversion(self) -> None:
        settlement_date = date(2026, 1, 2)
        foreign_event = replace(event("foreign", "10", START, "debit", currency="EUR"), settlement_date=settlement_date)
        rate = ExchangeRate(settlement_date, "EUR", "USD", Decimal("1.25"))
        result = ForecastEngine({(settlement_date, "EUR", "USD"): rate}).forecast(profile(), [foreign_event], START, horizon_days=2)
        self.assertEqual(Decimal("0"), result.daily_records[0].total_expenses)
        self.assertEqual(Decimal("12.50"), result.daily_records[1].total_expenses)

    def test_historical_and_inactive_events_are_excluded_and_final_day_is_included(self) -> None:
        historical = event("historical", "50", date(2025, 12, 31), "debit", status="settled")
        cancelled = event("cancelled", "50", START, "debit", status="cancelled")
        failed = event("failed", "50", START, "debit", status="failed")
        final_day = event("final-day", "10", date(2026, 1, 2), "debit")
        result = ForecastEngine({}).forecast(profile(), [historical, cancelled, failed, final_day], START, horizon_days=2)
        self.assertEqual(("historical",), result.excluded_historical_event_ids)
        self.assertEqual(Decimal("100"), result.daily_records[0].closing_balance)
        self.assertEqual(Decimal("90"), result.daily_records[1].closing_balance)


if __name__ == "__main__":
    unittest.main()
