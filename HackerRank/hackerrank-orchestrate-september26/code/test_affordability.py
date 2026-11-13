"""Focused tests for deterministic amount_safe_to_pay calculation."""

from datetime import date, timedelta
from decimal import Decimal
import unittest

from affordability import calculate_amount_safe_to_pay
from data_loading import ClassifiedEvent, FinancialEvent, FinancialProfile, FinancialState, Request, RequestContext
from forecast import ForecastEngine


START = date(2026, 1, 1)


def context(balance: str = "100", minimum: str = "20", events: tuple[FinancialEvent, ...] = ()) -> RequestContext:
    profile = FinancialProfile("user", "USD", Decimal(balance), Decimal(minimum), (), (), (), (), (), None)
    classified = tuple(ClassifiedEvent(event, False) for event in events)
    state = FinancialState(classified, (), (), (), (), (), classified, (), (), ())
    request = Request("request", "user", START, "purchase", Decimal("999"), START + timedelta(days=30), True, "")
    return RequestContext(request, profile, state, (), (), ())


def expense(event_id: str, amount: str, event_date: date, flexibility: str = "fixed") -> FinancialEvent:
    return FinancialEvent(event_id, "user", "expense", event_id, "test", "debit", Decimal(amount), "USD", event_date, event_date, "scheduled", None, flexibility, None)


class AmountSafeToPayTests(unittest.TestCase):
    def test_requested_amount_fully_safe_today(self) -> None:
        result = calculate_amount_safe_to_pay(ForecastEngine({}), context(), Decimal("50"))
        self.assertEqual(Decimal("50"), result.amount_safe_to_pay)
        self.assertTrue(result.payment_forecast.safe)
        self.assertEqual(Decimal("50"), result.payment_forecast.daily_records[0].candidate_payments)

    def test_only_fraction_is_safe_today(self) -> None:
        result = calculate_amount_safe_to_pay(ForecastEngine({}), context(), Decimal("200"))
        self.assertEqual(Decimal("80"), result.amount_safe_to_pay)

    def test_zero_when_no_positive_payment_is_safe(self) -> None:
        result = calculate_amount_safe_to_pay(ForecastEngine({}), context(balance="20"), Decimal("1"))
        self.assertEqual(Decimal("0"), result.amount_safe_to_pay)

    def test_future_cash_flow_limits_seemingly_safe_immediate_payment(self) -> None:
        future_expense = expense("upcoming-bill", "50", START + timedelta(days=10))
        result = calculate_amount_safe_to_pay(ForecastEngine({}), context(events=(future_expense,)), Decimal("80"))
        self.assertEqual(Decimal("30"), result.amount_safe_to_pay)

    def test_requested_amount_is_an_upper_cap(self) -> None:
        result = calculate_amount_safe_to_pay(ForecastEngine({}), context(), Decimal("0.01"))
        self.assertEqual(Decimal("0.01"), result.amount_safe_to_pay)

    def test_decimal_precision_is_preserved(self) -> None:
        future_expense = expense("precise", "0.20", START + timedelta(days=1))
        result = calculate_amount_safe_to_pay(ForecastEngine({}), context(balance="1.00", minimum="0.10", events=(future_expense,)), Decimal("1.00"))
        self.assertEqual(Decimal("0.70"), result.amount_safe_to_pay)

    def test_flexible_event_is_not_changed(self) -> None:
        flexible_expense = expense("flexible", "10", START + timedelta(days=1), flexibility="reducible")
        result = calculate_amount_safe_to_pay(ForecastEngine({}), context(events=(flexible_expense,)), Decimal("100"))
        self.assertEqual(Decimal("70"), result.amount_safe_to_pay)
        self.assertEqual(Decimal("10"), result.payment_forecast.daily_records[1].total_expenses)


if __name__ == "__main__":
    unittest.main()
