"""Deterministic payment-plan evaluation and ranking built on the forecast engine."""
from __future__ import annotations
from dataclasses import dataclass, replace
from datetime import date, timedelta
from decimal import Decimal
from itertools import combinations
from typing import Sequence
from affordability import calculate_amount_safe_to_pay
from data_loading import PaymentOption, RequestContext
from forecast import CandidatePayment, ForecastEngine, ForecastResult, RecurringCashFlow

@dataclass(frozen=True)
class ScheduledPayment:
    payment_date: date
    amount: Decimal

@dataclass(frozen=True)
class SpendingChange:
    action: str
    event_id: str
    new_amount: Decimal | None = None

@dataclass(frozen=True)
class PlanEvaluation:
    method: str
    payments: tuple[ScheduledPayment, ...]
    total_paid: Decimal
    payment_option_id: str | None
    spending_changes: tuple[SpendingChange, ...]
    safe: bool
    completes_by_deadline: bool
    forecast: ForecastResult | None
    @property
    def first_payment_date(self) -> date: return self.payments[0].payment_date
    @property
    def number_of_payments(self) -> int: return len(self.payments)

@dataclass(frozen=True)
class DecisionResult:
    request_id: str
    amount_safe_to_pay: Decimal
    earliest_date_for_full_payment: date | None
    affordability_status: str
    recommended_payment_method: str
    payment_plan: PlanEvaluation | None
    evaluated_plans: tuple[PlanEvaluation, ...]

class DecisionEngine:
    """Evaluate only supplied payment arrangements; no language model or heuristics."""
    def __init__(self, forecast_engine: ForecastEngine) -> None: self.engine = forecast_engine

    def decide(self, context: RequestContext, recurring_cash_flows: Sequence[RecurringCashFlow] = (), horizon_days: int = 90) -> DecisionResult:
        request, profile = context.request, context.financial_profile
        amount_safe = calculate_amount_safe_to_pay(self.engine, context, request.requested_amount, horizon_days=horizon_days, recurring_cash_flows=recurring_cash_flows).amount_safe_to_pay
        earliest = self._earliest_full_date(context, recurring_cash_flows, horizon_days)
        specs = self._base_specs(context, amount_safe, earliest, horizon_days)
        evaluations = [self._evaluate(context, *spec, (), recurring_cash_flows, horizon_days) for spec in specs]
        safe = [p for p in evaluations if p.safe and p.completes_by_deadline]
        if not safe:
            for changes in self._change_sets(context, recurring_cash_flows):
                changed_schedules = self._apply_changes(recurring_cash_flows, changes)
                batch = [self._evaluate(context, *spec, changes, changed_schedules, horizon_days) for spec in specs]
                evaluations.extend(batch)
                safe = [p for p in batch if p.safe and p.completes_by_deadline]
                if safe: break
        chosen = min(safe, key=self._rank_key) if safe else None
        if chosen is None:
            status, method = "not_affordable", "not_recommended"
        elif chosen.method == "full_payment" and not chosen.spending_changes and chosen.first_payment_date == request.request_date:
            status, method = "affordable_now", "full_payment"
        elif chosen.method == "wait": status, method = "affordable_later", "wait"
        else: status, method = "affordable_with_plan", chosen.method
        return DecisionResult(request.request_id, amount_safe, earliest, status, method, chosen, tuple(evaluations))

    def _earliest_full_date(self, context: RequestContext, schedules: Sequence[RecurringCashFlow], horizon: int) -> date | None:
        start, amount = context.request.request_date, context.request.requested_amount
        for offset in range(horizon):
            day = start + timedelta(days=offset)
            result = self.engine.forecast_request_context(context, (CandidatePayment(f"full-date:{offset}", day, amount),), schedules, horizon, start)
            if result.safe: return day
        return None

    def _base_specs(self, context: RequestContext, safe_today: Decimal, earliest: date | None, horizon: int):
        r, p = context.request, context.financial_profile
        specs = []
        if "full_payment" in p.payment_methods_user_will_consider:
            specs.append(("full_payment", (ScheduledPayment(r.request_date, r.requested_amount),), r.requested_amount, None))
            if earliest and earliest > r.request_date and earliest <= r.desired_completion_date:
                specs.append(("wait", (ScheduledPayment(earliest, r.requested_amount),), r.requested_amount, None))
        if r.allows_partial_payment and "partial_payment" in p.payment_methods_user_will_consider and Decimal("0") < safe_today < r.requested_amount and earliest and earliest <= r.desired_completion_date:
            specs.append(("partial_payment", (ScheduledPayment(r.request_date, safe_today), ScheduledPayment(earliest, r.requested_amount-safe_today)), r.requested_amount, None))
        end = r.request_date + timedelta(days=horizon-1)
        for option in context.payment_options:
            if option.payment_method != "installments" or "installments" not in p.payment_methods_user_will_consider: continue
            if p.max_installment_months is None or option.number_of_payments > p.max_installment_months or option.payment_frequency_days is None: continue
            payments = tuple(ScheduledPayment(option.first_payment_date + timedelta(days=i*option.payment_frequency_days), option.payment_amount) for i in range(option.number_of_payments))
            if payments[-1].payment_date > end or payments[-1].payment_date > r.desired_completion_date: continue
            if option.payment_amount * option.number_of_payments < option.total_payable_amount or option.total_payable_amount < r.requested_amount: continue
            specs.append(("installments", payments, option.total_payable_amount, option.payment_option_id))
        return specs

    def _evaluate(self, context, method, payments, total, option_id, changes, schedules, horizon):
        r = context.request
        candidates = tuple(CandidatePayment(f"{method}:{option_id or 'direct'}:{i}", x.payment_date, x.amount) for i,x in enumerate(payments))
        forecast = self.engine.forecast_request_context(context, candidates, schedules, horizon, r.request_date)
        return PlanEvaluation(method, payments, total, option_id, tuple(changes), forecast.safe, payments[-1].payment_date <= r.desired_completion_date, forecast)

    def _change_sets(self, context: RequestContext, schedules: Sequence[RecurringCashFlow]):
        p = context.financial_profile; by_id = {x.event.event_id:x.event for x in context.financial_state.all_events if x.is_recurring}
        actions=[]
        for flow in schedules:
            event=by_id.get(flow.source_event_id)
            if not event or event.category in p.expense_categories_to_protect: continue
            if event.flexibility in {"stoppable","reducible_or_stoppable"} and event.category in p.expense_categories_user_is_willing_to_stop:
                actions.append(SpendingChange("stop", event.event_id))
            if event.flexibility in {"reducible","reducible_or_stoppable"} and event.category in p.expense_categories_user_is_willing_to_reduce and event.minimum_allowed_amount is not None and event.minimum_allowed_amount < flow.amount:
                actions.append(SpendingChange("reduce_to", event.event_id, event.minimum_allowed_amount))
        actions.sort(key=lambda x:(x.event_id,x.action))
        for n in range(1, min(3,len(actions))+1):
            for combo in combinations(actions,n):
                if len({x.event_id for x in combo}) == n: yield combo

    @staticmethod
    def _apply_changes(schedules, changes):
        changed=[]; stop={x.event_id for x in changes if x.action=="stop"}; reduce={x.event_id:x.new_amount for x in changes if x.action=="reduce_to"}
        for flow in schedules:
            if flow.source_event_id in stop: continue
            changed.append(replace(flow, amount=reduce.get(flow.source_event_id, flow.amount)))
        return tuple(changed)

    @staticmethod
    def _rank_key(plan: PlanEvaluation):
        return (0 if plan.completes_by_deadline else 1, len(plan.spending_changes), plan.total_paid, plan.first_payment_date, plan.number_of_payments, plan.payment_option_id or "")
