"""Deterministic recurring cash-flow schedule constructor.

This module converts the data-loading layer's detected recurring event patterns
into explicit ``RecurringCashFlow`` objects that the forecast engine can simulate.

It does not parse CSVs, call external services, or make financial recommendations.
It does not mutate any repository or request-context state.
"""

from __future__ import annotations

import calendar
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from statistics import median
from typing import Sequence

from data_loading import CashState, ClassifiedEvent, FinancialEvent, RequestContext
from forecast import RecurringCashFlow


@dataclass(frozen=True)
class RecurrencePattern:
    """Intermediate deterministic summary of one recurring group.

    Exposed for diagnostics and testing; callers typically use
    :func:`build_recurring_cash_flows` directly.
    """

    group_key: tuple[str, str, str, str, str]
    settled_events: tuple[FinancialEvent, ...]
    settled_dates: tuple[date, ...]
    settled_amounts: tuple[Decimal, ...]
    median_interval_days: int
    conservative_amount: Decimal
    most_recent_settled_date: date
    representative_event: FinancialEvent
    is_monthly: bool = False
    canonical_day_of_month: int | None = None


def _cash_date(event: FinancialEvent) -> date:
    """Use settlement date when supplied; event date is the explicit fallback."""
    return event.settlement_date or event.event_date


def _group_key(event: FinancialEvent) -> tuple[str, str, str, str, str]:
    """Grouping key matching the data-loader's ``_recurring_event_ids``."""
    return (event.event_type, event.description, event.category, event.direction, event.currency)


def _add_months(d: date, n: int, target_day: int) -> date:
    """Advance date d by n calendar months, pinning to target_day (or last day of month)."""
    month = d.month - 1 + n
    year = d.year + month // 12
    month = month % 12 + 1
    max_days = calendar.monthrange(year, month)[1]
    day = min(target_day, max_days)
    return date(year, month, day)


def _detect_patterns(
    classified_events: Sequence[ClassifiedEvent],
) -> list[RecurrencePattern]:
    """Build intermediate recurrence patterns from already-classified events.

    Only groups that the data-loader has already flagged as recurring (via
    ``ClassifiedEvent.is_recurring``) are considered. Within each group, only
    settled events with a known amount form the basis for future projection.
    """
    groups: defaultdict[tuple[str, str, str, str, str], list[FinancialEvent]] = defaultdict(list)
    for item in classified_events:
        if not item.is_recurring:
            continue
        groups[_group_key(item.event)].append(item.event)

    patterns: list[RecurrencePattern] = []
    for key, events in groups.items():
        # Only settled events with known amounts are valid recurrence evidence.
        settled = sorted(
            (
                event
                for event in events
                if event.cash_state is CashState.SETTLED
                and event.amount is not None
                and event.amount > Decimal("0")
            ),
            key=_cash_date,
        )
        if len(settled) < 2:
            continue

        dates = tuple(_cash_date(event) for event in settled)
        amounts = tuple(event.amount for event in settled)  # type: ignore[misc]

        # Calculate intervals between consecutive settled dates.
        intervals = [
            (dates[i] - dates[i - 1]).days for i in range(1, len(dates))
        ]
        # Filter to plausible monthly intervals (20--40 days) for median.
        monthly_intervals = [iv for iv in intervals if 20 <= iv <= 40]
        if not monthly_intervals:
            continue
        interval = int(median(monthly_intervals))

        # Conservative amount rule:
        # For expenses (debits): use the maximum of the last 3 settled amounts.
        # For income (credits): use the minimum of the last 3 settled amounts.
        recent = amounts[-3:] if len(amounts) >= 3 else amounts
        if key[3] == "debit":
            conservative = max(recent)
        else:
            conservative = min(recent)

        # Check for monthly day-of-month alignment:
        # A pattern is genuinely monthly if the median interval is 28-31 days
        # and at least half the settled dates occur on the same day of the month.
        is_monthly = 28 <= interval <= 31
        day_counts = Counter(d.day for d in dates)
        canonical_day, count = day_counts.most_common(1)[0]
        strictly_monthly = is_monthly and (count >= len(dates) // 2)

        patterns.append(
            RecurrencePattern(
                group_key=key,
                settled_events=tuple(settled),
                settled_dates=dates,
                settled_amounts=amounts,
                median_interval_days=interval,
                conservative_amount=conservative,
                most_recent_settled_date=dates[-1],
                representative_event=settled[-1],
                is_monthly=strictly_monthly,
                canonical_day_of_month=canonical_day if strictly_monthly else None,
            )
        )
    return patterns


def _has_explicit_future_occurrence(
    pattern: RecurrencePattern,
    all_events: Sequence[ClassifiedEvent],
    start_date: date,
    end_date: date,
) -> bool:
    """Return True if the group already has a scheduled or pending event in the
    forecast window, meaning the forecast engine will pick it up directly.
    """
    key = pattern.group_key
    for item in all_events:
        event = item.event
        if _group_key(event) != key:
            continue
        if event.cash_state not in {CashState.SCHEDULED, CashState.PENDING}:
            continue
        cash = _cash_date(event)
        if start_date <= cash <= end_date:
            return True
    return False


def build_recurring_cash_flows(
    context: RequestContext,
    forecast_start_date: date | None = None,
    horizon_days: int = 90,
) -> tuple[RecurringCashFlow, ...]:
    """Construct future recurring cash-flow schedules from historical evidence.

    Parameters
    ----------
    context:
        The loaded and classified request context. Not mutated.
    forecast_start_date:
        The first day of the forecast window. Defaults to ``context.request.request_date``.
    horizon_days:
        Number of inclusive forecast days.

    Returns
    -------
    tuple[RecurringCashFlow, ...]
        One ``RecurringCashFlow`` per active recurring stream inside the forecast window.
    """
    start = forecast_start_date or context.request.request_date
    end = start + timedelta(days=horizon_days - 1)
    all_events = context.financial_state.all_events

    # Terminal stream detection:
    # If structured evidence indicates a stream has ended (e.g. "Final employer payroll"),
    # suppress future recurring projections for that category/direction.
    terminal_categories: set[tuple[str, str]] = set()
    for item in all_events:
        ev = item.event
        if ev.cash_state is CashState.SETTLED and "final" in ev.description.lower():
            terminal_categories.add((ev.category, ev.direction))

    # Category-aware scheduled salary detection:
    # Look for an explicit scheduled/pending salary credit in the forecast horizon.
    scheduled_salary: FinancialEvent | None = None
    for item in all_events:
        ev = item.event
        if (
            ev.cash_state in {CashState.SCHEDULED, CashState.PENDING}
            and ev.category == "salary"
            and ev.direction == "credit"
        ):
            cash = _cash_date(ev)
            if start <= cash <= end:
                scheduled_salary = ev
                break

    patterns = _detect_patterns(all_events)
    flows: list[RecurringCashFlow] = []

    # Single settled + scheduled salary continuation (e.g. new job with prorated first salary):
    # When user has 1 settled salary and 1 scheduled salary, continue projection after scheduled date.
    has_salary_group = any(p.group_key[2] == "salary" and p.group_key[3] == "credit" for p in patterns)
    if not has_salary_group and scheduled_salary and ("salary", "credit") not in terminal_categories:
        settled_sal = [
            item.event
            for item in all_events
            if item.event.cash_state is CashState.SETTLED
            and item.event.category == "salary"
            and item.event.direction == "credit"
            and item.event.amount is not None
            and item.event.amount > Decimal("0")
        ]
        if settled_sal:
            rep = scheduled_salary
            sched_date = _cash_date(scheduled_salary)
            target_day = sched_date.day
            first_d = _add_months(sched_date, 1, target_day)
            if first_d <= end:
                flows.append(
                    RecurringCashFlow(
                        flow_id=f"rec:{rep.event_id}",
                        first_occurrence_date=first_d,
                        amount=scheduled_salary.amount,
                        currency=rep.currency,
                        direction=rep.direction,
                        frequency_days=30,
                        occurrence_count=None,
                        source_event_id=rep.event_id,
                    )
                )

    for pattern in patterns:
        key = pattern.group_key

        # Skip stream if structured evidence shows it was terminated.
        if (key[2], key[3]) in terminal_categories:
            continue

        # Skip if there is already an exact-key explicit future occurrence in the window.
        if _has_explicit_future_occurrence(pattern, all_events, start, end):
            continue

        rep = pattern.representative_event
        interval = pattern.median_interval_days
        is_salary = (key[2] == "salary" and key[3] == "credit")
        sched_date = _cash_date(scheduled_salary) if (is_salary and scheduled_salary) else None

        if pattern.is_monthly and pattern.canonical_day_of_month is not None:
            # Genuinely monthly: project based on canonical day-of-month
            canonical_day = pattern.canonical_day_of_month
            last_date = pattern.most_recent_settled_date
            m_step = 1
            curr_date = _add_months(last_date, m_step, canonical_day)
            while curr_date < start:
                m_step += 1
                curr_date = _add_months(last_date, m_step, canonical_day)

            # Category-aware deduplication: if this occurrence coincides with the scheduled salary,
            # advance to the next month to let the explicit scheduled row take priority.
            if sched_date and (curr_date.year == sched_date.year and curr_date.month == sched_date.month):
                m_step += 1
                curr_date = _add_months(last_date, m_step, canonical_day)

            if curr_date > end:
                continue

            flows.append(
                RecurringCashFlow(
                    flow_id=f"rec:{rep.event_id}",
                    first_occurrence_date=curr_date,
                    amount=pattern.conservative_amount,
                    currency=rep.currency,
                    direction=rep.direction,
                    frequency_days=30,
                    occurrence_count=None,
                    source_event_id=rep.event_id,
                )
            )
        else:
            # Non-monthly cadence: preserve exact interval step
            next_date = pattern.most_recent_settled_date + timedelta(days=interval)
            while next_date < start:
                next_date += timedelta(days=interval)

            if sched_date and is_salary and abs((next_date - sched_date).days) <= 5:
                next_date += timedelta(days=interval)

            if next_date > end:
                continue

            flows.append(
                RecurringCashFlow(
                    flow_id=f"rec:{rep.event_id}",
                    first_occurrence_date=next_date,
                    amount=pattern.conservative_amount,
                    currency=rep.currency,
                    direction=rep.direction,
                    frequency_days=interval,
                    occurrence_count=None,
                    source_event_id=rep.event_id,
                )
            )

    return tuple(flows)

