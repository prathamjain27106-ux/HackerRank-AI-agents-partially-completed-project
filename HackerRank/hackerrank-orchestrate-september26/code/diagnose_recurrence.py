"""Diagnostic: show generated recurring schedules for real dataset users."""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from data_loading import DataRepository
from recurrence import build_recurring_cash_flows, _detect_patterns

DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"


def main() -> None:
    repository = DataRepository.load(DATASET_DIR)

    # Use a mix of sample users
    sample_ids = sorted(repository.sample_requests)[:8]
    print("=" * 120)
    print(f"{'request_id':15s} {'user_id':10s} {'req_date':12s} {'patterns':8s} {'flows':6s}  Recurring schedules")
    print("=" * 120)

    for request_id in sample_ids:
        context = repository.build_request_context(request_id)
        patterns = _detect_patterns(context.financial_state.all_events)
        flows = build_recurring_cash_flows(context)
        start = context.request.request_date
        end = start + timedelta(days=89)

        print(f"{request_id:15s} {context.request.user_id:10s} {str(start):12s} {len(patterns):8d} {len(flows):6d}")
        for flow in flows:
            occ_count = 0
            d = flow.first_occurrence_date
            while d <= end:
                occ_count += 1
                d += timedelta(days=flow.frequency_days)
            print(
                f"  {'DEBIT' if flow.direction == 'debit' else 'CREDIT':6s} "
                f"{str(flow.amount):>15s} {flow.currency:4s} "
                f"every {flow.frequency_days:2d}d "
                f"first={flow.first_occurrence_date} "
                f"occurrences={occ_count} "
                f"src={flow.source_event_id}"
            )
        if not flows and patterns:
            for p in patterns:
                print(
                    f"  [NO FLOW] {p.group_key[1][:30]:30s} "
                    f"interval={p.median_interval_days}d "
                    f"amount={p.conservative_amount} "
                    f"last_settled={p.most_recent_settled_date}"
                )

    # Summary statistics
    print("\n" + "=" * 120)
    print("GLOBAL STATS across all 250 evaluation requests:")
    total_patterns = 0
    total_flows = 0
    total_occurrences = 0
    errors = 0
    for request_id in sorted(repository.requests):
        try:
            context = repository.build_request_context(request_id)
            patterns = _detect_patterns(context.financial_state.all_events)
            flows = build_recurring_cash_flows(context)
            start = context.request.request_date
            end = start + timedelta(days=89)
            total_patterns += len(patterns)
            total_flows += len(flows)
            for flow in flows:
                d = flow.first_occurrence_date
                while d <= end:
                    total_occurrences += 1
                    d += timedelta(days=flow.frequency_days)
        except Exception as e:
            errors += 1
            print(f"  ERROR {request_id}: {e}")

    print(f"  Total recurring patterns detected: {total_patterns}")
    print(f"  Total RecurringCashFlow objects created: {total_flows}")
    print(f"  Total projected occurrences in 90-day windows: {total_occurrences}")
    print(f"  Errors: {errors}")


if __name__ == "__main__":
    main()
