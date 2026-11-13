"""Print a compact, deterministic context diagnostic for all public samples."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_loading import DataRepository


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path(__file__).resolve().parents[1] / "dataset")
    args = parser.parse_args()
    repository = DataRepository.load(args.dataset_dir)
    print("request_id,user_id,requested_amount,home_currency,current_available_balance,minimum_balance_to_keep,relevant_financial_events,payment_options,relevant_messages,relevant_images")
    for request_id in sorted(repository.sample_requests):
        context = repository.build_request_context(request_id)
        profile = context.financial_profile
        print(
            f"{context.request.request_id},{context.request.user_id},{context.request.requested_amount},"
            f"{profile.home_currency},{profile.current_available_balance},{profile.minimum_balance_to_keep},"
            f"{len(context.financial_state.all_events)},{len(context.payment_options)},"
            f"{len(context.messages)},{len(context.images)}"
        )
    if repository.integrity_issues:
        print("\nDATA-INTEGRITY ISSUES:")
        print("\n".join(f"- {issue}" for issue in repository.integrity_issues))
    else:
        print("\nDATA-INTEGRITY ISSUES: none")


if __name__ == "__main__":
    main()
