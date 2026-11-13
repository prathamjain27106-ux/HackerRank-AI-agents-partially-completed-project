"""Run non-recommendation forecast diagnostics for public sample contexts."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_loading import DataRepository
from forecast import ForecastEngine, ForecastInputError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path(__file__).resolve().parents[1] / "dataset")
    parser.add_argument("--limit", type=int, default=5, help="Number of public samples to simulate.")
    args = parser.parse_args()
    repository = DataRepository.load(args.dataset_dir)
    engine = ForecastEngine.from_repository(repository)
    print("request_id,user_id,safe,minimum_projected_balance,minimum_balance_date,ending_balance,unprojected_recurring_events")
    for request_id in sorted(repository.sample_requests)[:args.limit]:
        context = repository.build_request_context(request_id)
        try:
            result = engine.forecast_request_context(context)
            print(
                f"{request_id},{context.request.user_id},{result.safe},{result.minimum_projected_balance},"
                f"{result.minimum_balance_date},{result.ending_balance},{len(result.unprojected_recurring_event_ids)}"
            )
        except ForecastInputError as error:
            print(f"{request_id},{context.request.user_id},INPUT_ERROR,{error}")


if __name__ == "__main__":
    main()
