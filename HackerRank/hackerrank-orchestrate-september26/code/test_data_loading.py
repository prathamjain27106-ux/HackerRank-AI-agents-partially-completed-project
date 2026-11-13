"""Focused regression tests for deterministic data loading and request contexts."""

from pathlib import Path
import unittest

from data_loading import CashState, DataRepository


DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"


class DataLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository = DataRepository.load(DATASET_DIR)

    def test_all_public_samples_build_contexts(self) -> None:
        self.assertEqual(25, len(self.repository.sample_requests))
        for request_id in self.repository.sample_requests:
            context = self.repository.build_request_context(request_id)
            self.assertEqual(request_id, context.request.request_id)
            self.assertEqual(context.request.user_id, context.financial_profile.user_id)

    def test_blank_event_amounts_remain_unknown_and_have_image_evidence(self) -> None:
        blank_events = [event for event in self.repository.events_by_event_id.values() if event.amount is None]
        self.assertTrue(blank_events)
        for event in blank_events:
            self.assertIn(event.event_id, self.repository.images_by_related_event_id)

    def test_event_state_classification_preserves_inactive_records(self) -> None:
        self.assertTrue(any(event.cash_state is CashState.PENDING for event in self.repository.events_by_event_id.values()))
        self.assertTrue(any(event.cash_state is CashState.INACTIVE for event in self.repository.events_by_event_id.values()))
        self.assertTrue(any(event.cash_state is CashState.NON_CASH for event in self.repository.events_by_event_id.values()))


if __name__ == "__main__":
    unittest.main()
