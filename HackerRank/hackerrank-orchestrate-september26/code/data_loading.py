"""Deterministic CSV loading and financial-state reconstruction.

This module deliberately performs no affordability calculation or recommendation.
It is the typed data boundary for a later deterministic forecast engine.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Mapping, TypeVar


class DataValidationError(ValueError):
    """Raised when a required CSV field cannot be parsed safely."""


class CashState(str, Enum):
    """Whether an event should be considered active cash-flow evidence."""

    SETTLED = "settled"
    PENDING = "pending"
    SCHEDULED = "scheduled"
    INACTIVE = "inactive"
    NON_CASH = "non_cash"


class TransactionKind(str, Enum):
    """High-level financial meaning of an event, without forecasting it."""

    INCOME = "income"
    REFUND = "refund"
    EXPENSE = "expense"
    DEBT_PAYMENT = "debt_payment"
    SUBSCRIPTION = "subscription"
    INVESTMENT_PURCHASE = "investment_purchase"
    INVESTMENT_SALE = "investment_sale"
    INVESTMENT_VALUATION = "investment_valuation"
    OTHER = "other"


@dataclass(frozen=True)
class Request:
    request_id: str
    user_id: str
    request_date: date
    request_type: str
    requested_amount: Decimal
    desired_completion_date: date
    allows_partial_payment: bool
    request_text: str


@dataclass(frozen=True)
class FinancialProfile:
    user_id: str
    home_currency: str
    current_available_balance: Decimal
    minimum_balance_to_keep: Decimal
    financial_priorities: tuple[str, ...]
    expense_categories_to_protect: tuple[str, ...]
    expense_categories_user_is_willing_to_reduce: tuple[str, ...]
    expense_categories_user_is_willing_to_stop: tuple[str, ...]
    payment_methods_user_will_consider: tuple[str, ...]
    max_installment_months: int | None


@dataclass(frozen=True)
class FinancialEvent:
    event_id: str
    user_id: str
    event_type: str
    description: str
    category: str
    direction: str
    amount: Decimal | None
    currency: str
    event_date: date
    settlement_date: date | None
    status: str
    linked_event_id: str | None
    flexibility: str
    minimum_allowed_amount: Decimal | None

    @property
    def cash_state(self) -> CashState:
        """Classify the record's supplied cash state; do not infer a cash amount."""
        if self.direction == "non_cash" or self.status == "unrealized":
            return CashState.NON_CASH
        if self.status == "settled":
            return CashState.SETTLED
        if self.status == "pending":
            return CashState.PENDING
        if self.status == "scheduled":
            return CashState.SCHEDULED
        return CashState.INACTIVE

    @property
    def transaction_kind(self) -> TransactionKind:
        """Normalize the supplied event type for downstream financial logic."""
        try:
            return TransactionKind(self.event_type)
        except ValueError:
            return TransactionKind.OTHER

    @property
    def is_income(self) -> bool:
        return self.transaction_kind is TransactionKind.INCOME and self.direction == "credit"

    @property
    def is_expense(self) -> bool:
        return self.direction == "debit" and self.transaction_kind not in {
            TransactionKind.INVESTMENT_VALUATION,
        }

    @property
    def is_fixed_expense(self) -> bool:
        return self.is_expense and self.flexibility == "fixed"

    @property
    def is_flexible_expense(self) -> bool:
        return self.is_expense and self.flexibility in {
            "stoppable", "reducible", "reducible_or_stoppable"
        }


@dataclass(frozen=True)
class PaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: str
    payment_amount: Decimal
    number_of_payments: int
    first_payment_date: date
    payment_frequency_days: int | None
    financing_fee: Decimal
    total_payable_amount: Decimal


@dataclass(frozen=True)
class ExchangeRate:
    rate_date: date
    from_currency: str
    to_currency: str
    rate: Decimal


@dataclass(frozen=True)
class Message:
    message_id: str
    user_id: str
    request_id: str | None
    related_event_id: str | None
    sent_at: datetime
    source_type: str
    message_text: str


@dataclass(frozen=True)
class ImageEvidence:
    image_id: str
    user_id: str
    request_id: str | None
    related_event_id: str | None
    path: Path


@dataclass(frozen=True)
class ClassifiedEvent:
    """An event plus deterministic classifications used by forecast code later."""

    event: FinancialEvent
    is_recurring: bool


@dataclass(frozen=True)
class FinancialState:
    """Financial-event partition for one user at a specific data-loading stage."""

    all_events: tuple[ClassifiedEvent, ...]
    settled_events: tuple[ClassifiedEvent, ...]
    pending_events: tuple[ClassifiedEvent, ...]
    scheduled_events: tuple[ClassifiedEvent, ...]
    inactive_events: tuple[ClassifiedEvent, ...]
    recurring_expenses: tuple[ClassifiedEvent, ...]
    one_time_transactions: tuple[ClassifiedEvent, ...]
    income_events: tuple[ClassifiedEvent, ...]
    fixed_expenses: tuple[ClassifiedEvent, ...]
    flexible_expenses: tuple[ClassifiedEvent, ...]


@dataclass(frozen=True)
class RequestContext:
    """All deterministic, joined evidence available for a single request."""

    request: Request
    financial_profile: FinancialProfile
    financial_state: FinancialState
    payment_options: tuple[PaymentOption, ...]
    messages: tuple[Message, ...]
    images: tuple[ImageEvidence, ...]


T = TypeVar("T")


def _required(row: Mapping[str, str], name: str, source: Path, line: int) -> str:
    value = row.get(name, "").strip()
    if not value:
        raise DataValidationError(f"{source.name}:{line}: required field {name!r} is blank")
    return value


def _optional(row: Mapping[str, str], name: str) -> str | None:
    value = row.get(name, "").strip()
    return value or None


def _decimal(value: str | None, field: str, source: Path, line: int) -> Decimal | None:
    if value is None or not value.strip():
        return None
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise DataValidationError(
            f"{source.name}:{line}: invalid decimal in {field!r}: {value!r}"
        ) from exc


def _required_decimal(row: Mapping[str, str], name: str, source: Path, line: int) -> Decimal:
    value = _decimal(_required(row, name, source, line), name, source, line)
    assert value is not None
    return value


def _date(value: str | None, field: str, source: Path, line: int) -> date | None:
    if value is None or not value.strip():
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise DataValidationError(
            f"{source.name}:{line}: invalid ISO date in {field!r}: {value!r}"
        ) from exc


def _required_date(row: Mapping[str, str], name: str, source: Path, line: int) -> date:
    value = _date(_required(row, name, source, line), name, source, line)
    assert value is not None
    return value


def _datetime(value: str, field: str, source: Path, line: int) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DataValidationError(
            f"{source.name}:{line}: invalid ISO datetime in {field!r}: {value!r}"
        ) from exc


def _integer(value: str | None, field: str, source: Path, line: int) -> int | None:
    if value is None or not value.strip():
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise DataValidationError(
            f"{source.name}:{line}: invalid integer in {field!r}: {value!r}"
        ) from exc


def _required_integer(row: Mapping[str, str], name: str, source: Path, line: int) -> int:
    value = _integer(_required(row, name, source, line), name, source, line)
    assert value is not None
    return value


def _boolean(value: str, field: str, source: Path, line: int) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise DataValidationError(f"{source.name}:{line}: invalid boolean in {field!r}: {value!r}")


def _pipe_list(value: str | None) -> tuple[str, ...]:
    return tuple(part.strip() for part in (value or "").split("|") if part.strip())


def _read_rows(dataset_dir: Path, filename: str, fields: set[str]) -> tuple[Path, list[dict[str, str]]]:
    source = dataset_dir / filename
    if not source.is_file():
        raise DataValidationError(f"Required dataset file is missing: {source}")
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = set(reader.fieldnames or ())
        missing = fields - headers
        if missing:
            raise DataValidationError(f"{filename}: missing required columns {sorted(missing)}")
        return source, list(reader)


def _unique(items: Iterable[T], key: Callable[[T], str], label: str) -> dict[str, T]:
    result: dict[str, T] = {}
    for item in items:
        identifier = key(item)
        if identifier in result:
            raise DataValidationError(f"Duplicate {label}: {identifier}")
        result[identifier] = item
    return result


def _freeze_index(index: defaultdict[str, list[T]]) -> dict[str, tuple[T, ...]]:
    return {key: tuple(values) for key, values in index.items()}


class DataRepository:
    """Typed data store and indexes built from the immutable ``dataset/`` CSV files."""

    def __init__(self, dataset_dir: Path) -> None:
        self.dataset_dir = dataset_dir
        self.requests: dict[str, Request] = {}
        self.sample_requests: dict[str, Request] = {}
        self.profiles_by_user_id: dict[str, FinancialProfile] = {}
        self.events_by_event_id: dict[str, FinancialEvent] = {}
        self.events_by_user_id: dict[str, tuple[FinancialEvent, ...]] = {}
        self.events_by_linked_event_id: dict[str, tuple[FinancialEvent, ...]] = {}
        self.payment_options_by_request_id: dict[str, tuple[PaymentOption, ...]] = {}
        self.messages_by_user_id: dict[str, tuple[Message, ...]] = {}
        self.messages_by_request_id: dict[str, tuple[Message, ...]] = {}
        self.messages_by_related_event_id: dict[str, tuple[Message, ...]] = {}
        self.images_by_user_id: dict[str, tuple[ImageEvidence, ...]] = {}
        self.images_by_request_id: dict[str, tuple[ImageEvidence, ...]] = {}
        self.images_by_related_event_id: dict[str, tuple[ImageEvidence, ...]] = {}
        self.exchange_rates_by_key: dict[tuple[date, str, str], ExchangeRate] = {}
        self.integrity_issues: tuple[str, ...] = ()

    @classmethod
    def load(cls, dataset_dir: str | Path) -> "DataRepository":
        """Load every participant-facing source CSV and construct deterministic indexes."""
        repository = cls(Path(dataset_dir).resolve())
        repository._load_all()
        return repository

    def _load_all(self) -> None:
        self.requests = self._load_requests("requests.csv")
        self.sample_requests = self._load_requests("sample_requests.csv")
        overlap = set(self.requests) & set(self.sample_requests)
        if overlap:
            raise DataValidationError(f"request IDs overlap between input and samples: {sorted(overlap)}")
        self.profiles_by_user_id = self._load_profiles()
        self.events_by_event_id, self.events_by_user_id, self.events_by_linked_event_id = self._load_events()
        self.payment_options_by_request_id = self._load_payment_options()
        self.messages_by_user_id, self.messages_by_request_id, self.messages_by_related_event_id = self._load_messages()
        self.images_by_user_id, self.images_by_request_id, self.images_by_related_event_id = self._load_images()
        self.exchange_rates_by_key = self._load_exchange_rates()
        self.integrity_issues = self._collect_integrity_issues()

    def _load_requests(self, filename: str) -> dict[str, Request]:
        source, rows = _read_rows(self.dataset_dir, filename, {
            "request_id", "user_id", "request_date", "request_type", "requested_amount",
            "desired_completion_date", "allows_partial_payment", "request_text",
        })
        requests = []
        for line, row in enumerate(rows, start=2):
            requests.append(Request(
                request_id=_required(row, "request_id", source, line),
                user_id=_required(row, "user_id", source, line),
                request_date=_required_date(row, "request_date", source, line),
                request_type=_required(row, "request_type", source, line),
                requested_amount=_required_decimal(row, "requested_amount", source, line),
                desired_completion_date=_required_date(row, "desired_completion_date", source, line),
                allows_partial_payment=_boolean(_required(row, "allows_partial_payment", source, line), "allows_partial_payment", source, line),
                request_text=_required(row, "request_text", source, line),
            ))
        return _unique(requests, lambda item: item.request_id, "request_id")

    def _load_profiles(self) -> dict[str, FinancialProfile]:
        source, rows = _read_rows(self.dataset_dir, "financial_profiles.csv", {
            "user_id", "home_currency", "current_available_balance", "minimum_balance_to_keep",
            "financial_priorities", "expense_categories_to_protect",
            "expense_categories_user_is_willing_to_reduce", "expense_categories_user_is_willing_to_stop",
            "payment_methods_user_will_consider", "max_installment_months",
        })
        profiles = []
        for line, row in enumerate(rows, start=2):
            profiles.append(FinancialProfile(
                user_id=_required(row, "user_id", source, line), home_currency=_required(row, "home_currency", source, line),
                current_available_balance=_required_decimal(row, "current_available_balance", source, line),
                minimum_balance_to_keep=_required_decimal(row, "minimum_balance_to_keep", source, line),
                financial_priorities=_pipe_list(row.get("financial_priorities")),
                expense_categories_to_protect=_pipe_list(row.get("expense_categories_to_protect")),
                expense_categories_user_is_willing_to_reduce=_pipe_list(row.get("expense_categories_user_is_willing_to_reduce")),
                expense_categories_user_is_willing_to_stop=_pipe_list(row.get("expense_categories_user_is_willing_to_stop")),
                payment_methods_user_will_consider=_pipe_list(row.get("payment_methods_user_will_consider")),
                max_installment_months=_integer(row.get("max_installment_months"), "max_installment_months", source, line),
            ))
        return _unique(profiles, lambda item: item.user_id, "financial profile user_id")

    def _load_events(self) -> tuple[dict[str, FinancialEvent], dict[str, tuple[FinancialEvent, ...]], dict[str, tuple[FinancialEvent, ...]]]:
        source, rows = _read_rows(self.dataset_dir, "financial_events.csv", {
            "event_id", "user_id", "event_type", "description", "category", "direction", "amount",
            "currency", "event_date", "settlement_date", "status", "linked_event_id", "flexibility", "minimum_allowed_amount",
        })
        events = []
        for line, row in enumerate(rows, start=2):
            events.append(FinancialEvent(
                event_id=_required(row, "event_id", source, line), user_id=_required(row, "user_id", source, line),
                event_type=_required(row, "event_type", source, line), description=_required(row, "description", source, line),
                category=_required(row, "category", source, line), direction=_required(row, "direction", source, line),
                amount=_decimal(row.get("amount"), "amount", source, line), currency=_required(row, "currency", source, line),
                event_date=_required_date(row, "event_date", source, line), settlement_date=_date(row.get("settlement_date"), "settlement_date", source, line),
                status=_required(row, "status", source, line), linked_event_id=_optional(row, "linked_event_id"),
                flexibility=_required(row, "flexibility", source, line), minimum_allowed_amount=_decimal(row.get("minimum_allowed_amount"), "minimum_allowed_amount", source, line),
            ))
        by_user: defaultdict[str, list[FinancialEvent]] = defaultdict(list)
        by_link: defaultdict[str, list[FinancialEvent]] = defaultdict(list)
        for event in events:
            by_user[event.user_id].append(event)
            if event.linked_event_id:
                by_link[event.linked_event_id].append(event)
        return _unique(events, lambda item: item.event_id, "event_id"), _freeze_index(by_user), _freeze_index(by_link)

    def _load_payment_options(self) -> dict[str, tuple[PaymentOption, ...]]:
        source, rows = _read_rows(self.dataset_dir, "request_payment_options.csv", {
            "payment_option_id", "request_id", "payment_method", "payment_amount", "number_of_payments",
            "first_payment_date", "payment_frequency_days", "financing_fee", "total_payable_amount",
        })
        options = []
        for line, row in enumerate(rows, start=2):
            options.append(PaymentOption(
                payment_option_id=_required(row, "payment_option_id", source, line), request_id=_required(row, "request_id", source, line),
                payment_method=_required(row, "payment_method", source, line), payment_amount=_required_decimal(row, "payment_amount", source, line),
                number_of_payments=_required_integer(row, "number_of_payments", source, line), first_payment_date=_required_date(row, "first_payment_date", source, line),
                payment_frequency_days=_integer(row.get("payment_frequency_days"), "payment_frequency_days", source, line),
                financing_fee=_required_decimal(row, "financing_fee", source, line), total_payable_amount=_required_decimal(row, "total_payable_amount", source, line),
            ))
        _unique(options, lambda item: item.payment_option_id, "payment_option_id")
        by_request: defaultdict[str, list[PaymentOption]] = defaultdict(list)
        for option in options:
            by_request[option.request_id].append(option)
        return _freeze_index(by_request)

    def _load_messages(self) -> tuple[dict[str, tuple[Message, ...]], dict[str, tuple[Message, ...]], dict[str, tuple[Message, ...]]]:
        source, rows = _read_rows(self.dataset_dir, "messages.csv", {
            "message_id", "user_id", "request_id", "related_event_id", "sent_at", "source_type", "message_text",
        })
        messages = []
        for line, row in enumerate(rows, start=2):
            messages.append(Message(
                message_id=_required(row, "message_id", source, line), user_id=_required(row, "user_id", source, line),
                request_id=_optional(row, "request_id"), related_event_id=_optional(row, "related_event_id"),
                sent_at=_datetime(_required(row, "sent_at", source, line), "sent_at", source, line),
                source_type=_required(row, "source_type", source, line), message_text=_required(row, "message_text", source, line),
            ))
        _unique(messages, lambda item: item.message_id, "message_id")
        return self._indexes_for_evidence(messages, lambda item: item.user_id, lambda item: item.request_id, lambda item: item.related_event_id)

    def _load_images(self) -> tuple[dict[str, tuple[ImageEvidence, ...]], dict[str, tuple[ImageEvidence, ...]], dict[str, tuple[ImageEvidence, ...]]]:
        source, rows = _read_rows(self.dataset_dir, "images.csv", {"image_id", "user_id", "request_id", "related_event_id"})
        images = []
        for line, row in enumerate(rows, start=2):
            image_id = _required(row, "image_id", source, line)
            images.append(ImageEvidence(
                image_id=image_id, user_id=_required(row, "user_id", source, line), request_id=_optional(row, "request_id"),
                related_event_id=_optional(row, "related_event_id"), path=self.dataset_dir / "media" / "images" / f"{image_id}.png",
            ))
        _unique(images, lambda item: item.image_id, "image_id")
        return self._indexes_for_evidence(images, lambda item: item.user_id, lambda item: item.request_id, lambda item: item.related_event_id)

    def _indexes_for_evidence(self, items: Iterable[T], user_key: Callable[[T], str], request_key: Callable[[T], str | None], event_key: Callable[[T], str | None]) -> tuple[dict[str, tuple[T, ...]], dict[str, tuple[T, ...]], dict[str, tuple[T, ...]]]:
        by_user: defaultdict[str, list[T]] = defaultdict(list)
        by_request: defaultdict[str, list[T]] = defaultdict(list)
        by_event: defaultdict[str, list[T]] = defaultdict(list)
        for item in items:
            by_user[user_key(item)].append(item)
            if (identifier := request_key(item)):
                by_request[identifier].append(item)
            if (identifier := event_key(item)):
                by_event[identifier].append(item)
        return _freeze_index(by_user), _freeze_index(by_request), _freeze_index(by_event)

    def _load_exchange_rates(self) -> dict[tuple[date, str, str], ExchangeRate]:
        source, rows = _read_rows(self.dataset_dir, "exchange_rates.csv", {"rate_date", "from_currency", "to_currency", "rate"})
        rates = []
        for line, row in enumerate(rows, start=2):
            rates.append(ExchangeRate(
                rate_date=_required_date(row, "rate_date", source, line), from_currency=_required(row, "from_currency", source, line),
                to_currency=_required(row, "to_currency", source, line), rate=_required_decimal(row, "rate", source, line),
            ))
        result: dict[tuple[date, str, str], ExchangeRate] = {}
        for rate in rates:
            key = (rate.rate_date, rate.from_currency, rate.to_currency)
            if key in result:
                raise DataValidationError(f"Duplicate exchange rate: {key}")
            result[key] = rate
        return result

    def _collect_integrity_issues(self) -> tuple[str, ...]:
        issues: list[str] = []
        all_requests = {**self.requests, **self.sample_requests}
        for request in all_requests.values():
            if request.user_id not in self.profiles_by_user_id:
                issues.append(f"{request.request_id}: missing financial profile for {request.user_id}")
        for event in self.events_by_event_id.values():
            if event.user_id not in self.profiles_by_user_id:
                issues.append(f"{event.event_id}: missing financial profile for {event.user_id}")
            if event.amount is None and event.event_id not in self.images_by_related_event_id:
                issues.append(f"{event.event_id}: blank amount has no linked image evidence")
            if event.linked_event_id and event.linked_event_id not in self.events_by_event_id:
                issues.append(f"{event.event_id}: linked event {event.linked_event_id} is missing")
        for request_id in self.payment_options_by_request_id:
            if request_id not in all_requests:
                issues.append(f"{request_id}: payment options reference an unknown request")
        for evidence in (*self._all_messages(), *self._all_images()):
            if evidence.related_event_id and evidence.related_event_id not in self.events_by_event_id:
                issues.append(f"{evidence!r}: related event {evidence.related_event_id} is missing")
        for image in self._all_images():
            if not image.path.is_file():
                issues.append(f"{image.image_id}: referenced image file is missing")
        return tuple(issues)

    def _all_messages(self) -> tuple[Message, ...]:
        return tuple(message for values in self.messages_by_user_id.values() for message in values)

    def _all_images(self) -> tuple[ImageEvidence, ...]:
        return tuple(image for values in self.images_by_user_id.values() for image in values)

    def build_request_context(self, request_id: str) -> RequestContext:
        """Return joined evidence for an evaluation or public sample request.

        Evidence is included when it belongs to the request's user, directly names the
        request, or directly names one of that user's event IDs.  Items are deduplicated
        by their stable IDs and retain source CSV order.
        """
        request = self.requests.get(request_id) or self.sample_requests.get(request_id)
        if request is None:
            raise KeyError(f"Unknown request_id: {request_id}")
        try:
            profile = self.profiles_by_user_id[request.user_id]
        except KeyError as exc:
            raise DataValidationError(f"{request_id}: profile for {request.user_id} is missing") from exc
        events = self.events_by_user_id.get(request.user_id, ())
        classifications = self._classify_events(events)
        event_ids = {event.event_id for event in events}
        messages = self._relevant_evidence(request, event_ids, self.messages_by_user_id, self.messages_by_request_id, self.messages_by_related_event_id, "message_id")
        images = self._relevant_evidence(request, event_ids, self.images_by_user_id, self.images_by_request_id, self.images_by_related_event_id, "image_id")
        return RequestContext(
            request=request, financial_profile=profile, financial_state=self._financial_state(classifications),
            payment_options=self.payment_options_by_request_id.get(request_id, ()), messages=messages, images=images,
        )

    def _relevant_evidence(self, request: Request, event_ids: set[str], by_user: Mapping[str, tuple[T, ...]], by_request: Mapping[str, tuple[T, ...]], by_event: Mapping[str, tuple[T, ...]], identifier: str) -> tuple[T, ...]:
        selected: dict[str, T] = {}
        for item in by_user.get(request.user_id, ()): selected[getattr(item, identifier)] = item
        for item in by_request.get(request.request_id, ()): selected[getattr(item, identifier)] = item
        for event_id in event_ids:
            for item in by_event.get(event_id, ()): selected[getattr(item, identifier)] = item
        return tuple(selected.values())

    def _classify_events(self, events: tuple[FinancialEvent, ...]) -> tuple[ClassifiedEvent, ...]:
        recurring_ids = self._recurring_event_ids(events)
        return tuple(ClassifiedEvent(event=event, is_recurring=event.event_id in recurring_ids) for event in events)

    @staticmethod
    def _recurring_event_ids(events: tuple[FinancialEvent, ...]) -> set[str]:
        """Detect recurrence only from repeated settled history of the same event pattern.

        At least two settled occurrences must be 20--40 days apart. Pending and
        scheduled rows are marked recurring only when their pattern has such history.
        """
        groups: defaultdict[tuple[str, str, str, str, str], list[FinancialEvent]] = defaultdict(list)
        for event in events:
            groups[(event.event_type, event.description, event.category, event.direction, event.currency)].append(event)
        recurring: set[str] = set()
        for group in groups.values():
            settled = sorted((event for event in group if event.cash_state is CashState.SETTLED), key=lambda event: event.settlement_date or event.event_date)
            dates = [event.settlement_date or event.event_date for event in settled]
            if any(20 <= (later - earlier).days <= 40 for earlier, later in zip(dates, dates[1:])):
                recurring.update(event.event_id for event in group)
        return recurring

    @staticmethod
    def _financial_state(events: tuple[ClassifiedEvent, ...]) -> FinancialState:
        by_state = lambda state: tuple(item for item in events if item.event.cash_state is state)
        recurring = tuple(item for item in events if item.is_recurring and item.event.is_expense)
        return FinancialState(
            all_events=events, settled_events=by_state(CashState.SETTLED), pending_events=by_state(CashState.PENDING),
            scheduled_events=by_state(CashState.SCHEDULED), inactive_events=tuple(item for item in events if item.event.cash_state in {CashState.INACTIVE, CashState.NON_CASH}),
            recurring_expenses=recurring, one_time_transactions=tuple(item for item in events if not item.is_recurring),
            income_events=tuple(item for item in events if item.event.is_income), fixed_expenses=tuple(item for item in events if item.event.is_fixed_expense),
            flexible_expenses=tuple(item for item in events if item.event.is_flexible_expense),
        )
