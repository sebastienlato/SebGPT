"""Deterministic in-memory Unicode code-point inventory for extracted works."""

from __future__ import annotations

import json
import platform
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from sebgpt.data.shakespeare_extract import ExtractedWork


SPLIT_ORDER = ("train", "validation", "test")


class InventoryError(Exception):
    """An inventory failure containing structural and numeric facts only."""

    def __init__(
        self,
        invariant: str,
        *,
        expected: Any,
        observed: Any,
        work_id: str | None = None,
    ) -> None:
        self.details = {
            "invariant": invariant,
            "expected": expected,
            "observed": observed,
        }
        if work_id is not None:
            self.details["work_id"] = work_id
        super().__init__(json.dumps(self.details, ensure_ascii=True, sort_keys=True))


@dataclass(frozen=True)
class CodePointCount:
    """One code point's structural identity and frequency."""

    code_point: int
    notation: str
    name: str
    count: int


@dataclass(frozen=True)
class WorkCharacterInventory:
    work_id: str
    split: str
    manifest_order: int
    total_code_point_count: int
    distinct_code_point_count: int
    code_points: tuple[CodePointCount, ...] = field(repr=False)


@dataclass(frozen=True)
class SplitCharacterInventory:
    split: str
    work_count: int
    total_code_point_count: int
    distinct_code_point_count: int
    code_points: tuple[CodePointCount, ...] = field(repr=False)


@dataclass(frozen=True)
class GlobalCharacterInventory:
    work_count: int
    total_code_point_count: int
    distinct_code_point_count: int
    code_points: tuple[CodePointCount, ...] = field(repr=False)


@dataclass(frozen=True)
class CrossSplitRelationships:
    validation_not_in_train: tuple[CodePointCount, ...] = field(repr=False)
    test_not_in_train: tuple[CodePointCount, ...] = field(repr=False)
    test_not_in_train_or_validation: tuple[CodePointCount, ...] = field(repr=False)


@dataclass(frozen=True)
class CharacterInventorySummary:
    """A content-safe summary containing totals and cardinalities only."""

    python_version: str
    work_count: int
    train_work_count: int
    validation_work_count: int
    test_work_count: int
    train_total_code_points: int
    validation_total_code_points: int
    test_total_code_points: int
    global_total_code_points: int
    train_distinct_code_points: int
    validation_distinct_code_points: int
    test_distinct_code_points: int
    global_distinct_code_points: int
    validation_not_in_train_count: int
    test_not_in_train_count: int
    test_not_in_train_or_validation_count: int


@dataclass(frozen=True)
class CharacterInventoryReport:
    python_version: str
    work_inventories: tuple[WorkCharacterInventory, ...]
    split_inventories: tuple[SplitCharacterInventory, ...]
    global_inventory: GlobalCharacterInventory
    relationships: CrossSplitRelationships

    def safe_summary(self) -> CharacterInventorySummary:
        """Return only approved structural totals, never code-point identities."""

        split_by_name = {
            inventory.split: inventory for inventory in self.split_inventories
        }
        train = split_by_name["train"]
        validation = split_by_name["validation"]
        test = split_by_name["test"]
        return CharacterInventorySummary(
            python_version=self.python_version,
            work_count=len(self.work_inventories),
            train_work_count=train.work_count,
            validation_work_count=validation.work_count,
            test_work_count=test.work_count,
            train_total_code_points=train.total_code_point_count,
            validation_total_code_points=validation.total_code_point_count,
            test_total_code_points=test.total_code_point_count,
            global_total_code_points=self.global_inventory.total_code_point_count,
            train_distinct_code_points=train.distinct_code_point_count,
            validation_distinct_code_points=validation.distinct_code_point_count,
            test_distinct_code_points=test.distinct_code_point_count,
            global_distinct_code_points=self.global_inventory.distinct_code_point_count,
            validation_not_in_train_count=len(
                self.relationships.validation_not_in_train
            ),
            test_not_in_train_count=len(self.relationships.test_not_in_train),
            test_not_in_train_or_validation_count=len(
                self.relationships.test_not_in_train_or_validation
            ),
        )


def _require_equal(
    invariant: str,
    expected: Any,
    observed: Any,
    *,
    work_id: str | None = None,
) -> None:
    if observed != expected:
        raise InventoryError(
            invariant,
            expected=expected,
            observed=observed,
            work_id=work_id,
        )


def _freeze_counts(counts: Counter[int]) -> tuple[CodePointCount, ...]:
    return tuple(
        CodePointCount(
            code_point=code_point,
            notation=f"U+{code_point:04X}",
            name=unicodedata.name(chr(code_point), "UNNAMED"),
            count=counts[code_point],
        )
        for code_point in sorted(counts)
    )


def _inventory_work(work: ExtractedWork) -> WorkCharacterInventory:
    carriage_return_count = work.processed_text.count("\r")
    _require_equal(
        "work.processed_text.carriage_returns",
        0,
        carriage_return_count,
        work_id=work.work_id,
    )
    counts: Counter[int] = Counter(ord(character) for character in work.processed_text)
    total = sum(counts.values())
    _require_equal(
        "work.total_code_point_count",
        work.processed_code_point_count,
        total,
        work_id=work.work_id,
    )
    frozen_counts = _freeze_counts(counts)
    return WorkCharacterInventory(
        work_id=work.work_id,
        split=work.split,
        manifest_order=work.manifest_order,
        total_code_point_count=total,
        distinct_code_point_count=len(frozen_counts),
        code_points=frozen_counts,
    )


def _aggregate_counts(
    inventories: tuple[WorkCharacterInventory, ...],
) -> Counter[int]:
    counts: Counter[int] = Counter()
    for inventory in inventories:
        counts.update(
            {entry.code_point: entry.count for entry in inventory.code_points}
        )
    return counts


def _split_inventory(
    split: str,
    inventories: tuple[WorkCharacterInventory, ...],
) -> SplitCharacterInventory:
    selected = tuple(
        inventory for inventory in inventories if inventory.split == split
    )
    counts = _aggregate_counts(selected)
    frozen_counts = _freeze_counts(counts)
    return SplitCharacterInventory(
        split=split,
        work_count=len(selected),
        total_code_point_count=sum(counts.values()),
        distinct_code_point_count=len(frozen_counts),
        code_points=frozen_counts,
    )


def _select_code_points(
    inventory: SplitCharacterInventory,
    selected: set[int],
) -> tuple[CodePointCount, ...]:
    return tuple(
        entry for entry in inventory.code_points if entry.code_point in selected
    )


def inventory_extracted_works(
    works: tuple[ExtractedWork, ...],
) -> CharacterInventoryReport:
    """Count exact code points without changing text or invoking earlier stages."""

    _require_equal("collection.input_type_tuple", True, isinstance(works, tuple))
    _require_equal("collection.nonempty", True, bool(works))
    _require_equal(
        "collection.unique_work_ids",
        len(works),
        len({work.work_id for work in works}),
    )
    _require_equal(
        "collection.unique_manifest_orders",
        len(works),
        len({work.manifest_order for work in works}),
    )
    _require_equal(
        "collection.known_splits",
        0,
        sum(work.split not in SPLIT_ORDER for work in works),
    )

    ordered_works = tuple(sorted(works, key=lambda work: work.manifest_order))
    work_inventories = tuple(_inventory_work(work) for work in ordered_works)
    split_inventories = tuple(
        _split_inventory(split, work_inventories) for split in SPLIT_ORDER
    )
    split_by_name = {
        inventory.split: inventory for inventory in split_inventories
    }

    global_counts = _aggregate_counts(work_inventories)
    global_code_points = _freeze_counts(global_counts)
    global_inventory = GlobalCharacterInventory(
        work_count=len(work_inventories),
        total_code_point_count=sum(global_counts.values()),
        distinct_code_point_count=len(global_code_points),
        code_points=global_code_points,
    )

    train = split_by_name["train"]
    validation = split_by_name["validation"]
    test = split_by_name["test"]
    train_values = {entry.code_point for entry in train.code_points}
    validation_values = {entry.code_point for entry in validation.code_points}
    test_values = {entry.code_point for entry in test.code_points}
    relationships = CrossSplitRelationships(
        validation_not_in_train=_select_code_points(
            validation, validation_values - train_values
        ),
        test_not_in_train=_select_code_points(test, test_values - train_values),
        test_not_in_train_or_validation=_select_code_points(
            test, test_values - (train_values | validation_values)
        ),
    )

    _require_equal(
        "aggregate.global_total",
        sum(inventory.total_code_point_count for inventory in split_inventories),
        global_inventory.total_code_point_count,
    )
    for split_inventory in split_inventories:
        expected_total = sum(
            inventory.total_code_point_count
            for inventory in work_inventories
            if inventory.split == split_inventory.split
        )
        _require_equal(
            "aggregate.split_total",
            expected_total,
            split_inventory.total_code_point_count,
        )

    return CharacterInventoryReport(
        python_version=platform.python_version(),
        work_inventories=work_inventories,
        split_inventories=split_inventories,
        global_inventory=global_inventory,
        relationships=relationships,
    )
