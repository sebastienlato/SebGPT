"""Tests for deterministic in-memory Unicode character inventory."""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from dataclasses import FrozenInstanceError, fields
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from sebgpt.data.shakespeare_extract import (  # noqa: E402
    ByteStage,
    ExtractedWork,
    OuterRawRange,
    extract_shakespeare_in_memory,
)
from sebgpt.data.shakespeare_inventory import (  # noqa: E402
    CharacterInventoryReport,
    CodePointCount,
    InventoryError,
    inventory_extracted_works,
)
from sebgpt.data.shakespeare_preflight import Position  # noqa: E402


SECRET_TEST_PROSE = "SEALED TEST PROSE MUST NEVER APPEAR"


def _work(
    work_id: str,
    split: str,
    order: int,
    text: str,
) -> ExtractedWork:
    encoded = text.encode("utf-8")
    stage = ByteStage(hashlib.sha256(encoded).hexdigest(), len(encoded))
    position = Position(1, 0, 0)
    return ExtractedWork(
        work_id=work_id,
        title=f"Work {order}",
        split=split,
        manifest_order=order,
        body_marker=f"WORK {order}",
        successor_marker=f"NEXT {order}",
        processed_text=text,
        outer_range=OuterRawRange(position, len(encoded), len(text)),
        successor_position=position,
        local_contents_position=position,
        dramatis_position=position,
        outer_raw=stage,
        retained_raw=stage,
        processed=stage,
        removed_local_contents_byte_count=0,
        processed_code_point_count=len(text),
        processed_line_count=text.count("\n"),
        processed_word_count=len(text.split()),
    )


def _frequency_map(inventory) -> dict[int, CodePointCount]:
    return {entry.code_point: entry for entry in inventory.code_points}


def _relationship_values(entries: tuple[CodePointCount, ...]) -> tuple[int, ...]:
    return tuple(entry.code_point for entry in entries)


def _corpus() -> tuple[ExtractedWork, ...]:
    return (
        _work("train-two", "train", 2, "bé\n"),
        _work("test", "test", 4, "aΩЖ\u00a0\n"),
        _work("train-one", "train", 1, "aa \n"),
        _work("validation", "validation", 3, "aΩ\n"),
    )


class CharacterInventoryTests(unittest.TestCase):
    def test_exact_code_point_counts_and_structural_representation(self) -> None:
        text = "aa \t\n\u00a0'’\"“–—æ\n"
        report = inventory_extracted_works((_work("one", "train", 1, text),))
        inventory = report.work_inventories[0]
        counts = _frequency_map(inventory)

        self.assertEqual(inventory.total_code_point_count, len(text))
        self.assertEqual(inventory.distinct_code_point_count, len(set(text)))
        self.assertEqual(counts[ord("a")].count, 2)
        self.assertEqual(counts[0x000A].count, 2)
        self.assertEqual(counts[0x0009].count, 1)
        self.assertEqual(counts[0x0020].count, 1)
        self.assertEqual(counts[0x00A0].count, 1)
        self.assertEqual(counts[0x00E6].notation, "U+00E6")
        self.assertEqual(counts[0x00E6].name, "LATIN SMALL LETTER AE")
        self.assertEqual(counts[0x000A].name, "UNNAMED")

    def test_unicode_and_punctuation_variants_remain_distinct(self) -> None:
        text = "'’\"“ -–— é e\u0301\n"
        report = inventory_extracted_works((_work("one", "train", 1, text),))
        values = {entry.code_point for entry in report.work_inventories[0].code_points}

        expected_distinct_values = {
            0x0022,
            0x0027,
            0x002D,
            0x0065,
            0x00E9,
            0x0301,
            0x2013,
            0x2014,
            0x2019,
            0x201C,
        }
        self.assertTrue(expected_distinct_values.issubset(values))

    def test_per_work_split_and_global_aggregation(self) -> None:
        report = inventory_extracted_works(_corpus())
        split_by_name = {
            inventory.split: inventory for inventory in report.split_inventories
        }

        self.assertEqual(
            tuple(inventory.work_id for inventory in report.work_inventories),
            ("train-one", "train-two", "validation", "test"),
        )
        self.assertEqual(
            tuple(inventory.split for inventory in report.split_inventories),
            ("train", "validation", "test"),
        )
        self.assertEqual(split_by_name["train"].work_count, 2)
        self.assertEqual(split_by_name["validation"].work_count, 1)
        self.assertEqual(split_by_name["test"].work_count, 1)
        self.assertEqual(
            split_by_name["train"].total_code_point_count,
            sum(len(work.processed_text) for work in _corpus() if work.split == "train"),
        )
        self.assertEqual(
            report.global_inventory.total_code_point_count,
            sum(len(work.processed_text) for work in _corpus()),
        )

    def test_cross_split_relationships_are_exact_and_sorted(self) -> None:
        report = inventory_extracted_works(_corpus())

        self.assertEqual(
            _relationship_values(report.relationships.validation_not_in_train),
            (0x03A9,),
        )
        self.assertEqual(
            _relationship_values(report.relationships.test_not_in_train),
            (0x00A0, 0x03A9, 0x0416),
        )
        self.assertEqual(
            _relationship_values(
                report.relationships.test_not_in_train_or_validation
            ),
            (0x00A0, 0x0416),
        )

    def test_code_points_are_numerically_ordered(self) -> None:
        report = inventory_extracted_works(_corpus())

        for inventory in (
            *report.work_inventories,
            *report.split_inventories,
            report.global_inventory,
        ):
            values = tuple(entry.code_point for entry in inventory.code_points)
            self.assertEqual(values, tuple(sorted(values)))

    def test_repeated_inventory_is_deterministic(self) -> None:
        works = _corpus()

        self.assertEqual(
            inventory_extracted_works(works),
            inventory_extracted_works(works),
        )

    def test_results_are_frozen_and_do_not_store_literal_characters(self) -> None:
        report = inventory_extracted_works(_corpus())
        entry = report.global_inventory.code_points[0]

        with self.assertRaises(FrozenInstanceError):
            entry.count = 99  # type: ignore[misc]
        self.assertNotIn("character", {item.name for item in fields(CodePointCount)})
        self.assertIsInstance(report.work_inventories, tuple)
        self.assertIsInstance(report.global_inventory.code_points, tuple)

    def test_repr_and_safe_summary_hide_sealed_details(self) -> None:
        relationship_report = inventory_extracted_works(_corpus())
        works = _corpus() + (
            _work("sealed-extra", "test", 5, SECRET_TEST_PROSE + "\n"),
        )
        report = inventory_extracted_works(works)
        summary = report.safe_summary()

        report_repr_is_safe = SECRET_TEST_PROSE not in repr(report)
        summary_repr_is_safe = SECRET_TEST_PROSE not in repr(summary)
        test_identity_is_hidden = (
            "U+0416" not in repr(relationship_report)
            and "CYRILLIC CAPITAL LETTER ZHE" not in repr(relationship_report)
        )
        self.assertTrue(report_repr_is_safe)
        self.assertTrue(summary_repr_is_safe)
        self.assertTrue(test_identity_is_hidden)
        self.assertEqual(summary.work_count, 5)
        self.assertFalse(hasattr(summary, "test_not_in_train"))

    def test_carriage_return_is_rejected_without_exposing_prose(self) -> None:
        work = _work("sealed", "test", 1, SECRET_TEST_PROSE + "\r\n")

        with self.assertRaises(InventoryError) as raised:
            inventory_extracted_works((work,))

        payload_is_safe = SECRET_TEST_PROSE not in str(raised.exception)
        self.assertTrue(payload_is_safe)
        self.assertEqual(
            raised.exception.details["invariant"],
            "work.processed_text.carriage_returns",
        )
        self.assertEqual(raised.exception.details["observed"], 1)


class ProductionCharacterInventoryTests(unittest.TestCase):
    def test_production_inventory_uses_eight_in_memory_works_without_outputs(self) -> None:
        processed_root = REPOSITORY_ROOT / "data/processed"
        self.assertFalse(processed_root.exists())
        works = extract_shakespeare_in_memory(REPOSITORY_ROOT)

        report = inventory_extracted_works(works)
        summary = report.safe_summary()

        self.assertEqual(len(report.work_inventories), 8)
        self.assertEqual(summary.train_work_count, 6)
        self.assertEqual(summary.validation_work_count, 1)
        self.assertEqual(summary.test_work_count, 1)
        totals_match = all(
            inventory.total_code_point_count == work.processed_code_point_count
            for inventory, work in zip(report.work_inventories, works)
        )
        self.assertTrue(totals_match)
        self.assertEqual(
            summary.global_total_code_points,
            summary.train_total_code_points
            + summary.validation_total_code_points
            + summary.test_total_code_points,
        )
        manifest = json.loads(
            (
                REPOSITORY_ROOT
                / "docs/data/shakespeare-eight-play-manifest.json"
            ).read_text(encoding="utf-8")
        )
        audit = manifest["processing"]["character_inventory_audit"]
        self.assertEqual(audit["status"], "completed")
        self.assertEqual(
            audit["validation_not_in_train_cardinality"],
            summary.validation_not_in_train_count,
        )
        self.assertEqual(
            audit["test_not_in_train_cardinality"],
            summary.test_not_in_train_count,
        )
        self.assertEqual(
            audit["test_not_in_train_or_validation_cardinality"],
            summary.test_not_in_train_or_validation_count,
        )
        self.assertEqual(audit["occurrence_location_candidate_count"], 0)
        self.assertEqual(audit["occurrence_location_entries"], [])
        self.assertEqual(
            audit["occurrence_location_reporting"],
            "not_applicable_for_pinned_corpus",
        )
        self.assertFalse(audit["test_only_code_point_identities_emitted"])
        self.assertFalse(audit["test_prose_emitted"])
        self.assertEqual(report.relationships.validation_not_in_train, ())
        self.assertEqual(report.relationships.test_not_in_train, ())
        self.assertEqual(report.relationships.test_not_in_train_or_validation, ())
        self.assertFalse(processed_root.exists())


if __name__ == "__main__":
    unittest.main()
