"""Tests for deterministic in-memory Shakespeare extraction."""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from sebgpt.data.shakespeare_extract import (  # noqa: E402
    ExtractionError,
    _validate_only_crlf_to_lf,
    extract_shakespeare_in_memory,
    extract_validated_source,
)
from sebgpt.data.shakespeare_preflight import (  # noqa: E402
    GlobalContentsSpec,
    Position,
    PreflightError,
    PreflightSpec,
    RawExpectations,
    WorkSpec,
    WrapperSpec,
    validate_source,
)


START = "*** START FIXTURE ***"
END = "*** END FIXTURE ***"
ENTRIES = ("THE SONNETS", "WORK A", "WORK B")
SECRET_TEST_PROSE = "SEALED TEST PROSE MUST NEVER APPEAR"


def _fixture_lines() -> list[str]:
    return [
        START,
        "  Contents",
        "",
        "  THE SONNETS",
        "  WORK A",
        "  WORK B",
        "",
        "",
        "THE SONNETS",
        "WORK A",
        "",
        "Contents",
        "ACT LIST",
        "Dramatis Personæ",
        "  INDENTED",
        "TRAILING ",
        "Unicode: Æ—’",
        "FINIS A",
        "",
        "",
        "NEXT A",
        "WORK B",
        "Contents",
        "INDEX B",
        " Dramatis Personæ",
        SECRET_TEST_PROSE,
        "FINIS B",
        "",
        "",
        "NEXT B",
        END,
    ]


def _encode(lines: list[str]) -> bytes:
    return ("\r\n".join(lines) + "\r\n").encode("utf-8")


def _positions(lines: list[str]) -> list[Position]:
    positions: list[Position] = []
    byte_offset = 0
    code_point_offset = 0
    for line_number, value in enumerate(lines, start=1):
        positions.append(Position(line_number, byte_offset, code_point_offset))
        byte_offset += len(value.encode("utf-8")) + 2
        code_point_offset += len(value) + 2
    return positions


def _validated_fixture(lines: list[str]):
    raw_bytes = _encode(lines)
    text = raw_bytes.decode("utf-8")
    positions = _positions(lines)
    heading_index = lines.index("  Contents")
    first_entry_index = heading_index + 2
    end_exclusive_index = first_entry_index + len(ENTRIES)
    first_body_index = lines.index("THE SONNETS", end_exclusive_index)
    crlf_count = raw_bytes.count(b"\r\n")

    works = []
    for order, (work_id, title, split, body, successor, dramatis_indent) in enumerate(
        (
            ("work-a", "Work A", "train", "WORK A", "NEXT A", 0),
            ("twelfth-night", "Work B", "test", "WORK B", "NEXT B", 1),
        ),
        start=1,
    ):
        body_index = lines.index(body, first_body_index)
        successor_index = lines.index(successor, first_body_index)
        works.append(
            WorkSpec(
                work_id=work_id,
                body_marker=body,
                successor_marker=successor,
                body_position=positions[body_index],
                successor_line=positions[successor_index].line_number,
                expected_empty_lines_before_successor=2,
                expected_whitespace_only_lines_before_successor=0,
                title=title,
                split=split,
                manifest_order=order,
                dramatis_leading_u0020_count=dramatis_indent,
            )
        )

    spec = PreflightSpec(
        raw_path=Path("fixture.txt"),
        raw=RawExpectations(
            sha256=hashlib.sha256(raw_bytes).hexdigest(),
            byte_count=len(raw_bytes),
            unicode_code_points=len(text),
            crlf_pairs=crlf_count,
            lone_cr=0,
            lone_lf=0,
            ends_with_crlf=True,
            nul_bytes=0,
            utf8_bom_present=False,
        ),
        wrapper=WrapperSpec(
            start_marker=START,
            end_marker=END,
            required_occurrences_each=1,
            start_position=positions[lines.index(START)],
            end_position=positions[lines.index(END)],
        ),
        global_contents=GlobalContentsSpec(
            heading="  Contents",
            required_heading_occurrences=1,
            empty_lines_after_heading=1,
            entry_prefix="  ",
            entries=ENTRIES,
            expected_entry_count=len(ENTRIES),
            empty_lines_after_entries=2,
            first_body_marker="THE SONNETS",
            heading_position=positions[heading_index],
            first_entry_position=positions[first_entry_index],
            end_exclusive_position=positions[end_exclusive_index],
            first_body_position=positions[first_body_index],
        ),
        works=tuple(works),
        local_contents_marker="Contents",
        dramatis_canonical_text="Dramatis Personæ",
        required_dramatis_occurrences=1,
    )
    return validate_source(raw_bytes, spec)


def _extract_fixture(lines: list[str] | None = None):
    source = _validated_fixture(_fixture_lines() if lines is None else lines)
    return source, extract_validated_source(source)


class InMemoryExtractionTests(unittest.TestCase):
    def test_outer_range_and_successor_separator_are_exact(self) -> None:
        lines = _fixture_lines()
        source, works = _extract_fixture(lines)
        work = works[0]
        body_index = lines.index("WORK A", lines.index("THE SONNETS", 6))
        outer_end_index = lines.index("NEXT A") - 2
        expected_outer = _encode(lines[body_index:outer_end_index])

        self.assertEqual(work.outer_raw.byte_count, len(expected_outer))
        self.assertEqual(work.outer_raw.sha256, hashlib.sha256(expected_outer).hexdigest())
        self.assertEqual(
            work.outer_range.end_byte_offset,
            source.logical_lines[outer_end_index].position.byte_offset,
        )

    def test_local_contents_is_removed_and_dramatis_forms_are_retained(self) -> None:
        _, works = _extract_fixture()
        work_a, work_b = works

        self.assertNotIn("\nContents\n", work_a.processed_text)
        self.assertIn("\nDramatis Personæ\n", work_a.processed_text)
        sealed_marker_is_retained = "\n Dramatis Personæ\n" in work_b.processed_text
        self.assertTrue(sealed_marker_is_retained)

    def test_extra_dramatis_indentation_variant_is_rejected(self) -> None:
        lines = _fixture_lines()
        lines.insert(lines.index("Dramatis Personæ") + 1, "  Dramatis Personæ")

        with self.assertRaises(ExtractionError) as raised:
            _extract_fixture(lines)

        self.assertEqual(
            raised.exception.details["invariant"],
            "dramatis_personae.unexpected_variant_count",
        )

    def test_trailing_space_dramatis_variant_is_rejected(self) -> None:
        lines = _fixture_lines()
        lines.insert(lines.index("Dramatis Personæ") + 1, "Dramatis Personæ ")

        with self.assertRaises(ExtractionError) as raised:
            _extract_fixture(lines)

        self.assertEqual(
            raised.exception.details["invariant"],
            "dramatis_personae.unexpected_variant_count",
        )

    def test_zero_space_variant_is_rejected_for_one_space_work(self) -> None:
        lines = _fixture_lines()
        lines.insert(lines.index(" Dramatis Personæ") + 1, "Dramatis Personæ")

        with self.assertRaises(ExtractionError) as raised:
            _extract_fixture(lines)

        self.assertEqual(raised.exception.details["work_id"], "twelfth-night")
        self.assertEqual(
            raised.exception.details["invariant"],
            "dramatis_personae.unexpected_variant_count",
        )

    def test_missing_local_contents_is_rejected(self) -> None:
        lines = _fixture_lines()
        lines[lines.index("Contents", lines.index("WORK A"))] = "CONTENTS"

        with self.assertRaises(ExtractionError) as raised:
            _extract_fixture(lines)

        self.assertEqual(raised.exception.details["invariant"], "local_contents.cardinality")

    def test_duplicate_local_contents_is_rejected(self) -> None:
        lines = _fixture_lines()
        lines.insert(lines.index("ACT LIST") + 1, "Contents")

        with self.assertRaises(ExtractionError) as raised:
            _extract_fixture(lines)

        self.assertEqual(raised.exception.details["invariant"], "local_contents.cardinality")

    def test_missing_dramatis_is_rejected(self) -> None:
        lines = _fixture_lines()
        lines[lines.index("Dramatis Personæ")] = "DRAMATIS PERSONAE"

        with self.assertRaises(ExtractionError) as raised:
            _extract_fixture(lines)

        self.assertEqual(raised.exception.details["invariant"], "dramatis_personae.cardinality")

    def test_duplicate_dramatis_is_rejected(self) -> None:
        lines = _fixture_lines()
        lines.insert(lines.index("Dramatis Personæ") + 1, "Dramatis Personæ")

        with self.assertRaises(ExtractionError) as raised:
            _extract_fixture(lines)

        self.assertEqual(raised.exception.details["invariant"], "dramatis_personae.cardinality")

    def test_reordered_local_markers_are_rejected(self) -> None:
        lines = _fixture_lines()
        contents_index = lines.index("Contents", lines.index("WORK A"))
        dramatis_index = lines.index("Dramatis Personæ")
        lines[contents_index], lines[dramatis_index] = (
            lines[dramatis_index],
            lines[contents_index],
        )

        with self.assertRaises(ExtractionError) as raised:
            _extract_fixture(lines)

        self.assertEqual(raised.exception.details["invariant"], "dramatis_personae.cardinality")

    def test_title_formatting_unicode_and_minimal_normalization_are_preserved(self) -> None:
        lines = _fixture_lines()
        _, works = _extract_fixture(lines)
        work = works[0]
        body_index = lines.index("WORK A", lines.index("THE SONNETS", 6))
        contents_index = lines.index("Contents", body_index)
        dramatis_index = lines.index("Dramatis Personæ", contents_index)
        outer_end_index = lines.index("NEXT A") - 2
        retained_lines = lines[body_index:contents_index] + lines[dramatis_index:outer_end_index]
        expected_text = "\n".join(retained_lines) + "\n"

        self.assertEqual(work.processed_text, expected_text)
        self.assertTrue(work.processed_text.startswith("WORK A\n\n"))
        self.assertIn("  INDENTED\n", work.processed_text)
        self.assertIn("TRAILING \n", work.processed_text)
        self.assertIn("Unicode: Æ—’\n", work.processed_text)
        self.assertNotIn("NEXT A", work.processed_text)
        self.assertNotIn("\r", work.processed_text)
        self.assertTrue(work.processed_text.endswith("\n"))
        self.assertFalse(work.processed_text.endswith("\n\n"))

    def test_three_provenance_stages_use_their_exact_byte_domains(self) -> None:
        lines = _fixture_lines()
        _, works = _extract_fixture(lines)
        work = works[0]
        body_index = lines.index("WORK A", lines.index("THE SONNETS", 6))
        contents_index = lines.index("Contents", body_index)
        dramatis_index = lines.index("Dramatis Personæ", contents_index)
        outer_end_index = lines.index("NEXT A") - 2
        outer = _encode(lines[body_index:outer_end_index])
        retained = _encode(
            lines[body_index:contents_index] + lines[dramatis_index:outer_end_index]
        )
        processed = retained.replace(b"\r\n", b"\n")

        self.assertEqual(work.outer_raw.sha256, hashlib.sha256(outer).hexdigest())
        self.assertEqual(work.retained_raw.sha256, hashlib.sha256(retained).hexdigest())
        self.assertEqual(work.processed.sha256, hashlib.sha256(processed).hexdigest())
        self.assertEqual(
            len({work.outer_raw.sha256, work.retained_raw.sha256, work.processed.sha256}),
            3,
        )

    def test_normalization_failure_diagnostics_are_content_safe(self) -> None:
        retained_raw = SECRET_TEST_PROSE.encode("utf-8") + b"\r\n"
        mismatched_processed = b"different\n"

        with self.assertRaises(ExtractionError) as raised:
            _validate_only_crlf_to_lf(
                retained_raw,
                mismatched_processed,
                "twelfth-night",
            )

        payload = str(raised.exception)
        payload_is_safe = SECRET_TEST_PROSE not in payload
        self.assertTrue(payload_is_safe)
        self.assertEqual(
            json.loads(payload),
            {
                "expected": True,
                "invariant": "processed.only_crlf_to_lf",
                "observed": False,
                "work_id": "twelfth-night",
            },
        )

    def test_repeated_extraction_is_deterministic(self) -> None:
        source = _validated_fixture(_fixture_lines())

        self.assertEqual(
            extract_validated_source(source),
            extract_validated_source(source),
        )

    def test_extracted_work_repr_hides_sealed_text(self) -> None:
        _, works = _extract_fixture()

        sealed_text_is_hidden = SECRET_TEST_PROSE not in repr(works[1])
        self.assertTrue(sealed_text_is_hidden)

    def test_sealed_test_error_hides_source_text(self) -> None:
        lines = _fixture_lines()
        second_contents = lines.index("Contents", lines.index("WORK B"))
        lines[second_contents] = "CONTENTS"

        with self.assertRaises(ExtractionError) as raised:
            _extract_fixture(lines)

        self.assertEqual(raised.exception.details["work_id"], "twelfth-night")
        sealed_text_is_hidden = SECRET_TEST_PROSE not in str(raised.exception)
        self.assertTrue(sealed_text_is_hidden)

    def test_whitespace_only_successor_separator_is_rejected_by_preflight(self) -> None:
        lines = _fixture_lines()
        lines[lines.index("NEXT A") - 1] = " "

        with self.assertRaises(PreflightError) as raised:
            _extract_fixture(lines)

        self.assertEqual(
            raised.exception.details["invariant"],
            "work.successor_separator.whitespace_only_count",
        )


class ProductionInMemoryExtractionTests(unittest.TestCase):
    def test_production_extraction_returns_eight_independent_works_without_outputs(self) -> None:
        processed_root = REPOSITORY_ROOT / "data/processed"
        self.assertFalse(processed_root.exists())

        works = extract_shakespeare_in_memory(REPOSITORY_ROOT)

        self.assertEqual(len(works), 8)
        self.assertEqual(len({work.work_id for work in works}), 8)
        self.assertEqual(len({work.manifest_order for work in works}), 8)
        self.assertEqual(sum(work.split == "train" for work in works), 6)
        self.assertEqual(sum(work.split == "validation" for work in works), 1)
        self.assertEqual(sum(work.split == "test" for work in works), 1)
        self.assertEqual(len({work.processed.sha256 for work in works}), 8)
        self.assertFalse(processed_root.exists())


if __name__ == "__main__":
    unittest.main()
