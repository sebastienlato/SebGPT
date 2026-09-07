"""Focused tests for the read-only Shakespeare structural preflight."""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from sebgpt.data.shakespeare_preflight import (  # noqa: E402
    GlobalContentsSpec,
    Position,
    PreflightError,
    PreflightSpec,
    RawExpectations,
    WorkSpec,
    WrapperSpec,
    run_preflight,
    validate_preflight,
    validate_source,
)


START = "*** START FIXTURE ***"
END = "*** END FIXTURE ***"
ENTRIES = ("THE SONNETS", "WORK A", "WORK B")
SECRET_TEST_PROSE = "SEALED TEST PROSE MUST NEVER APPEAR"


def _tree_identity(root: Path):
    if not root.exists():
        return None
    directories = tuple(
        sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_dir())
    )
    files = tuple(
        sorted(
            (
                path.relative_to(root).as_posix(),
                path.stat().st_ino,
                path.stat().st_size,
                path.stat().st_mtime_ns,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in root.rglob("*")
            if path.is_file()
        )
    )
    return directories, files


def _fixture_lines(test_prose: str = "SAFE B") -> list[str]:
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
        "SAFE A",
        "FILLER",
        "",
        "",
        "NEXT A",
        "WORK B",
        test_prose,
        "",
        "",
        "NEXT B",
        END,
    ]


def _encode(lines: list[str]) -> bytes:
    return ("\r\n".join(lines) + "\r\n").encode("utf-8")


def _positions(lines: list[str]) -> list[Position]:
    result: list[Position] = []
    byte_offset = 0
    code_point_offset = 0
    for line_number, value in enumerate(lines, start=1):
        result.append(Position(line_number, byte_offset, code_point_offset))
        byte_offset += len(value.encode("utf-8")) + 2
        code_point_offset += len(value) + 2
    return result


def _raw_expectations(raw_bytes: bytes) -> RawExpectations:
    text = raw_bytes.decode("utf-8")
    crlf_pairs = raw_bytes.count(b"\r\n")
    return RawExpectations(
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        byte_count=len(raw_bytes),
        unicode_code_points=len(text),
        crlf_pairs=crlf_pairs,
        lone_cr=raw_bytes.count(b"\r") - crlf_pairs,
        lone_lf=raw_bytes.count(b"\n") - crlf_pairs,
        ends_with_crlf=True,
        nul_bytes=0,
        utf8_bom_present=False,
    )


def _spec_for_lines(lines: list[str]) -> tuple[bytes, PreflightSpec]:
    raw_bytes = _encode(lines)
    positions = _positions(lines)
    heading_index = lines.index("  Contents")
    first_entry_index = heading_index + 2
    end_exclusive_index = first_entry_index + len(ENTRIES)
    first_body_index = lines.index("THE SONNETS", end_exclusive_index)

    works = []
    for work_id, body_marker, successor_marker in (
        ("work-a", "WORK A", "NEXT A"),
        ("twelfth-night", "WORK B", "NEXT B"),
    ):
        body_index = lines.index(body_marker, first_body_index)
        successor_index = lines.index(successor_marker, first_body_index)
        works.append(
            WorkSpec(
                work_id=work_id,
                body_marker=body_marker,
                successor_marker=successor_marker,
                body_position=positions[body_index],
                successor_line=positions[successor_index].line_number,
                expected_empty_lines_before_successor=2,
                expected_whitespace_only_lines_before_successor=0,
            )
        )

    spec = PreflightSpec(
        raw_path=Path("fixture.txt"),
        raw=_raw_expectations(raw_bytes),
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
    )
    return raw_bytes, spec


def _with_matching_raw_identity(spec: PreflightSpec, raw_bytes: bytes) -> PreflightSpec:
    return replace(spec, raw=_raw_expectations(raw_bytes))


class SyntheticPreflightTests(unittest.TestCase):
    def test_correct_hash_is_accepted(self) -> None:
        raw_bytes, spec = _spec_for_lines(_fixture_lines())

        report = validate_preflight(raw_bytes, spec)

        self.assertEqual(report.raw_sha256, hashlib.sha256(raw_bytes).hexdigest())

    def test_incorrect_hash_is_rejected_before_decoding(self) -> None:
        _, spec = _spec_for_lines(_fixture_lines())

        with self.assertRaises(PreflightError) as raised:
            validate_preflight(b"\xff", spec)

        self.assertEqual(raised.exception.details["invariant"], "raw.sha256")

    def test_invalid_utf8_is_rejected_after_matching_hash(self) -> None:
        _, spec = _spec_for_lines(_fixture_lines())
        invalid_utf8 = b"\xff\r\n"
        raw = RawExpectations(
            sha256=hashlib.sha256(invalid_utf8).hexdigest(),
            byte_count=len(invalid_utf8),
            unicode_code_points=0,
            crlf_pairs=1,
            lone_cr=0,
            lone_lf=0,
            ends_with_crlf=True,
            nul_bytes=0,
            utf8_bom_present=False,
        )

        with self.assertRaises(PreflightError) as raised:
            validate_preflight(invalid_utf8, replace(spec, raw=raw))

        self.assertEqual(raised.exception.details["invariant"], "raw.strict_utf8")

    def test_lone_lf_is_rejected_before_marker_scanning(self) -> None:
        raw_bytes, spec = _spec_for_lines(_fixture_lines())
        changed = raw_bytes.replace(b"\r\n", b"\n", 1)
        changed_raw = replace(
            spec.raw,
            sha256=hashlib.sha256(changed).hexdigest(),
            byte_count=len(changed),
        )

        with self.assertRaises(PreflightError) as raised:
            validate_preflight(changed, replace(spec, raw=changed_raw))

        self.assertEqual(raised.exception.details["invariant"], "raw.crlf_pairs")

    def test_unexpected_utf8_bom_is_rejected_without_stripping(self) -> None:
        raw_bytes, spec = _spec_for_lines(_fixture_lines())
        changed = b"\xef\xbb\xbf" + raw_bytes
        changed_raw = replace(
            spec.raw,
            sha256=hashlib.sha256(changed).hexdigest(),
            byte_count=len(changed),
        )

        with self.assertRaises(PreflightError) as raised:
            validate_preflight(changed, replace(spec, raw=changed_raw))

        self.assertEqual(
            raised.exception.details["invariant"], "raw.utf8_bom_present"
        )

    def test_missing_work_marker_is_rejected(self) -> None:
        lines = _fixture_lines()
        raw_bytes, spec = _spec_for_lines(lines)
        lines[lines.index("WORK A", lines.index("THE SONNETS", 6))] = "BROKEN"
        changed = _encode(lines)

        with self.assertRaises(PreflightError) as raised:
            validate_preflight(changed, _with_matching_raw_identity(spec, changed))

        self.assertEqual(
            raised.exception.details["invariant"], "work.body_marker.cardinality"
        )

    def test_duplicate_work_marker_is_rejected(self) -> None:
        lines = _fixture_lines()
        raw_bytes, spec = _spec_for_lines(lines)
        lines[lines.index("FILLER")] = "WORK A"
        changed = _encode(lines)

        with self.assertRaises(PreflightError) as raised:
            validate_preflight(changed, _with_matching_raw_identity(spec, changed))

        self.assertEqual(
            raised.exception.details["invariant"], "work.body_marker.cardinality"
        )

    def test_reordered_work_and_successor_markers_are_rejected(self) -> None:
        lines = _fixture_lines()
        raw_bytes, spec = _spec_for_lines(lines)
        body_index = lines.index("WORK A", lines.index("THE SONNETS", 6))
        successor_index = lines.index("NEXT A")
        lines[body_index], lines[successor_index] = (
            lines[successor_index],
            lines[body_index],
        )
        changed = _encode(lines)

        with self.assertRaises(PreflightError) as raised:
            validate_preflight(changed, _with_matching_raw_identity(spec, changed))

        self.assertEqual(raised.exception.details["invariant"], "work.marker_order")

    def test_incorrect_global_contents_indentation_is_rejected(self) -> None:
        lines = _fixture_lines()
        raw_bytes, spec = _spec_for_lines(lines)
        lines[lines.index("  Contents")] = " Contents "
        changed = _encode(lines)

        with self.assertRaises(PreflightError) as raised:
            validate_preflight(changed, _with_matching_raw_identity(spec, changed))

        self.assertEqual(
            raised.exception.details["invariant"],
            "global_contents.heading.cardinality",
        )

    def test_incorrect_global_entry_order_is_rejected(self) -> None:
        lines = _fixture_lines()
        raw_bytes, spec = _spec_for_lines(lines)
        first = lines.index("  WORK A")
        second = lines.index("  WORK B")
        lines[first], lines[second] = lines[second], lines[first]
        changed = _encode(lines)

        with self.assertRaises(PreflightError) as raised:
            validate_preflight(changed, _with_matching_raw_identity(spec, changed))

        self.assertEqual(
            raised.exception.details["invariant"], "global_contents.entry_sequence"
        )

    def test_incorrect_global_entry_count_is_rejected(self) -> None:
        lines = _fixture_lines()
        lines.pop(lines.index("  WORK B"))
        changed, spec = _spec_for_lines(lines)

        with self.assertRaises(PreflightError) as raised:
            validate_preflight(changed, spec)

        self.assertEqual(
            raised.exception.details["invariant"], "global_contents.entry_sequence"
        )

    def test_successor_separator_count_mismatch_is_rejected(self) -> None:
        lines = _fixture_lines()
        next_a_index = lines.index("NEXT A")
        lines.pop(next_a_index - 1)
        changed, spec = _spec_for_lines(lines)

        with self.assertRaises(PreflightError) as raised:
            validate_preflight(changed, spec)

        self.assertEqual(
            raised.exception.details["invariant"],
            "work.successor_separator.empty_count",
        )

    def test_whitespace_only_successor_separator_is_rejected(self) -> None:
        lines = _fixture_lines()
        lines[lines.index("NEXT A") - 1] = " "
        changed, spec = _spec_for_lines(lines)

        with self.assertRaises(PreflightError) as raised:
            validate_preflight(changed, spec)

        self.assertEqual(
            raised.exception.details["invariant"],
            "work.successor_separator.whitespace_only_count",
        )

    def test_assertion_offset_mismatch_is_rejected(self) -> None:
        raw_bytes, spec = _spec_for_lines(_fixture_lines())
        first_work = spec.works[0]
        wrong_position = replace(
            first_work.body_position,
            byte_offset=first_work.body_position.byte_offset + 1,
        )
        wrong_work = replace(first_work, body_position=wrong_position)

        with self.assertRaises(PreflightError) as raised:
            validate_preflight(raw_bytes, replace(spec, works=(wrong_work, *spec.works[1:])))

        self.assertEqual(
            raised.exception.details["invariant"], "work.body_marker.position"
        )

    def test_sealed_test_prose_is_absent_from_reports_and_errors(self) -> None:
        raw_bytes, spec = _spec_for_lines(_fixture_lines(SECRET_TEST_PROSE))
        report_json = json.dumps(validate_preflight(raw_bytes, spec).to_dict())
        self.assertNotIn(SECRET_TEST_PROSE, report_json)

        test_work = spec.works[1]
        wrong_position = replace(
            test_work.body_position,
            code_point_offset=test_work.body_position.code_point_offset + 1,
        )
        wrong_test_work = replace(test_work, body_position=wrong_position)
        with self.assertRaises(PreflightError) as raised:
            validate_preflight(
                raw_bytes,
                replace(spec, works=(spec.works[0], wrong_test_work)),
            )

        self.assertNotIn(SECRET_TEST_PROSE, str(raised.exception))
        self.assertEqual(raised.exception.details["work_id"], "twelfth-night")

    def test_validated_source_keeps_exact_bytes_and_hides_text_from_repr(self) -> None:
        raw_bytes, spec = _spec_for_lines(_fixture_lines(SECRET_TEST_PROSE))

        source = validate_source(raw_bytes, spec)

        self.assertIs(source.raw_bytes, raw_bytes)
        self.assertNotIn(SECRET_TEST_PROSE, repr(source))


class ProductionPreflightTests(unittest.TestCase):
    def test_pinned_source_passes_without_mutating_processed_outputs(self) -> None:
        processed_root = REPOSITORY_ROOT / "data/processed"
        processed_before = _tree_identity(processed_root)

        report = run_preflight(REPOSITORY_ROOT)

        self.assertEqual(
            report.raw_sha256,
            "3cf4b3d44ee14cff4e14e78e2ad3318eff76f3f7f2afc3cee6bb925879110a37",
        )
        self.assertEqual(report.global_entry_count, 44)
        self.assertEqual(len(report.works), 8)
        self.assertEqual(_tree_identity(processed_root), processed_before)


if __name__ == "__main__":
    unittest.main()
