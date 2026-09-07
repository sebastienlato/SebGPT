"""Read-only validation of the pinned Shakespeare source.

This module deliberately stops before extraction.  It proves that the raw bytes
and the structural anchors match the approved repository contract, then returns
only structural and numeric facts.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = Path("docs/data/shakespeare-eight-play-manifest.json")


class PreflightError(Exception):
    """A contract failure whose message never contains source text."""

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
class Position:
    """One-based line number and zero-based raw byte/code-point offsets."""

    line_number: int
    byte_offset: int
    code_point_offset: int


@dataclass(frozen=True)
class RawExpectations:
    sha256: str
    byte_count: int
    unicode_code_points: int
    crlf_pairs: int
    lone_cr: int
    lone_lf: int
    ends_with_crlf: bool
    nul_bytes: int
    utf8_bom_present: bool


@dataclass(frozen=True)
class WrapperSpec:
    start_marker: str
    end_marker: str
    required_occurrences_each: int
    start_position: Position
    end_position: Position


@dataclass(frozen=True)
class GlobalContentsSpec:
    heading: str
    required_heading_occurrences: int
    empty_lines_after_heading: int
    entry_prefix: str
    entries: tuple[str, ...]
    expected_entry_count: int
    empty_lines_after_entries: int
    first_body_marker: str
    heading_position: Position
    first_entry_position: Position
    end_exclusive_position: Position
    first_body_position: Position


@dataclass(frozen=True)
class WorkSpec:
    work_id: str
    body_marker: str
    successor_marker: str
    body_position: Position
    successor_line: int
    expected_empty_lines_before_successor: int
    expected_whitespace_only_lines_before_successor: int
    title: str = ""
    split: str = ""
    manifest_order: int = 0
    dramatis_leading_u0020_count: int = 0


@dataclass(frozen=True)
class PreflightSpec:
    raw_path: Path
    raw: RawExpectations
    wrapper: WrapperSpec
    global_contents: GlobalContentsSpec
    works: tuple[WorkSpec, ...]
    local_contents_marker: str = "Contents"
    dramatis_canonical_text: str = "Dramatis Personæ"
    required_dramatis_occurrences: int = 1


@dataclass(frozen=True)
class WorkPreflightResult:
    work_id: str
    body_position: Position
    successor_position: Position
    empty_lines_before_successor: int
    whitespace_only_lines_before_successor: int


@dataclass(frozen=True)
class PreflightReport:
    raw_sha256: str
    raw_byte_count: int
    raw_unicode_code_points: int
    raw_crlf_pairs: int
    logical_line_count: int
    start_position: Position
    end_position: Position
    global_heading_position: Position
    global_entry_count: int
    first_body_position: Position
    works: tuple[WorkPreflightResult, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready structural facts without any source excerpts."""

        return asdict(self)


@dataclass(frozen=True)
class LogicalLine:
    value: str = field(repr=False)
    position: Position


@dataclass(frozen=True)
class ValidatedWorkMarkers:
    """Safe numeric indexes discovered by generic preflight matching."""

    work_id: str
    body_line_index: int
    successor_line_index: int
    outer_end_line_index: int
    outer_end_byte_offset: int
    outer_end_code_point_offset: int


@dataclass(frozen=True)
class ValidatedSource:
    """Immutable handoff binding extraction to the exact preflighted bytes."""

    spec: PreflightSpec
    raw_bytes: bytes = field(repr=False)
    logical_lines: tuple[LogicalLine, ...] = field(repr=False)
    work_markers: tuple[ValidatedWorkMarkers, ...]
    report: PreflightReport


def _require_equal(
    invariant: str,
    expected: Any,
    observed: Any,
    *,
    work_id: str | None = None,
) -> None:
    if observed != expected:
        raise PreflightError(
            invariant, expected=expected, observed=observed, work_id=work_id
        )


def _position_from_mapping(
    values: dict[str, Any], prefix: str = "observed_"
) -> Position:
    return Position(
        line_number=values[f"{prefix}line"],
        byte_offset=values[f"{prefix}byte"],
        code_point_offset=values[f"{prefix}code_point"],
    )


def load_preflight_spec(repository_root: Path = REPOSITORY_ROOT) -> PreflightSpec:
    """Load the approved machine-readable contract from the repository."""

    manifest_file = repository_root / MANIFEST_PATH
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    source = manifest["source"]
    raw_verification = manifest["raw_verification"]
    contract = manifest["processing"]["extraction_contract"]
    wrapper = contract["gutenberg_wrapper"]
    global_contents = contract["global_contents"]
    local_contents = contract["play_local_contents_marker"]
    dramatis = contract["dramatis_personae_marker"]

    raw = RawExpectations(
        sha256=source["raw_sha256"],
        byte_count=source["raw_byte_count"],
        unicode_code_points=raw_verification["raw_unicode_code_points"],
        crlf_pairs=raw_verification["crlf_pairs"],
        lone_cr=raw_verification["lone_cr"],
        lone_lf=raw_verification["lone_lf"],
        ends_with_crlf=raw_verification["ends_with_crlf"],
        nul_bytes=raw_verification["nul_bytes"],
        utf8_bom_present=raw_verification["utf8_bom_present"],
    )
    wrapper_spec = WrapperSpec(
        start_marker=wrapper["start_marker"],
        end_marker=wrapper["end_marker"],
        required_occurrences_each=wrapper["required_occurrences_each"],
        start_position=_position_from_mapping(wrapper, "observed_start_"),
        end_position=_position_from_mapping(wrapper, "observed_end_"),
    )
    global_spec = GlobalContentsSpec(
        heading=(
            " " * global_contents["heading_leading_u0020_count"]
            + global_contents["heading_text"]
            + " " * global_contents["heading_trailing_space_count"]
        ),
        required_heading_occurrences=global_contents["required_heading_occurrences"],
        empty_lines_after_heading=global_contents["empty_crlf_lines_after_heading"],
        entry_prefix=" " * global_contents["entry_leading_u0020_count"],
        entries=tuple(global_contents["expected_entries_without_prefix"]),
        expected_entry_count=global_contents["expected_entry_count"],
        empty_lines_after_entries=global_contents["empty_crlf_lines_after_entries"],
        first_body_marker=global_contents["first_body_marker_after_region"],
        heading_position=_position_from_mapping(global_contents, "observed_heading_"),
        first_entry_position=_position_from_mapping(
            global_contents, "observed_first_entry_"
        ),
        end_exclusive_position=_position_from_mapping(
            global_contents, "observed_end_exclusive_"
        ),
        first_body_position=_position_from_mapping(
            global_contents, "observed_first_body_marker_"
        ),
    )
    works = tuple(
        WorkSpec(
            work_id=work["work_id"],
            body_marker=work["body_title_marker"],
            successor_marker=work["successor_title_marker"],
            body_position=Position(
                line_number=work["observed_start_line"],
                byte_offset=work["observed_start_byte"],
                code_point_offset=work["observed_start_code_point"],
            ),
            successor_line=work["observed_successor_line"],
            expected_empty_lines_before_successor=work[
                "expected_empty_crlf_lines_before_successor"
            ],
            expected_whitespace_only_lines_before_successor=work[
                "expected_whitespace_only_lines_before_successor"
            ],
            title=work["title"],
            split=work["split"],
            manifest_order=work["order"],
            dramatis_leading_u0020_count=work[
                "dramatis_leading_u0020_count"
            ],
        )
        for work in manifest["works"]
    )
    return PreflightSpec(
        raw_path=Path(source["raw_repository_path"]),
        raw=raw,
        wrapper=wrapper_spec,
        global_contents=global_spec,
        works=works,
        local_contents_marker=local_contents["logical_line"],
        dramatis_canonical_text=dramatis["canonical_text"],
        required_dramatis_occurrences=dramatis[
            "required_occurrences_per_outer_range"
        ],
    )


def _validate_raw_bytes(raw_bytes: bytes, expected: RawExpectations) -> str:
    """Validate byte identity before attempting any text decoding."""

    observed_hash = hashlib.sha256(raw_bytes).hexdigest()
    _require_equal("raw.sha256", expected.sha256, observed_hash)
    _require_equal("raw.byte_count", expected.byte_count, len(raw_bytes))

    crlf_pairs = raw_bytes.count(b"\r\n")
    lone_cr = raw_bytes.count(b"\r") - crlf_pairs
    lone_lf = raw_bytes.count(b"\n") - crlf_pairs
    raw_facts = (
        ("raw.crlf_pairs", expected.crlf_pairs, crlf_pairs),
        ("raw.lone_cr", expected.lone_cr, lone_cr),
        ("raw.lone_lf", expected.lone_lf, lone_lf),
        ("raw.ends_with_crlf", expected.ends_with_crlf, raw_bytes.endswith(b"\r\n")),
        ("raw.nul_bytes", expected.nul_bytes, raw_bytes.count(b"\x00")),
        (
            "raw.utf8_bom_present",
            expected.utf8_bom_present,
            raw_bytes.startswith(b"\xef\xbb\xbf"),
        ),
    )
    for invariant, expected_value, observed_value in raw_facts:
        _require_equal(invariant, expected_value, observed_value)

    try:
        text = raw_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise PreflightError(
            "raw.strict_utf8",
            expected={"decodes": True},
            observed={
                "decodes": False,
                "start_byte": error.start,
                "end_byte": error.end,
                "reason": error.reason,
            },
        ) from None

    _require_equal("raw.unicode_code_points", expected.unicode_code_points, len(text))
    return text


def _logical_lines(text: str) -> tuple[LogicalLine, ...]:
    """Split only at CRLF while retaining explicit raw positions."""

    parts = text.split("\r\n")
    # The raw contract requires a terminal CRLF, so the final split item is a
    # sentinel after the last terminator rather than another logical line.
    values = parts[:-1]
    lines: list[LogicalLine] = []
    byte_offset = 0
    code_point_offset = 0
    for line_number, value in enumerate(values, start=1):
        lines.append(
            LogicalLine(
                value=value,
                position=Position(line_number, byte_offset, code_point_offset),
            )
        )
        byte_offset += len(value.encode("utf-8")) + 2
        code_point_offset += len(value) + 2
    return tuple(lines)


def _find_line_indices(
    lines: tuple[LogicalLine, ...],
    marker: str,
    *,
    start: int = 0,
    stop: int | None = None,
) -> list[int]:
    upper_bound = len(lines) if stop is None else stop
    return [
        index
        for index in range(start, upper_bound)
        if lines[index].value == marker
    ]


def _assert_position(
    invariant: str,
    expected: Position,
    observed: Position,
    *,
    work_id: str | None = None,
) -> None:
    _require_equal(
        invariant,
        asdict(expected),
        asdict(observed),
        work_id=work_id,
    )


def _validate_wrapper(
    lines: tuple[LogicalLine, ...], spec: WrapperSpec
) -> tuple[int, int]:
    start_indices = _find_line_indices(lines, spec.start_marker)
    end_indices = _find_line_indices(lines, spec.end_marker)
    _require_equal(
        "wrapper.start.cardinality",
        spec.required_occurrences_each,
        len(start_indices),
    )
    _require_equal(
        "wrapper.end.cardinality", spec.required_occurrences_each, len(end_indices)
    )
    start_index = start_indices[0]
    end_index = end_indices[0]
    _assert_position(
        "wrapper.start.position", spec.start_position, lines[start_index].position
    )
    _assert_position("wrapper.end.position", spec.end_position, lines[end_index].position)
    _require_equal("wrapper.order", True, start_index < end_index)
    return start_index, end_index


def _validate_global_contents(
    lines: tuple[LogicalLine, ...], spec: GlobalContentsSpec
) -> tuple[int, int]:
    _require_equal(
        "global_contents.contract_entry_count",
        spec.expected_entry_count,
        len(spec.entries),
    )
    heading_indices = _find_line_indices(lines, spec.heading)
    _require_equal(
        "global_contents.heading.cardinality",
        spec.required_heading_occurrences,
        len(heading_indices),
    )
    heading_index = heading_indices[0]
    _assert_position(
        "global_contents.heading.position",
        spec.heading_position,
        lines[heading_index].position,
    )

    first_entry_index = heading_index + 1 + spec.empty_lines_after_heading
    for offset in range(1, spec.empty_lines_after_heading + 1):
        line_index = heading_index + offset
        observed_empty = line_index < len(lines) and lines[line_index].value == ""
        _require_equal(
            "global_contents.empty_line_after_heading",
            True,
            observed_empty,
        )

    _assert_position(
        "global_contents.first_entry.position",
        spec.first_entry_position,
        lines[first_entry_index].position,
    )
    for entry_number, expected_title in enumerate(spec.entries, start=1):
        line_index = first_entry_index + entry_number - 1
        matches = (
            line_index < len(lines)
            and lines[line_index].value == spec.entry_prefix + expected_title
        )
        _require_equal(
            "global_contents.entry_sequence",
            {"matches": True, "entry_number": entry_number},
            {"matches": matches, "entry_number": entry_number},
        )

    end_exclusive_index = first_entry_index + len(spec.entries)
    _assert_position(
        "global_contents.end_exclusive.position",
        spec.end_exclusive_position,
        lines[end_exclusive_index].position,
    )
    for offset in range(spec.empty_lines_after_entries):
        line_index = end_exclusive_index + offset
        observed_empty = line_index < len(lines) and lines[line_index].value == ""
        _require_equal(
            "global_contents.empty_line_after_entries",
            True,
            observed_empty,
        )

    first_body_index = end_exclusive_index + spec.empty_lines_after_entries
    first_body_matches = (
        first_body_index < len(lines)
        and lines[first_body_index].value == spec.first_body_marker
    )
    _require_equal("global_contents.first_body_marker", True, first_body_matches)
    _assert_position(
        "global_contents.first_body.position",
        spec.first_body_position,
        lines[first_body_index].position,
    )
    return heading_index, first_body_index


def _separator_counts(
    lines: tuple[LogicalLine, ...], successor_index: int
) -> tuple[int, int]:
    empty_count = 0
    whitespace_only_count = 0
    index = successor_index - 1
    while index >= 0:
        value = lines[index].value
        if value == "":
            empty_count += 1
        elif all(character.isspace() for character in value):
            whitespace_only_count += 1
        else:
            break
        index -= 1
    return empty_count, whitespace_only_count


def _validate_works(
    lines: tuple[LogicalLine, ...],
    specs: tuple[WorkSpec, ...],
    global_spec: GlobalContentsSpec,
    first_body_index: int,
    end_index: int,
) -> tuple[tuple[WorkPreflightResult, ...], tuple[ValidatedWorkMarkers, ...]]:
    indexed_ranges: list[tuple[int, int, str]] = []
    results: list[WorkPreflightResult] = []
    markers: list[ValidatedWorkMarkers] = []

    for work in specs:
        _require_equal(
            "work.global_contents.cardinality",
            1,
            global_spec.entries.count(work.body_marker),
            work_id=work.work_id,
        )
        body_indices = _find_line_indices(
            lines, work.body_marker, start=first_body_index, stop=end_index
        )
        successor_indices = _find_line_indices(
            lines, work.successor_marker, start=first_body_index, stop=end_index
        )
        _require_equal(
            "work.body_marker.cardinality",
            1,
            len(body_indices),
            work_id=work.work_id,
        )
        _require_equal(
            "work.successor_marker.cardinality",
            1,
            len(successor_indices),
            work_id=work.work_id,
        )
        body_index = body_indices[0]
        successor_index = successor_indices[0]
        _require_equal(
            "work.marker_order",
            True,
            first_body_index <= body_index < successor_index < end_index,
            work_id=work.work_id,
        )
        _assert_position(
            "work.body_marker.position",
            work.body_position,
            lines[body_index].position,
            work_id=work.work_id,
        )
        _require_equal(
            "work.successor_marker.line",
            work.successor_line,
            lines[successor_index].position.line_number,
            work_id=work.work_id,
        )

        empty_count, whitespace_only_count = _separator_counts(lines, successor_index)
        _require_equal(
            "work.successor_separator.whitespace_only_count",
            work.expected_whitespace_only_lines_before_successor,
            whitespace_only_count,
            work_id=work.work_id,
        )
        _require_equal(
            "work.successor_separator.empty_count",
            work.expected_empty_lines_before_successor,
            empty_count,
            work_id=work.work_id,
        )
        indexed_ranges.append((body_index, successor_index, work.work_id))
        outer_end_line_index = successor_index - empty_count
        markers.append(
            ValidatedWorkMarkers(
                work_id=work.work_id,
                body_line_index=body_index,
                successor_line_index=successor_index,
                outer_end_line_index=outer_end_line_index,
                outer_end_byte_offset=lines[outer_end_line_index].position.byte_offset,
                outer_end_code_point_offset=lines[
                    outer_end_line_index
                ].position.code_point_offset,
            )
        )
        results.append(
            WorkPreflightResult(
                work_id=work.work_id,
                body_position=lines[body_index].position,
                successor_position=lines[successor_index].position,
                empty_lines_before_successor=empty_count,
                whitespace_only_lines_before_successor=whitespace_only_count,
            )
        )

    ranges_in_source_order = sorted(indexed_ranges)
    for current, following in zip(ranges_in_source_order, ranges_in_source_order[1:]):
        _require_equal(
            "work.outer_ranges_non_overlapping",
            True,
            current[1] <= following[0],
            work_id=current[2],
        )
    return tuple(results), tuple(markers)


def validate_preflight(raw_bytes: bytes, spec: PreflightSpec) -> PreflightReport:
    """Validate source identity and structure without producing derivatives."""

    return validate_source(raw_bytes, spec).report


def validate_source(raw_bytes: bytes, spec: PreflightSpec) -> ValidatedSource:
    """Return an immutable source bound to the exact bytes that passed preflight."""

    text = _validate_raw_bytes(raw_bytes, spec.raw)
    lines = _logical_lines(text)
    start_index, end_index = _validate_wrapper(lines, spec.wrapper)
    heading_index, first_body_index = _validate_global_contents(
        lines, spec.global_contents
    )
    _require_equal(
        "global_contents.wrapper_order",
        True,
        start_index < heading_index < first_body_index < end_index,
    )
    works, work_markers = _validate_works(
        lines,
        spec.works,
        spec.global_contents,
        first_body_index,
        end_index,
    )
    report = PreflightReport(
        raw_sha256=spec.raw.sha256,
        raw_byte_count=len(raw_bytes),
        raw_unicode_code_points=len(text),
        raw_crlf_pairs=raw_bytes.count(b"\r\n"),
        logical_line_count=len(lines),
        start_position=lines[start_index].position,
        end_position=lines[end_index].position,
        global_heading_position=lines[heading_index].position,
        global_entry_count=len(spec.global_contents.entries),
        first_body_position=lines[first_body_index].position,
        works=works,
    )
    return ValidatedSource(
        spec=spec,
        raw_bytes=raw_bytes,
        logical_lines=lines,
        work_markers=work_markers,
        report=report,
    )


def load_validated_source(
    repository_root: Path = REPOSITORY_ROOT,
) -> ValidatedSource:
    """Read the production source once and return that exact validated object."""

    spec = load_preflight_spec(repository_root)
    raw_path = repository_root / spec.raw_path
    try:
        raw_bytes = raw_path.read_bytes()
    except OSError as error:
        raise PreflightError(
            "raw.read",
            expected={"readable": True, "path": spec.raw_path.as_posix()},
            observed={"readable": False, "error_type": type(error).__name__},
        ) from None
    return validate_source(raw_bytes, spec)


def run_preflight(repository_root: Path = REPOSITORY_ROOT) -> PreflightReport:
    """Run the production preflight using only tracked repository inputs."""

    return load_validated_source(repository_root).report


def main() -> int:
    try:
        report = run_preflight()
    except (PreflightError, KeyError, TypeError, ValueError) as error:
        if isinstance(error, PreflightError):
            message = str(error)
        else:
            message = json.dumps(
                {
                    "invariant": "manifest.schema",
                    "expected": {"valid": True},
                    "observed": {"valid": False, "error_type": type(error).__name__},
                },
                sort_keys=True,
            )
        print(message, file=sys.stderr)
        return 1
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
