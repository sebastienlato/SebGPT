"""Deterministic, in-memory extraction from a preflight-validated source."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sebgpt.data.shakespeare_preflight import (
    REPOSITORY_ROOT,
    Position,
    ValidatedSource,
    ValidatedWorkMarkers,
    WorkSpec,
    load_validated_source,
)


NORMALIZATION_ID = "crlf_to_lf_only"


class ExtractionError(Exception):
    """An extraction failure containing no source text or excerpts."""

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
class ByteStage:
    """Identity and size of one exact provenance byte domain."""

    sha256: str
    byte_count: int


@dataclass(frozen=True)
class OuterRawRange:
    start_position: Position
    end_byte_offset: int
    end_code_point_offset: int


@dataclass(frozen=True)
class ExtractedWork:
    """One immutable processed work; its text is intentionally repr-hidden."""

    work_id: str
    title: str
    split: str
    manifest_order: int
    body_marker: str
    successor_marker: str
    processed_text: str = field(repr=False)
    outer_range: OuterRawRange
    successor_position: Position
    local_contents_position: Position
    dramatis_position: Position
    outer_raw: ByteStage
    retained_raw: ByteStage
    processed: ByteStage
    removed_local_contents_byte_count: int
    processed_code_point_count: int
    processed_line_count: int
    processed_word_count: int
    normalization: str = NORMALIZATION_ID


def _require(
    invariant: str,
    expected: Any,
    observed: Any,
    *,
    work_id: str | None = None,
) -> None:
    if observed != expected:
        raise ExtractionError(
            invariant, expected=expected, observed=observed, work_id=work_id
        )


def _stage(data: bytes) -> ByteStage:
    return ByteStage(hashlib.sha256(data).hexdigest(), len(data))


def _line_indices(
    source: ValidatedSource,
    marker: str,
    *,
    start: int,
    stop: int,
) -> list[int]:
    return [
        index
        for index in range(start, stop)
        if source.logical_lines[index].value == marker
    ]


def _has_only_u0020_padding(value: str, canonical_text: str) -> bool:
    """Classify forbidden marker variants without changing extraction matching."""

    start = 0
    while start < len(value) and value[start] == " ":
        start += 1
    end = len(value)
    while end > start and value[end - 1] == " ":
        end -= 1
    return value[start:end] == canonical_text


def _validate_retained_raw(retained_raw: bytes, work_id: str) -> str:
    crlf_count = retained_raw.count(b"\r\n")
    _require(
        "retained_raw.lone_cr",
        0,
        retained_raw.count(b"\r") - crlf_count,
        work_id=work_id,
    )
    _require(
        "retained_raw.lone_lf",
        0,
        retained_raw.count(b"\n") - crlf_count,
        work_id=work_id,
    )
    _require(
        "retained_raw.terminal_crlf",
        True,
        retained_raw.endswith(b"\r\n")
        and not retained_raw.endswith(b"\r\n\r\n"),
        work_id=work_id,
    )
    try:
        return retained_raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ExtractionError(
            "retained_raw.strict_utf8",
            expected={"decodes": True},
            observed={
                "decodes": False,
                "start_byte": error.start,
                "end_byte": error.end,
                "reason": error.reason,
            },
            work_id=work_id,
        ) from None


def _validate_only_crlf_to_lf(
    retained_raw: bytes, processed_bytes: bytes, work_id: str
) -> None:
    """Verify normalization while keeping source bytes out of diagnostics."""

    _require(
        "processed.only_crlf_to_lf",
        True,
        retained_raw.replace(b"\r\n", b"\n") == processed_bytes,
        work_id=work_id,
    )


def _extract_work(
    source: ValidatedSource,
    work: WorkSpec,
    markers: ValidatedWorkMarkers,
) -> ExtractedWork:
    lines = source.logical_lines
    body_index = markers.body_line_index
    outer_end_index = markers.outer_end_line_index
    outer_start = lines[body_index].position
    outer_end_byte = markers.outer_end_byte_offset
    outer_end_code_point = markers.outer_end_code_point_offset

    _require(
        "outer_range.end_position",
        {
            "byte_offset": outer_end_byte,
            "code_point_offset": outer_end_code_point,
        },
        {
            "byte_offset": lines[outer_end_index].position.byte_offset,
            "code_point_offset": lines[outer_end_index].position.code_point_offset,
        },
        work_id=work.work_id,
    )
    outer_raw = source.raw_bytes[outer_start.byte_offset:outer_end_byte]
    _require(
        "outer_range.starts_with_title",
        True,
        outer_raw.startswith(work.body_marker.encode("utf-8") + b"\r\n"),
        work_id=work.work_id,
    )
    _require(
        "outer_range.terminal_crlf",
        True,
        outer_raw.endswith(b"\r\n") and not outer_raw.endswith(b"\r\n\r\n"),
        work_id=work.work_id,
    )

    contents_indices = _line_indices(
        source,
        source.spec.local_contents_marker,
        start=body_index + 1,
        stop=outer_end_index,
    )
    _require(
        "local_contents.cardinality",
        1,
        len(contents_indices),
        work_id=work.work_id,
    )
    contents_index = contents_indices[0]
    dramatis_marker = (
        " " * work.dramatis_leading_u0020_count
        + source.spec.dramatis_canonical_text
    )
    dramatis_indices = _line_indices(
        source,
        dramatis_marker,
        start=contents_index + 1,
        stop=outer_end_index,
    )
    _require(
        "dramatis_personae.cardinality",
        source.spec.required_dramatis_occurrences,
        len(dramatis_indices),
        work_id=work.work_id,
    )
    unexpected_dramatis_variants = sum(
        _has_only_u0020_padding(
            lines[index].value, source.spec.dramatis_canonical_text
        )
        and lines[index].value != dramatis_marker
        for index in range(contents_index + 1, outer_end_index)
    )
    _require(
        "dramatis_personae.unexpected_variant_count",
        0,
        unexpected_dramatis_variants,
        work_id=work.work_id,
    )
    dramatis_index = dramatis_indices[0]
    _require(
        "local_marker.order",
        True,
        body_index < contents_index < dramatis_index < outer_end_index,
        work_id=work.work_id,
    )

    contents_start = lines[contents_index].position.byte_offset
    dramatis_start = lines[dramatis_index].position.byte_offset
    retained_raw = (
        source.raw_bytes[outer_start.byte_offset:contents_start]
        + source.raw_bytes[dramatis_start:outer_end_byte]
    )
    retained_text = _validate_retained_raw(retained_raw, work.work_id)
    processed_text = retained_text.replace("\r\n", "\n")
    processed_bytes = processed_text.encode("utf-8")

    _validate_only_crlf_to_lf(retained_raw, processed_bytes, work.work_id)
    _require(
        "processed.carriage_returns",
        0,
        processed_text.count("\r"),
        work_id=work.work_id,
    )
    _require(
        "processed.single_terminal_lf",
        True,
        processed_text.endswith("\n") and not processed_text.endswith("\n\n"),
        work_id=work.work_id,
    )
    first_line, separator, _ = processed_text.partition("\n")
    _require(
        "processed.top_level_title",
        True,
        separator == "\n" and first_line == work.body_marker,
        work_id=work.work_id,
    )
    processed_lines = processed_text.split("\n")[:-1]
    _require(
        "processed.local_contents_removed",
        0,
        processed_lines.count(source.spec.local_contents_marker),
        work_id=work.work_id,
    )
    _require(
        "processed.dramatis_retained",
        source.spec.required_dramatis_occurrences,
        processed_lines.count(dramatis_marker),
        work_id=work.work_id,
    )

    return ExtractedWork(
        work_id=work.work_id,
        title=work.title,
        split=work.split,
        manifest_order=work.manifest_order,
        body_marker=work.body_marker,
        successor_marker=work.successor_marker,
        processed_text=processed_text,
        outer_range=OuterRawRange(
            start_position=outer_start,
            end_byte_offset=outer_end_byte,
            end_code_point_offset=outer_end_code_point,
        ),
        successor_position=lines[markers.successor_line_index].position,
        local_contents_position=lines[contents_index].position,
        dramatis_position=lines[dramatis_index].position,
        outer_raw=_stage(outer_raw),
        retained_raw=_stage(retained_raw),
        processed=_stage(processed_bytes),
        removed_local_contents_byte_count=dramatis_start - contents_start,
        processed_code_point_count=len(processed_text),
        processed_line_count=processed_text.count("\n"),
        processed_word_count=len(processed_text.split()),
    )


def extract_validated_source(source: ValidatedSource) -> tuple[ExtractedWork, ...]:
    """Extract every configured work from one immutable validated source."""

    marker_by_id = {marker.work_id: marker for marker in source.work_markers}
    _require(
        "collection.marker_count",
        len(source.spec.works),
        len(marker_by_id),
    )
    extracted = tuple(
        _extract_work(source, work, marker_by_id[work.work_id])
        for work in source.spec.works
    )
    _require("collection.work_count", len(source.spec.works), len(extracted))
    _require(
        "collection.unique_work_ids",
        len(extracted),
        len({work.work_id for work in extracted}),
    )
    _require(
        "collection.unique_manifest_orders",
        len(extracted),
        len({work.manifest_order for work in extracted}),
    )
    _require(
        "collection.distinct_processed_hashes",
        len(extracted),
        len({work.processed.sha256 for work in extracted}),
    )
    return extracted


def extract_shakespeare_in_memory(
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[ExtractedWork, ...]:
    """Preflight one raw read, then return the eight works only in memory."""

    source = load_validated_source(repository_root)
    extracted = extract_validated_source(source)
    _require("production.work_count", 8, len(extracted))
    _require(
        "production.train_work_count",
        6,
        sum(work.split == "train" for work in extracted),
    )
    _require(
        "production.validation_work_count",
        1,
        sum(work.split == "validation" for work in extracted),
    )
    _require(
        "production.test_work_count",
        1,
        sum(work.split == "test" for work in extracted),
    )
    return extracted
