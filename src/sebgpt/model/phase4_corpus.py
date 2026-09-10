"""Safe construction of the seven-work Phase 4 permitted corpus."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final

from sebgpt.data.shakespeare_extract import ByteStage, ExtractedWork, OuterRawRange
from sebgpt.data.shakespeare_preflight import Position, REPOSITORY_ROOT
from sebgpt.data.shifted_examples import Phase4GovernanceError, Phase4TypeError
from sebgpt.model.phase4_experiment import (
    CONTEXT_LENGTH,
    PHASE_1_MANIFEST_SHA256,
    TRAINING_EXAMPLES,
    TRAINING_SOURCE_TOKENS,
    TRAINING_TARGETS,
    VALIDATION_EXAMPLES,
    VALIDATION_SOURCE_TOKENS,
    VALIDATION_TARGETS,
    Phase4PermittedCorpus,
)


PHASE_1_MANIFEST_PATH: Final = Path(
    "docs/data/shakespeare-eight-play-manifest.json"
)
PROCESSING_MANIFEST_PATH: Final = Path(
    "data/processed/shakespeare-eight-play/processing-manifest.json"
)
PROCESSING_MANIFEST_SHA256: Final = (
    "bbf938e565022dde72f26470e2fa7214f2fe1735afbc62ace320f1e6ebf372cc"
)
RAW_SOURCE_SHA256: Final = (
    "3cf4b3d44ee14cff4e14e78e2ad3318eff76f3f7f2afc3cee6bb925879110a37"
)
NORMALIZATION: Final = "crlf_to_lf_only"
PERMITTED_IDENTITIES: Final = (
    (1, "hamlet", "train"),
    (2, "romeo-and-juliet", "train"),
    (3, "macbeth", "train"),
    (4, "a-midsummer-nights-dream", "train"),
    (5, "much-ado-about-nothing", "train"),
    (6, "henry-v", "train"),
    (7, "the-tempest", "validation"),
)
SEALED_IDENTITY: Final = (8, "twelfth-night", "test")
ALL_IDENTITIES: Final = (*PERMITTED_IDENTITIES, SEALED_IDENTITY)


def _governance(condition: bool, invariant: str, **safe_facts: object) -> None:
    if not condition:
        raise Phase4GovernanceError(invariant, **safe_facts)


def _mapping(value: object, invariant: str) -> Mapping[str, Any]:
    _governance(isinstance(value, Mapping), invariant)
    return value


def _list(value: object, invariant: str) -> list[Any]:
    _governance(type(value) is list, invariant)
    return value


def _exact_int(value: object, invariant: str) -> int:
    _governance(type(value) is int, invariant)
    return value


def _exact_str(value: object, invariant: str) -> str:
    _governance(type(value) is str, invariant)
    return value


def _position(value: object, invariant: str) -> Position:
    record = _mapping(value, invariant)
    return Position(
        line_number=_exact_int(record.get("line_number"), invariant),
        byte_offset=_exact_int(record.get("byte_offset"), invariant),
        code_point_offset=_exact_int(record.get("code_point_offset"), invariant),
    )


def _byte_stage(value: object, invariant: str) -> ByteStage:
    record = _mapping(value, invariant)
    sha256 = _exact_str(record.get("sha256"), invariant)
    byte_count = _exact_int(record.get("byte_count"), invariant)
    _governance(
        len(sha256) == 64
        and all(character in "0123456789abcdef" for character in sha256),
        invariant,
    )
    _governance(byte_count >= 0, invariant)
    return ByteStage(sha256=sha256, byte_count=byte_count)


def _read_bytes(
    repository_root: Path,
    relative_path: Path,
    *,
    invariant: str,
) -> bytes:
    try:
        root_status = os.lstat(repository_root)
    except OSError:
        raise Phase4GovernanceError(
            "phase4_corpus.repository_root.directory",
        ) from None
    _governance(
        stat.S_ISDIR(root_status.st_mode)
        and not stat.S_ISLNK(root_status.st_mode),
        "phase4_corpus.repository_root.directory",
    )
    try:
        resolved_root = repository_root.resolve(strict=True)
    except OSError:
        raise Phase4GovernanceError(
            "phase4_corpus.repository_root.resolve",
        ) from None
    pure_path = PurePosixPath(relative_path.as_posix())
    _governance(
        not pure_path.is_absolute()
        and pure_path.parts
        and ".." not in pure_path.parts,
        invariant,
    )
    candidate = resolved_root
    for component_index, component in enumerate(pure_path.parts):
        candidate = candidate / component
        try:
            candidate_status = os.lstat(candidate)
        except OSError:
            raise Phase4GovernanceError(
                invariant,
                path=relative_path.as_posix(),
                component_index=component_index,
            ) from None
        _governance(
            not stat.S_ISLNK(candidate_status.st_mode),
            f"{invariant}.symlink_component",
            path=relative_path.as_posix(),
            component_index=component_index,
        )
        if component_index < len(pure_path.parts) - 1:
            _governance(
                stat.S_ISDIR(candidate_status.st_mode),
                f"{invariant}.ancestor_directory",
                path=relative_path.as_posix(),
                component_index=component_index,
            )
        else:
            _governance(
                stat.S_ISREG(candidate_status.st_mode),
                f"{invariant}.regular_file",
                path=relative_path.as_posix(),
            )
    try:
        resolved_candidate = candidate.resolve(strict=True)
    except OSError:
        raise Phase4GovernanceError(
            f"{invariant}.resolve",
            path=relative_path.as_posix(),
        ) from None
    _governance(
        resolved_candidate.is_relative_to(resolved_root),
        f"{invariant}.contained",
        path=relative_path.as_posix(),
    )
    try:
        return resolved_candidate.read_bytes()
    except OSError:
        raise Phase4GovernanceError(
            invariant,
            path=relative_path.as_posix(),
        ) from None


def _read_verified_json(
    repository_root: Path,
    relative_path: Path,
    *,
    expected_sha256: str,
    invariant: str,
) -> Mapping[str, Any]:
    content = _read_bytes(repository_root, relative_path, invariant=invariant)
    _governance(
        hashlib.sha256(content).hexdigest() == expected_sha256,
        f"{invariant}.sha256",
        path=relative_path.as_posix(),
    )
    try:
        value = json.loads(content.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise Phase4GovernanceError(f"{invariant}.json") from None
    return _mapping(value, f"{invariant}.mapping")


@dataclass(frozen=True)
class _PermittedWorkRecord:
    manifest_order: int
    work_id: str
    split: str
    title: str
    body_marker: str
    successor_marker: str
    start_position: Position
    successor_line: int
    empty_lines_before_successor: int
    dramatis_leading_spaces: int
    output_path: Path
    outer_raw: ByteStage
    retained_raw: ByteStage
    processed: ByteStage
    processed_code_point_count: int
    processed_line_count: int
    processed_word_count: int
    excluded_start: Position
    excluded_end: Position
    excluded_byte_count: int


def _identity(record: object, invariant: str) -> tuple[int, str, str]:
    value = _mapping(record, invariant)
    return (
        _exact_int(value.get("order", value.get("manifest_order")), invariant),
        _exact_str(value.get("work_id"), invariant),
        _exact_str(value.get("split"), invariant),
    )


def _require_exact_identity_sequence(
    records: list[Any],
    *,
    invariant: str,
) -> tuple[Mapping[str, Any], ...]:
    typed = tuple(_mapping(record, invariant) for record in records)
    identities = tuple(_identity(record, invariant) for record in typed)
    _governance(identities == ALL_IDENTITIES, invariant)
    return typed


def _expected_output_path(split: str, work_id: str) -> Path:
    return Path(f"data/processed/shakespeare-eight-play/{split}/{work_id}.txt")


def _select_permitted_records(
    phase_1_manifest: Mapping[str, Any],
    processing_manifest: Mapping[str, Any],
) -> tuple[_PermittedWorkRecord, ...]:
    """Validate all metadata and select seven records before any text read."""

    _governance(
        phase_1_manifest.get("schema_version") == 4,
        "phase4_corpus.phase_1_manifest.schema",
    )
    _governance(
        phase_1_manifest.get("dataset_id") == "shakespeare-eight-play",
        "phase4_corpus.phase_1_manifest.dataset",
    )
    manifest_works = _require_exact_identity_sequence(
        _list(
            phase_1_manifest.get("works"),
            "phase4_corpus.phase_1_manifest.works",
        ),
        invariant="phase4_corpus.phase_1_manifest.identities",
    )
    processing = _mapping(
        phase_1_manifest.get("processing"),
        "phase4_corpus.phase_1_manifest.processing",
    )
    generated_manifest = _mapping(
        processing.get("generated_processing_manifest"),
        "phase4_corpus.phase_1_manifest.generated",
    )
    _governance(
        generated_manifest.get("relative_path")
        == PROCESSING_MANIFEST_PATH.as_posix()
        and generated_manifest.get("schema_version") == 1
        and generated_manifest.get("sha256") == PROCESSING_MANIFEST_SHA256,
        "phase4_corpus.phase_1_manifest.generated",
    )
    results = _require_exact_identity_sequence(
        _list(
            processing.get("per_work_results"),
            "phase4_corpus.phase_1_manifest.results",
        ),
        invariant="phase4_corpus.phase_1_manifest.result_identities",
    )
    excluded_ranges = _list(
        processing.get("excluded_ranges"),
        "phase4_corpus.phase_1_manifest.excluded_ranges",
    )
    _governance(
        tuple(
            _exact_str(
                _mapping(record, "phase4_corpus.excluded_range").get("work_id"),
                "phase4_corpus.excluded_range.work_id",
            )
            for record in excluded_ranges
        )
        == tuple(work_id for _, work_id, _ in ALL_IDENTITIES),
        "phase4_corpus.phase_1_manifest.excluded_identities",
    )

    _governance(
        processing_manifest.get("schema_version") == 1
        and processing_manifest.get("dataset_id") == "shakespeare-eight-play"
        and processing_manifest.get("raw_sha256") == RAW_SOURCE_SHA256,
        "phase4_corpus.processing_manifest.identity",
    )
    published_processing = _mapping(
        processing_manifest.get("processing"),
        "phase4_corpus.processing_manifest.processing",
    )
    _governance(
        published_processing.get("normalization") == NORMALIZATION,
        "phase4_corpus.processing_manifest.normalization",
    )
    published_works = _require_exact_identity_sequence(
        _list(
            processing_manifest.get("works"),
            "phase4_corpus.processing_manifest.works",
        ),
        invariant="phase4_corpus.processing_manifest.identities",
    )

    selected: list[_PermittedWorkRecord] = []
    for position, (expected, work, result, excluded, published) in enumerate(
        zip(
            ALL_IDENTITIES,
            manifest_works,
            results,
            excluded_ranges,
            published_works,
            strict=True,
        )
    ):
        order, work_id, split = expected
        expected_path = _expected_output_path(split, work_id)
        excluded_record = _mapping(excluded, "phase4_corpus.excluded_range")
        _governance(
            work.get("processed_path") == expected_path.as_posix()
            and result.get("output_path") == expected_path.as_posix()
            and published.get("output_path") == expected_path.as_posix(),
            "phase4_corpus.output_path",
            position=position,
        )
        _governance(
            result.get("title") == work.get("title") == published.get("title"),
            "phase4_corpus.title",
            position=position,
        )
        _governance(
            result.get("normalization")
            == published.get("normalization")
            == NORMALIZATION,
            "phase4_corpus.normalization",
            position=position,
        )
        for stage_name in ("outer_raw", "retained_raw"):
            _governance(
                result.get(stage_name) == published.get(stage_name),
                "phase4_corpus.stage_identity",
                position=position,
                stage=stage_name,
            )
        _governance(
            result.get("processed") == published.get("processed"),
            "phase4_corpus.processed_identity",
            position=position,
        )
        published_excluded = _mapping(
            published.get("excluded_local_contents"),
            "phase4_corpus.processing_manifest.excluded",
        )
        authoritative_excluded = {
            key: value for key, value in excluded_record.items() if key != "work_id"
        }
        _governance(
            published_excluded == authoritative_excluded,
            "phase4_corpus.excluded_identity",
            position=position,
        )
        if expected == SEALED_IDENTITY:
            continue

        processed_record = _mapping(
            result.get("processed"),
            "phase4_corpus.processed",
        )
        selected.append(
            _PermittedWorkRecord(
                manifest_order=order,
                work_id=work_id,
                split=split,
                title=_exact_str(work.get("title"), "phase4_corpus.title"),
                body_marker=_exact_str(
                    work.get("body_title_marker"),
                    "phase4_corpus.body_marker",
                ),
                successor_marker=_exact_str(
                    work.get("successor_title_marker"),
                    "phase4_corpus.successor_marker",
                ),
                start_position=Position(
                    line_number=_exact_int(
                        work.get("observed_start_line"),
                        "phase4_corpus.start",
                    ),
                    byte_offset=_exact_int(
                        work.get("observed_start_byte"),
                        "phase4_corpus.start",
                    ),
                    code_point_offset=_exact_int(
                        work.get("observed_start_code_point"),
                        "phase4_corpus.start",
                    ),
                ),
                successor_line=_exact_int(
                    work.get("observed_successor_line"),
                    "phase4_corpus.successor_line",
                ),
                empty_lines_before_successor=_exact_int(
                    work.get("expected_empty_crlf_lines_before_successor"),
                    "phase4_corpus.successor_separator",
                ),
                dramatis_leading_spaces=_exact_int(
                    work.get("dramatis_leading_u0020_count"),
                    "phase4_corpus.dramatis_indent",
                ),
                output_path=expected_path,
                outer_raw=_byte_stage(result.get("outer_raw"), "phase4_corpus.outer_raw"),
                retained_raw=_byte_stage(
                    result.get("retained_raw"),
                    "phase4_corpus.retained_raw",
                ),
                processed=_byte_stage(
                    processed_record,
                    "phase4_corpus.processed",
                ),
                processed_code_point_count=_exact_int(
                    processed_record.get("code_point_count"),
                    "phase4_corpus.processed.code_point_count",
                ),
                processed_line_count=_exact_int(
                    processed_record.get("line_count"),
                    "phase4_corpus.processed.line_count",
                ),
                processed_word_count=_exact_int(
                    processed_record.get("word_count"),
                    "phase4_corpus.processed.word_count",
                ),
                excluded_start=_position(
                    excluded_record.get("start"),
                    "phase4_corpus.excluded.start",
                ),
                excluded_end=_position(
                    excluded_record.get("end_exclusive"),
                    "phase4_corpus.excluded.end",
                ),
                excluded_byte_count=_exact_int(
                    excluded_record.get("byte_count"),
                    "phase4_corpus.excluded.byte_count",
                ),
            )
        )

    _governance(
        tuple((record.manifest_order, record.work_id, record.split) for record in selected)
        == PERMITTED_IDENTITIES,
        "phase4_corpus.selected_identities",
    )
    return tuple(selected)


def _load_permitted_work(
    repository_root: Path,
    record: _PermittedWorkRecord,
) -> ExtractedWork:
    content = _read_bytes(
        repository_root,
        record.output_path,
        invariant="phase4_corpus.processed_file",
    )
    _governance(
        hashlib.sha256(content).hexdigest() == record.processed.sha256
        and len(content) == record.processed.byte_count,
        "phase4_corpus.processed_file.identity",
        work_id=record.work_id,
    )
    try:
        text = content.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise Phase4GovernanceError(
            "phase4_corpus.processed_file.utf8",
            work_id=record.work_id,
        ) from None
    _governance(
        len(text) == record.processed_code_point_count
        and text.count("\n") == record.processed_line_count
        and len(text.split()) == record.processed_word_count,
        "phase4_corpus.processed_file.counts",
        work_id=record.work_id,
    )
    _governance(
        "\r" not in text
        and text.endswith("\n")
        and not text.endswith("\n\n")
        and text.partition("\n")[0] == record.body_marker,
        "phase4_corpus.processed_file.normalization",
        work_id=record.work_id,
    )
    dramatis_marker = " " * record.dramatis_leading_spaces + "Dramatis Personæ"
    _governance(
        text.split("\n")[:-1].count("Contents") == 0
        and text.split("\n")[:-1].count(dramatis_marker) == 1,
        "phase4_corpus.processed_file.extraction_semantics",
        work_id=record.work_id,
    )

    excluded_code_points = (
        record.excluded_end.code_point_offset
        - record.excluded_start.code_point_offset
    )
    outer_code_points = (
        record.processed_code_point_count
        + record.processed_line_count
        + excluded_code_points
    )
    outer_end_byte = record.start_position.byte_offset + record.outer_raw.byte_count
    outer_end_code_point = (
        record.start_position.code_point_offset + outer_code_points
    )
    separator_code_points = record.empty_lines_before_successor * 2
    successor_position = Position(
        line_number=record.successor_line,
        byte_offset=outer_end_byte + separator_code_points,
        code_point_offset=outer_end_code_point + separator_code_points,
    )
    return ExtractedWork(
        work_id=record.work_id,
        title=record.title,
        split=record.split,
        manifest_order=record.manifest_order,
        body_marker=record.body_marker,
        successor_marker=record.successor_marker,
        processed_text=text,
        outer_range=OuterRawRange(
            start_position=record.start_position,
            end_byte_offset=outer_end_byte,
            end_code_point_offset=outer_end_code_point,
        ),
        successor_position=successor_position,
        local_contents_position=record.excluded_start,
        dramatis_position=record.excluded_end,
        outer_raw=record.outer_raw,
        retained_raw=record.retained_raw,
        processed=record.processed,
        removed_local_contents_byte_count=record.excluded_byte_count,
        processed_code_point_count=record.processed_code_point_count,
        processed_line_count=record.processed_line_count,
        processed_word_count=record.processed_word_count,
        normalization=NORMALIZATION,
    )


def _counts(works: tuple[ExtractedWork, ...]) -> tuple[int, int, int]:
    source_tokens = sum(work.processed_code_point_count for work in works)
    targets = sum(work.processed_code_point_count - 1 for work in works)
    examples = sum(
        math.ceil((work.processed_code_point_count - 1) / CONTEXT_LENGTH)
        for work in works
    )
    return source_tokens, targets, examples


def load_phase4_permitted_corpus(
    repository_root: Path = REPOSITORY_ROOT,
) -> Phase4PermittedCorpus:
    """Load exactly six train works and one validation work, never the test file."""

    if not isinstance(repository_root, Path):
        raise Phase4TypeError("phase4_corpus.repository_root.path")
    phase_1_manifest = _read_verified_json(
        repository_root,
        PHASE_1_MANIFEST_PATH,
        expected_sha256=PHASE_1_MANIFEST_SHA256,
        invariant="phase4_corpus.phase_1_manifest",
    )
    processing_manifest = _read_verified_json(
        repository_root,
        PROCESSING_MANIFEST_PATH,
        expected_sha256=PROCESSING_MANIFEST_SHA256,
        invariant="phase4_corpus.processing_manifest",
    )
    selected = _select_permitted_records(phase_1_manifest, processing_manifest)
    works = tuple(_load_permitted_work(repository_root, record) for record in selected)
    training_works = tuple(work for work in works if work.split == "train")
    validation_works = tuple(work for work in works if work.split == "validation")
    training_counts = _counts(training_works)
    validation_counts = _counts(validation_works)
    _governance(
        training_counts
        == (TRAINING_SOURCE_TOKENS, TRAINING_TARGETS, TRAINING_EXAMPLES),
        "phase4_corpus.training_counts",
    )
    _governance(
        validation_counts
        == (VALIDATION_SOURCE_TOKENS, VALIDATION_TARGETS, VALIDATION_EXAMPLES),
        "phase4_corpus.validation_counts",
    )
    return Phase4PermittedCorpus(
        phase_1_manifest=phase_1_manifest,
        phase_1_manifest_sha256=PHASE_1_MANIFEST_SHA256,
        training_works=training_works,
        validation_works=validation_works,
        training_source_tokens=training_counts[0],
        training_targets=training_counts[1],
        training_examples=training_counts[2],
        validation_source_tokens=validation_counts[0],
        validation_targets=validation_counts[1],
        validation_examples=validation_counts[2],
        sealed_test_supplier=None,
    )


__all__ = ["load_phase4_permitted_corpus"]
