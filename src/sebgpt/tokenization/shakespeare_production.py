"""Canonical vocabulary artifact and permitted Phase 2 statistics."""

from __future__ import annotations

import hashlib
import json
import os
from bisect import bisect_left
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from sebgpt.data.shakespeare_extract import (
    ExtractedWork,
    extract_validated_source,
)
from sebgpt.data.shakespeare_preflight import (
    REPOSITORY_ROOT,
    load_validated_source,
)
from sebgpt.tokenization.code_point import CodePointTokenizer, code_point_uplus
from sebgpt.tokenization.shakespeare import (
    EXPECTED_DATASET_ID,
    EXPECTED_MANIFEST_SCHEMA_VERSION,
    EXPECTED_NORMALIZATION,
    EXPECTED_TRAINING_WORKS,
    build_shakespeare_training_vocabulary,
)


VOCABULARY_RELATIVE_PATH = Path(
    "artifacts/tokenizers/shakespeare-code-point-v1/vocabulary.json"
)
TOKENIZER_ID = "shakespeare-code-point-v1"
TOKEN_UNIT = "python_str_code_point"
NORMALIZATION = "none"
UNKNOWN_POLICY = "error"
SPECIAL_TOKEN_POLICY = "none"
ID_ORDER = "ascending_unicode_code_point"
CONTRACT_DECISION = "DEC-0014"
ARTIFACT_SCHEMA_VERSION = 1
EXPECTED_IMPLEMENTATION_COMMIT = "de7a7f096fbbd8607c944412eaef30be9b686b56"
EXPECTED_VALIDATION_WORK = (7, "the-tempest")

_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "tokenizer_id",
        "token_unit",
        "normalization",
        "unknown_policy",
        "special_token_policy",
        "id_order",
        "contract_decision",
        "phase_1_manifest_sha256",
        "implementation_git_commit",
        "training_works",
        "vocabulary_size",
        "vocabulary",
    }
)
_TRAINING_WORK_KEYS = frozenset(
    {"work_id", "manifest_order", "processed_sha256"}
)
_VOCABULARY_KEYS = frozenset(
    {"token_id", "code_point", "code_point_uplus"}
)


class ProductionTokenizationError(ValueError):
    """A content-safe production-checkpoint failure."""

    __slots__ = ("_details",)

    def __init__(self, invariant: str, *, position: int | None = None) -> None:
        details: dict[str, Any] = {"invariant": invariant}
        if position is not None:
            details["position"] = position
        self._details = MappingProxyType(details)
        super().__init__(json.dumps(details, ensure_ascii=True, sort_keys=True))

    @property
    def details(self) -> Mapping[str, Any]:
        return self._details


@dataclass(frozen=True)
class VocabularyArtifactBuild:
    content: bytes = field(repr=False)
    sha256: str
    vocabulary_size: int


@dataclass(frozen=True)
class WorkTokenizationStatistics:
    work_id: str
    token_count: int
    coverage: float
    unknown_count: int
    round_trip_success: bool


@dataclass(frozen=True)
class TokenFrequency:
    token_id: int
    code_point: int
    count: int


@dataclass(frozen=True)
class TrainingTokenizationStatistics:
    works: tuple[WorkTokenizationStatistics, ...]
    total_token_count: int
    coverage: float
    unknown_count: int
    frequencies: tuple[TokenFrequency, ...] = field(repr=False)


@dataclass(frozen=True)
class ValidationTokenizationStatistics:
    work_count: int
    token_count: int
    coverage: float
    unknown_count: int
    round_trip_success: bool | None


def _require(
    condition: bool,
    invariant: str,
    *,
    position: int | None = None,
) -> None:
    if not condition:
        raise ProductionTokenizationError(invariant, position=position)


def _exact_int(value: Any) -> bool:
    return type(value) is int


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _git_commit_text(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _mapping(value: Any, invariant: str) -> Mapping[str, Any]:
    _require(type(value) is dict, invariant)
    return value


def _list(value: Any, invariant: str) -> list[Any]:
    _require(type(value) is list, invariant)
    return value


def _load_json_object(data: bytes, invariant: str) -> dict[str, Any]:
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise ProductionTokenizationError(f"{invariant}.utf8") from None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        raise ProductionTokenizationError(f"{invariant}.json") from None
    _require(type(value) is dict, f"{invariant}.root_dict")
    return value


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _training_records(
    works: tuple[ExtractedWork, ...],
) -> list[dict[str, Any]]:
    return [
        {
            "work_id": work.work_id,
            "manifest_order": work.manifest_order,
            "processed_sha256": work.processed.sha256,
        }
        for work in works
    ]


def extract_permitted_phase_2_works(
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[ExtractedWork, ...]:
    """Return training and validation works without extracting the sealed test."""

    source = load_validated_source(repository_root)
    permitted_specs = tuple(
        work for work in source.spec.works if work.split in ("train", "validation")
    )
    permitted_ids = {work.work_id for work in permitted_specs}
    _require(len(permitted_specs) == 7, "permitted_works.spec_count")
    _require(
        all(work.split != "test" for work in permitted_specs),
        "permitted_works.no_test_spec",
    )
    filtered_source = replace(
        source,
        spec=replace(source.spec, works=permitted_specs),
        work_markers=tuple(
            marker
            for marker in source.work_markers
            if marker.work_id in permitted_ids
        ),
        report=replace(
            source.report,
            works=tuple(
                work for work in source.report.works if work.work_id in permitted_ids
            ),
        ),
    )
    extracted = extract_validated_source(filtered_source)
    _require(len(extracted) == 7, "permitted_works.extracted_count")
    _require(
        sum(work.split == "train" for work in extracted) == 6,
        "permitted_works.train_count",
    )
    _require(
        sum(work.split == "validation" for work in extracted) == 1,
        "permitted_works.validation_count",
    )
    _require(
        all(work.split != "test" for work in extracted),
        "permitted_works.no_test_extracted",
    )
    return extracted


def _vocabulary_records(vocabulary: tuple[int, ...]) -> list[dict[str, Any]]:
    return [
        {
            "token_id": token_id,
            "code_point": code_point,
            "code_point_uplus": code_point_uplus(code_point),
        }
        for token_id, code_point in enumerate(vocabulary)
    ]


def build_vocabulary_artifact(
    works: tuple[ExtractedWork, ...],
    phase_1_manifest_bytes: bytes,
    *,
    implementation_git_commit: str = EXPECTED_IMPLEMENTATION_COMMIT,
) -> VocabularyArtifactBuild:
    """Construct canonical bytes after accepted orchestration validates text."""

    _require(
        _git_commit_text(implementation_git_commit),
        "artifact.implementation_git_commit",
    )
    manifest = _load_json_object(phase_1_manifest_bytes, "phase_1_manifest")
    vocabulary = build_shakespeare_training_vocabulary(works, manifest)
    tokenizer = CodePointTokenizer(vocabulary)

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "tokenizer_id": TOKENIZER_ID,
        "token_unit": TOKEN_UNIT,
        "normalization": NORMALIZATION,
        "unknown_policy": UNKNOWN_POLICY,
        "special_token_policy": SPECIAL_TOKEN_POLICY,
        "id_order": ID_ORDER,
        "contract_decision": CONTRACT_DECISION,
        "phase_1_manifest_sha256": _sha256(phase_1_manifest_bytes),
        "implementation_git_commit": implementation_git_commit,
        "training_works": _training_records(works),
        "vocabulary_size": tokenizer.vocabulary_size,
        "vocabulary": _vocabulary_records(tokenizer.code_points),
    }
    content = _canonical_json_bytes(artifact)
    verify_vocabulary_artifact(
        content,
        works,
        phase_1_manifest_bytes,
        implementation_git_commit=implementation_git_commit,
    )
    return VocabularyArtifactBuild(
        content=content,
        sha256=_sha256(content),
        vocabulary_size=tokenizer.vocabulary_size,
    )


def verify_vocabulary_artifact(
    content: bytes,
    works: tuple[ExtractedWork, ...],
    phase_1_manifest_bytes: bytes,
    *,
    implementation_git_commit: str = EXPECTED_IMPLEMENTATION_COMMIT,
) -> None:
    """Verify canonical bytes and every accepted schema invariant."""

    _require(content.endswith(b"\n"), "artifact.terminal_lf")
    _require(not content.endswith(b"\n\n"), "artifact.one_terminal_lf")
    _require(b"\r" not in content, "artifact.no_cr")
    artifact = _load_json_object(content, "artifact")
    _require(
        content == _canonical_json_bytes(artifact),
        "artifact.canonical_serialization",
    )
    _require(frozenset(artifact) == _ROOT_KEYS, "artifact.root_keys")

    fixed_values = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "tokenizer_id": TOKENIZER_ID,
        "token_unit": TOKEN_UNIT,
        "normalization": NORMALIZATION,
        "unknown_policy": UNKNOWN_POLICY,
        "special_token_policy": SPECIAL_TOKEN_POLICY,
        "id_order": ID_ORDER,
        "contract_decision": CONTRACT_DECISION,
        "implementation_git_commit": implementation_git_commit,
        "phase_1_manifest_sha256": _sha256(phase_1_manifest_bytes),
    }
    for key, expected in fixed_values.items():
        observed = artifact.get(key)
        if type(expected) is int:
            _require(
                _exact_int(observed) and observed == expected,
                f"artifact.{key}",
            )
        else:
            _require(type(observed) is str and observed == expected, f"artifact.{key}")

    _require(
        _sha256_text(artifact.get("phase_1_manifest_sha256")),
        "artifact.phase_1_manifest_sha256.grammar",
    )
    _require(
        _git_commit_text(artifact.get("implementation_git_commit")),
        "artifact.implementation_git_commit.grammar",
    )

    manifest = _load_json_object(phase_1_manifest_bytes, "phase_1_manifest")
    expected_vocabulary = build_shakespeare_training_vocabulary(works, manifest)
    training_works = _list(artifact.get("training_works"), "artifact.training_works")
    _require(
        len(training_works) == len(EXPECTED_TRAINING_WORKS),
        "artifact.training_works.count",
    )
    for position, (record, work, expected_identity) in enumerate(
        zip(training_works, works, EXPECTED_TRAINING_WORKS, strict=True)
    ):
        record = _mapping(record, "artifact.training_work.dict")
        _require(
            frozenset(record) == _TRAINING_WORK_KEYS,
            "artifact.training_work.keys",
            position=position,
        )
        order, work_id = expected_identity
        _require(
            type(record.get("work_id")) is str
            and record.get("work_id") == work_id
            and record.get("work_id") == work.work_id,
            "artifact.training_work.work_id",
            position=position,
        )
        _require(
            _exact_int(record.get("manifest_order"))
            and record.get("manifest_order") == order
            and record.get("manifest_order") == work.manifest_order,
            "artifact.training_work.manifest_order",
            position=position,
        )
        _require(
            _sha256_text(record.get("processed_sha256"))
            and record.get("processed_sha256") == work.processed.sha256,
            "artifact.training_work.processed_sha256",
            position=position,
        )

    vocabulary_size = artifact.get("vocabulary_size")
    vocabulary_records = _list(
        artifact.get("vocabulary"),
        "artifact.vocabulary",
    )
    _require(
        _exact_int(vocabulary_size) and vocabulary_size > 0,
        "artifact.vocabulary_size",
    )
    _require(
        vocabulary_size == len(vocabulary_records),
        "artifact.vocabulary_size.records",
    )
    _require(
        vocabulary_size == len(expected_vocabulary),
        "artifact.vocabulary_size.expected",
    )

    previous: int | None = None
    for position, (record, expected_code_point) in enumerate(
        zip(vocabulary_records, expected_vocabulary, strict=True)
    ):
        record = _mapping(record, "artifact.vocabulary.record_dict")
        _require(
            frozenset(record) == _VOCABULARY_KEYS,
            "artifact.vocabulary.record_keys",
            position=position,
        )
        token_id = record.get("token_id")
        code_point = record.get("code_point")
        notation = record.get("code_point_uplus")
        _require(
            _exact_int(token_id) and token_id == position,
            "artifact.vocabulary.token_id",
            position=position,
        )
        _require(
            _exact_int(code_point)
            and 0 <= code_point <= 0x10FFFF
            and code_point == expected_code_point,
            "artifact.vocabulary.code_point",
            position=position,
        )
        _require(
            previous is None or code_point > previous,
            "artifact.vocabulary.strictly_increasing",
            position=position,
        )
        _require(
            type(notation) is str and notation == code_point_uplus(code_point),
            "artifact.vocabulary.code_point_uplus",
            position=position,
        )
        previous = code_point


def write_vocabulary_artifact(
    build: VocabularyArtifactBuild,
    repository_root: Path,
) -> Path:
    """Create the sole canonical artifact, or accept an exact existing file."""

    path = repository_root / VOCABULARY_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.is_file(), "artifact.existing.regular_file")
        _require(path.read_bytes() == build.content, "artifact.existing.exact")
        return path

    try:
        with path.open("xb") as artifact_file:
            artifact_file.write(build.content)
            artifact_file.flush()
            os.fsync(artifact_file.fileno())
    except FileExistsError:
        raise ProductionTokenizationError("artifact.exclusive_create") from None
    _require(path.read_bytes() == build.content, "artifact.write_verify")
    return path


def inspect_training_statistics(
    works: tuple[ExtractedWork, ...],
    phase_1_manifest: Mapping[str, Any],
    tokenizer: CodePointTokenizer,
) -> TrainingTokenizationStatistics:
    """Compute permitted ephemeral statistics for the six training works."""

    expected_vocabulary = build_shakespeare_training_vocabulary(
        works,
        phase_1_manifest,
    )
    _require(
        tokenizer.code_points == expected_vocabulary,
        "statistics.training.vocabulary",
    )

    frequencies: Counter[int] = Counter()
    work_statistics: list[WorkTokenizationStatistics] = []
    for position, work in enumerate(works):
        token_ids = tokenizer.encode(work.processed_text)
        round_trip = tokenizer.decode(token_ids) == work.processed_text
        _require(round_trip, "statistics.training.round_trip", position=position)
        _require(
            len(token_ids) == work.processed_code_point_count,
            "statistics.training.token_count",
            position=position,
        )
        frequencies.update(token_ids)
        work_statistics.append(
            WorkTokenizationStatistics(
                work_id=work.work_id,
                token_count=len(token_ids),
                coverage=1.0,
                unknown_count=0,
                round_trip_success=True,
            )
        )

    frozen_frequencies = tuple(
        TokenFrequency(
            token_id=token_id,
            code_point=tokenizer.code_points[token_id],
            count=frequencies[token_id],
        )
        for token_id in range(tokenizer.vocabulary_size)
    )
    _require(
        all(entry.count > 0 for entry in frozen_frequencies),
        "statistics.training.frequency.positive",
    )
    total = sum(work.token_count for work in work_statistics)
    _require(
        sum(entry.count for entry in frozen_frequencies) == total,
        "statistics.training.frequency.total",
    )
    return TrainingTokenizationStatistics(
        works=tuple(work_statistics),
        total_token_count=total,
        coverage=1.0,
        unknown_count=0,
        frequencies=frozen_frequencies,
    )


def _validation_expected(
    manifest: Mapping[str, Any],
) -> tuple[str, int, str, int, int]:
    _require(
        manifest.get("schema_version") == EXPECTED_MANIFEST_SCHEMA_VERSION,
        "validation.manifest.schema_version",
    )
    _require(
        manifest.get("dataset_id") == EXPECTED_DATASET_ID,
        "validation.manifest.dataset_id",
    )
    order, work_id = EXPECTED_VALIDATION_WORK
    works = _list(manifest.get("works"), "validation.manifest.works")
    matching_works = [
        entry
        for entry in works
        if type(entry) is dict and entry.get("work_id") == work_id
    ]
    _require(len(matching_works) == 1, "validation.manifest.work.count")
    work_record = matching_works[0]
    _require(work_record.get("order") == order, "validation.manifest.work.order")
    _require(
        work_record.get("split") == "validation",
        "validation.manifest.work.split",
    )

    processing = _mapping(
        manifest.get("processing"),
        "validation.manifest.processing",
    )
    results = _list(
        processing.get("per_work_results"),
        "validation.manifest.results",
    )
    matching_results = [
        entry
        for entry in results
        if type(entry) is dict and entry.get("work_id") == work_id
    ]
    _require(len(matching_results) == 1, "validation.manifest.result.count")
    result = matching_results[0]
    _require(
        result.get("manifest_order") == order,
        "validation.manifest.result.order",
    )
    _require(
        result.get("split") == "validation",
        "validation.manifest.result.split",
    )
    _require(
        result.get("normalization") == EXPECTED_NORMALIZATION,
        "validation.manifest.result.normalization",
    )
    processed = _mapping(
        result.get("processed"),
        "validation.manifest.result.processed",
    )
    sha256 = processed.get("sha256")
    byte_count = processed.get("byte_count")
    code_point_count = processed.get("code_point_count")
    _require(_sha256_text(sha256), "validation.manifest.result.sha256")
    _require(
        _exact_int(byte_count) and byte_count >= 0,
        "validation.manifest.result.byte_count",
    )
    _require(
        _exact_int(code_point_count) and code_point_count >= 0,
        "validation.manifest.result.code_point_count",
    )
    return work_id, order, sha256, byte_count, code_point_count


def _unknown_count(text: str, tokenizer: CodePointTokenizer) -> int:
    count = 0
    for character in text:
        code_point = ord(character)
        position = bisect_left(tokenizer.code_points, code_point)
        if (
            position == tokenizer.vocabulary_size
            or tokenizer.code_points[position] != code_point
        ):
            count += 1
    return count


def inspect_validation_statistics(
    work: ExtractedWork,
    phase_1_manifest: Mapping[str, Any],
    tokenizer: CodePointTokenizer,
) -> ValidationTokenizationStatistics:
    """Compute one aggregate validation result, rejecting test before text."""

    try:
        split = work.split
    except AttributeError:
        raise ProductionTokenizationError("validation.work.split.present") from None
    _require(split == "validation", "validation.work.split")

    work_id, order, sha256, byte_count, code_point_count = _validation_expected(
        phase_1_manifest
    )
    _require(work.work_id == work_id, "validation.work.id")
    _require(work.manifest_order == order, "validation.work.order")
    _require(work.normalization == EXPECTED_NORMALIZATION, "validation.work.normalization")
    _require(work.processed.sha256 == sha256, "validation.work.sha256")
    _require(work.processed.byte_count == byte_count, "validation.work.byte_count")
    _require(
        work.processed_code_point_count == code_point_count,
        "validation.work.code_point_count",
    )

    processed_text = work.processed_text
    _require(isinstance(processed_text, str), "validation.work.processed_text.str")
    try:
        processed_bytes = processed_text.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        raise ProductionTokenizationError("validation.work.strict_utf8") from None
    _require(len(processed_bytes) == byte_count, "validation.recomputed.byte_count")
    _require(_sha256(processed_bytes) == sha256, "validation.recomputed.sha256")
    _require(len(processed_text) == code_point_count, "validation.recomputed.code_points")

    unknown_count = _unknown_count(processed_text, tokenizer)
    token_count = len(processed_text)
    covered_count = token_count - unknown_count
    coverage = 1.0 if token_count == 0 else covered_count / token_count
    round_trip_success: bool | None = None
    if unknown_count == 0:
        token_ids = tokenizer.encode(processed_text)
        round_trip_success = tokenizer.decode(token_ids) == processed_text
        _require(round_trip_success, "validation.round_trip")
    return ValidationTokenizationStatistics(
        work_count=1,
        token_count=token_count,
        coverage=coverage,
        unknown_count=unknown_count,
        round_trip_success=round_trip_success,
    )
