"""Shakespeare governance for permitted Phase 4 shifted examples."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from sebgpt.data.shakespeare_extract import ByteStage, ExtractedWork
from sebgpt.data.shifted_examples import (
    CONTEXT_LENGTH,
    STRIDE,
    Phase4ContractError,
    Phase4GovernanceError,
    Phase4TypeError,
    ShiftedTokenExample,
    iter_shifted_examples,
)
from sebgpt.tokenization.code_point import CodePointTokenizer
from sebgpt.tokenization.vocabulary_artifact import (
    VocabularyBinding,
    _is_verified_vocabulary_binding,
)


EXPECTED_MANIFEST_SCHEMA_VERSION = 4
EXPECTED_DATASET_ID = "shakespeare-eight-play"
EXPECTED_NORMALIZATION = "crlf_to_lf_only"
EXPECTED_WORKS = MappingProxyType({
    "train": (
        (1, "hamlet"),
        (2, "romeo-and-juliet"),
        (3, "macbeth"),
        (4, "a-midsummer-nights-dream"),
        (5, "much-ado-about-nothing"),
        (6, "henry-v"),
    ),
    "validation": ((7, "the-tempest"),),
})


@dataclass(frozen=True)
class DocumentShiftedExample:
    """One safe work identity wrapped around one immutable shifted example."""

    work_id: str
    manifest_order: int
    split: str
    example: ShiftedTokenExample = field(repr=True)


@dataclass(frozen=True)
class _ExpectedWork:
    work_id: str
    manifest_order: int
    split: str
    normalization: str
    processed_sha256: str
    processed_byte_count: int
    processed_code_point_count: int


def _governance(
    condition: bool,
    invariant: str,
    *,
    position: int | None = None,
) -> None:
    if not condition:
        facts: dict[str, Any] = {}
        if position is not None:
            facts["position"] = position
        raise Phase4GovernanceError(invariant, **facts)


def _mapping(value: Any, invariant: str) -> Mapping[str, Any]:
    _governance(isinstance(value, Mapping), invariant)
    return value


def _list(value: Any, invariant: str) -> list[Any]:
    _governance(type(value) is list, invariant)
    return value


def _expected_works(
    manifest: Mapping[str, Any],
    split: str,
) -> tuple[_ExpectedWork, ...]:
    identities = EXPECTED_WORKS[split]
    work_records = _list(manifest.get("works"), "manifest.works.list")
    selected_work_records = tuple(
        _mapping(record, "manifest.work.mapping")
        for record in work_records
        if isinstance(record, Mapping) and record.get("split") == split
    )
    _governance(
        len(selected_work_records) == len(identities),
        "manifest.works.split_count",
    )

    processing = _mapping(
        manifest.get("processing"),
        "manifest.processing.mapping",
    )
    result_records = _list(
        processing.get("per_work_results"),
        "manifest.processing.per_work_results.list",
    )
    selected_results = tuple(
        _mapping(record, "manifest.result.mapping")
        for record in result_records
        if isinstance(record, Mapping) and record.get("split") == split
    )
    _governance(
        len(selected_results) == len(identities),
        "manifest.results.split_count",
    )

    results_by_id: dict[str, Mapping[str, Any]] = {}
    for result in selected_results:
        work_id = result.get("work_id")
        _governance(type(work_id) is str, "manifest.result.work_id")
        _governance(work_id not in results_by_id, "manifest.result.work_id.unique")
        results_by_id[work_id] = result

    expected: list[_ExpectedWork] = []
    for position, ((order, work_id), work_record) in enumerate(
        zip(identities, selected_work_records, strict=True)
    ):
        _governance(
            type(work_record.get("order")) is int
            and work_record.get("order") == order,
            "manifest.work.order",
            position=position,
        )
        _governance(
            work_record.get("work_id") == work_id,
            "manifest.work.id",
            position=position,
        )
        _governance(
            work_record.get("split") == split,
            "manifest.work.split",
            position=position,
        )
        _governance(
            work_id in results_by_id,
            "manifest.result.present",
            position=position,
        )
        result = results_by_id[work_id]
        _governance(
            type(result.get("manifest_order")) is int
            and result.get("manifest_order") == order,
            "manifest.result.order",
            position=position,
        )
        _governance(
            result.get("split") == split,
            "manifest.result.split",
            position=position,
        )
        _governance(
            result.get("normalization") == EXPECTED_NORMALIZATION,
            "manifest.result.normalization",
            position=position,
        )
        processed = _mapping(
            result.get("processed"),
            "manifest.result.processed.mapping",
        )
        sha256 = processed.get("sha256")
        byte_count = processed.get("byte_count")
        code_point_count = processed.get("code_point_count")
        _governance(
            type(sha256) is str
            and len(sha256) == 64
            and all(character in "0123456789abcdef" for character in sha256),
            "manifest.result.processed.sha256",
            position=position,
        )
        _governance(
            type(byte_count) is int and byte_count >= 0,
            "manifest.result.processed.byte_count",
            position=position,
        )
        _governance(
            type(code_point_count) is int and code_point_count >= 0,
            "manifest.result.processed.code_point_count",
            position=position,
        )
        expected.append(
            _ExpectedWork(
                work_id=work_id,
                manifest_order=order,
                split=split,
                normalization=EXPECTED_NORMALIZATION,
                processed_sha256=sha256,
                processed_byte_count=byte_count,
                processed_code_point_count=code_point_count,
            )
        )
    return tuple(expected)


def _validate_work_metadata(
    work: ExtractedWork,
    expected: _ExpectedWork,
    *,
    position: int,
) -> None:
    _governance(work.work_id == expected.work_id, "work.id", position=position)
    _governance(
        type(work.manifest_order) is int
        and work.manifest_order == expected.manifest_order,
        "work.order",
        position=position,
    )
    _governance(work.split == expected.split, "work.split", position=position)
    _governance(
        work.normalization == expected.normalization,
        "work.normalization",
        position=position,
    )
    _governance(
        isinstance(work.processed, ByteStage),
        "work.processed.stage",
        position=position,
    )
    _governance(
        work.processed.sha256 == expected.processed_sha256,
        "work.processed.sha256",
        position=position,
    )
    _governance(
        type(work.processed.byte_count) is int
        and work.processed.byte_count == expected.processed_byte_count,
        "work.processed.byte_count",
        position=position,
    )
    _governance(
        type(work.processed_code_point_count) is int
        and work.processed_code_point_count
        == expected.processed_code_point_count,
        "work.processed.code_point_count",
        position=position,
    )


def _iterate_document_examples(
    works: tuple[ExtractedWork, ...],
    expected_works: tuple[_ExpectedWork, ...],
    vocabulary: VocabularyBinding,
    *,
    context_length: int,
    stride: int,
) -> Iterator[DocumentShiftedExample]:
    tokenizer = CodePointTokenizer(vocabulary.code_points)
    for position, (work, expected) in enumerate(
        zip(works, expected_works, strict=True)
    ):
        text = work.processed_text
        if not isinstance(text, str):
            raise Phase4TypeError(
                "work.processed_text.isinstance_str",
                position=position,
            )
        try:
            encoded = text.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            raise Phase4GovernanceError(
                "work.processed_text.strict_utf8",
                position=position,
            ) from None
        _governance(
            hashlib.sha256(encoded).hexdigest() == expected.processed_sha256,
            "work.recomputed.sha256",
            position=position,
        )
        _governance(
            len(encoded) == expected.processed_byte_count,
            "work.recomputed.byte_count",
            position=position,
        )
        _governance(
            len(text) == expected.processed_code_point_count,
            "work.recomputed.code_point_count",
            position=position,
        )
        token_ids = tokenizer.encode(text)
        for example in iter_shifted_examples(
            token_ids,
            context_length=context_length,
            stride=stride,
        ):
            yield DocumentShiftedExample(
                work_id=expected.work_id,
                manifest_order=expected.manifest_order,
                split=expected.split,
                example=example,
            )


def iter_shakespeare_phase_4_examples(
    works: tuple[ExtractedWork, ...],
    vocabulary: VocabularyBinding,
    phase_1_manifest: Mapping[str, object],
    *,
    split: str,
) -> Iterator[DocumentShiftedExample]:
    """Validate governance eagerly, then yield permitted work examples."""

    if type(works) is not tuple:
        raise Phase4TypeError("works.exact_tuple")
    if not isinstance(phase_1_manifest, Mapping):
        raise Phase4TypeError("manifest.mapping")
    if type(split) is not str:
        raise Phase4TypeError("split.exact_str")
    if type(vocabulary) is not VocabularyBinding:
        raise Phase4TypeError("vocabulary.binding.type")
    if not _is_verified_vocabulary_binding(vocabulary):
        raise Phase4ContractError("vocabulary.binding.verified")
    if split not in EXPECTED_WORKS:
        raise Phase4GovernanceError("split.permitted")
    _governance(
        type(phase_1_manifest.get("schema_version")) is int
        and phase_1_manifest.get("schema_version")
        == EXPECTED_MANIFEST_SCHEMA_VERSION,
        "manifest.schema_version",
    )
    _governance(
        phase_1_manifest.get("dataset_id") == EXPECTED_DATASET_ID,
        "manifest.dataset_id",
    )

    identities = EXPECTED_WORKS[split]
    _governance(len(works) == len(identities), "works.count")
    for position, work in enumerate(works):
        if not isinstance(work, ExtractedWork):
            raise Phase4TypeError("work.extracted_work", position=position)

    manifest = phase_1_manifest
    expected_works = _expected_works(manifest, split)
    for position, (work, expected) in enumerate(
        zip(works, expected_works, strict=True)
    ):
        _validate_work_metadata(work, expected, position=position)

    return _iterate_document_examples(
        works,
        expected_works,
        vocabulary,
        context_length=CONTEXT_LENGTH,
        stride=STRIDE,
    )


__all__ = [
    "DocumentShiftedExample",
    "iter_shakespeare_phase_4_examples",
]
