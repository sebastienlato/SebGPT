"""Shakespeare-specific tokenization governance without production execution."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from sebgpt.data.shakespeare_extract import ExtractedWork
from sebgpt.tokenization.code_point import build_code_point_vocabulary


EXPECTED_MANIFEST_SCHEMA_VERSION = 4
EXPECTED_DATASET_ID = "shakespeare-eight-play"
EXPECTED_NORMALIZATION = "crlf_to_lf_only"
EXPECTED_TRAINING_WORKS = (
    (1, "hamlet"),
    (2, "romeo-and-juliet"),
    (3, "macbeth"),
    (4, "a-midsummer-nights-dream"),
    (5, "much-ado-about-nothing"),
    (6, "henry-v"),
)


class ShakespeareTokenizationError(ValueError):
    """A content-safe orchestration failure containing structural facts only."""

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
class _ExpectedTrainingWork:
    work_id: str
    manifest_order: int
    processed_sha256: str
    processed_byte_count: int
    processed_code_point_count: int
    normalization: str


def _require(
    condition: bool,
    invariant: str,
    *,
    position: int | None = None,
) -> None:
    if not condition:
        raise ShakespeareTokenizationError(invariant, position=position)


def _is_exact_int(value: Any) -> bool:
    return type(value) is int


def _is_exact_int_equal(value: Any, expected: int) -> bool:
    return type(value) is int and value == expected


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _mapping(value: Any, invariant: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), invariant)
    return value


def _list(value: Any, invariant: str) -> list[Any]:
    _require(type(value) is list, invariant)
    return value


def _expected_training_works(
    manifest: Mapping[str, Any],
) -> tuple[_ExpectedTrainingWork, ...]:
    _require(
        _is_exact_int_equal(
            manifest.get("schema_version"),
            EXPECTED_MANIFEST_SCHEMA_VERSION,
        ),
        "manifest.schema_version",
    )
    _require(
        manifest.get("dataset_id") == EXPECTED_DATASET_ID,
        "manifest.dataset_id",
    )

    work_entries = _list(manifest.get("works"), "manifest.works.list")
    training_entries = tuple(
        _mapping(entry, "manifest.work.mapping")
        for entry in work_entries
        if isinstance(entry, Mapping) and entry.get("split") == "train"
    )
    _require(
        len(training_entries) == len(EXPECTED_TRAINING_WORKS),
        "manifest.training_works.count",
    )

    processing = _mapping(manifest.get("processing"), "manifest.processing.mapping")
    result_entries = _list(
        processing.get("per_work_results"),
        "manifest.processing.per_work_results.list",
    )
    results_by_id: dict[str, Mapping[str, Any]] = {}
    for entry in result_entries:
        result = _mapping(entry, "manifest.processing.work.mapping")
        work_id = result.get("work_id")
        if isinstance(work_id, str):
            _require(
                work_id not in results_by_id,
                "manifest.processing.work_id.unique",
            )
            results_by_id[work_id] = result

    expected: list[_ExpectedTrainingWork] = []
    for position, ((order, work_id), work_entry) in enumerate(
        zip(EXPECTED_TRAINING_WORKS, training_entries, strict=True)
    ):
        _require(
            _is_exact_int_equal(work_entry.get("order"), order),
            "manifest.work.order",
            position=position,
        )
        _require(
            work_entry.get("work_id") == work_id,
            "manifest.work.id",
            position=position,
        )
        _require(
            work_entry.get("split") == "train",
            "manifest.work.split",
            position=position,
        )

        _require(
            work_id in results_by_id,
            "manifest.result.present",
            position=position,
        )
        result = results_by_id[work_id]
        _require(
            _is_exact_int_equal(result.get("manifest_order"), order),
            "manifest.result.order",
            position=position,
        )
        _require(
            result.get("split") == "train",
            "manifest.result.split",
            position=position,
        )
        _require(
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
        _require(
            _is_sha256(sha256),
            "manifest.result.processed.sha256",
            position=position,
        )
        _require(
            _is_exact_int(byte_count) and byte_count >= 0,
            "manifest.result.processed.byte_count",
            position=position,
        )
        _require(
            _is_exact_int(code_point_count) and code_point_count >= 0,
            "manifest.result.processed.code_point_count",
            position=position,
        )
        expected.append(
            _ExpectedTrainingWork(
                work_id=work_id,
                manifest_order=order,
                processed_sha256=sha256,
                processed_byte_count=byte_count,
                processed_code_point_count=code_point_count,
                normalization=EXPECTED_NORMALIZATION,
            )
        )
    return tuple(expected)


def _safe_attribute(
    value: Any,
    name: str,
    invariant: str,
    *,
    position: int,
) -> Any:
    try:
        return getattr(value, name)
    except AttributeError:
        raise ShakespeareTokenizationError(invariant, position=position) from None


def build_shakespeare_training_vocabulary(
    works: tuple[ExtractedWork, ...],
    authoritative_manifest: Mapping[str, Any],
) -> tuple[int, ...]:
    """Validate synthetic/accepted training works before reading their text."""

    _require(type(works) is tuple, "works.exact_tuple")
    _require(
        len(works) == len(EXPECTED_TRAINING_WORKS),
        "works.count",
    )
    manifest = _mapping(authoritative_manifest, "manifest.mapping")
    expected_works = _expected_training_works(manifest)

    for position, (work, expected) in enumerate(
        zip(works, expected_works, strict=True)
    ):
        split = _safe_attribute(
            work,
            "split",
            "work.split.present",
            position=position,
        )
        _require(split == "train", "work.split.train", position=position)
        _require(
            _safe_attribute(work, "work_id", "work.id.present", position=position)
            == expected.work_id,
            "work.id",
            position=position,
        )
        _require(
            _is_exact_int_equal(
                _safe_attribute(
                    work,
                    "manifest_order",
                    "work.order.present",
                    position=position,
                ),
                expected.manifest_order,
            ),
            "work.order",
            position=position,
        )
        _require(
            _safe_attribute(
                work,
                "normalization",
                "work.normalization.present",
                position=position,
            )
            == expected.normalization,
            "work.normalization",
            position=position,
        )
        processed = _safe_attribute(
            work,
            "processed",
            "work.processed.present",
            position=position,
        )
        _require(
            _safe_attribute(
                processed,
                "sha256",
                "work.processed.sha256.present",
                position=position,
            )
            == expected.processed_sha256,
            "work.processed.sha256",
            position=position,
        )
        _require(
            _is_exact_int_equal(
                _safe_attribute(
                    processed,
                    "byte_count",
                    "work.processed.byte_count.present",
                    position=position,
                ),
                expected.processed_byte_count,
            ),
            "work.processed.byte_count",
            position=position,
        )
        _require(
            _is_exact_int_equal(
                _safe_attribute(
                    work,
                    "processed_code_point_count",
                    "work.processed_code_point_count.present",
                    position=position,
                ),
                expected.processed_code_point_count,
            ),
            "work.processed_code_point_count",
            position=position,
        )

    training_texts: list[str] = []
    for position, (work, expected) in enumerate(
        zip(works, expected_works, strict=True)
    ):
        processed_text = _safe_attribute(
            work,
            "processed_text",
            "work.processed_text.present",
            position=position,
        )
        _require(
            isinstance(processed_text, str),
            "work.processed_text.str",
            position=position,
        )
        try:
            processed_bytes = processed_text.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            raise ShakespeareTokenizationError(
                "work.processed_text.strict_utf8",
                position=position,
            ) from None

        _require(
            len(processed_bytes) == expected.processed_byte_count,
            "work.recomputed.byte_count",
            position=position,
        )
        _require(
            hashlib.sha256(processed_bytes).hexdigest()
            == expected.processed_sha256,
            "work.recomputed.sha256",
            position=position,
        )
        _require(
            len(processed_text) == expected.processed_code_point_count,
            "work.recomputed.code_point_count",
            position=position,
        )
        training_texts.append(processed_text)

    vocabulary = build_code_point_vocabulary(tuple(training_texts))
    _require(bool(vocabulary), "vocabulary.nonempty")
    return vocabulary
