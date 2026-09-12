"""Document-local Phase 8 windows, ordering, and logical batches."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from typing import Any

import torch

from sebgpt.data.shakespeare_extract import ByteStage, ExtractedWork
from sebgpt.tokenization.code_point import CodePointTokenizer
from sebgpt.tokenization.vocabulary_artifact import (
    VocabularyBinding,
    _is_verified_vocabulary_binding,
)
from sebgpt.training.phase8_types import (
    Phase8ContractError,
    Phase8GovernanceError,
    Phase8LogicalBatch,
    Phase8TypeError,
    Phase8Window,
)


CONTEXT_LENGTH = 256
STRIDE = 256
LOGICAL_BATCH_CAPACITY = 8
MAXIMUM_EPOCHS = 10
EXPECTED_MANIFEST_SCHEMA_VERSION = 4
EXPECTED_DATASET_ID = "shakespeare-eight-play"
EXPECTED_NORMALIZATION = "crlf_to_lf_only"
EXPECTED_WORKS = {
    "train": (
        (1, "hamlet"),
        (2, "romeo-and-juliet"),
        (3, "macbeth"),
        (4, "a-midsummer-nights-dream"),
        (5, "much-ado-about-nothing"),
        (6, "henry-v"),
    ),
    "validation": ((7, "the-tempest"),),
}


def _require(condition: bool, invariant: str, **facts: object) -> None:
    if not condition:
        raise Phase8ContractError(invariant, **facts)


def _governance(condition: bool, invariant: str, **facts: object) -> None:
    if not condition:
        raise Phase8GovernanceError(invariant, **facts)


def _exact_dict(value: object, *, field: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise Phase8TypeError("phase8.type.record", field=field)
    return value


def _exact_list(value: object, *, field: str) -> list[Any]:
    if type(value) is not list:
        raise Phase8TypeError("phase8.type.record", field=field)
    return value


def _expected_metadata(
    manifest: Mapping[str, object],
    split: str,
) -> tuple[tuple[int, str, str, int, int], ...]:
    works = _exact_list(manifest.get("works"), field="manifest.works")
    processing = _exact_dict(
        manifest.get("processing"),
        field="manifest.processing",
    )
    results = _exact_list(
        processing.get("per_work_results"),
        field="manifest.processing.per_work_results",
    )
    selected_works = [
        _exact_dict(value, field="manifest.work")
        for value in works
        if type(value) is dict and value.get("split") == split
    ]
    selected_results = [
        _exact_dict(value, field="manifest.result")
        for value in results
        if type(value) is dict and value.get("split") == split
    ]
    identities = EXPECTED_WORKS[split]
    _governance(
        len(selected_works) == len(identities)
        and len(selected_results) == len(identities),
        "phase8.governance.dataset",
        field="split_count",
    )
    result_by_id: dict[str, dict[str, Any]] = {}
    for value in selected_results:
        work_id = value.get("work_id")
        _governance(
            type(work_id) is str and work_id not in result_by_id,
            "phase8.governance.dataset",
            field="result_identity",
        )
        assert isinstance(work_id, str)
        result_by_id[work_id] = value

    expected: list[tuple[int, str, str, int, int]] = []
    for position, ((order, work_id), work_record) in enumerate(
        zip(identities, selected_works, strict=True)
    ):
        _governance(
            work_record.get("order") == order
            and work_record.get("work_id") == work_id
            and work_record.get("split") == split,
            "phase8.governance.work",
            position=position,
        )
        result = result_by_id.get(work_id)
        _governance(
            type(result) is dict
            and result.get("manifest_order") == order
            and result.get("split") == split
            and result.get("normalization") == EXPECTED_NORMALIZATION,
            "phase8.governance.work",
            position=position,
        )
        assert isinstance(result, dict)
        processed = _exact_dict(
            result.get("processed"),
            field="manifest.result.processed",
        )
        digest = processed.get("sha256")
        byte_count = processed.get("byte_count")
        code_point_count = processed.get("code_point_count")
        _governance(
            type(digest) is str
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            and type(byte_count) is int
            and byte_count >= 0
            and type(code_point_count) is int
            and code_point_count >= 0,
            "phase8.governance.dataset",
            position=position,
        )
        assert isinstance(digest, str)
        assert isinstance(byte_count, int)
        assert isinstance(code_point_count, int)
        expected.append((order, work_id, digest, byte_count, code_point_count))
    return tuple(expected)


def build_phase8_windows(
    works: tuple[ExtractedWork, ...],
    vocabulary: VocabularyBinding,
    phase_1_manifest: Mapping[str, object],
    *,
    split: str,
) -> tuple[Phase8Window, ...]:
    if type(works) is not tuple:
        raise Phase8TypeError("phase8.type.argument", field="works")
    if type(vocabulary) is not VocabularyBinding:
        raise Phase8TypeError("phase8.type.argument", field="vocabulary")
    if not isinstance(phase_1_manifest, Mapping):
        raise Phase8TypeError("phase8.type.argument", field="phase_1_manifest")
    if type(split) is not str:
        raise Phase8TypeError("phase8.type.argument", field="split")
    _governance(
        split in ("train", "validation"),
        "phase8.governance.split",
    )
    _require(
        _is_verified_vocabulary_binding(vocabulary),
        "phase8.contract.window",
        field="vocabulary",
    )
    _governance(
        phase_1_manifest.get("schema_version") == EXPECTED_MANIFEST_SCHEMA_VERSION
        and phase_1_manifest.get("dataset_id") == EXPECTED_DATASET_ID,
        "phase8.governance.dataset",
        field="manifest_identity",
    )
    expected = _expected_metadata(phase_1_manifest, split)
    _governance(
        len(works) == len(expected),
        "phase8.governance.work",
        field="count",
    )
    for position, work in enumerate(works):
        if not isinstance(work, ExtractedWork):
            raise Phase8TypeError(
                "phase8.type.record",
                field="work",
                position=position,
            )
        order, work_id, digest, byte_count, code_point_count = expected[position]
        _governance(
            work.manifest_order == order
            and work.work_id == work_id
            and work.split == split
            and work.normalization == EXPECTED_NORMALIZATION
            and isinstance(work.processed, ByteStage)
            and work.processed.sha256 == digest
            and work.processed.byte_count == byte_count
            and work.processed_code_point_count == code_point_count,
            "phase8.governance.work",
            position=position,
        )

    tokenizer = CodePointTokenizer(vocabulary.code_points)
    output: list[Phase8Window] = []
    for position, work in enumerate(works):
        _, work_id, digest, byte_count, code_point_count = expected[position]
        text = work.processed_text
        if not isinstance(text, str):
            raise Phase8TypeError(
                "phase8.type.record",
                field="processed_text",
                position=position,
            )
        try:
            encoded = text.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            raise Phase8GovernanceError(
                "phase8.governance.work",
                field="strict_utf8",
                position=position,
            ) from None
        _governance(
            hashlib.sha256(encoded).hexdigest() == digest
            and len(encoded) == byte_count
            and len(text) == code_point_count,
            "phase8.governance.work",
            field="content_identity",
            position=position,
        )
        token_ids = tokenizer.encode(text)
        for start_index in range(0, len(token_ids) - 1, STRIDE):
            length = min(CONTEXT_LENGTH, len(token_ids) - 1 - start_index)
            output.append(
                Phase8Window(
                    work_id=work_id,
                    manifest_order=work.manifest_order,
                    split=split,
                    start_index=start_index,
                    input_ids=token_ids[start_index : start_index + length],
                    target_ids=token_ids[
                        start_index + 1 : start_index + length + 1
                    ],
                )
            )
    return tuple(output)


def create_phase8_epoch_order(
    number_of_windows: int,
    *,
    epoch: int,
    generator: torch.Generator,
) -> tuple[int, ...]:
    if type(number_of_windows) is not int:
        raise Phase8TypeError("phase8.type.argument", field="number_of_windows")
    if type(epoch) is not int:
        raise Phase8TypeError("phase8.type.argument", field="epoch")
    if type(generator) is not torch.Generator:
        raise Phase8TypeError("phase8.type.generator")
    _require(number_of_windows > 0, "phase8.contract.order", field="population")
    _require(
        1 <= epoch <= MAXIMUM_EPOCHS,
        "phase8.contract.order",
        field="epoch",
    )
    _require(
        generator.device == torch.device("cpu"),
        "phase8.contract.order",
        field="generator_device",
    )
    global_before = torch.get_rng_state().clone()
    generator_before = generator.get_state().clone()
    try:
        value = torch.randperm(
            number_of_windows,
            generator=generator,
            device="cpu",
        )
        result = tuple(int(item) for item in value.tolist())
        _require(
            len(result) == number_of_windows
            and tuple(sorted(result)) == tuple(range(number_of_windows)),
            "phase8.contract.order",
            field="permutation",
        )
    except BaseException:
        generator.set_state(generator_before)
        raise
    _require(
        torch.equal(torch.get_rng_state(), global_before),
        "phase8.contract.order",
        field="global_rng",
    )
    return result


def _iterate_batches(
    windows: tuple[Phase8Window, ...],
    order: tuple[int, ...],
    epoch: int,
) -> Iterator[Phase8LogicalBatch]:
    for batch_index, start in enumerate(range(0, len(order), LOGICAL_BATCH_CAPACITY)):
        indices = order[start : start + LOGICAL_BATCH_CAPACITY]
        yield Phase8LogicalBatch(
            epoch=epoch,
            batch_index=batch_index,
            window_indices=indices,
            target_count=sum(windows[index].length for index in indices),
        )


def iter_phase8_logical_batches(
    windows: tuple[Phase8Window, ...],
    order: tuple[int, ...],
    *,
    epoch: int,
) -> Iterator[Phase8LogicalBatch]:
    if type(windows) is not tuple:
        raise Phase8TypeError("phase8.type.argument", field="windows")
    if type(order) is not tuple:
        raise Phase8TypeError("phase8.type.argument", field="order")
    if type(epoch) is not int:
        raise Phase8TypeError("phase8.type.argument", field="epoch")
    _require(1 <= epoch <= MAXIMUM_EPOCHS, "phase8.contract.batch", field="epoch")
    for position, window in enumerate(windows):
        if type(window) is not Phase8Window:
            raise Phase8TypeError(
                "phase8.type.record",
                field="window",
                position=position,
            )
        _require(
            1 <= window.length <= CONTEXT_LENGTH
            and len(window.target_ids) == window.length,
            "phase8.contract.window",
            position=position,
        )
    for position, index in enumerate(order):
        if type(index) is not int:
            raise Phase8TypeError(
                "phase8.type.record",
                field="order_index",
                position=position,
            )
    _require(
        len(order) == len(windows)
        and tuple(sorted(order)) == tuple(range(len(windows))),
        "phase8.contract.order",
        field="permutation",
    )
    return _iterate_batches(windows, order, epoch)


__all__ = [
    "build_phase8_windows",
    "create_phase8_epoch_order",
    "iter_phase8_logical_batches",
]
