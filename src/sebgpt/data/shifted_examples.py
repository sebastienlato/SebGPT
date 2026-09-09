"""Deterministic, document-local shifted next-token examples."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


VOCABULARY_SIZE = 81
CONTEXT_LENGTH = 64
STRIDE = 64


def _safe_error_details(
    invariant: str,
    safe_facts: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str]:
    details = {"invariant": invariant, **safe_facts}
    return (
        MappingProxyType(details),
        json.dumps(details, ensure_ascii=True, sort_keys=True),
    )


class Phase4TypeError(TypeError):
    """An exact public Python/object type boundary failed."""

    __slots__ = ("_details",)

    def __init__(self, invariant: str, **safe_facts: Any) -> None:
        self._details, message = _safe_error_details(invariant, safe_facts)
        super().__init__(message)

    @property
    def details(self) -> Mapping[str, Any]:
        return self._details

    def __repr__(self) -> str:
        return str(self)


class Phase4ContractError(ValueError):
    """A typed Phase 4 value or configuration violated its contract."""

    __slots__ = ("_details",)

    def __init__(self, invariant: str, **safe_facts: Any) -> None:
        self._details, message = _safe_error_details(invariant, safe_facts)
        super().__init__(message)

    @property
    def details(self) -> Mapping[str, Any]:
        return self._details

    def __repr__(self) -> str:
        return str(self)


class Phase4GovernanceError(ValueError):
    """A split, provenance, or sealed-data policy boundary failed."""

    __slots__ = ("_details",)

    def __init__(self, invariant: str, **safe_facts: Any) -> None:
        self._details, message = _safe_error_details(invariant, safe_facts)
        super().__init__(message)

    @property
    def details(self) -> Mapping[str, Any]:
        return self._details

    def __repr__(self) -> str:
        return str(self)


@dataclass(frozen=True, repr=False)
class ShiftedTokenExample:
    """One immutable input tuple and its one-token-shifted target tuple."""

    start_index: int
    input_ids: tuple[int, ...] = field(repr=False)
    target_ids: tuple[int, ...] = field(repr=False)

    @property
    def length(self) -> int:
        return len(self.input_ids)

    def __repr__(self) -> str:
        return (
            "ShiftedTokenExample("
            f"start_index={self.start_index}, length={self.length})"
        )


def _iterate_shifted_examples(
    token_ids: tuple[int, ...],
    *,
    context_length: int,
    stride: int,
) -> Iterator[ShiftedTokenExample]:
    for start_index in range(0, len(token_ids) - 1, stride):
        length = min(context_length, len(token_ids) - 1 - start_index)
        yield ShiftedTokenExample(
            start_index=start_index,
            input_ids=token_ids[start_index : start_index + length],
            target_ids=token_ids[start_index + 1 : start_index + length + 1],
        )


def iter_shifted_examples(
    token_ids: tuple[int, ...],
    *,
    context_length: int,
    stride: int,
) -> Iterator[ShiftedTokenExample]:
    """Validate eagerly, then return an on-demand document-local iterator."""

    if type(token_ids) is not tuple:
        raise Phase4TypeError("example.token_ids.exact_tuple")
    if type(context_length) is not int:
        raise Phase4TypeError("example.context_length.exact_int")
    if context_length != CONTEXT_LENGTH:
        raise Phase4ContractError(
            "example.context_length.value",
            expected=CONTEXT_LENGTH,
        )
    if type(stride) is not int:
        raise Phase4TypeError("example.stride.exact_int")
    if stride != STRIDE:
        raise Phase4ContractError(
            "example.stride.value",
            expected=STRIDE,
        )
    for position, token_id in enumerate(token_ids):
        if type(token_id) is not int:
            raise Phase4TypeError(
                "example.token_id.exact_int",
                position=position,
            )
        if not 0 <= token_id < VOCABULARY_SIZE:
            raise Phase4ContractError(
                "example.token_id.range",
                position=position,
                lower_bound=0,
                upper_bound_exclusive=VOCABULARY_SIZE,
            )

    return _iterate_shifted_examples(
        token_ids,
        context_length=context_length,
        stride=stride,
    )


__all__ = [
    "CONTEXT_LENGTH",
    "Phase4ContractError",
    "Phase4GovernanceError",
    "Phase4TypeError",
    "STRIDE",
    "ShiftedTokenExample",
    "VOCABULARY_SIZE",
    "iter_shifted_examples",
]
