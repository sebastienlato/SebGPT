"""Explicit token and learned absolute-position embeddings."""

from __future__ import annotations

import json
import math
from types import MappingProxyType
from typing import Any, Mapping

import torch
from torch import nn

from sebgpt.tokenization.vocabulary_artifact import (
    VocabularyBinding,
    _is_verified_vocabulary_binding,
)


EMBEDDING_DIM = 32
MAX_POSITIONS = 256
INITIALIZATION_SEED = 1337
VOCABULARY_SIZE = 81
PARAMETER_DTYPE = torch.float32
PARAMETER_DEVICE = torch.device("cpu")


class EmbeddingTypeError(TypeError):
    """A deterministic type failure containing no input representation."""

    __slots__ = ("_details",)

    def __init__(self, invariant: str, **safe_facts: Any) -> None:
        details = {"invariant": invariant, **safe_facts}
        self._details = MappingProxyType(details)
        super().__init__(json.dumps(details, ensure_ascii=True, sort_keys=True))

    @property
    def details(self) -> Mapping[str, Any]:
        return self._details


class EmbeddingContractError(ValueError):
    """A deterministic contract failure containing only safe facts."""

    __slots__ = ("_details",)

    def __init__(self, invariant: str, **safe_facts: Any) -> None:
        details = {"invariant": invariant, **safe_facts}
        self._details = MappingProxyType(details)
        super().__init__(json.dumps(details, ensure_ascii=True, sort_keys=True))

    @property
    def details(self) -> Mapping[str, Any]:
        return self._details


def _require(condition: bool, invariant: str, **safe_facts: Any) -> None:
    if not condition:
        raise EmbeddingContractError(invariant, **safe_facts)


def _require_exact_configuration(
    *,
    embedding_dim: int,
    max_positions: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    _require(
        type(embedding_dim) is int and embedding_dim == EMBEDDING_DIM,
        "configuration.embedding_dim",
        expected=EMBEDDING_DIM,
    )
    _require(
        type(max_positions) is int and max_positions == MAX_POSITIONS,
        "configuration.max_positions",
        expected=MAX_POSITIONS,
    )
    _require(
        type(seed) is int and seed == INITIALIZATION_SEED,
        "configuration.seed",
        expected=INITIALIZATION_SEED,
    )
    _require(
        type(device) is torch.device and device == PARAMETER_DEVICE,
        "configuration.device",
        expected=str(PARAMETER_DEVICE),
    )
    _require(
        dtype is PARAMETER_DTYPE,
        "configuration.dtype",
        expected=str(PARAMETER_DTYPE),
    )


class TokenPositionEmbedding(nn.Module):
    """Add explicit token lookup and learned absolute-position lookup."""

    def __init__(
        self,
        vocabulary: VocabularyBinding,
        *,
        embedding_dim: int,
        max_positions: int,
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        _require(
            _is_verified_vocabulary_binding(vocabulary),
            "vocabulary.binding",
        )
        _require_exact_configuration(
            embedding_dim=embedding_dim,
            max_positions=max_positions,
            seed=seed,
            device=device,
            dtype=dtype,
        )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        standard_deviation = 1.0 / math.sqrt(embedding_dim)

        token_weights = torch.empty(
            (vocabulary.vocabulary_size, embedding_dim),
            device=device,
            dtype=dtype,
        )
        nn.init.normal_(
            token_weights,
            mean=0.0,
            std=standard_deviation,
            generator=generator,
        )
        self.token_embeddings = nn.Parameter(token_weights)

        position_weights = torch.empty(
            (max_positions, embedding_dim),
            device=device,
            dtype=dtype,
        )
        nn.init.normal_(
            position_weights,
            mean=0.0,
            std=standard_deviation,
            generator=generator,
        )
        self.position_embeddings = nn.Parameter(position_weights)
        self.vocabulary = vocabulary
        self.embedding_dim = embedding_dim
        self.max_positions = max_positions

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return token plus position vectors after complete validation."""

        if not isinstance(token_ids, torch.Tensor):
            raise EmbeddingTypeError("input.tensor")
        _require(
            token_ids.dim() in (1, 2),
            "input.rank",
            expected=(1, 2),
            observed=token_ids.dim(),
        )
        _require(
            token_ids.dtype is torch.long,
            "input.dtype",
            expected=str(torch.long),
            observed=str(token_ids.dtype),
        )
        _require(
            token_ids.device == PARAMETER_DEVICE
            and self.token_embeddings.device == PARAMETER_DEVICE
            and self.position_embeddings.device == PARAMETER_DEVICE,
            "input.device",
            expected=str(PARAMETER_DEVICE),
            observed=str(token_ids.device),
        )

        sequence_length = token_ids.shape[-1]
        _require(
            sequence_length <= self.max_positions,
            "input.sequence_length",
            expected_max=self.max_positions,
            observed=sequence_length,
        )
        if token_ids.numel() > 0:
            valid_ids = torch.all(
                (token_ids >= 0) & (token_ids < self.vocabulary.vocabulary_size)
            )
            _require(
                bool(valid_ids.item()),
                "input.token_id_range",
                lower_bound=0,
                upper_bound_exclusive=self.vocabulary.vocabulary_size,
                shape=tuple(token_ids.shape),
            )

        position_ids = torch.arange(
            sequence_length,
            dtype=torch.long,
            device=token_ids.device,
        )
        token_vectors = self.token_embeddings[token_ids]
        position_vectors = self.position_embeddings[position_ids]
        return token_vectors + position_vectors
