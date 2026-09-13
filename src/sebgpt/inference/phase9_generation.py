"""Transparent Phase 9 autoregressive generation and sampling mechanics."""

from __future__ import annotations

import hashlib
import math
from typing import Literal

import torch

from sebgpt.model.mini_gpt import MiniGPT
from sebgpt.tokenization.code_point import CodePointTokenizer
from sebgpt.tokenization.vocabulary_artifact import (
    VocabularyBinding,
    _is_verified_vocabulary_binding,
)
from sebgpt.inference.phase9_types import (
    Phase9CheckpointIdentity,
    Phase9ContractError,
    Phase9Generation,
    Phase9InferenceBundle,
    Phase9NumericalError,
    Phase9TypeError,
    _is_inference_bundle,
    _validate_phase9_minigpt_structure,
)


VOCABULARY_SIZE = 81
CONTEXT_CAPACITY = 256
MAXIMUM_GENERATED_TOKENS = 1024
MAXIMUM_SEED = 9_223_372_036_854_775_807
RUN_ID = "EXP-20260912-01"
CHECKPOINT_SHA256 = "6990c89166d48732b0601aeae1edd57a9f0e22eec1c7c1160d5be1b8281c9148"
CATALOG_SHA256 = "6d590f0355ec950d770e77b4c542d65cda15012fbea56349b97b154a0d614241"
VALIDATION_LOSS_HEX = "0x1.3f94b678f5807p+1"


def _all_true(value: torch.Tensor) -> bool:
    return bool(torch.all(value).item())


def _validate_logits(value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise Phase9TypeError("phase9.type.argument", field="logits")
    if value.dim() != 1:
        raise Phase9ContractError("phase9.sampling.logits", field="rank")
    if tuple(value.shape) != (VOCABULARY_SIZE,):
        raise Phase9ContractError("phase9.sampling.logits", field="shape")
    if value.dtype is not torch.float32:
        raise Phase9ContractError("phase9.sampling.logits", field="dtype")
    if value.device != torch.device("cpu"):
        raise Phase9ContractError("phase9.sampling.logits", field="device")
    if not _all_true(torch.isfinite(value)):
        raise Phase9NumericalError("phase9.sampling.logits", field="finite")
    return value


def select_greedy_token_id(next_token_logits: torch.Tensor) -> int:
    value = _validate_logits(next_token_logits)
    return int(torch.argmax(value).item())


def _validate_probability_vector(
    probabilities: torch.Tensor,
    retained: tuple[int, ...],
) -> None:
    tolerance = 8 * torch.finfo(torch.float32).eps
    retained_set = set(retained)
    if (
        type(probabilities) is not torch.Tensor
        or tuple(probabilities.shape) != (VOCABULARY_SIZE,)
        or probabilities.dtype is not torch.float32
        or probabilities.device != torch.device("cpu")
        or not _all_true(torch.isfinite(probabilities) & (probabilities >= 0))
        or abs(float(probabilities.sum().item()) - 1.0) > tolerance
        or any(float(probabilities[index].item()) <= 0.0 for index in retained)
        or any(
            float(probabilities[index].item()) != 0.0
            for index in range(VOCABULARY_SIZE)
            if index not in retained_set
        )
    ):
        raise Phase9NumericalError(
            "phase9.sampling.probabilities", field="probabilities"
        )


def build_categorical_probabilities(
    next_token_logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    value = _validate_logits(next_token_logits)
    if type(temperature) is not float:
        raise Phase9TypeError("phase9.type.argument", field="temperature")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise Phase9ContractError("phase9.sampling.temperature", field="temperature")
    if type(top_k) is not int:
        raise Phase9TypeError("phase9.type.argument", field="top_k")
    if not 1 <= top_k <= VOCABULARY_SIZE:
        raise Phase9ContractError("phase9.sampling.top_k", field="top_k")

    scaled = value / temperature
    if not _all_true(torch.isfinite(scaled)):
        raise Phase9NumericalError("phase9.sampling.temperature", field="temperature")
    ordered_ids = sorted(
        range(VOCABULARY_SIZE),
        key=lambda token_id: (-float(scaled[token_id].item()), token_id),
    )
    retained = ordered_ids[:top_k]
    weights = torch.zeros((VOCABULARY_SIZE,), dtype=torch.float32, device="cpu")
    retained_indices = torch.tensor(retained, dtype=torch.long, device="cpu")
    retained_logits = scaled[retained_indices]
    retained_shifted = retained_logits - retained_logits.max()
    retained_weights = torch.exp(retained_shifted)
    if not _all_true(torch.isfinite(retained_weights) & (retained_weights > 0)):
        raise Phase9NumericalError("phase9.sampling.probabilities", field="probabilities")
    weights[retained_indices] = retained_weights
    total = weights.sum()
    if not bool(torch.isfinite(total).item()) or float(total.item()) <= 0.0:
        raise Phase9NumericalError("phase9.sampling.probabilities", field="probabilities")
    probabilities = weights / total
    _validate_probability_vector(probabilities, tuple(retained))
    return probabilities.detach()


def _validate_bundle(bundle: object) -> Phase9InferenceBundle:
    if type(bundle) is not Phase9InferenceBundle:
        raise Phase9TypeError("phase9.type.record", field="bundle")
    if not _is_inference_bundle(bundle):
        raise Phase9ContractError("phase9.contract.bundle")
    if type(bundle.model) is not MiniGPT:
        raise Phase9TypeError("phase9.type.record", field="model")
    if type(bundle.tokenizer) is not CodePointTokenizer:
        raise Phase9TypeError("phase9.type.record", field="tokenizer")
    if type(bundle.vocabulary) is not VocabularyBinding:
        raise Phase9TypeError("phase9.type.record", field="vocabulary")
    if not _is_verified_vocabulary_binding(bundle.vocabulary):
        raise Phase9ContractError("phase9.contract.bundle", field="vocabulary")
    if bundle.tokenizer.code_points != bundle.vocabulary.code_points:
        raise Phase9ContractError("phase9.contract.bundle", field="tokenizer")
    checkpoint = bundle.checkpoint
    if type(checkpoint) is not Phase9CheckpointIdentity:
        raise Phase9TypeError("phase9.type.record", field="checkpoint")
    if (
        type(checkpoint.run_id) is not str or checkpoint.run_id != RUN_ID
        or checkpoint.role != "best_validation"
        or checkpoint.logical_id != "epoch-0010"
        or type(checkpoint.epoch) is not int or checkpoint.epoch != 10
        or checkpoint.validation_loss_hex != VALIDATION_LOSS_HEX
        or checkpoint.checkpoint_sha256 != CHECKPOINT_SHA256
        or checkpoint.checkpoint_relative_path != f"objects/{CHECKPOINT_SHA256}.pt"
        or checkpoint.catalog_sha256 != CATALOG_SHA256
    ):
        raise Phase9ContractError("phase9.contract.bundle", field="checkpoint")
    _validate_phase9_minigpt_structure(bundle.model)
    return bundle


def _model_snapshot(model: MiniGPT) -> tuple[tuple[int, torch.Tensor], ...]:
    return tuple((id(parameter), parameter.detach().clone()) for parameter in model.parameters())


def _dropout_snapshot(model: MiniGPT) -> tuple[torch.Tensor, ...]:
    output: list[torch.Tensor] = []
    for block in model.blocks:
        output.append(block.attention_dropout._generator.get_state().clone())
        output.append(block.feed_forward_dropout._generator.get_state().clone())
    return tuple(output)


def _restore_state(
    model: MiniGPT,
    parameters: tuple[tuple[int, torch.Tensor], ...],
    modes: tuple[tuple[object, bool], ...],
    dropout: tuple[torch.Tensor, ...],
    global_rng: torch.Tensor,
) -> None:
    with torch.no_grad():
        for parameter, (_, expected) in zip(model.parameters(), parameters, strict=True):
            if parameter.requires_grad is not True:
                parameter.requires_grad_(True)
            if not torch.equal(parameter.detach(), expected):
                parameter.copy_(expected)
            parameter.grad = None
    for module, training in modes:
        module.training = training  # type: ignore[attr-defined]
    generators = tuple(
        generator
        for block in model.blocks
        for generator in (block.attention_dropout._generator, block.feed_forward_dropout._generator)
    )
    for generator, state in zip(generators, dropout, strict=True):
        if not torch.equal(generator.get_state(), state):
            generator.set_state(state)
    if not torch.equal(torch.get_rng_state(), global_rng):
        torch.set_rng_state(global_rng)


def _state_digest(value: torch.Tensor) -> str:
    return hashlib.sha256(bytes(value.tolist())).hexdigest()


def _require_unchanged(
    model: MiniGPT,
    parameters: tuple[tuple[int, torch.Tensor], ...],
    dropout: tuple[torch.Tensor, ...],
    global_rng: torch.Tensor,
) -> None:
    current = tuple(model.parameters())
    if (
        len(current) != len(parameters)
        or any(id(value) != expected_id for value, (expected_id, _) in zip(current, parameters, strict=True))
        or any(not torch.equal(value.detach(), expected) for value, (_, expected) in zip(current, parameters, strict=True))
        or any(value.grad is not None for value in current)
        or any(value.requires_grad is not True for value in current)
        or model.training
        or any(module.training for _, module in model.named_modules())
        or any(
            not torch.equal(observed, expected)
            for observed, expected in zip(_dropout_snapshot(model), dropout, strict=True)
        )
        or not torch.equal(torch.get_rng_state(), global_rng)
    ):
        raise Phase9ContractError("phase9.generation.mutation", field="mutation")


def generate_phase9_text(
    bundle: Phase9InferenceBundle,
    prompt: str,
    *,
    generated_token_count: int,
    mode: Literal["greedy", "categorical"],
    temperature: float | None,
    top_k: int | None,
    seed: int | None,
) -> Phase9Generation:
    value = _validate_bundle(bundle)
    if not isinstance(prompt, str):
        raise Phase9TypeError("phase9.type.argument", field="prompt")
    if not prompt or len(prompt) > CONTEXT_CAPACITY:
        raise Phase9ContractError("phase9.generation.prompt", field="prompt")
    if type(generated_token_count) is not int:
        raise Phase9TypeError("phase9.type.argument", field="generated_token_count")
    if not 0 <= generated_token_count <= MAXIMUM_GENERATED_TOKENS:
        raise Phase9ContractError("phase9.generation.count", field="generated_token_count")
    if type(mode) is not str:
        raise Phase9TypeError("phase9.type.argument", field="mode")
    if mode not in ("greedy", "categorical"):
        raise Phase9ContractError("phase9.generation.mode", field="mode")
    if mode == "greedy":
        if temperature is not None or top_k is not None or seed is not None:
            raise Phase9ContractError("phase9.generation.settings", field="mode")
    else:
        if type(temperature) is not float:
            raise Phase9TypeError("phase9.type.argument", field="temperature")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise Phase9ContractError("phase9.sampling.temperature", field="temperature")
        if type(top_k) is not int:
            raise Phase9TypeError("phase9.type.argument", field="top_k")
        if not 1 <= top_k <= VOCABULARY_SIZE:
            raise Phase9ContractError("phase9.sampling.top_k", field="top_k")
        if type(seed) is not int:
            raise Phase9TypeError("phase9.type.argument", field="seed")
        if not 0 <= seed <= MAXIMUM_SEED:
            raise Phase9ContractError("phase9.generation.settings", field="seed")

    prompt_ids = value.tokenizer.encode(prompt)
    if len(prompt_ids) != len(prompt) or value.tokenizer.decode(prompt_ids) != prompt:
        raise Phase9ContractError("phase9.generation.prompt", field="prompt")
    parameters = _model_snapshot(value.model)
    modes = tuple((module, module.training) for _, module in value.model.named_modules())
    dropout = _dropout_snapshot(value.model)
    global_rng = torch.get_rng_state().clone()
    generated: list[int] = []
    local_generator: torch.Generator | None = None
    if mode == "categorical" and generated_token_count > 0:
        local_generator = torch.Generator(device="cpu")
        assert seed is not None
        local_generator.manual_seed(seed)
    try:
        history = list(prompt_ids)
        for _ in range(generated_token_count):
            context = history[-CONTEXT_CAPACITY:]
            token_ids = torch.tensor(context, dtype=torch.long, device="cpu")
            with torch.inference_mode():
                logits = value.model(token_ids)
            if type(logits) is not torch.Tensor:
                raise Phase9TypeError("phase9.type.record", field="logits")
            if (
                tuple(logits.shape) != (len(context), VOCABULARY_SIZE)
                or logits.dtype is not torch.float32
                or logits.device != torch.device("cpu")
            ):
                raise Phase9ContractError("phase9.generation.output", field="logits")
            if not _all_true(torch.isfinite(logits)):
                raise Phase9NumericalError("phase9.sampling.logits", field="logits")
            final_row = logits[-1, :]
            if mode == "greedy":
                next_id = select_greedy_token_id(final_row)
            else:
                assert temperature is not None and top_k is not None and local_generator is not None
                probabilities = build_categorical_probabilities(
                    final_row,
                    temperature=temperature,
                    top_k=top_k,
                )
                sampled = torch.multinomial(
                    probabilities,
                    num_samples=1,
                    replacement=True,
                    generator=local_generator,
                )
                next_id = int(sampled.item())
                if not 0 <= next_id < VOCABULARY_SIZE:
                    raise Phase9NumericalError("phase9.sampling.generator", field="output")
            history.append(next_id)
            generated.append(next_id)

        generated_ids = tuple(generated)
        generated_text = value.tokenizer.decode(generated_ids)
        full_text = value.tokenizer.decode((*prompt_ids, *generated_ids))
        if (
            full_text != prompt + generated_text
            or len(generated_ids) != generated_token_count
            or len(prompt_ids) != len(prompt)
            or value.tokenizer.decode(prompt_ids) != prompt
            or value.tokenizer.decode((*prompt_ids, *generated_ids)) != full_text
        ):
            raise Phase9ContractError("phase9.generation.output", field="output")
        final_digest = (
            _state_digest(local_generator.get_state()) if local_generator is not None else None
        )
        result = Phase9Generation(
            prompt=prompt,
            generated_text=generated_text,
            full_text=full_text,
            prompt_token_ids=prompt_ids,
            generated_token_ids=generated_ids,
            mode=mode,
            generated_token_count=generated_token_count,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
            context_capacity=CONTEXT_CAPACITY,
            local_generator_final_state_sha256=final_digest,
        )
        _require_unchanged(value.model, parameters, dropout, global_rng)
        return result
    finally:
        _restore_state(value.model, parameters, modes, dropout, global_rng)


__all__ = [
    "select_greedy_token_id",
    "build_categorical_probabilities",
    "generate_phase9_text",
]
