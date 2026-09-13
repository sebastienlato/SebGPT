"""Small corpus-neutral Phase 9 evaluation and baseline mechanics."""

from __future__ import annotations

import hashlib
import json
import math
import sys

import torch

from sebgpt.inference.phase9_types import (
    Phase9ContractError,
    Phase9LaplaceUnigramBaseline,
    Phase9Metrics,
    Phase9NumericalError,
    Phase9TokenDocument,
    Phase9TypeError,
    Phase9Window,
    _is_laplace_unigram,
    _make_laplace_unigram,
    _validate_phase9_minigpt_structure,
)
from sebgpt.model.mini_gpt import MiniGPT
from sebgpt.model.simple_language_model import explicit_cross_entropy


def _exact_positive_int(value: object, field: str) -> int:
    if type(value) is not int:
        raise Phase9TypeError("phase9.type.argument", field=field)
    if value < 1:
        raise Phase9ContractError("phase9.contract.window", field=field)
    return value


def build_phase9_windows(
    documents: tuple[Phase9TokenDocument, ...],
    *,
    vocabulary_size: int,
    context_length: int,
    stride: int,
) -> tuple[Phase9Window, ...]:
    if type(documents) is not tuple:
        raise Phase9TypeError("phase9.type.argument", field="documents")
    vocabulary_size = _exact_positive_int(vocabulary_size, "vocabulary_size")
    context_length = _exact_positive_int(context_length, "context_length")
    stride = _exact_positive_int(stride, "stride")
    if context_length > 256 or stride != context_length:
        raise Phase9ContractError("phase9.contract.window", field="stride")
    if not documents:
        raise Phase9ContractError("phase9.contract.window", field="documents")

    seen: set[str] = set()
    split: str | None = None
    previous_order: int | None = None
    result: list[Phase9Window] = []
    for position, document in enumerate(documents):
        if type(document) is not Phase9TokenDocument:
            raise Phase9TypeError("phase9.type.record", field="documents", position=position)
        if type(document.document_id) is not str or not document.document_id:
            raise Phase9ContractError("phase9.contract.window", field="documents", position=position)
        if document.document_id in seen:
            raise Phase9ContractError("phase9.contract.window", field="documents", position=position)
        seen.add(document.document_id)
        if type(document.document_order) is not int or document.document_order < 0:
            raise Phase9ContractError("phase9.contract.window", field="documents", position=position)
        if previous_order is not None and document.document_order <= previous_order:
            raise Phase9ContractError("phase9.contract.window", field="documents", position=position)
        previous_order = document.document_order
        if type(document.split) is not str or not document.split or document.split == "test":
            raise Phase9ContractError("phase9.contract.window", field="split", position=position)
        if split is None:
            split = document.split
        elif document.split != split:
            raise Phase9ContractError("phase9.contract.window", field="split", position=position)
        if type(document.token_ids) is not tuple or len(document.token_ids) < 2:
            raise Phase9ContractError("phase9.contract.window", field="token", position=position)
        for token in document.token_ids:
            if type(token) is not int or not 0 <= token < vocabulary_size:
                raise Phase9ContractError("phase9.contract.window", field="token", position=position)
        for start in range(0, len(document.token_ids) - 1, stride):
            length = min(context_length, len(document.token_ids) - 1 - start)
            result.append(
                Phase9Window(
                    document_id=document.document_id,
                    document_order=document.document_order,
                    split=document.split,
                    start_index=start,
                    input_ids=document.token_ids[start : start + length],
                    target_ids=document.token_ids[start + 1 : start + length + 1],
                )
            )
    if not result:
        raise Phase9ContractError("phase9.contract.window", field="documents")
    return tuple(result)


def _canonical_window_bytes(windows: tuple[Phase9Window, ...]) -> bytes:
    value = [
        {
            "document_id": window.document_id,
            "document_order": window.document_order,
            "split": window.split,
            "start_index": window.start_index,
            "input_ids": list(window.input_ids),
            "target_ids": list(window.target_ids),
        }
        for window in windows
    ]
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def _validate_windows(
    windows: object,
    *,
    split: object,
    vocabulary_size: int | None = None,
    allow_test: bool = False,
) -> tuple[tuple[Phase9Window, ...], int, str]:
    if type(windows) is not tuple:
        raise Phase9TypeError("phase9.type.argument", field="windows")
    if type(split) is not str:
        raise Phase9TypeError("phase9.type.argument", field="split")
    if not split or (split == "test" and not allow_test):
        raise Phase9ContractError("phase9.evaluation.split", field="split")
    if not windows:
        raise Phase9ContractError("phase9.evaluation.windows", field="windows")
    previous: tuple[int, int] | None = None
    seen_ids: dict[int, str] = {}
    seen_orders: dict[str, int] = {}
    document_stride: dict[str, int] = {}
    prior_window: Phase9Window | None = None
    targets = 0
    for position, window in enumerate(windows):
        if type(window) is not Phase9Window:
            raise Phase9TypeError("phase9.type.record", field="windows", position=position)
        if (
            type(window.document_id) is not str
            or not window.document_id
            or type(window.document_order) is not int
            or window.document_order < 0
            or type(window.split) is not str
            or window.split != split
            or type(window.start_index) is not int
            or window.start_index < 0
            or type(window.input_ids) is not tuple
            or type(window.target_ids) is not tuple
            or not 1 <= len(window.input_ids) <= 256
            or len(window.input_ids) != len(window.target_ids)
            or window.input_ids[1:] != window.target_ids[:-1]
        ):
            raise Phase9ContractError("phase9.evaluation.windows", field="windows", position=position)
        if window.document_order in seen_ids and seen_ids[window.document_order] != window.document_id:
            raise Phase9ContractError("phase9.evaluation.windows", field="windows", position=position)
        if window.document_id in seen_orders and seen_orders[window.document_id] != window.document_order:
            raise Phase9ContractError("phase9.evaluation.windows", field="windows", position=position)
        seen_ids[window.document_order] = window.document_id
        seen_orders[window.document_id] = window.document_order
        key = (window.document_order, window.start_index)
        if previous is not None and key <= previous:
            raise Phase9ContractError("phase9.evaluation.windows", field="windows", position=position)
        if prior_window is None or window.document_id != prior_window.document_id:
            if window.start_index != 0:
                raise Phase9ContractError("phase9.evaluation.windows", field="windows", position=position)
            document_stride[window.document_id] = len(window.input_ids)
        else:
            expected_stride = document_stride[window.document_id]
            if (
                len(prior_window.input_ids) != expected_stride
                or window.start_index != prior_window.start_index + expected_stride
                or len(window.input_ids) > expected_stride
                or prior_window.target_ids[-1] != window.input_ids[0]
            ):
                raise Phase9ContractError(
                    "phase9.evaluation.windows", field="windows", position=position
                )
        previous = key
        prior_window = window
        for token in (*window.input_ids, *window.target_ids):
            if type(token) is not int or token < 0 or (
                vocabulary_size is not None and token >= vocabulary_size
            ):
                raise Phase9ContractError("phase9.evaluation.windows", field="token", position=position)
        targets += len(window.target_ids)
    digest = hashlib.sha256(_canonical_window_bytes(windows)).hexdigest()
    return windows, targets, digest


def _metrics(
    *,
    predictor: str,
    split: str,
    window_sha256: str,
    total_nll: float,
    correct_count: int,
    target_count: int,
    window_count: int,
) -> Phase9Metrics:
    nll = total_nll / target_count
    try:
        perplexity = math.exp(nll)
    except OverflowError:
        raise Phase9NumericalError("phase9.evaluation.metric", field="perplexity") from None
    accuracy = correct_count / target_count
    if (
        not math.isfinite(nll)
        or nll < 0.0
        or not math.isfinite(perplexity)
        or perplexity < 1.0
        or not math.isfinite(accuracy)
        or not 0.0 <= accuracy <= 1.0
    ):
        raise Phase9NumericalError("phase9.evaluation.metric", field="metric")
    return Phase9Metrics(
        predictor=predictor,
        split=split,
        window_sha256=window_sha256,
        nll=nll,
        perplexity=perplexity,
        correct_count=correct_count,
        target_count=target_count,
        top1_accuracy=accuracy,
        window_count=window_count,
    )


def fit_phase9_laplace_unigram(
    training_windows: tuple[Phase9Window, ...],
    *,
    vocabulary_size: int,
) -> Phase9LaplaceUnigramBaseline:
    vocabulary_size = _exact_positive_int(vocabulary_size, "vocabulary_size")
    if (
        type(training_windows) is tuple
        and training_windows
        and type(training_windows[0]) is Phase9Window
        and training_windows[0].split != "train"
    ):
        raise Phase9ContractError("phase9.baseline.training_only", field="split")
    windows, target_count, _ = _validate_windows(
        training_windows,
        split="train",
        vocabulary_size=vocabulary_size,
    )
    counts = [0] * vocabulary_size
    for window in windows:
        for token in window.target_ids:
            counts[token] += 1
    smoothed = tuple(value + 1 for value in counts)
    denominator = target_count + vocabulary_size
    probabilities = tuple(value / denominator for value in smoothed)
    if (
        sum(counts) != target_count
        or sum(smoothed) != denominator
        or any(not math.isfinite(value) or value <= 0.0 for value in probabilities)
        or abs(math.fsum(probabilities) - 1.0) > 8 * sys.float_info.epsilon
    ):
        raise Phase9NumericalError("phase9.baseline.probabilities", field="probabilities")
    top_token = min(range(vocabulary_size), key=lambda index: (-counts[index], index))
    return _make_laplace_unigram(
        vocabulary_size=vocabulary_size,
        training_target_count=target_count,
        raw_counts=tuple(counts),
        smoothed_counts=smoothed,
        probabilities=probabilities,
        top_token_id=top_token,
    )


def evaluate_phase9_uniform(
    windows: tuple[Phase9Window, ...],
    *,
    vocabulary_size: int,
    split: str,
) -> Phase9Metrics:
    return _evaluate_uniform_internal(
        windows,
        vocabulary_size=vocabulary_size,
        split=split,
        allow_test=False,
    )


def _evaluate_uniform_internal(
    windows: tuple[Phase9Window, ...],
    *,
    vocabulary_size: int,
    split: str,
    allow_test: bool,
) -> Phase9Metrics:
    vocabulary_size = _exact_positive_int(vocabulary_size, "vocabulary_size")
    values, target_count, digest = _validate_windows(
        windows,
        split=split,
        vocabulary_size=vocabulary_size,
        allow_test=allow_test,
    )
    per_target = math.log(vocabulary_size)
    contributions = [per_target * len(window.target_ids) for window in values]
    correct = sum(token == 0 for window in values for token in window.target_ids)
    return _metrics(
        predictor=f"uniform_{vocabulary_size}",
        split=split,
        window_sha256=digest,
        total_nll=math.fsum(contributions),
        correct_count=correct,
        target_count=target_count,
        window_count=len(values),
    )


def evaluate_phase9_laplace_unigram(
    baseline: Phase9LaplaceUnigramBaseline,
    windows: tuple[Phase9Window, ...],
    *,
    split: str,
) -> Phase9Metrics:
    return _evaluate_laplace_internal(
        baseline,
        windows,
        split=split,
        allow_test=False,
    )


def _evaluate_laplace_internal(
    baseline: Phase9LaplaceUnigramBaseline,
    windows: tuple[Phase9Window, ...],
    *,
    split: str,
    allow_test: bool,
) -> Phase9Metrics:
    if type(baseline) is not Phase9LaplaceUnigramBaseline:
        raise Phase9TypeError("phase9.type.record", field="baseline")
    if not _is_laplace_unigram(baseline):
        raise Phase9ContractError("phase9.baseline.counts", field="baseline")
    if (
        type(baseline.vocabulary_size) is not int
        or baseline.vocabulary_size < 1
        or type(baseline.training_target_count) is not int
        or baseline.training_target_count < 1
        or type(baseline.raw_counts) is not tuple
        or type(baseline.smoothed_counts) is not tuple
        or type(baseline.probabilities) is not tuple
        or len(baseline.raw_counts) != baseline.vocabulary_size
        or len(baseline.smoothed_counts) != baseline.vocabulary_size
        or len(baseline.probabilities) != baseline.vocabulary_size
        or sum(baseline.raw_counts) != baseline.training_target_count
        or any(type(value) is not int or value < 0 for value in baseline.raw_counts)
        or any(type(value) is not int or value <= 0 for value in baseline.smoothed_counts)
        or any(smoothed != raw + 1 for raw, smoothed in zip(baseline.raw_counts, baseline.smoothed_counts, strict=True))
        or sum(baseline.smoothed_counts) != baseline.training_target_count + baseline.vocabulary_size
        or any(type(value) is not float or not math.isfinite(value) or value <= 0 for value in baseline.probabilities)
        or abs(math.fsum(baseline.probabilities) - 1.0) > 8 * sys.float_info.epsilon
        or any(
            probability
            != smoothed / (baseline.training_target_count + baseline.vocabulary_size)
            for smoothed, probability in zip(
                baseline.smoothed_counts, baseline.probabilities, strict=True
            )
        )
        or type(baseline.top_token_id) is not int
        or baseline.top_token_id != min(range(baseline.vocabulary_size), key=lambda index: (-baseline.raw_counts[index], index))
    ):
        raise Phase9ContractError("phase9.baseline.counts", field="baseline")
    values, target_count, digest = _validate_windows(
        windows,
        split=split,
        vocabulary_size=baseline.vocabulary_size,
        allow_test=allow_test,
    )
    contributions = [
        -math.log(baseline.probabilities[token])
        for window in values
        for token in window.target_ids
    ]
    correct = sum(
        token == baseline.top_token_id for window in values for token in window.target_ids
    )
    return _metrics(
        predictor="training_laplace_unigram_add_one",
        split=split,
        window_sha256=digest,
        total_nll=math.fsum(contributions),
        correct_count=correct,
        target_count=target_count,
        window_count=len(values),
    )


def _dropout_states(model: MiniGPT) -> tuple[torch.Tensor, ...]:
    states: list[torch.Tensor] = []
    for block in model.blocks:
        states.append(block.attention_dropout._generator.get_state().clone())
        states.append(block.feed_forward_dropout._generator.get_state().clone())
    return tuple(states)


def _restore_model(
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


def evaluate_phase9_model(
    model: MiniGPT,
    windows: tuple[Phase9Window, ...],
    *,
    predictor: str,
    split: str,
) -> Phase9Metrics:
    return _evaluate_model_internal(
        model,
        windows,
        predictor=predictor,
        split=split,
        allow_test=False,
    )


def _evaluate_model_internal(
    model: MiniGPT,
    windows: tuple[Phase9Window, ...],
    *,
    predictor: str,
    split: str,
    allow_test: bool,
) -> Phase9Metrics:
    model = _validate_phase9_minigpt_structure(model)
    if type(predictor) is not str:
        raise Phase9TypeError("phase9.type.argument", field="predictor")
    if not predictor:
        raise Phase9ContractError("phase9.evaluation.metric", field="predictor")
    named_parameters = tuple(model.named_parameters())
    values, target_count, digest = _validate_windows(
        windows,
        split=split,
        vocabulary_size=81,
        allow_test=allow_test,
    )
    parameters = tuple((id(value), value.detach().clone()) for _, value in named_parameters)
    dropout = _dropout_states(model)
    modes = tuple((module, module.training) for _, module in model.named_modules())
    global_rng = torch.get_rng_state().clone()
    weighted: list[float] = []
    correct = 0
    try:
        for window in values:
            inputs = torch.tensor(window.input_ids, dtype=torch.long, device="cpu")
            targets = torch.tensor(window.target_ids, dtype=torch.long, device="cpu")
            with torch.inference_mode():
                logits = model(inputs)
                if (
                    type(logits) is not torch.Tensor
                    or tuple(logits.shape) != (len(window.input_ids), 81)
                    or logits.device != torch.device("cpu")
                    or logits.dtype is not torch.float32
                    or not bool(torch.all(torch.isfinite(logits)).item())
                ):
                    raise Phase9NumericalError("phase9.evaluation.metric", field="logits")
                mean_loss = explicit_cross_entropy(logits, targets)
            if not bool(torch.isfinite(mean_loss).item()):
                raise Phase9NumericalError("phase9.evaluation.metric", field="metric")
            weighted.append(float(mean_loss.item()) * len(window.target_ids))
            correct += int((torch.argmax(logits, dim=1) == targets).sum().item())
        result = _metrics(
            predictor=predictor,
            split=split,
            window_sha256=digest,
            total_nll=math.fsum(weighted),
            correct_count=correct,
            target_count=target_count,
            window_count=len(values),
        )
        current = tuple(model.parameters())
        if (
            any(id(value) != expected_id or not torch.equal(value.detach(), expected) for value, (expected_id, expected) in zip(current, parameters, strict=True))
            or any(value.grad is not None for value in current)
            or any(value.requires_grad is not True for value in current)
            or model.training
            or any(module.training for _, module in model.named_modules())
            or any(not torch.equal(a, b) for a, b in zip(_dropout_states(model), dropout, strict=True))
            or not torch.equal(torch.get_rng_state(), global_rng)
        ):
            raise Phase9ContractError("phase9.evaluation.mutation", field="mutation")
        return result
    finally:
        _restore_model(model, parameters, modes, dropout, global_rng)


__all__ = [
    "build_phase9_windows",
    "fit_phase9_laplace_unigram",
    "evaluate_phase9_model",
    "evaluate_phase9_uniform",
    "evaluate_phase9_laplace_unigram",
]
