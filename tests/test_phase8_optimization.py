"""Synthetic tests for Phase 8 AdamW, gradients, and evaluation."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from sebgpt.model.mini_gpt import MiniGPT  # noqa: E402
import sebgpt.training.phase8_optimization as optimization  # noqa: E402
from sebgpt.model.simple_language_model import explicit_cross_entropy  # noqa: E402
from sebgpt.tokenization.vocabulary_artifact import (  # noqa: E402
    load_accepted_vocabulary_binding,
)
from sebgpt.training import (  # noqa: E402
    Phase8ContractError,
    Phase8LogicalBatch,
    Phase8NumericalError,
    Phase8Progress,
    Phase8Window,
    configure_phase8_runtime,
    create_phase8_optimizer,
    evaluate_phase8_split,
    run_phase8_logical_batch,
)


def _model() -> MiniGPT:
    vocabulary = load_accepted_vocabulary_binding(REPOSITORY_ROOT)
    return MiniGPT(
        vocabulary,
        model_width=32,
        max_sequence_length=256,
        number_of_blocks=4,
        number_of_heads=4,
        head_width=8,
        hidden_width=128,
        layer_norm_epsilon=1e-5,
        dropout_probability=0.1,
        embedding_seed=1337,
        block_parameter_seeds=(7001, 7002, 7003, 7004),
        output_head_seed=7005,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ).train()


def _windows() -> tuple[Phase8Window, ...]:
    return (
        Phase8Window("hamlet", 1, "train", 0, (1, 2, 3), (2, 3, 4)),
        Phase8Window("hamlet", 1, "train", 256, (5,), (6,)),
    )


def _validation_windows() -> tuple[Phase8Window, ...]:
    return (
        Phase8Window("the-tempest", 7, "validation", 0, (1, 2), (2, 3)),
    )


def _evaluation_snapshot(model, optimizer, generator, progress):
    return {
        "parameters": tuple(
            parameter.detach().clone() for parameter in model.parameters()
        ),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "progress": progress,
        "order_rng": generator.get_state().clone(),
        "global_rng": torch.get_rng_state().clone(),
        "dropout_rngs": tuple(
            value
            for block in model.blocks
            for value in (
                block.attention_dropout._generator.get_state().clone(),
                block.feed_forward_dropout._generator.get_state().clone(),
            )
        ),
        "modes": tuple(
            (name, module.training) for name, module in model.named_modules()
        ),
    }


def _assert_evaluation_snapshot(
    test,
    snapshot,
    model,
    optimizer,
    generator,
    progress,
    *,
    expect_modes_restored=True,
):
    for expected, observed in zip(
        snapshot["parameters"], model.parameters(), strict=True
    ):
        test.assertTrue(torch.equal(expected, observed))
    test.assertTrue(
        optimization._objects_equal(snapshot["optimizer"], optimizer.state_dict())
    )
    test.assertIs(progress, snapshot["progress"])
    test.assertTrue(torch.equal(snapshot["order_rng"], generator.get_state()))
    test.assertTrue(torch.equal(snapshot["global_rng"], torch.get_rng_state()))
    observed_dropout = tuple(
        value
        for block in model.blocks
        for value in (
            block.attention_dropout._generator.get_state(),
            block.feed_forward_dropout._generator.get_state(),
        )
    )
    test.assertTrue(
        all(
            torch.equal(expected, observed)
            for expected, observed in zip(
                snapshot["dropout_rngs"], observed_dropout, strict=True
            )
        )
    )
    if expect_modes_restored:
        test.assertEqual(
            tuple((name, module.training) for name, module in model.named_modules()),
            snapshot["modes"],
        )


class Phase8OptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        configure_phase8_runtime()

    def test_optimizer_has_exact_supported_group_schema(self) -> None:
        model = _model()
        optimizer = create_phase8_optimizer(model)
        self.assertEqual(
            tuple(optimizer.state_dict()["param_groups"][0]),
            (
                "lr",
                "betas",
                "eps",
                "weight_decay",
                "amsgrad",
                "maximize",
                "foreach",
                "capturable",
                "differentiable",
                "fused",
                "decoupled_weight_decay",
                "params",
            ),
        )
        self.assertIs(optimizer.param_groups[0]["decoupled_weight_decay"], True)
        self.assertEqual(optimizer.state_dict()["state"], {})
        self.assertEqual(len(optimizer.param_groups[0]["params"]), 90)

    def test_logical_batch_produces_one_step_and_none_gradients(self) -> None:
        model = _model()
        optimizer = create_phase8_optimizer(model)
        windows = _windows()
        batch = Phase8LogicalBatch(1, 0, (0, 1), 4)
        result = run_phase8_logical_batch(model, optimizer, windows, batch)
        self.assertEqual(result.target_count, 4)
        self.assertEqual(result.sequence_count, 2)
        self.assertTrue(result.pre_clip_global_norm >= 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        state = optimizer.state_dict()["state"]
        self.assertEqual(tuple(state), tuple(range(90)))
        for parameter_state in state.values():
            step = parameter_state["step"]
            self.assertEqual(step.shape, torch.Size([]))
            self.assertEqual(step.device, torch.device("cpu"))
            self.assertIs(step.dtype, torch.float32)
            self.assertEqual(float(step.item()), 1.0)

    def test_token_weighted_accumulation_matches_independent_control(self) -> None:
        public_model = _model()
        control_model = _model()
        public_optimizer = create_phase8_optimizer(public_model)
        control_optimizer = create_phase8_optimizer(control_model)
        windows = _windows()
        run_phase8_logical_batch(
            public_model,
            public_optimizer,
            windows,
            Phase8LogicalBatch(1, 0, (0, 1), 4),
        )

        control_optimizer.zero_grad(set_to_none=True)
        for window in windows:
            inputs = torch.tensor(window.input_ids, dtype=torch.long)
            targets = torch.tensor(window.target_ids, dtype=torch.long)
            loss = explicit_cross_entropy(control_model(inputs), targets)
            (loss * (window.length / 4)).backward()
        torch.nn.utils.clip_grad_norm_(
            tuple(control_model.parameters()),
            max_norm=1.0,
            norm_type=2.0,
            error_if_nonfinite=True,
            foreach=False,
        )
        control_optimizer.step()
        control_optimizer.zero_grad(set_to_none=True)

        for public, control in zip(
            public_model.parameters(),
            control_model.parameters(),
            strict=True,
        ):
            self.assertTrue(torch.equal(public, control))
        public_state = public_optimizer.state_dict()
        control_state = control_optimizer.state_dict()
        for position in range(90):
            for name in ("step", "exp_avg", "exp_avg_sq"):
                self.assertTrue(
                    torch.equal(
                        public_state["state"][position][name],
                        control_state["state"][position][name],
                    )
                )

    def test_clipping_oracle_includes_supported_epsilon(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        parameter.grad = torch.tensor([1.0])
        observed = torch.nn.utils.clip_grad_norm_(
            (parameter,),
            max_norm=1.0,
            norm_type=2.0,
            error_if_nonfinite=True,
            foreach=False,
        )
        expected_coefficient = torch.clamp(
            torch.tensor(1.0) / (torch.tensor(1.0) + 1e-6),
            max=1.0,
        )
        self.assertEqual(float(observed.item()), 1.0)
        self.assertTrue(torch.equal(parameter.grad, expected_coefficient.reshape(1)))
        self.assertLess(float(parameter.grad.item()), 1.0)

    def test_clipping_boundaries_match_literal_supported_formula(self) -> None:
        for gradient_value in (0.25, 1.0 - 5e-7, 1.0, 2.0):
            with self.subTest(gradient_value=gradient_value):
                parameter = torch.nn.Parameter(torch.tensor([0.0]))
                parameter.grad = torch.tensor([gradient_value])
                expected_norm = torch.linalg.vector_norm(parameter.grad, 2.0)
                expected_coefficient = torch.clamp(
                    torch.tensor(1.0) / (expected_norm + 1e-6),
                    max=1.0,
                )
                expected_gradient = parameter.grad * expected_coefficient
                observed = torch.nn.utils.clip_grad_norm_(
                    (parameter,),
                    max_norm=1.0,
                    norm_type=2.0,
                    error_if_nonfinite=True,
                    foreach=False,
                )
                self.assertTrue(torch.equal(observed, expected_norm))
                self.assertTrue(torch.equal(parameter.grad, expected_gradient))

    def test_nonfinite_clipping_norm_is_rejected(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([0.0]))
        parameter.grad = torch.tensor([float("inf")])
        with self.assertRaises(RuntimeError):
            torch.nn.utils.clip_grad_norm_(
                (parameter,),
                max_norm=1.0,
                norm_type=2.0,
                error_if_nonfinite=True,
                foreach=False,
            )

    def test_independent_single_expression_gradient_oracle_matches_accumulation(self) -> None:
        observed_model = _model()
        control_model = _model()
        observed_optimizer = create_phase8_optimizer(observed_model)
        windows = _windows()
        _, evidence = optimization._run_phase8_logical_batch(
            observed_model,
            observed_optimizer,
            windows,
            Phase8LogicalBatch(1, 0, (0, 1), 4),
            capture_evidence=True,
        )
        self.assertIsNotNone(evidence)
        control_losses = []
        for window in windows:
            logits = control_model(torch.tensor(window.input_ids, dtype=torch.long))
            targets = torch.tensor(window.target_ids, dtype=torch.long)
            control_losses.append(
                explicit_cross_entropy(logits, targets) * window.length
            )
        (sum(control_losses) / 4).backward()
        for expected, parameter in zip(
            evidence.accumulated_gradients,  # type: ignore[union-attr]
            control_model.parameters(),
            strict=True,
        ):
            self.assertTrue(torch.equal(expected, parameter.grad))

    def test_training_authority_rejections_precede_forward_and_preserve_state(self) -> None:
        cases = (
            (
                (
                    Phase8Window(
                        "the-tempest", 7, "validation", 0, (1, 2), (2, 3)
                    ),
                ),
                Phase8LogicalBatch(1, 0, (0,), 2),
            ),
            (
                (
                    _windows()[0],
                    Phase8Window(
                        "the-tempest", 7, "validation", 0, (1,), (2,)
                    ),
                ),
                Phase8LogicalBatch(1, 0, (0, 1), 4),
            ),
            (_windows(), Phase8LogicalBatch(1, 0, (0, 0), 6)),
            (_windows(), Phase8LogicalBatch(1, 1, (0, 1), 4)),
        )
        for windows, batch in cases:
            with self.subTest(batch=batch):
                model = _model()
                optimizer = create_phase8_optimizer(model)
                parameters_before = tuple(
                    parameter.detach().clone() for parameter in model.parameters()
                )
                optimizer_before = copy.deepcopy(optimizer.state_dict())
                with patch.object(model, "forward", wraps=model.forward) as forward:
                    with self.assertRaises(Phase8ContractError):
                        run_phase8_logical_batch(model, optimizer, windows, batch)
                forward.assert_not_called()
                self.assertEqual(optimizer.state_dict(), optimizer_before)
                for expected, observed in zip(
                    parameters_before,
                    model.parameters(),
                    strict=True,
                ):
                    self.assertTrue(torch.equal(expected, observed))

    def test_post_clip_invalid_gradient_prevents_step_and_clears_failure_state(self) -> None:
        model = _model()
        optimizer = create_phase8_optimizer(model)
        parameters_before = tuple(
            parameter.detach().clone() for parameter in model.parameters()
        )
        optimizer_before = copy.deepcopy(optimizer.state_dict())
        real_clip = torch.nn.utils.clip_grad_norm_

        def poison_gradients(parameters, *args, **kwargs):
            parameters = tuple(parameters)
            result = real_clip(parameters, *args, **kwargs)
            parameters[0].grad.fill_(float("nan"))
            return result

        with patch.object(
            optimization.torch.nn.utils,
            "clip_grad_norm_",
            side_effect=poison_gradients,
        ):
            with self.assertRaises(Phase8NumericalError):
                run_phase8_logical_batch(
                    model,
                    optimizer,
                    _windows(),
                    Phase8LogicalBatch(1, 0, (0, 1), 4),
                )
        self.assertEqual(optimizer.state_dict(), optimizer_before)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        for expected, observed in zip(
            parameters_before,
            model.parameters(),
            strict=True,
        ):
            self.assertTrue(torch.equal(expected, observed))

    def test_evaluation_is_target_weighted_and_nonmutating(self) -> None:
        model = _model()
        optimizer = create_phase8_optimizer(model)
        windows = _windows()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(8001)
        progress = Phase8Progress(0, 1, 0, 0, 0, 0)
        parameters_before = tuple(parameter.detach().clone() for parameter in model.parameters())
        optimizer_before = copy.deepcopy(optimizer.state_dict())
        order_before = generator.get_state().clone()
        dropout_before = tuple(
            state
            for block in model.blocks
            for state in (
                block.attention_dropout._generator.get_state().clone(),
                block.feed_forward_dropout._generator.get_state().clone(),
            )
        )
        result = evaluate_phase8_split(
            model,
            optimizer,
            windows,
            generator,
            progress,
            split="train",
        )
        self.assertEqual((result.target_count, result.window_count), (4, 2))
        self.assertTrue(model.training)
        self.assertEqual(optimizer.state_dict(), optimizer_before)
        self.assertTrue(torch.equal(generator.get_state(), order_before))
        for before, after in zip(parameters_before, model.parameters(), strict=True):
            self.assertTrue(torch.equal(before, after))
        dropout_after = tuple(
            state
            for block in model.blocks
            for state in (
                block.attention_dropout._generator.get_state(),
                block.feed_forward_dropout._generator.get_state(),
            )
        )
        self.assertTrue(
            all(
                torch.equal(before, after)
                for before, after in zip(dropout_before, dropout_after, strict=True)
            )
        )

    def test_existing_gradient_is_rejected_before_update(self) -> None:
        model = _model()
        optimizer = create_phase8_optimizer(model)
        first = next(model.parameters())
        first.grad = torch.zeros_like(first)
        before = tuple(parameter.detach().clone() for parameter in model.parameters())
        with self.assertRaises(Phase8ContractError):
            run_phase8_logical_batch(
                model,
                optimizer,
                _windows(),
                Phase8LogicalBatch(1, 0, (0, 1), 4),
            )
        for expected, observed in zip(before, model.parameters(), strict=True):
            self.assertTrue(torch.equal(expected, observed))

    def test_evaluation_failure_restores_training_mode(self) -> None:
        model = _model()
        optimizer = create_phase8_optimizer(model)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(8001)
        with patch.object(MiniGPT, "forward", side_effect=RuntimeError("forced")):
            with self.assertRaises(RuntimeError):
                evaluate_phase8_split(
                    model,
                    optimizer,
                    _windows(),
                    generator,
                    Phase8Progress(0, 1, 0, 0, 0, 0),
                    split="train",
                )
        self.assertTrue(all(module.training for _, module in model.named_modules()))

    def test_evaluation_failure_matrix_preserves_each_defined_boundary(self) -> None:
        cases = ("entry", "train_forward", "validation_forward", "aggregation", "packaging")
        for case in cases:
            with self.subTest(case=case):
                model = _model()
                optimizer = create_phase8_optimizer(model)
                generator = torch.Generator(device="cpu")
                generator.manual_seed(8001)
                progress = Phase8Progress(0, 1, 0, 0, 0, 0)
                windows = _validation_windows() if case == "validation_forward" else _windows()
                split = "validation" if case == "validation_forward" else "train"
                snapshot = _evaluation_snapshot(model, optimizer, generator, progress)
                patches = []
                if case in ("train_forward", "validation_forward"):
                    patches.append(
                        patch.object(model, "forward", side_effect=RuntimeError(case))
                    )
                elif case == "aggregation":
                    patches.append(
                        patch.object(optimization.math, "fsum", side_effect=RuntimeError(case))
                    )
                elif case == "packaging":
                    patches.append(
                        patch.object(
                            optimization,
                            "Phase8Evaluation",
                            side_effect=RuntimeError(case),
                        )
                    )
                for active_patch in patches:
                    active_patch.start()
                try:
                    with self.assertRaises(Exception):
                        evaluate_phase8_split(
                            model,
                            optimizer,
                            windows if case != "entry" else [],  # type: ignore[arg-type]
                            generator,
                            progress,
                            split=split,
                        )
                finally:
                    for active_patch in reversed(patches):
                        active_patch.stop()
                _assert_evaluation_snapshot(
                    self,
                    snapshot,
                    model,
                    optimizer,
                    generator,
                    progress,
                )

    def test_evaluation_mode_restoration_failure_is_attempted_without_new_rollback_claim(self) -> None:
        model = _model()
        optimizer = create_phase8_optimizer(model)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(8001)
        progress = Phase8Progress(0, 1, 0, 0, 0, 0)
        snapshot = _evaluation_snapshot(model, optimizer, generator, progress)
        real_train = model.train
        calls: list[bool] = []

        def fail_restoration(mode=True):
            calls.append(mode)
            if mode is True:
                raise RuntimeError("restore")
            return real_train(mode)

        with patch.object(model, "train", side_effect=fail_restoration):
            with self.assertRaises(RuntimeError):
                evaluate_phase8_split(
                    model,
                    optimizer,
                    _windows(),
                    generator,
                    progress,
                    split="train",
                )
        self.assertEqual(calls, [False, True])
        self.assertTrue(all(not module.training for _, module in model.named_modules()))
        _assert_evaluation_snapshot(
            self,
            snapshot,
            model,
            optimizer,
            generator,
            progress,
            expect_modes_restored=False,
        )


if __name__ == "__main__":
    unittest.main()
