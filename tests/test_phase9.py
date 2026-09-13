"""Independent synthetic contract tests for Phase 9."""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import math
import platform
import copy
import subprocess
import sys
import tempfile
import unittest
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import sebgpt.inference as inference  # noqa: E402
import sebgpt.inference.phase9_checkpoint as checkpoint  # noqa: E402
import sebgpt.inference.phase9_experiment as experiment  # noqa: E402
from sebgpt.inference.phase9_types import (  # noqa: E402
    Phase9CheckpointIdentity,
    Phase9ContractError,
    Phase9EvidenceReference,
    Phase9GovernanceError,
    Phase9PublicationError,
    Phase9TokenDocument,
    Phase9TypeError,
    _make_inference_bundle,
)
from sebgpt.model.mini_gpt import MiniGPT  # noqa: E402
from sebgpt.tokenization.code_point import CodePointTokenizer  # noqa: E402
from sebgpt.tokenization.vocabulary_artifact import load_accepted_vocabulary_binding  # noqa: E402


EXPECTED_EXPORTS = (
    "Phase9TypeError", "Phase9ContractError", "Phase9GovernanceError",
    "Phase9CheckpointError", "Phase9NumericalError", "Phase9PublicationError",
    "Phase9CheckpointIdentity", "Phase9InferenceBundle", "Phase9Generation",
    "Phase9Window", "Phase9TokenDocument", "Phase9Metrics",
    "Phase9LaplaceUnigramBaseline", "Phase9EvidenceReference",
    "load_phase9_inference_bundle", "select_greedy_token_id",
    "build_categorical_probabilities", "generate_phase9_text",
    "build_phase9_windows", "fit_phase9_laplace_unigram",
    "evaluate_phase9_model", "evaluate_phase9_uniform",
    "evaluate_phase9_laplace_unigram", "run_fixed_phase9_development_evaluation",
    "run_fixed_phase9_sealed_test_evaluation",
)


def _model() -> tuple[MiniGPT, object]:
    vocabulary = load_accepted_vocabulary_binding(REPOSITORY_ROOT)
    model = MiniGPT(
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
    ).eval()
    return model, vocabulary


def _bundle():
    model, vocabulary = _model()
    identity = Phase9CheckpointIdentity(
        "EXP-20260912-01", "best_validation", "epoch-0010", 10,
        "0x1.3f94b678f5807p+1", checkpoint.CHECKPOINT_SHA256,
        f"objects/{checkpoint.CHECKPOINT_SHA256}.pt", checkpoint.CATALOG_SHA256,
    )
    return _make_inference_bundle(
        model=model,
        tokenizer=CodePointTokenizer(vocabulary.code_points),
        vocabulary=vocabulary,
        checkpoint=identity,
    )


def _windows(split: str = "validation"):
    documents = (
        Phase9TokenDocument("a", 0, split, (0, 1, 2, 3, 4)),
        Phase9TokenDocument("b", 1, split, (4, 3, 2)),
    )
    return inference.build_phase9_windows(
        documents, vocabulary_size=5, context_length=3, stride=3
    )


def _payload(model: MiniGPT, loss_hex: str = "0x1.0000000000000p+0") -> dict[str, object]:
    lock_sha = hashlib.sha256((REPOSITORY_ROOT / "requirements.lock").read_bytes()).hexdigest()
    runtime = {
        "schema_version": 1,
        "python_version": "3.14.4",
        "python_implementation": "CPython",
        "torch_version": "2.14.0",
        "operating_system": "Darwin",
        "operating_system_release": platform.release(),
        "machine": "arm64",
        "processor": platform.processor(),
        "device": "cpu",
        "parameter_dtype": "torch.float32",
        "token_dtype": "torch.long",
        "torch_intra_op_threads": 1,
        "torch_inter_op_threads": 1,
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "float32_matmul_precision": "highest",
        "default_dtype": "torch.float32",
        "default_device": "cpu",
        "grad_mode_enabled": True,
        "cpu_autocast_enabled": False,
        "inference_mode_enabled": False,
        "mkldnn_enabled": False,
        "torch_build_config_sha256": hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
        "torch_parallel_info_sha256": hashlib.sha256(torch.__config__.parallel_info().encode()).hexdigest(),
        "requirements_lock_sha256": lock_sha,
    }
    works = tuple(
        {
            "manifest_order": order,
            "work_id": work_id,
            "split": split,
            "processed_sha256": f"{order}" * 64,
        }
        for order, work_id, split in (
            (1, "hamlet", "train"),
            (2, "romeo-and-juliet", "train"),
            (3, "macbeth", "train"),
            (4, "a-midsummer-nights-dream", "train"),
            (5, "much-ado-about-nothing", "train"),
            (6, "henry-v", "train"),
            (7, "the-tempest", "validation"),
        )
    )
    configuration = {
        "schema_version": 1,
        "runtime": runtime,
        "context_length": 256,
        "stride": 256,
        "retain_unpadded_tail": True,
        "logical_batch_capacity": 8,
        "allow_final_partial_batch": True,
        "maximum_epochs": 10,
        "order_seed": 8001,
        "order_algorithm": "torch_randperm_local_cpu_generator",
        "optimizer_name": "AdamW",
        "learning_rate": 3e-4,
        "betas": (0.9, 0.999),
        "epsilon": 1e-8,
        "weight_decay": 0.01,
        "amsgrad": False,
        "maximize": False,
        "foreach": False,
        "capturable": False,
        "differentiable": False,
        "fused": False,
        "scheduler_name": None,
        "warmup_steps": 0,
        "gradient_clip_norm_type": 2.0,
        "gradient_clip_max_norm": 1.0,
        "gradient_clip_epsilon": 1e-6,
        "evaluate_initialized_state": True,
        "evaluation_interval_epochs": 1,
        "evaluation_split_order": ("train", "validation"),
        "best_comparison": "strict_lower",
        "validation_early_stopping": False,
        "parameter_device": "cpu",
        "parameter_dtype": "torch.float32",
        "token_dtype": "torch.long",
        "checkpoint_schema_version": 1,
        "catalog_schema_version": 1,
        "maximum_catalog_bytes": 16_384,
        "maximum_checkpoint_object_bytes": 67_108_864,
    }
    training_metric = {"split": "train", "loss": 1.25, "loss_hex": (1.25).hex(), "target_count": 16, "window_count": 8}
    validation_loss = float.fromhex(loss_hex)
    validation_metric = {"split": "validation", "loss": validation_loss, "loss_hex": loss_hex, "target_count": 8, "window_count": 4}
    model_state = OrderedDict((name, value.detach().clone()) for name, value in model.state_dict().items())
    optimizer_state = {
        position: {
            "step": torch.tensor(1.0),
            "exp_avg": torch.zeros_like(value),
            "exp_avg_sq": torch.zeros_like(value),
        }
        for position, value in enumerate(model_state.values())
    }
    order_generator = torch.Generator(device="cpu").manual_seed(8001)
    dropout_generators = {}
    for position, name in enumerate(checkpoint._DROPOUT_NAMES):
        seed = 6007 if position % 2 == 0 else 6008
        dropout_generators[name] = torch.Generator(device="cpu").manual_seed(seed).get_state()
    return {
        "schema_version": 1,
        "run": {"run_id": "EXP-20260912-01", "logical_id": "epoch-0001", "completed_epoch": 1, "code_commit": checkpoint.PHASE8_IMPLEMENTATION_COMMIT},
        "authority": {
            "phase7_closure_commit": checkpoint.PHASE7_CLOSURE_COMMIT,
            "phase7_contract_commit": checkpoint.PHASE7_CONTRACT_COMMIT,
            "phase7_implementation_commit": checkpoint.PHASE7_IMPLEMENTATION_COMMIT,
            "mini_gpt_spec_sha256": checkpoint.MINI_GPT_SPEC_SHA256,
            "mini_gpt_source_sha256": checkpoint.MINI_GPT_SOURCE_SHA256,
            "mini_gpt_test_sha256": checkpoint.MINI_GPT_TEST_SHA256,
            "phase8_contract_commit": checkpoint.PHASE8_CONTRACT_COMMIT,
            "phase8_spec_sha256": checkpoint.PHASE8_SPEC_SHA256,
            "requirements_lock_sha256": lock_sha,
            "tokenizer": {
                "implementation_commit": checkpoint.TOKENIZER_IMPLEMENTATION_COMMIT,
                "tokenizer_id": "shakespeare-code-point-v1",
                "schema_version": 1,
                "vocabulary_size": 81,
                "artifact_path": "artifacts/tokenizers/shakespeare-code-point-v1/vocabulary.json",
                "artifact_sha256": checkpoint.VOCABULARY_SHA256,
            },
            "dataset": {
                "dataset_id": "shakespeare-eight-play",
                "manifest_sha256": checkpoint.PHASE1_MANIFEST_SHA256,
                "processing_manifest_sha256": checkpoint.PROCESSING_MANIFEST_SHA256,
                "training_works": works[:6],
                "validation_works": works[6:],
            },
            "model": {
                "model_width": 32, "max_sequence_length": 256,
                "number_of_blocks": 4, "number_of_heads": 4, "head_width": 8,
                "hidden_width": 128, "layer_norm_epsilon": 1e-5,
                "dropout_probability": 0.1, "embedding_seed": 1337,
                "block_parameter_seeds": (7001, 7002, 7003, 7004),
                "output_head_seed": 7005, "parameter_device": "cpu",
                "parameter_dtype": "torch.float32", "trainable_tensor_count": 90,
                "parameter_count": 63_825,
            },
            "runtime": runtime,
            "sealed_test_access": "none",
        },
        "configuration": configuration,
        "model_state": model_state,
        "optimizer_state": {
            "state": optimizer_state,
            "param_groups": [{
                "lr": 3e-4, "betas": (0.9, 0.999), "eps": 1e-8,
                "weight_decay": 0.01, "amsgrad": False, "maximize": False,
                "foreach": False, "capturable": False, "differentiable": False,
                "fused": False, "decoupled_weight_decay": True,
                "params": list(range(90)),
            }],
        },
        "progress": {
            "completed_epochs": 1, "next_epoch": 2, "next_example_offset": 0,
            "optimizer_updates": 1, "examples_processed": 8, "targets_processed": 16,
            "per_epoch_optimizer_updates": (1,), "per_epoch_example_counts": (8,),
            "per_epoch_target_counts": (16,),
            "next_ordering_action": "draw_next_epoch_permutation",
        },
        "random_state": {
            "order_generator": order_generator.get_state(),
            "global_cpu": torch.get_rng_state().clone(),
            "dropout_generators": dropout_generators,
        },
        "mode_state": {"training": True, "named_modules": tuple((name, True) for name in checkpoint._MODE_NAMES)},
        "metric_state": {
            "initialized_training": training_metric,
            "initialized_validation": validation_metric,
            "epoch_training": (training_metric,),
            "epoch_validation": (validation_metric,),
            "current_training": training_metric,
            "current_validation": validation_metric,
            "best_validation_loss": validation_loss,
            "best_validation_loss_hex": loss_hex,
            "best_validation_epoch": 1,
            "best_logical_id": "epoch-0001",
            "best_comparison": "strict_lower",
        },
        "resume_lineage": {"root_run_id": "EXP-20260912-01", "resumed_from_checkpoint_sha256": None, "resume_count": 0, "checkpoint_sha256s": ()},
    }


def _serialized_payload(payload: dict[str, object]) -> bytes:
    stream = io.BytesIO()
    torch.save(payload, stream)
    return stream.getvalue()


def _epoch10_payload(model: MiniGPT) -> dict[str, object]:
    payload = _payload(model, checkpoint.VALIDATION_LOSS_HEX)
    payload["run"].update({"logical_id": "epoch-0010", "completed_epoch": 10})  # type: ignore[union-attr]
    progress = payload["progress"]
    progress.update({  # type: ignore[union-attr]
        "completed_epochs": 10, "next_epoch": 11, "optimizer_updates": 10,
        "examples_processed": 80, "targets_processed": 160,
        "per_epoch_optimizer_updates": (1,) * 10,
        "per_epoch_example_counts": (8,) * 10,
        "per_epoch_target_counts": (16,) * 10,
    })
    for state in payload["optimizer_state"]["state"].values():  # type: ignore[index,union-attr]
        state["step"] = torch.tensor(10.0)
    metrics = payload["metric_state"]
    training = metrics["current_training"]  # type: ignore[index]
    validation_values = tuple(
        {
            "split": "validation", "loss": loss,
            "loss_hex": loss.hex(), "target_count": 8, "window_count": 4,
        }
        for loss in (*[3.0 - index / 20 for index in range(9)], float.fromhex(checkpoint.VALIDATION_LOSS_HEX))
    )
    metrics.update({  # type: ignore[union-attr]
        "epoch_training": (training,) * 10,
        "epoch_validation": validation_values,
        "current_validation": validation_values[-1],
        "best_validation_loss": validation_values[-1]["loss"],
        "best_validation_loss_hex": checkpoint.VALIDATION_LOSS_HEX,
        "best_validation_epoch": 10,
        "best_logical_id": "epoch-0010",
    })
    return payload


class SurfaceTests(unittest.TestCase):
    def test_package_exports_and_signatures(self) -> None:
        self.assertEqual(tuple(inference.__all__), EXPECTED_EXPORTS)
        signature = inspect.signature(inference.run_fixed_phase9_development_evaluation)
        self.assertEqual(tuple(signature.parameters), ("repository_root", "evaluation_id"))
        self.assertEqual(signature.parameters["evaluation_id"].kind, inspect.Parameter.KEYWORD_ONLY)

    def test_all_module_exports_and_public_parameter_shapes(self) -> None:
        import sebgpt.inference.phase9_evaluation as evaluation
        import sebgpt.inference.phase9_generation as generation
        import sebgpt.inference.phase9_types as types

        self.assertEqual(tuple(checkpoint.__all__), ("load_phase9_inference_bundle",))
        self.assertEqual(tuple(generation.__all__), EXPECTED_EXPORTS[15:18])
        self.assertEqual(tuple(evaluation.__all__), EXPECTED_EXPORTS[18:23])
        self.assertEqual(tuple(experiment.__all__), EXPECTED_EXPORTS[23:])
        self.assertEqual(tuple(types.__all__), EXPECTED_EXPORTS[:14])
        expected = {
            "load_phase9_inference_bundle": ("repository_root",),
            "select_greedy_token_id": ("next_token_logits",),
            "build_categorical_probabilities": ("next_token_logits", "temperature", "top_k"),
            "generate_phase9_text": ("bundle", "prompt", "generated_token_count", "mode", "temperature", "top_k", "seed"),
            "build_phase9_windows": ("documents", "vocabulary_size", "context_length", "stride"),
            "fit_phase9_laplace_unigram": ("training_windows", "vocabulary_size"),
            "evaluate_phase9_model": ("model", "windows", "predictor", "split"),
            "evaluate_phase9_uniform": ("windows", "vocabulary_size", "split"),
            "evaluate_phase9_laplace_unigram": ("baseline", "windows", "split"),
            "run_fixed_phase9_development_evaluation": ("repository_root", "evaluation_id"),
            "run_fixed_phase9_sealed_test_evaluation": ("repository_root", "evaluation_id"),
        }
        for name, parameters in expected.items():
            signature = inspect.signature(getattr(inference, name))
            self.assertEqual(tuple(signature.parameters), parameters)
            self.assertTrue(all(value.default is inspect.Parameter.empty for value in signature.parameters.values()))

    def test_record_fields_and_factory_boundaries(self) -> None:
        self.assertEqual(
            tuple(value.name for value in fields(inference.Phase9EvidenceReference)),
            ("evaluation_id", "kind", "status", "relative_path", "byte_count", "sha256"),
        )
        with self.assertRaises(Phase9ContractError):
            inference.Phase9InferenceBundle()  # type: ignore[call-arg]
        with self.assertRaises(Phase9ContractError):
            inference.Phase9LaplaceUnigramBaseline()  # type: ignore[call-arg]
        record = Phase9TokenDocument("a", 0, "train", (1, 2))
        with self.assertRaises(FrozenInstanceError):
            record.document_id = "b"  # type: ignore[misc]
        self.assertNotIn("token_ids", repr(record))

    def test_content_safe_errors(self) -> None:
        error = Phase9TypeError("phase9.type.argument", field="prompt")
        self.assertEqual(error.details["field"], "prompt")
        self.assertNotIn("secret", str(error))

    def test_all_evidence_reference_rows_and_cross_pairs(self) -> None:
        for (kind, status), (filename, _) in experiment._EVIDENCE_ROWS.items():
            reference = Phase9EvidenceReference(
                "EXP-20260913-01", kind, status,
                f"experiments/EXP-20260913-01/artifacts/{filename}", 1, "a" * 64,
            )
            self.assertIs(experiment._validate_reference(reference), reference)
            wrong = Phase9EvidenceReference(reference.evaluation_id, kind, "unknown", reference.relative_path, 1, "a" * 64)
            with self.assertRaises(Phase9ContractError):
                experiment._validate_reference(wrong)

    def test_evidence_reference_rejects_wrong_types_bounds_and_hashes(self) -> None:
        valid = Phase9EvidenceReference(
            "EXP-20260913-01", "phase9_development", "completed",
            "experiments/EXP-20260913-01/artifacts/phase9-development-evidence.json",
            1, "a" * 64,
        )
        for replacement in (
            {"evaluation_id": 1}, {"kind": 1}, {"status": 1},
            {"relative_path": 1}, {"byte_count": True}, {"byte_count": 0},
            {"sha256": "A" * 64}, {"relative_path": "artifacts/x"},
        ):
            values = {field.name: getattr(valid, field.name) for field in fields(valid)}
            values.update(replacement)
            with self.assertRaises((Phase9TypeError, Phase9ContractError)):
                experiment._validate_reference(Phase9EvidenceReference(**values))

    def test_static_scope_has_no_advanced_decoding_or_training_dependency(self) -> None:
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted((REPOSITORY_ROOT / "src/sebgpt/inference").glob("*.py"))
        ).lower()
        for forbidden in (
            "transformers", "huggingface", "openai", "hosted_model", "top_p",
            "beam_search", "repetition_penalty", "frequency_penalty", "kv_cache",
            "backward(", "torch.optim", "phase10",
        ):
            self.assertNotIn(forbidden, source)
        checkpoint_source = (
            REPOSITORY_ROOT / "src/sebgpt/inference/phase9_checkpoint.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("sebgpt.training", checkpoint_source)


class GenerationTests(unittest.TestCase):
    def test_greedy_tie_and_validation(self) -> None:
        logits = torch.zeros(81, dtype=torch.float32)
        logits[4] = logits[7] = 2.0
        self.assertEqual(inference.select_greedy_token_id(logits), 4)
        with self.assertRaises(Phase9ContractError):
            inference.select_greedy_token_id(torch.zeros(80))

    def test_top_k_ties_temperature_and_full_distribution(self) -> None:
        logits = torch.zeros(81, dtype=torch.float32)
        logits[3] = logits[5] = 2.0
        probabilities = inference.build_categorical_probabilities(logits, temperature=1.0, top_k=1)
        self.assertEqual(float(probabilities[3]), 1.0)
        self.assertEqual(int(torch.count_nonzero(probabilities)), 1)
        full = inference.build_categorical_probabilities(logits, temperature=0.8, top_k=81)
        self.assertTrue(torch.all(full > 0))
        self.assertAlmostEqual(float(full.sum()), 1.0, places=6)

    def test_temperature_top_k_type_and_range_errors(self) -> None:
        logits = torch.zeros(81)
        for temperature in (0.0, -1.0, float("inf")):
            with self.assertRaises(Phase9ContractError):
                inference.build_categorical_probabilities(logits, temperature=temperature, top_k=1)
        with self.assertRaises(Phase9TypeError):
            inference.build_categorical_probabilities(logits, temperature=1.0, top_k=True)  # type: ignore[arg-type]

    def test_probability_math_nonaliasing_and_input_nonmutation(self) -> None:
        logits = torch.arange(81, dtype=torch.float32)
        original = logits.clone()
        probabilities = inference.build_categorical_probabilities(logits, temperature=2.0, top_k=2)
        expected = torch.softmax(torch.tensor([79.0, 80.0]) / 2.0, dim=0)
        self.assertAlmostEqual(float(probabilities[79]), float(expected[0]), places=7)
        self.assertAlmostEqual(float(probabilities[80]), float(expected[1]), places=7)
        self.assertEqual(int(torch.count_nonzero(probabilities)), 2)
        self.assertNotEqual(probabilities.data_ptr(), logits.data_ptr())
        self.assertTrue(torch.equal(logits, original))
        before = torch.get_rng_state().clone()
        self.assertIs(type(inference.select_greedy_token_id(logits)), int)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_tiny_temperature_overflow_is_numerical_error(self) -> None:
        logits = torch.zeros(81, dtype=torch.float32)
        logits[0] = torch.finfo(torch.float32).max
        with self.assertRaises(inference.Phase9NumericalError):
            inference.build_categorical_probabilities(logits, temperature=float.fromhex("0x1p-126"), top_k=81)

    def test_zero_generation_uses_no_model_or_rng(self) -> None:
        bundle = _bundle()
        before = torch.get_rng_state().clone()
        with patch.object(MiniGPT, "forward", side_effect=AssertionError("called")):
            result = inference.generate_phase9_text(
                bundle, "A", generated_token_count=0, mode="categorical",
                temperature=1.0, top_k=81, seed=0,
            )
        self.assertEqual(result.generated_text, "")
        self.assertIsNone(result.local_generator_final_state_sha256)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_rolling_context_final_row_and_prompt_preservation(self) -> None:
        bundle = _bundle()
        observed: list[tuple[int, ...]] = []

        def fake_forward(_model, token_ids):
            observed.append(tuple(token_ids.tolist()))
            logits = torch.zeros((len(token_ids), 81), dtype=torch.float32)
            logits[-1, 1] = 1.0
            return logits

        with patch.object(MiniGPT, "forward", new=fake_forward):
            result = inference.generate_phase9_text(
                bundle, "A", generated_token_count=257, mode="greedy",
                temperature=None, top_k=None, seed=None,
            )
        self.assertEqual(len(observed), 257)
        self.assertEqual(len(observed[-1]), 256)
        self.assertEqual(result.full_text, "A" + result.generated_text)
        self.assertEqual(len(result.generated_token_ids), 257)

    def test_categorical_same_seed_and_global_rng_isolation(self) -> None:
        first_bundle = _bundle()
        second_bundle = _bundle()
        before = torch.get_rng_state().clone()

        def fake_forward(_model, token_ids):
            return torch.zeros((len(token_ids), 81), dtype=torch.float32)

        with patch.object(MiniGPT, "forward", new=fake_forward):
            first = inference.generate_phase9_text(first_bundle, "A", generated_token_count=10, mode="categorical", temperature=1.0, top_k=81, seed=9)
            second = inference.generate_phase9_text(second_bundle, "A", generated_token_count=10, mode="categorical", temperature=1.0, top_k=81, seed=9)
        self.assertEqual(first.generated_token_ids, second.generated_token_ids)
        self.assertEqual(first.local_generator_final_state_sha256, second.local_generator_final_state_sha256)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_prompt_and_count_boundaries(self) -> None:
        bundle = _bundle()
        for prompt in ("", "A" * 257):
            with self.assertRaises(Phase9ContractError):
                inference.generate_phase9_text(bundle, prompt, generated_token_count=0, mode="greedy", temperature=None, top_k=None, seed=None)
        for count in (-1, 1025):
            with self.assertRaises(Phase9ContractError):
                inference.generate_phase9_text(bundle, "A", generated_token_count=count, mode="greedy", temperature=None, top_k=None, seed=None)

    def test_prompt_count_and_setting_type_matrix(self) -> None:
        bundle = _bundle()
        invalid_calls = (
            (1, 0, "greedy", None, None, None, Phase9TypeError),
            ("A", True, "greedy", None, None, None, Phase9TypeError),
            ("A", 0, 1, None, None, None, Phase9TypeError),
            ("A", 0, "greedy", 1.0, None, None, Phase9ContractError),
            ("A", 0, "categorical", 1, 1, 0, Phase9TypeError),
            ("A", 0, "categorical", 1.0, 0, 0, Phase9ContractError),
            ("A", 0, "categorical", 1.0, 82, 0, Phase9ContractError),
            ("A", 0, "categorical", 1.0, 1, True, Phase9TypeError),
            ("A", 0, "categorical", 1.0, 1, -1, Phase9ContractError),
            ("A", 0, "categorical", 1.0, 1, 9_223_372_036_854_775_808, Phase9ContractError),
        )
        for prompt, count, mode, temperature, top_k, seed, error in invalid_calls:
            with self.assertRaises(error):
                inference.generate_phase9_text(bundle, prompt, generated_token_count=count, mode=mode, temperature=temperature, top_k=top_k, seed=seed)  # type: ignore[arg-type]

    def test_prompt_length_whitespace_and_unknown_character_boundaries(self) -> None:
        bundle = _bundle()
        for size in (1, 255, 256):
            prompt = "A" * size
            result = inference.generate_phase9_text(bundle, prompt, generated_token_count=0, mode="greedy", temperature=None, top_k=None, seed=None)
            self.assertEqual(result.full_text, prompt)
            self.assertEqual(len(result.prompt_token_ids), size)
        whitespace = " \n "
        result = inference.generate_phase9_text(bundle, whitespace, generated_token_count=0, mode="greedy", temperature=None, top_k=None, seed=None)
        self.assertEqual(result.full_text, whitespace)
        from sebgpt.tokenization.code_point import UnknownCodePointError
        with self.assertRaises(UnknownCodePointError):
            inference.generate_phase9_text(bundle, "☃", generated_token_count=0, mode="greedy", temperature=None, top_k=None, seed=None)

    def test_failure_restores_global_rng_and_model_state(self) -> None:
        bundle = _bundle()
        parameter = next(bundle.model.parameters())
        expected = parameter.detach().clone()
        before = torch.get_rng_state().clone()

        def failing_forward(_model, _tokens):
            torch.rand(1)
            with torch.no_grad():
                parameter.add_(1.0)
            raise RuntimeError("synthetic")

        with patch.object(MiniGPT, "forward", new=failing_forward):
            with self.assertRaises(RuntimeError):
                inference.generate_phase9_text(bundle, "A", generated_token_count=1, mode="greedy", temperature=None, top_k=None, seed=None)
        self.assertTrue(torch.equal(parameter, expected))
        self.assertTrue(torch.equal(torch.get_rng_state(), before))


class EvaluationTests(unittest.TestCase):
    def test_window_boundaries_tails_and_document_isolation(self) -> None:
        documents = (
            Phase9TokenDocument("a", 0, "train", tuple(range(6))),
            Phase9TokenDocument("b", 1, "train", (1, 2)),
        )
        windows = inference.build_phase9_windows(documents, vocabulary_size=6, context_length=3, stride=3)
        self.assertEqual(tuple((w.document_id, w.start_index, len(w.input_ids)) for w in windows), (("a", 0, 3), ("a", 3, 2), ("b", 0, 1)))
        self.assertFalse(any(w.input_ids[-1:] == (5,) and w.target_ids[-1:] == (1,) for w in windows))

    def test_window_argument_and_reserved_test_rejections(self) -> None:
        with self.assertRaises(Phase9ContractError):
            inference.build_phase9_windows((Phase9TokenDocument("a", 0, "test", (1, 2)),), vocabulary_size=3, context_length=2, stride=2)
        with self.assertRaises(Phase9ContractError):
            inference.build_phase9_windows((Phase9TokenDocument("a", 0, "train", (1, 2)),), vocabulary_size=3, context_length=2, stride=1)

    def test_exact_window_formula_length_matrix(self) -> None:
        for length, expected_lengths in (
            (2, (1,)), (256, (255,)), (257, (256,)),
            (258, (256, 1)), (512, (256, 255)), (513, (256, 256)),
        ):
            tokens = tuple(index % 7 for index in range(length))
            windows = inference.build_phase9_windows((Phase9TokenDocument("a", 0, "train", tokens),), vocabulary_size=7, context_length=256, stride=256)
            self.assertEqual(tuple(len(window.input_ids) for window in windows), expected_lengths)
            flattened = tuple(token for window in windows for token in window.target_ids)
            self.assertEqual(flattened, tokens[1:])

    def test_laplace_unigram_and_metrics(self) -> None:
        train = inference.build_phase9_windows((Phase9TokenDocument("a", 0, "train", (0, 1, 1, 2)),), vocabulary_size=3, context_length=2, stride=2)
        baseline = inference.fit_phase9_laplace_unigram(train, vocabulary_size=3)
        self.assertEqual(baseline.raw_counts, (0, 2, 1))
        self.assertEqual(baseline.smoothed_counts, (1, 3, 2))
        self.assertAlmostEqual(math.fsum(baseline.probabilities), 1.0)
        validation = inference.build_phase9_windows((Phase9TokenDocument("v", 0, "validation", (0, 1, 2)),), vocabulary_size=3, context_length=2, stride=2)
        metric = inference.evaluate_phase9_laplace_unigram(baseline, validation, split="validation")
        expected = (-math.log(3 / 6) - math.log(2 / 6)) / 2
        self.assertAlmostEqual(metric.nll, expected)
        self.assertAlmostEqual(metric.perplexity, math.exp(expected))

    def test_uniform_baseline_weighting_and_accuracy(self) -> None:
        windows = _windows()
        metric = inference.evaluate_phase9_uniform(windows, vocabulary_size=5, split="validation")
        self.assertAlmostEqual(metric.nll, math.log(5))
        self.assertAlmostEqual(metric.perplexity, 5.0)
        targets = [token for window in windows for token in window.target_ids]
        self.assertEqual(metric.correct_count, targets.count(0))

    def test_model_metric_is_finite_and_nonmutating(self) -> None:
        model, _ = _model()
        windows = inference.build_phase9_windows((Phase9TokenDocument("v", 0, "validation", (0, 1, 2)),), vocabulary_size=81, context_length=2, stride=2)
        before = tuple(value.detach().clone() for value in model.parameters())
        metric = inference.evaluate_phase9_model(model, windows, predictor="synthetic", split="validation")
        self.assertTrue(math.isfinite(metric.nll))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True)))
        self.assertTrue(all(value.grad is None for value in model.parameters()))

    def test_model_metric_target_weighting_ties_and_cross_entropy_calls(self) -> None:
        model, _ = _model()
        windows = (
            inference.Phase9Window("a", 0, "validation", 0, (0, 1), (1, 2)),
            inference.Phase9Window("a", 0, "validation", 2, (2,), (0,)),
        )
        losses = iter((torch.tensor(1.0), torch.tensor(4.0)))
        with patch.object(MiniGPT, "forward", side_effect=(torch.zeros((2, 81)), torch.zeros((1, 81)))), patch("sebgpt.inference.phase9_evaluation.explicit_cross_entropy", side_effect=lambda *_: next(losses)) as cross_entropy:
            metric = inference.evaluate_phase9_model(model, windows, predictor="synthetic", split="validation")
        self.assertEqual(cross_entropy.call_count, 2)
        self.assertAlmostEqual(metric.nll, 2.0)
        self.assertAlmostEqual(metric.perplexity, math.exp(2.0))
        self.assertEqual(metric.correct_count, 1)
        self.assertAlmostEqual(metric.top1_accuracy, 1 / 3)

    def test_all_public_evaluators_reject_reserved_test(self) -> None:
        test_windows = (inference.Phase9Window("a", 0, "test", 0, (0,), (1,)),)
        model, _ = _model()
        training = inference.build_phase9_windows((Phase9TokenDocument("t", 0, "train", (0, 1)),), vocabulary_size=2, context_length=1, stride=1)
        baseline = inference.fit_phase9_laplace_unigram(training, vocabulary_size=2)
        for call in (
            lambda: inference.evaluate_phase9_uniform(test_windows, vocabulary_size=2, split="test"),
            lambda: inference.evaluate_phase9_laplace_unigram(baseline, test_windows, split="test"),
            lambda: inference.evaluate_phase9_model(model, test_windows, predictor="synthetic", split="test"),
        ):
            with self.assertRaises(Phase9ContractError):
                call()

    def test_laplace_tie_break_training_only_and_immutability(self) -> None:
        training = inference.build_phase9_windows((Phase9TokenDocument("t", 0, "train", (0, 1, 2)),), vocabulary_size=3, context_length=2, stride=2)
        baseline = inference.fit_phase9_laplace_unigram(training, vocabulary_size=3)
        self.assertEqual(baseline.top_token_id, 1)
        with self.assertRaises(FrozenInstanceError):
            baseline.top_token_id = 2  # type: ignore[misc]
        validation = inference.build_phase9_windows((Phase9TokenDocument("v", 0, "validation", (0, 1)),), vocabulary_size=3, context_length=1, stride=1)
        with self.assertRaises(Phase9ContractError):
            inference.fit_phase9_laplace_unigram(validation, vocabulary_size=3)

    def test_cross_predictor_comparison_rejects_identity_mismatch(self) -> None:
        base = inference.Phase9Metrics("a", "validation", "a" * 64, 1.0, math.e, 1, 2, 0.5, 1)
        mismatch = inference.Phase9Metrics("b", "validation", "b" * 64, 1.0, math.e, 1, 2, 0.5, 1)
        with self.assertRaises(Phase9ContractError):
            experiment._compare_metrics(base, base, mismatch)


class CheckpointTests(unittest.TestCase):
    def test_protected_import_closure_is_exact(self) -> None:
        self.assertEqual(len(checkpoint.PROTECTED_PATHS), 19)
        self.assertIn("src/sebgpt/__init__.py", checkpoint.PROTECTED_PATHS)
        self.assertIn("src/sebgpt/data/__init__.py", checkpoint.PROTECTED_PATHS)
        self.assertNotIn("tests/test_mini_gpt.py", checkpoint.PROTECTED_PATHS)

    def test_accepted_phase8_result_record_is_exact_and_append_stable(self) -> None:
        content = (REPOSITORY_ROOT / "EXPERIMENT_LOG.md").read_bytes()
        checkpoint._validate_phase8_result_record(content)
        checkpoint._validate_phase8_result_record(content + "\n### EXP-20260913-01 — Phase 9 development plan\n".encode())
        heading = "### EXP-20260912-01 — Completed Phase 8 fixed training\n".encode()
        start = content.index(heading)
        mutated = content[:start] + content[start:].replace(b'`PASS`', b'`FAIL`', 1)
        with self.assertRaises(Phase9GovernanceError):
            checkpoint._validate_phase8_result_record(mutated)

    def test_synthetic_catalog_selects_best_and_validates_hash(self) -> None:
        model, _ = _model()
        payload_bytes = _serialized_payload(_payload(model))
        digest = hashlib.sha256(payload_bytes).hexdigest()
        base = {"run_id": "EXP-20260912-01", "logical_id": "epoch-0001", "epoch": 1, "validation_loss_hex": "0x1.0000000000000p+0", "sha256": digest, "relative_path": f"objects/{digest}.pt"}
        catalog = {"schema_version": 1, "run_id": "EXP-20260912-01", "latest": {**base, "role": "latest"}, "best_validation": {**base, "role": "best_validation"}}
        content = checkpoint._canonical_catalog(catalog)
        reference, payload = checkpoint._validate_graph(content, lambda _: payload_bytes, expected_catalog_sha256=hashlib.sha256(content).hexdigest(), expected_checkpoint_sha256=digest)
        self.assertEqual(reference["role"], "best_validation")
        self.assertEqual(payload["run"]["logical_id"], "epoch-0001")  # type: ignore[index]
        with self.assertRaises(inference.Phase9CheckpointError):
            checkpoint._validate_graph(content + b" ", lambda _: payload_bytes, expected_catalog_sha256=None, expected_checkpoint_sha256=digest)

    def test_synthetic_distinct_latest_and_best_graph_selects_best(self) -> None:
        model, _ = _model()
        best_payload = _payload(model, (1.0).hex())
        latest_payload = copy.deepcopy(best_payload)
        latest_payload["run"].update({"logical_id": "epoch-0002", "completed_epoch": 2})  # type: ignore[union-attr]
        latest_payload["progress"].update({  # type: ignore[union-attr]
            "completed_epochs": 2, "next_epoch": 3, "optimizer_updates": 2,
            "examples_processed": 16, "targets_processed": 32,
            "per_epoch_optimizer_updates": (1, 1),
            "per_epoch_example_counts": (8, 8),
            "per_epoch_target_counts": (16, 16),
        })
        for state in latest_payload["optimizer_state"]["state"].values():  # type: ignore[index,union-attr]
            state["step"] = torch.tensor(2.0)
        training = latest_payload["metric_state"]["current_training"]  # type: ignore[index]
        second_validation = {"split": "validation", "loss": 2.0, "loss_hex": (2.0).hex(), "target_count": 8, "window_count": 4}
        latest_payload["metric_state"].update({  # type: ignore[union-attr]
            "epoch_training": (training, training),
            "epoch_validation": (latest_payload["metric_state"]["current_validation"], second_validation),  # type: ignore[index]
            "current_validation": second_validation,
        })
        best_bytes = _serialized_payload(best_payload)
        latest_bytes = _serialized_payload(latest_payload)
        best_digest = hashlib.sha256(best_bytes).hexdigest()
        latest_digest = hashlib.sha256(latest_bytes).hexdigest()
        def reference(role, epoch, loss, digest):
            return {"run_id": checkpoint.RUN_ID, "role": role, "logical_id": f"epoch-{epoch:04d}", "epoch": epoch, "validation_loss_hex": loss.hex(), "sha256": digest, "relative_path": f"objects/{digest}.pt"}
        catalog = {
            "schema_version": 1, "run_id": checkpoint.RUN_ID,
            "latest": reference("latest", 2, 2.0, latest_digest),
            "best_validation": reference("best_validation", 1, 1.0, best_digest),
        }
        content = checkpoint._canonical_catalog(catalog)
        objects = {latest_digest: latest_bytes, best_digest: best_bytes}
        selected, payload = checkpoint._validate_graph(content, lambda value: objects[value["sha256"]], expected_catalog_sha256=hashlib.sha256(content).hexdigest(), expected_checkpoint_sha256=best_digest)
        self.assertEqual(selected["role"], "best_validation")
        self.assertEqual(payload["run"]["logical_id"], "epoch-0001")  # type: ignore[index]

    def test_descriptor_relative_reader_rejects_symlink_and_size_edges(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "value").write_bytes(b"abc")
            (root / "empty").write_bytes(b"")
            (root / "link").symlink_to(root / "value")
            fd = experiment.os.open(root, experiment.os.O_RDONLY | experiment.os.O_DIRECTORY)
            try:
                self.assertEqual(checkpoint._read_fd(fd, "value", 3), b"abc")
                for name, maximum in (("value", 2), ("empty", 3), ("link", 3)):
                    with self.assertRaises(inference.Phase9CheckpointError):
                        checkpoint._read_fd(fd, name, maximum)
            finally:
                experiment.os.close(fd)

    def test_public_loader_reconstructs_synthetic_state_without_training_restore(self) -> None:
        source_model, _ = _model()
        payload = _epoch10_payload(source_model)
        reference = {
            "run_id": checkpoint.RUN_ID, "role": "best_validation",
            "logical_id": "epoch-0010", "epoch": 10,
            "validation_loss_hex": checkpoint.VALIDATION_LOSS_HEX,
            "sha256": checkpoint.CHECKPOINT_SHA256,
            "relative_path": f"objects/{checkpoint.CHECKPOINT_SHA256}.pt",
        }
        before = torch.get_rng_state().clone()
        with patch.object(checkpoint, "_validate_repository", return_value=REPOSITORY_ROOT), patch.object(checkpoint, "_load_validated_artifacts", return_value=(reference, payload)), patch.object(checkpoint, "_construct_model", wraps=checkpoint._construct_model) as constructor:
            bundle = inference.load_phase9_inference_bundle(REPOSITORY_ROOT)
        self.assertEqual(constructor.call_count, 1)
        self.assertFalse(bundle.model.training)
        self.assertTrue(all(not module.training for _, module in bundle.model.named_modules()))
        self.assertTrue(all(parameter.grad is None for parameter in bundle.model.parameters()))
        self.assertTrue(all(torch.equal(left, right) for left, right in zip(bundle.model.state_dict().values(), payload["model_state"].values(), strict=True)))  # type: ignore[union-attr]
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_loader_failure_restores_global_rng(self) -> None:
        before = torch.get_rng_state().clone()
        def fail(_root):
            torch.rand(1)
            raise inference.Phase9CheckpointError("phase9.checkpoint.schema", field="payload")
        with patch.object(checkpoint, "_validate_repository", return_value=REPOSITORY_ROOT), patch.object(checkpoint, "_load_validated_artifacts", side_effect=fail):
            with self.assertRaises(inference.Phase9CheckpointError):
                inference.load_phase9_inference_bundle(REPOSITORY_ROOT)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_payload_mutations_are_rejected(self) -> None:
        model, _ = _model()
        payload = _payload(model)
        payload["run"]["code_commit"] = "bad"  # type: ignore[index]
        with self.assertRaises(inference.Phase9CheckpointError):
            checkpoint._validate_payload(payload)

    def test_nested_payload_mutation_matrix_is_rejected(self) -> None:
        model, _ = _model()
        mutations = (
            lambda value: value["authority"]["tokenizer"].__setitem__("vocabulary_size", 80),
            lambda value: value["configuration"].__setitem__("stride", 128),
            lambda value: value["model_state"].__setitem__(next(iter(value["model_state"])), torch.zeros(1)),
            lambda value: value["optimizer_state"]["param_groups"][0].__setitem__("params", list(reversed(range(90)))),
            lambda value: value["progress"].__setitem__("targets_processed", 15),
            lambda value: value["random_state"]["dropout_generators"].pop(checkpoint._DROPOUT_NAMES[-1]),
            lambda value: value["mode_state"].__setitem__("training", False),
            lambda value: value["metric_state"].__setitem__("best_logical_id", "epoch-0002"),
            lambda value: value["resume_lineage"].__setitem__("resume_count", 1),
        )
        for mutate in mutations:
            payload = copy.deepcopy(_payload(model))
            mutate(payload)
            with self.assertRaises(inference.Phase9CheckpointError):
                checkpoint._validate_payload(payload)

    def test_phase8_public_loader_source_is_unchanged(self) -> None:
        expected = (REPOSITORY_ROOT / "src/sebgpt/training/phase8_checkpoint.py").read_bytes()
        import subprocess
        observed = subprocess.run(("git", "show", f"{checkpoint.PHASE8_CLOSURE_COMMIT}:src/sebgpt/training/phase8_checkpoint.py"), cwd=REPOSITORY_ROOT, check=True, capture_output=True).stdout
        self.assertEqual(expected, observed)


def _render_record(evaluation_id: str, kind: str, values: dict[str, object]) -> bytes:
    suffix = next(key for key, value in experiment._KIND_BY_SUFFIX.items() if value == kind)
    lines = [f"### {evaluation_id} — Phase 9 {suffix}\n", "\n"]
    for label in experiment._LABELS[kind]:
        lines.append(f"- **{label}:** {json.dumps(values[label], ensure_ascii=True, sort_keys=True, separators=(',', ':'))}\n")
    lines.append("\n")
    return "".join(lines).encode("utf-8")


def _values(kind: str, evaluation_id: str) -> dict[str, object]:
    values = {label: "x" for label in experiment._LABELS[kind]}
    values.update({"Evaluation ID": evaluation_id, "Entry kind": kind})
    if kind == "development_plan":
        values.update({"Scope": "development", "Status": "planned", "Predecessor evaluation": None})
    elif kind == "development_authorization":
        values.update({"Scope": "development", "Status": "authorized_once"})
    elif kind == "development_result":
        values.update({"Scope": "development", "Status": "failed", "Predecessor evaluation": None})
    elif kind == "development_supersession":
        values.update({"Scope": "development", "Status": "superseded"})
    else:
        values.update({"Scope": "sealed_test", "Status": "planned"})
    return values


class ExperimentTests(unittest.TestCase):
    def test_record_parser_and_explicit_selection(self) -> None:
        evaluation_id = "EXP-20260913-01"
        plan_values = _values("development_plan", evaluation_id)
        plan_bytes = _render_record(evaluation_id, "development_plan", plan_values)
        plan_hash = hashlib.sha256(plan_bytes).hexdigest()
        authorization_values = _values("development_authorization", evaluation_id)
        authorization_values["Development plan record SHA-256"] = plan_hash
        content = plan_bytes + _render_record(evaluation_id, "development_authorization", authorization_values)
        records = experiment._parse_records(content)
        selected = experiment._development_authority(records, evaluation_id)
        self.assertEqual(selected[0].sha256, plan_hash)
        with self.assertRaises(Phase9GovernanceError):
            experiment._development_authority(records, "EXP-20260913-02")

    def test_duplicate_and_self_predecessor_are_rejected(self) -> None:
        evaluation_id = "EXP-20260913-01"
        plan_values = _values("development_plan", evaluation_id)
        plan_values["Predecessor evaluation"] = evaluation_id
        plan = _render_record(evaluation_id, "development_plan", plan_values)
        authorization_values = _values("development_authorization", evaluation_id)
        authorization_values["Development plan record SHA-256"] = hashlib.sha256(plan).hexdigest()
        records = experiment._parse_records(plan + _render_record(evaluation_id, "development_authorization", authorization_values))
        with self.assertRaises(Phase9GovernanceError):
            experiment._development_authority(records, evaluation_id)

    def test_valid_predecessor_chain_and_fork_rejection(self) -> None:
        prior = "EXP-20260913-01"
        current = "EXP-20260913-02"
        prior_plan_values = _values("development_plan", prior)
        prior_plan = _render_record(prior, "development_plan", prior_plan_values)
        result_values = _values("development_result", prior)
        result = _render_record(prior, "development_result", result_values)
        supersession_values = _values("development_supersession", prior)
        supersession_values["Prior result record SHA-256"] = hashlib.sha256(result).hexdigest()
        supersession_values["Successor evaluation ID"] = current
        supersession = _render_record(prior, "development_supersession", supersession_values)
        current_plan_values = _values("development_plan", current)
        current_plan_values["Predecessor evaluation"] = prior
        current_plan = _render_record(current, "development_plan", current_plan_values)
        authorization_values = _values("development_authorization", current)
        authorization_values["Development plan record SHA-256"] = hashlib.sha256(current_plan).hexdigest()
        authorization = _render_record(current, "development_authorization", authorization_values)
        records = experiment._parse_records(prior_plan + result + supersession + current_plan + authorization)
        self.assertEqual(experiment._development_authority(records, current)[0].evaluation_id, current)
        with self.assertRaises(Phase9GovernanceError):
            experiment._development_authority((*records, records[2]), current)

    def test_publication_known_present_absent_unknown_and_no_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            content = experiment._canonical_json({"ok": True})
            result = experiment._publish(root, "EXP-20260913-01", filename="phase9-development-attempt.json", kind="phase9_development_attempt", status="consumed", content=content)
            self.assertEqual(result.state, experiment.KNOWN_PRESENT)
            self.assertIsNotNone(result.reference)
            second = experiment._publish(root, "EXP-20260913-01", filename="phase9-development-attempt.json", kind="phase9_development_attempt", status="consumed", content=content)
            self.assertEqual(second.state, experiment.UNKNOWN)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch.object(experiment.os, "write", side_effect=OSError):
                result = experiment._publish(root, "EXP-20260913-02", filename="phase9-development-attempt.json", kind="phase9_development_attempt", status="consumed", content=content)
            self.assertEqual(result.state, experiment.KNOWN_ABSENT)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            real_fsync = experiment.os.fsync
            calls = 0
            def ambiguous_fsync(fd):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError
                return real_fsync(fd)
            with patch.object(experiment.os, "fsync", side_effect=ambiguous_fsync):
                result = experiment._publish(root, "EXP-20260913-03", filename="phase9-development-attempt.json", kind="phase9_development_attempt", status="consumed", content=content)
            self.assertEqual(result.state, experiment.UNKNOWN)
            self.assertFalse((root / "experiments/EXP-20260913-03/artifacts/phase9-development-attempt.json").exists())

    def test_terminal_opposites_are_mutually_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            content = experiment._canonical_json({"ok": True})
            success = experiment._publish_terminal(root, "EXP-20260913-01", filename="phase9-development-evidence.json", opposite_filename="phase9-development-failure.json", kind="phase9_development", status="completed", content=content)
            self.assertEqual(success.state, experiment.KNOWN_PRESENT)
            failure = experiment._publish_terminal(root, "EXP-20260913-01", filename="phase9-development-failure.json", opposite_filename="phase9-development-evidence.json", kind="phase9_development_failure", status="failed", content=content)
            self.assertEqual(failure.state, experiment.UNKNOWN)
            self.assertFalse((root / "experiments/EXP-20260913-01/artifacts/phase9-development-failure.json").exists())

    def test_runner_invalid_id_cannot_touch_sealed_content(self) -> None:
        with patch.object(experiment, "_validate_repository", return_value=REPOSITORY_ROOT), patch.object(Path, "read_bytes", side_effect=AssertionError("content touched")):
            with self.assertRaises(Phase9GovernanceError):
                inference.run_fixed_phase9_sealed_test_evaluation(REPOSITORY_ROOT, evaluation_id="bad")

    def test_qualitative_matrix_is_exact_and_rolls_context(self) -> None:
        self.assertEqual(tuple(row[0] for row in experiment.QUALITATIVE_MATRIX), ("romeo-greedy", "romeo-focused", "romeo-full", "to-be-greedy", "to-be-focused", "to-be-full"))
        self.assertEqual(tuple(row[2] for row in experiment.QUALITATIVE_MATRIX), (256,) * 6)
        self.assertEqual(tuple(row[5] for row in experiment.QUALITATIVE_MATRIX), (None, 20, 81, None, 20, 81))
        self.assertEqual(tuple(row[6] for row in experiment.QUALITATIVE_MATRIX), (None, 9001, 9002, None, 9003, 9004))
        vocabulary = load_accepted_vocabulary_binding(REPOSITORY_ROOT)
        tokenizer = CodePointTokenizer(vocabulary.code_points)
        for _, prompt, count, *_ in experiment.QUALITATIVE_MATRIX:
            self.assertEqual(len(tokenizer.encode(prompt)), len(prompt))
            self.assertGreater(len(prompt) + count, 256)

    def test_synthetic_development_runner_marks_before_work_and_publishes_once(self) -> None:
        evaluation_id = "EXP-20260913-01"
        plan_values = _values("development_plan", evaluation_id)
        plan_values.update({
            "Question": "Does the accepted Phase 9 best-validation model produce the fixed development metrics, baseline comparisons, and complete qualitative matrix under the accepted contract?",
            "Evaluation configuration": experiment._configuration_mapping(),
            "Qualitative matrix": experiment._configuration_mapping()["qualitative_matrix"],
            "Sealed-test access": "none",
            "Next gate": "independent_pre_registration_review",
        })
        plan = _render_record(evaluation_id, "development_plan", plan_values)
        authorization_values = _values("development_authorization", evaluation_id)
        authorization_values.update({
            "Development plan record SHA-256": hashlib.sha256(plan).hexdigest(),
            "Development pre-registration commit": "a" * 40,
            "Phase 9 implementation commit": "b" * 40,
            "Checkpoint SHA-256": checkpoint.CHECKPOINT_SHA256,
            "Authorization": "Authorize exactly one governed development attempt for this evaluation ID; sealed-test access remains unauthorized.",
            "Next gate": "execute_authorized_development_attempt",
        })
        authorization = _render_record(evaluation_id, "development_authorization", authorization_values)
        bundle = _bundle()
        train_documents = (Phase9TokenDocument("t", 0, "train", (0, 1)),)
        validation_documents = (Phase9TokenDocument("v", 0, "validation", (0, 1)),)
        long_window = SimpleNamespace(target_ids=range(792_699))
        empty_window = SimpleNamespace(target_ids=())
        training_windows = (long_window, *(empty_window for _ in range(3_098)))
        validation_windows = (SimpleNamespace(target_ids=range(98_295)), *(empty_window for _ in range(383)))
        metric = inference.Phase9Metrics("mini_gpt_best_validation", "validation", "a" * 64, float.fromhex(checkpoint.VALIDATION_LOSS_HEX), math.exp(float.fromhex(checkpoint.VALIDATION_LOSS_HEX)), 1, 98_295, 1 / 98_295, 384)
        uniform = inference.Phase9Metrics("uniform_81", "validation", "a" * 64, math.log(81), 81.0, 1, 98_295, 1 / 98_295, 384)
        unigram = inference.Phase9Metrics("training_laplace_unigram_add_one", "validation", "a" * 64, 3.0, math.exp(3.0), 1, 98_295, 1 / 98_295, 384)
        baseline = SimpleNamespace(vocabulary_size=81, training_target_count=792_699, raw_counts=(0,) * 81, smoothed_counts=(1,) * 81, probabilities=(1 / 81,) * 81, top_token_id=0)
        generation = SimpleNamespace(generated_text="A" * 256, full_text="ROMEO:\n" + "A" * 256, local_generator_final_state_sha256=None)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "EXPERIMENT_LOG.md").write_bytes(plan + authorization)
            marker_path = root / f"experiments/{evaluation_id}/artifacts/phase9-development-attempt.json"
            def load_after_marker(_root):
                self.assertTrue(marker_path.exists())
                return bundle
            with (
                patch.object(experiment, "_validate_repository", return_value=root),
                patch.object(experiment, "_validate_predecessor_chain"),
                patch.object(experiment, "_validate_development_launch", return_value="c" * 40),
                patch.object(experiment, "_git", side_effect=lambda _root, *args: "c" * 40 if args[:2] == ("rev-parse", "HEAD") else ""),
                patch.object(experiment, "load_phase9_inference_bundle", side_effect=load_after_marker),
                patch.object(experiment, "_load_development_corpus", return_value=object()),
                patch.object(experiment, "_documents", return_value=(train_documents, validation_documents)),
                patch.object(experiment, "build_phase9_windows", side_effect=(training_windows, validation_windows)),
                patch.object(experiment, "fit_phase9_laplace_unigram", return_value=baseline),
                patch.object(experiment, "evaluate_phase9_model", return_value=metric),
                patch.object(experiment, "evaluate_phase9_uniform", return_value=uniform),
                patch.object(experiment, "evaluate_phase9_laplace_unigram", return_value=unigram),
                patch.object(experiment, "generate_phase9_text", return_value=generation) as generate,
                patch.object(experiment, "_validate_development_evidence_mapping"),
            ):
                reference = inference.run_fixed_phase9_development_evaluation(root, evaluation_id=evaluation_id)
            self.assertEqual(reference.kind, "phase9_development")
            self.assertEqual(generate.call_count, 6)
            self.assertTrue(marker_path.exists())
            self.assertTrue((root / f"experiments/{evaluation_id}/artifacts/phase9-development-evidence.json").exists())
            self.assertFalse((root / f"experiments/{evaluation_id}/artifacts/phase9-development-failure.json").exists())

    def test_synthetic_sealed_marker_precedes_guarded_content_and_consumes_failure(self) -> None:
        evaluation_id = "EXP-20260913-01"
        result_values = _values("development_result", evaluation_id)
        result_values["Status"] = "completed"
        development_result = _render_record(evaluation_id, "development_result", result_values)
        development_evidence = experiment._canonical_json({
            "authority": {},
            "laplace_unigram_baseline": {
                "vocabulary_size": 81, "training_target_count": 0,
                "raw_counts": [0] * 81, "smoothed_counts": [1] * 81,
                "probabilities": [1 / 81] * 81, "top_token_id": 0,
            },
        })
        development_digest = hashlib.sha256(development_evidence).hexdigest()
        plan_values = _values("sealed_test_plan", evaluation_id)
        plan_values.update({
            "Status": "planned",
            "Development result record SHA-256": hashlib.sha256(development_result).hexdigest(),
            "Development evidence SHA-256": development_digest,
            "Sealed-test access": "pending_explicit_authorization",
            "Next gate": "independent_sealed_plan_review",
        })
        plan = _render_record(evaluation_id, "sealed_test_plan", plan_values)
        authorization_values = _values("sealed_test_authorization", evaluation_id)
        authorization_values.update({
            "Status": "authorized_once",
            "Sealed plan record SHA-256": hashlib.sha256(plan).hexdigest(),
            "Sealed pre-registration commit": "a" * 40,
            "Development evidence SHA-256": development_digest,
            "Phase 9 implementation commit": "b" * 40,
            "Checkpoint SHA-256": checkpoint.CHECKPOINT_SHA256,
            "Authorization": "Authorize one durable-marker-consumed Twelfth Night metric evaluation for this evaluation ID with no generation, tuning, or retry.",
            "Next gate": "execute_authorized_sealed_test_once",
        })
        authorization = _render_record(evaluation_id, "sealed_test_authorization", authorization_values)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "EXPERIMENT_LOG.md").write_bytes(development_result + plan + authorization)
            artifacts = root / f"experiments/{evaluation_id}/artifacts"
            artifacts.mkdir(parents=True)
            (artifacts / "phase9-development-evidence.json").write_bytes(development_evidence)
            marker_path = artifacts / "phase9-sealed-test-access.json"
            def guarded_read(_root, _metadata):
                self.assertTrue(marker_path.exists())
                raise Phase9GovernanceError("phase9.governance.dataset", field="path")
            with (
                patch.object(experiment, "_validate_repository", return_value=root),
                patch.object(experiment, "_parse_selected_chain", return_value=experiment._parse_records(development_result + plan + authorization)),
                patch.object(experiment, "_validate_predecessor_chain"),
                patch.object(experiment, "_validate_sealed_launch", return_value=("c" * 40, {})),
                patch.object(experiment, "_sealed_test_metadata", return_value={}),
                patch.object(experiment, "_validate_baseline_mapping"),
                patch.object(experiment, "_validate_sealed_evidence_mapping"),
                patch.object(experiment, "load_phase9_inference_bundle", return_value=_bundle()),
                patch.object(experiment, "_read_sealed_test_once", side_effect=guarded_read),
            ):
                reference = inference.run_fixed_phase9_sealed_test_evaluation(root, evaluation_id=evaluation_id)
            self.assertEqual((reference.kind, reference.status), ("phase9_sealed_test_failure", "failed"))
            self.assertTrue(marker_path.exists())
            self.assertTrue((artifacts / "phase9-sealed-test-failure.json").exists())
            self.assertFalse((artifacts / "phase9-sealed-test-evidence.json").exists())


class CorrectionRegressionTests(unittest.TestCase):
    def test_error_fact_vocabulary_rejects_forbidden_content(self) -> None:
        with self.assertRaises(TypeError):
            Phase9TypeError(
                "phase9.type.argument", field="prompt", leaked="SECRET_PROMPT"
            )
        with self.assertRaises(TypeError):
            Phase9TypeError("phase9.type.argument", field="SECRET_PROMPT")

    def test_direct_windows_reject_gap_overlap_discontinuity_and_early_tail(self) -> None:
        invalid = (
            (
                inference.Phase9Window("a", 0, "validation", 0, (0, 1), (1, 2)),
                inference.Phase9Window("a", 0, "validation", 3, (2,), (3,)),
            ),
            (
                inference.Phase9Window("a", 0, "validation", 0, (0, 1), (1, 2)),
                inference.Phase9Window("a", 0, "validation", 1, (2,), (3,)),
            ),
            (
                inference.Phase9Window("a", 0, "validation", 0, (0, 1), (1, 2)),
                inference.Phase9Window("a", 0, "validation", 2, (7,), (8,)),
            ),
            (
                inference.Phase9Window("a", 0, "validation", 0, (0,), (1,)),
                inference.Phase9Window("a", 0, "validation", 1, (1, 2), (2, 3)),
            ),
        )
        for windows in invalid:
            with self.assertRaises(Phase9ContractError):
                inference.evaluate_phase9_uniform(
                    windows, vocabulary_size=9, split="validation"
                )

    def test_baseline_probabilities_must_be_derived_from_smoothed_counts(self) -> None:
        from sebgpt.inference.phase9_types import _make_laplace_unigram

        baseline = _make_laplace_unigram(
            vocabulary_size=2,
            training_target_count=2,
            raw_counts=(1, 1),
            smoothed_counts=(2, 2),
            probabilities=(0.75, 0.25),
            top_token_id=0,
        )
        windows = (inference.Phase9Window("v", 0, "validation", 0, (0,), (1,)),)
        with self.assertRaises(Phase9ContractError):
            inference.evaluate_phase9_laplace_unigram(
                baseline, windows, split="validation"
            )

    def test_generation_exact_rolling_slices_and_one_draw_per_token(self) -> None:
        bundle = _bundle()
        prompt = "A" * 256
        prompt_ids = bundle.tokenizer.encode(prompt)
        observed: list[tuple[int, ...]] = []

        def fake_forward(_model, token_ids):
            observed.append(tuple(token_ids.tolist()))
            return torch.zeros((len(token_ids), 81), dtype=torch.float32)

        real_multinomial = torch.multinomial
        generator_ids: list[int] = []

        def traced_multinomial(*args, **kwargs):
            generator_ids.append(id(kwargs["generator"]))
            return real_multinomial(*args, **kwargs)

        with (
            patch.object(MiniGPT, "forward", new=fake_forward),
            patch("sebgpt.inference.phase9_generation.torch.multinomial", side_effect=traced_multinomial),
        ):
            generated = inference.generate_phase9_text(
                bundle,
                prompt,
                generated_token_count=3,
                mode="categorical",
                temperature=1.0,
                top_k=1,
                seed=0,
            )
        self.assertEqual(len(observed), 3)
        self.assertEqual(observed[0], prompt_ids)
        self.assertEqual(observed[1], (*prompt_ids[1:], generated.generated_token_ids[0]))
        self.assertEqual(
            observed[2],
            (*prompt_ids[2:], *generated.generated_token_ids[:2]),
        )
        self.assertEqual(len(generator_ids), 3)
        self.assertEqual(len(set(generator_ids)), 1)

    def test_probability_injected_nonfinite_weight_fails_before_sampling(self) -> None:
        logits = torch.zeros(81, dtype=torch.float32)
        with patch(
            "sebgpt.inference.phase9_generation.torch.exp",
            return_value=torch.full((2,), float("nan"), dtype=torch.float32),
        ):
            with self.assertRaises(inference.Phase9NumericalError):
                inference.build_categorical_probabilities(
                    logits, temperature=1.0, top_k=2
                )
        import sebgpt.inference.phase9_generation as generation

        for invalid in (
            torch.full((81,), float("nan"), dtype=torch.float32),
            torch.zeros(81, dtype=torch.float32),
            torch.full((81,), 1 / 81, dtype=torch.float32),
        ):
            with self.assertRaises(inference.Phase9NumericalError):
                generation._validate_probability_vector(invalid, (0,))

    def test_publication_rejects_traversal_and_symlinked_ancestor(self) -> None:
        content = experiment._canonical_json({"ok": True})
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory).resolve()
            with self.assertRaises(Phase9ContractError):
                experiment._publish(
                    root,
                    "EXP-20260913-01",
                    filename="../phase9-development-attempt.json",
                    kind="phase9_development_attempt",
                    status="consumed",
                    content=content,
                )
            (root / "experiments").symlink_to(Path(outside).resolve(), target_is_directory=True)
            result = experiment._publish(
                root,
                "EXP-20260913-01",
                filename="phase9-development-attempt.json",
                kind="phase9_development_attempt",
                status="consumed",
                content=content,
            )
            self.assertEqual(result.state, experiment.UNKNOWN)
            self.assertFalse(
                (Path(outside) / "EXP-20260913-01/artifacts/phase9-development-attempt.json").exists()
            )

    def test_publication_directory_replacement_is_unknown(self) -> None:
        content = experiment._canonical_json({"ok": True})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch.object(
                experiment,
                "_revalidate_publication_directories",
                side_effect=OSError,
            ):
                result = experiment._publish(
                    root,
                    "EXP-20260913-01",
                    filename="phase9-development-attempt.json",
                    kind="phase9_development_attempt",
                    status="consumed",
                    content=content,
                )
            self.assertEqual(result.state, experiment.UNKNOWN)

    def test_checkpoint_entry_revalidation_detects_file_and_directory_swaps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            parent_fd = experiment.os.open(
                root, experiment.os.O_RDONLY | experiment.os.O_DIRECTORY
            )
            try:
                (root / "child").mkdir()
                child_meta = (root / "child").stat()
                directory_entry = (
                    parent_fd,
                    "child",
                    (child_meta.st_dev, child_meta.st_ino, child_meta.st_mode),
                )
                (root / "value").write_bytes(b"a")
                value_meta = (root / "value").stat()
                file_entry = (
                    parent_fd,
                    "value",
                    (value_meta.st_dev, value_meta.st_ino, value_meta.st_mode, value_meta.st_size),
                )
                (root / "value").rename(root / "old-value")
                (root / "value").write_bytes(b"a")
                with self.assertRaises(inference.Phase9CheckpointError):
                    checkpoint._revalidate_entries((directory_entry,), (file_entry,))
                (root / "child").rename(root / "old-child")
                (root / "child").mkdir()
                with self.assertRaises(inference.Phase9CheckpointError):
                    checkpoint._revalidate_entries((directory_entry,), ())
            finally:
                experiment.os.close(parent_fd)

    def test_each_of_19_protected_paths_is_independently_checked(self) -> None:
        for mutated_path in checkpoint.PROTECTED_PATHS:
            with self.subTest(path=mutated_path), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                for relative in checkpoint.PROTECTED_PATHS:
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"mutated" if relative == mutated_path else b"accepted")
                (root / "EXPERIMENT_LOG.md").write_bytes(b"synthetic")

                def fake_git(_root, *args, binary=False):
                    if args == ("rev-parse", "--show-toplevel"):
                        return str(root)
                    if args == ("branch", "--show-current"):
                        return "main"
                    if args in (("rev-parse", "HEAD"), ("rev-parse", "origin/main")):
                        return "a" * 40
                    if args[:2] == ("status", "--porcelain=v2"):
                        return ""
                    if args and args[0] == "show":
                        return b"accepted"
                    raise AssertionError(args)

                with (
                    patch.object(checkpoint, "_git", side_effect=fake_git),
                    patch.object(checkpoint.subprocess, "run", return_value=SimpleNamespace()),
                    patch.object(checkpoint, "_sha_file", side_effect=lambda path: checkpoint.TRACKED_IDENTITY_SHA256S.get(str(path.relative_to(root)), "x")),
                    patch.object(checkpoint, "_validate_phase8_result_record"),
                    patch.object(checkpoint, "_validate_runtime"),
                ):
                    with self.assertRaises(Phase9GovernanceError) as caught:
                        checkpoint._validate_repository(root)
                self.assertEqual(caught.exception.details["field"], "protected_path")

    def test_scoped_parser_ignores_unrelated_body_but_rejects_competing_edge(self) -> None:
        selected = "EXP-20260913-01"
        plan_values = _values("development_plan", selected)
        plan = _render_record(selected, "development_plan", plan_values)
        authorization_values = _values("development_authorization", selected)
        authorization_values["Development plan record SHA-256"] = hashlib.sha256(plan).hexdigest()
        authorization = _render_record(selected, "development_authorization", authorization_values)
        unrelated = b"### EXP-20260913-99 \xe2\x80\x94 Phase 9 development plan\n\n- **broken:** nope\n\n"
        records = experiment._parse_selected_chain(plan + authorization + unrelated, selected)
        self.assertEqual(experiment._one(records, selected, "development_plan").evaluation_id, selected)
        competing = (
            b"### EXP-20260913-98 \xe2\x80\x94 Phase 9 development supersession\n\n"
            + selected.encode("ascii")
            + b"\n"
        )
        with self.assertRaises(Phase9GovernanceError):
            experiment._parse_selected_chain(plan + authorization + competing, selected)

    def test_predecessor_64_edge_boundary_cycle_merge_and_broken_reciprocal(self) -> None:
        def chain(edge_count: int):
            ids = tuple(f"EXP-20260913-{index:02d}" for index in range(1, edge_count + 2))
            rendered: list[bytes] = []
            for index, evaluation_id in enumerate(ids):
                plan_values = _values("development_plan", evaluation_id)
                plan_values["Predecessor evaluation"] = ids[index - 1] if index else None
                plan = _render_record(evaluation_id, "development_plan", plan_values)
                rendered.append(plan)
                if index < len(ids) - 1:
                    result_values = _values("development_result", evaluation_id)
                    result_values["Status"] = "failed"
                    result = _render_record(evaluation_id, "development_result", result_values)
                    rendered.append(result)
                    supersession_values = _values("development_supersession", evaluation_id)
                    supersession_values.update({
                        "Prior result record SHA-256": hashlib.sha256(result).hexdigest(),
                        "Successor evaluation ID": ids[index + 1],
                    })
                    rendered.append(
                        _render_record(
                            evaluation_id, "development_supersession", supersession_values
                        )
                    )
            current_plan = rendered[-1]
            authorization_values = _values("development_authorization", ids[-1])
            authorization_values["Development plan record SHA-256"] = hashlib.sha256(current_plan).hexdigest()
            rendered.append(
                _render_record(ids[-1], "development_authorization", authorization_values)
            )
            return ids, experiment._parse_records(b"".join(rendered))

        ids, records = chain(64)
        self.assertEqual(
            experiment._development_authority(records, ids[-1])[0].evaluation_id,
            ids[-1],
        )
        ids_65, records_65 = chain(65)
        with self.assertRaises(Phase9GovernanceError):
            experiment._development_authority(records_65, ids_65[-1])

        cycle_records = list(records)
        terminal_index = next(
            index for index, record in enumerate(cycle_records)
            if record.evaluation_id == ids[0] and record.kind == "development_plan"
        )
        terminal = cycle_records[terminal_index]
        terminal_values = dict(terminal.values)
        terminal_values["Predecessor evaluation"] = ids[-1]
        cycle_records[terminal_index] = experiment._Record(
            terminal.evaluation_id, terminal.kind, terminal_values, terminal.raw, terminal.sha256
        )
        with self.assertRaises(Phase9GovernanceError):
            experiment._development_authority(tuple(cycle_records), ids[-1])

        broken = list(records)
        edge_index = next(
            index for index, record in enumerate(broken)
            if record.evaluation_id == ids[-2] and record.kind == "development_supersession"
        )
        edge = broken[edge_index]
        edge_values = dict(edge.values)
        edge_values["Successor evaluation ID"] = ids[0]
        broken[edge_index] = experiment._Record(
            edge.evaluation_id, edge.kind, edge_values, edge.raw, edge.sha256
        )
        with self.assertRaises(Phase9GovernanceError):
            experiment._development_authority(tuple(broken), ids[-1])

        other = "EXP-20260913-99"
        other_values = _values("development_supersession", other)
        other_values.update({
            "Prior result record SHA-256": "a" * 64,
            "Successor evaluation ID": ids[-1],
        })
        other_record = experiment._parse_records(
            _render_record(other, "development_supersession", other_values)
        )[0]
        with self.assertRaises(Phase9GovernanceError):
            experiment._development_authority((*records, other_record), ids[-1])

        with self.assertRaises(Phase9GovernanceError):
            experiment._development_authority(
                tuple(record for record in records if record.kind != "development_authorization"),
                ids[-1],
            )
        authorization = experiment._one(records, ids[-1], "development_authorization")
        with self.assertRaises(Phase9GovernanceError):
            experiment._development_authority((*records, authorization), ids[-1])

    def _valid_development_evidence(self):
        evaluation_id = "EXP-20260913-01"
        marker = Phase9EvidenceReference(
            evaluation_id,
            "phase9_development_attempt",
            "consumed",
            f"experiments/{evaluation_id}/artifacts/phase9-development-attempt.json",
            100,
            "a" * 64,
        )
        metric = inference.Phase9Metrics(
            "mini_gpt_best_validation", "validation", "b" * 64,
            float.fromhex(checkpoint.VALIDATION_LOSS_HEX),
            math.exp(float.fromhex(checkpoint.VALIDATION_LOSS_HEX)),
            1, 98_295, 1 / 98_295, 384,
        )
        uniform = inference.Phase9Metrics(
            "uniform_81", "validation", "b" * 64,
            math.log(81), math.exp(math.log(81)),
            1, 98_295, 1 / 98_295, 384,
        )
        unigram = inference.Phase9Metrics(
            "training_laplace_unigram_add_one", "validation", "b" * 64,
            2.0, math.exp(2.0), 1, 98_295, 1 / 98_295, 384,
        )
        raw = (792_699, *((0,) * 80))
        smoothed = tuple(value + 1 for value in raw)
        denominator = 792_699 + 81
        baseline = SimpleNamespace(
            vocabulary_size=81,
            training_target_count=792_699,
            raw_counts=raw,
            smoothed_counts=smoothed,
            probabilities=tuple(value / denominator for value in smoothed),
            top_token_id=0,
        )
        samples = []
        for sample_id, prompt, count, mode, temperature, top_k, seed in experiment.QUALITATIVE_MATRIX:
            generated = "A" * count
            samples.append({
                "sample_id": sample_id,
                "prompt": prompt,
                "generated_token_count": count,
                "mode": mode,
                "temperature": temperature,
                "temperature_hex": temperature.hex() if temperature is not None else None,
                "top_k": top_k,
                "seed": seed,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "checkpoint_role": "best_validation",
                "checkpoint_sha256": checkpoint.CHECKPOINT_SHA256,
                "generated_text": generated,
                "generated_byte_count": len(generated.encode()),
                "generated_code_point_count": count,
                "generated_text_sha256": hashlib.sha256(generated.encode()).hexdigest(),
                "full_text_sha256": hashlib.sha256((prompt + generated).encode()).hexdigest(),
                "local_generator_final_state_sha256": None if mode == "greedy" else "c" * 64,
            })
        authority = experiment._development_authority_mapping(
            experiment._Record(
                evaluation_id, "development_plan", {}, b"plan", "d" * 64
            ),
            experiment._Record(
                evaluation_id,
                "development_authorization",
                {
                    "Phase 9 implementation commit": "1" * 40,
                    "Development pre-registration commit": "2" * 40,
                },
                b"authorization",
                "e" * 64,
            ),
            "3" * 40,
        )
        configuration = experiment._configuration_mapping()
        evidence = experiment._development_evidence(
            evaluation_id,
            marker,
            authority,
            configuration,
            metric,
            uniform,
            baseline,
            unigram,
            samples,
        )
        return evidence, marker, authority, configuration

    def test_development_evidence_validator_rejects_independent_mutations(self) -> None:
        evidence, marker, authority, configuration = self._valid_development_evidence()
        experiment._validate_development_evidence_mapping(
            evidence,
            evaluation_id=evidence["evaluation_id"],
            marker_sha256=marker.sha256,
            authority=authority,
            configuration=configuration,
            completed=True,
        )
        mutations = (
            lambda value: value.pop("limitations"),
            lambda value: value.__setitem__("schema_version", True),
            lambda value: value["model_validation"].__setitem__("nll_hex", "bad"),
            lambda value: value["laplace_unigram_baseline"]["probabilities"].__setitem__(0, 0.5),
            lambda value: value["samples"][0].__setitem__("sample_id", "wrong"),
            lambda value: value["samples"][1].__setitem__("local_generator_final_state_sha256", None),
            lambda value: value.__setitem__("sealed_test_access", "opened"),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                candidate = copy.deepcopy(evidence)
                mutate(candidate)
                with self.assertRaises((Phase9TypeError, Phase9ContractError)):
                    experiment._validate_development_evidence_mapping(
                        candidate,
                        evaluation_id=evidence["evaluation_id"],
                        marker_sha256=marker.sha256,
                        authority=authority,
                        configuration=configuration,
                        completed=True,
                    )

    def test_failure_schema_is_stage_specific_and_content_safe(self) -> None:
        error = Phase9ContractError("phase9.generation.output", field="output")
        failure = experiment._failure("sample_romeo_greedy", error)
        self.assertEqual(
            set(failure["safe_facts"]),
            {"sample_id", "field", "completed_token_count"},
        )
        experiment._validate_failure_mapping(failure, sealed=False)
        broken = copy.deepcopy(failure)
        broken["safe_facts"].pop("sample_id")
        with self.assertRaises(Phase9ContractError):
            experiment._validate_failure_mapping(broken, sealed=False)

    def test_attempt_and_access_marker_validators_reject_cross_field_mutation(self) -> None:
        evaluation_id = "EXP-20260913-01"
        plan = experiment._Record(evaluation_id, "development_plan", {}, b"plan", "a" * 64)
        authorization = experiment._Record(
            evaluation_id,
            "development_authorization",
            {
                "Development pre-registration commit": "b" * 40,
                "Phase 9 implementation commit": "c" * 40,
                "Sealed pre-registration commit": "d" * 40,
            },
            b"authorization",
            "e" * 64,
        )
        marker = {
            "schema_version": 1,
            "evaluation_id": evaluation_id,
            "kind": "phase9_development_attempt",
            "development_plan_record_sha256": plan.sha256,
            "development_pre_registration_commit": "b" * 40,
            "development_authorization_record_sha256": authorization.sha256,
            "development_authorization_commit": "f" * 40,
            "phase9_contract_sha256": checkpoint.PHASE9_SPEC_SHA256,
            "phase9_implementation_commit": "c" * 40,
            "checkpoint_sha256": checkpoint.CHECKPOINT_SHA256,
            "created_at_utc": "2026-09-13T12:00:00Z",
            "consumed": True,
        }
        experiment._validate_attempt_marker(
            marker,
            evaluation_id=evaluation_id,
            plan=plan,
            authorization=authorization,
            authorization_commit="f" * 40,
        )
        for field in marker:
            candidate = copy.deepcopy(marker)
            candidate[field] = None
            with self.subTest(field=field), self.assertRaises(Phase9ContractError):
                experiment._validate_attempt_marker(
                    candidate,
                    evaluation_id=evaluation_id,
                    plan=plan,
                    authorization=authorization,
                    authorization_commit="f" * 40,
                )
        access = {
            "schema_version": 1,
            "evaluation_id": evaluation_id,
            "kind": "phase9_sealed_test_access",
            "sealed_plan_record_sha256": plan.sha256,
            "sealed_pre_registration_commit": "d" * 40,
            "sealed_authorization_record_sha256": authorization.sha256,
            "sealed_authorization_commit": "f" * 40,
            "phase9_contract_sha256": checkpoint.PHASE9_SPEC_SHA256,
            "phase9_implementation_commit": "c" * 40,
            "checkpoint_sha256": checkpoint.CHECKPOINT_SHA256,
            "development_evidence_sha256": "1" * 64,
            "created_at_utc": "2026-09-13T12:00:00Z",
            "consumed": True,
        }
        experiment._validate_access_marker(
            access,
            evaluation_id=evaluation_id,
            plan=plan,
            authorization=authorization,
            authorization_commit="f" * 40,
            development_evidence_sha256="1" * 64,
        )
        broken_access = copy.deepcopy(access)
        broken_access["checkpoint_sha256"] = "0" * 64
        with self.assertRaises(Phase9ContractError):
            experiment._validate_access_marker(
                broken_access,
                evaluation_id=evaluation_id,
                plan=plan,
                authorization=authorization,
                authorization_commit="f" * 40,
                development_evidence_sha256="1" * 64,
            )

    def test_development_runner_exact_governed_event_order(self) -> None:
        evaluation_id = "EXP-20260913-01"
        plan_values = _values("development_plan", evaluation_id)
        plan = _render_record(evaluation_id, "development_plan", plan_values)
        authorization_values = _values("development_authorization", evaluation_id)
        authorization_values["Development plan record SHA-256"] = hashlib.sha256(plan).hexdigest()
        authorization = _render_record(evaluation_id, "development_authorization", authorization_values)
        records = experiment._parse_records(plan + authorization)
        bundle = _bundle()
        trace: list[str] = []
        marker_reference = Phase9EvidenceReference(
            evaluation_id,
            "phase9_development_attempt",
            "consumed",
            f"experiments/{evaluation_id}/artifacts/phase9-development-attempt.json",
            1,
            "a" * 64,
        )
        completed_reference = Phase9EvidenceReference(
            evaluation_id,
            "phase9_development",
            "completed",
            f"experiments/{evaluation_id}/artifacts/phase9-development-evidence.json",
            1,
            "b" * 64,
        )
        long_window = SimpleNamespace(target_ids=range(792_699))
        empty_window = SimpleNamespace(target_ids=())
        training_windows = (long_window, *(empty_window for _ in range(3_098)))
        validation_windows = (SimpleNamespace(target_ids=range(98_295)), *(empty_window for _ in range(383)))
        model_metric = inference.Phase9Metrics(
            "mini_gpt_best_validation", "validation", "c" * 64,
            float.fromhex(checkpoint.VALIDATION_LOSS_HEX),
            math.exp(float.fromhex(checkpoint.VALIDATION_LOSS_HEX)), 1, 98_295,
            1 / 98_295, 384,
        )
        uniform_metric = inference.Phase9Metrics(
            "uniform_81", "validation", "c" * 64, math.log(81), 81.0,
            1, 98_295, 1 / 98_295, 384,
        )
        unigram_metric = inference.Phase9Metrics(
            "training_laplace_unigram_add_one", "validation", "c" * 64,
            3.0, math.exp(3.0), 1, 98_295, 1 / 98_295, 384,
        )
        baseline = SimpleNamespace(
            vocabulary_size=81, training_target_count=792_699,
            raw_counts=(792_699, *((0,) * 80)),
            smoothed_counts=(792_700, *((1,) * 80)),
            probabilities=(1.0, *((0.0,) * 80)), top_token_id=0,
        )

        def publish(*_args, **_kwargs):
            trace.append("marker_publication")
            return experiment._Publication(experiment.KNOWN_PRESENT, marker_reference)

        def build(_documents, **_kwargs):
            name = "training_windows" if not trace or trace[-1] == "documents" else "validation_windows"
            trace.append(name)
            return training_windows if name == "training_windows" else validation_windows

        def generate(_bundle, prompt, **kwargs):
            sample_id = next(row[0] for row in experiment.QUALITATIVE_MATRIX if row[1] == prompt and row[3] == kwargs["mode"] and row[5] == kwargs["top_k"])
            trace.append(sample_id)
            generated = "A" * 256
            return SimpleNamespace(
                generated_text=generated,
                full_text=prompt + generated,
                local_generator_final_state_sha256=None if kwargs["mode"] == "greedy" else "d" * 64,
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "EXPERIMENT_LOG.md").write_bytes(b"synthetic")
            with (
                patch.object(experiment, "_validate_repository", return_value=root),
                patch.object(experiment, "_parse_selected_chain", return_value=records),
                patch.object(experiment, "_validate_predecessor_chain"),
                patch.object(experiment, "_validate_development_launch", return_value="e" * 40),
                patch.object(experiment, "_require_artifacts_absent"),
                patch.object(experiment, "_validate_attempt_marker"),
                patch.object(experiment, "_publish", side_effect=publish),
                patch.object(experiment, "load_phase9_inference_bundle", side_effect=lambda _root: (trace.append("inference_load"), bundle)[1]),
                patch.object(experiment, "_load_development_corpus", side_effect=lambda _root: (trace.append("development_corpus"), object())[1]),
                patch.object(experiment, "_documents", side_effect=lambda *_: (trace.append("documents"), ((object(),), (object(),)))[1]),
                patch.object(experiment, "build_phase9_windows", side_effect=build),
                patch.object(experiment, "fit_phase9_laplace_unigram", side_effect=lambda *_args, **_kwargs: (trace.append("unigram_fit"), baseline)[1]),
                patch.object(experiment, "evaluate_phase9_model", side_effect=lambda *_args, **_kwargs: (trace.append("model_validation"), model_metric)[1]),
                patch.object(experiment, "evaluate_phase9_uniform", side_effect=lambda *_args, **_kwargs: (trace.append("uniform_validation"), uniform_metric)[1]),
                patch.object(experiment, "evaluate_phase9_laplace_unigram", side_effect=lambda *_args, **_kwargs: (trace.append("unigram_validation"), unigram_metric)[1]),
                patch.object(experiment, "generate_phase9_text", side_effect=generate),
                patch.object(experiment, "_compare_metrics", side_effect=lambda *_: trace.append("comparison")),
                patch.object(experiment, "_validate_development_evidence_mapping", side_effect=lambda *_args, **_kwargs: trace.append("evidence_validation")),
                patch.object(experiment, "_publish_terminal", side_effect=lambda *_args, **_kwargs: (trace.append("terminal_publication"), experiment._Publication(experiment.KNOWN_PRESENT, completed_reference))[1]),
            ):
                result = inference.run_fixed_phase9_development_evaluation(
                    root, evaluation_id=evaluation_id
                )
        self.assertIs(result, completed_reference)
        self.assertEqual(
            trace,
            [
                "marker_publication", "inference_load", "development_corpus", "documents",
                "training_windows", "validation_windows", "unigram_fit",
                "model_validation", "uniform_validation", "unigram_validation",
                "romeo-greedy", "romeo-focused", "romeo-full", "to-be-greedy",
                "to-be-focused", "to-be-full", "comparison", "evidence_validation",
                "terminal_publication",
            ],
        )

    def _sealed_launch_fixture(self):
        evaluation_id = "EXP-20260913-01"
        implementation_commit = "1" * 40
        development_plan = experiment._Record(
            evaluation_id, "development_plan", {}, b"development plan", "1" * 64
        )
        development_authorization = experiment._Record(
            evaluation_id,
            "development_authorization",
            {
                "Evaluation ID": evaluation_id,
                "Entry kind": "development_authorization",
                "Scope": "development",
                "Status": "authorized_once",
                "Development plan record SHA-256": development_plan.sha256,
                "Development pre-registration commit": "2" * 40,
                "Phase 9 implementation commit": implementation_commit,
                "Checkpoint SHA-256": checkpoint.CHECKPOINT_SHA256,
                "Authorization": experiment.DEVELOPMENT_AUTHORIZATION,
                "Next gate": "execute_authorized_development_attempt",
            },
            b"development authorization",
            "2" * 64,
        )
        development_authorization_commit = "3" * 40
        authority = experiment._development_authority_mapping(
            development_plan,
            development_authorization,
            development_authorization_commit,
        )
        development, _, _, _ = self._valid_development_evidence()
        development["authority"] = authority
        development_bytes = experiment._canonical_json(development)
        development_digest = hashlib.sha256(development_bytes).hexdigest()
        marker_reference = {
            "evaluation_id": evaluation_id,
            "kind": "phase9_development_attempt",
            "status": "consumed",
            "relative_path": f"experiments/{evaluation_id}/artifacts/phase9-development-attempt.json",
            "byte_count": 100,
            "sha256": development["attempt_marker_sha256"],
        }
        evidence_reference = {
            "evaluation_id": evaluation_id,
            "kind": "phase9_development",
            "status": "completed",
            "relative_path": f"experiments/{evaluation_id}/artifacts/phase9-development-evidence.json",
            "byte_count": len(development_bytes),
            "sha256": development_digest,
        }
        evidence_object = Phase9EvidenceReference(**evidence_reference)
        result_values = _values("development_result", evaluation_id)
        result_values.update({
            "Scope": "development",
            "Status": "completed",
            "Recorded at UTC": "2026-09-13T12:00:02Z",
            "Started at UTC": "2026-09-13T12:00:00Z",
            "Finished at UTC": "2026-09-13T12:00:01Z",
            "Predecessor evaluation": None,
            "Development plan record SHA-256": development_plan.sha256,
            "Development pre-registration commit": "2" * 40,
            "Development authorization record SHA-256": development_authorization.sha256,
            "Development authorization commit": development_authorization_commit,
            "Attempt marker": marker_reference,
            "Development evidence": evidence_reference,
            "Model validation": development["model_validation"],
            "Uniform validation": development["uniform_validation"],
            "Laplace unigram baseline": experiment._expected_baseline_result_reference(
                development["laplace_unigram_baseline"], evidence_object
            ),
            "Laplace unigram validation": development["laplace_unigram_validation"],
            "Qualitative samples": [
                experiment._expected_sample_reference(sample, index, evidence_object)
                for index, sample in enumerate(development["samples"])
            ],
            "Sealed-test access": "none",
            "Failure": None,
            "Observations": [],
            "Conclusions": [],
            "Limitations": list(experiment.LIMITATIONS),
            "Next gate": "independent_development_result_review",
        })
        development_result = experiment._Record(
            evaluation_id,
            "development_result",
            result_values,
            b"development result",
            "6" * 64,
        )
        marker_policy = {
            "schema_version": 1,
            "relative_path": f"experiments/{evaluation_id}/artifacts/phase9-sealed-test-access.json",
            "maximum_bytes": 16_384,
            "no_clobber": True,
            "commit_point": "known_present",
        }
        evidence_policy = {
            "schema_version": 1,
            "completed_relative_path": f"experiments/{evaluation_id}/artifacts/phase9-sealed-test-evidence.json",
            "failure_relative_path": f"experiments/{evaluation_id}/artifacts/phase9-sealed-test-failure.json",
            "maximum_bytes": 262_144,
            "canonical_json": "utf8_ensure_ascii_sorted_keys_indent_2_allow_nan_false_final_lf",
            "generated_text_storage": "none",
            "tracked_record_storage": "references_and_hashes_without_generated_or_sealed_text",
        }
        plan_values = {
            "Evaluation ID": evaluation_id,
            "Entry kind": "sealed_test_plan",
            "Scope": "sealed_test",
            "Status": "planned",
            "Accepted development result commit": "7" * 40,
            "Development result record SHA-256": development_result.sha256,
            "Development evidence SHA-256": development_digest,
            "Frozen authority": authority,
            "Frozen evaluation configuration": experiment._configuration_mapping(),
            "Frozen qualitative matrix": experiment._configuration_mapping()["qualitative_matrix"],
            "Access marker policy": marker_policy,
            "Sealed evidence policy": evidence_policy,
            "Interpretation policy": experiment.SEALED_INTERPRETATION_POLICY,
            "Sealed-test access": "pending_explicit_authorization",
            "Next gate": "independent_sealed_plan_review",
        }
        plan = experiment._Record(
            evaluation_id, "sealed_test_plan", plan_values, b"sealed plan", "8" * 64
        )
        authorization_values = {
            "Evaluation ID": evaluation_id,
            "Entry kind": "sealed_test_authorization",
            "Scope": "sealed_test",
            "Status": "authorized_once",
            "Sealed plan record SHA-256": plan.sha256,
            "Sealed pre-registration commit": "9" * 40,
            "Development evidence SHA-256": development_digest,
            "Phase 9 implementation commit": implementation_commit,
            "Checkpoint SHA-256": checkpoint.CHECKPOINT_SHA256,
            "Authorization": experiment.SEALED_AUTHORIZATION,
            "Next gate": "execute_authorized_sealed_test_once",
        }
        authorization = experiment._Record(
            evaluation_id,
            "sealed_test_authorization",
            authorization_values,
            b"sealed authorization",
            "9" * 64,
        )
        records = (
            development_plan, development_authorization, development_result, plan, authorization
        )
        return records, development_result, plan, authorization, development, development_digest

    def test_every_sealed_frozen_prerequisite_fails_before_consumption(self) -> None:
        records, result, plan, authorization, development, digest = self._sealed_launch_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()

            def validate(candidate_plan, candidate_authorization, candidate_result=result):
                candidate_records = tuple(
                    candidate_plan if item is plan else candidate_authorization if item is authorization else candidate_result if item is result else item
                    for item in records
                )
                with (
                    patch.object(experiment, "_validate_plan_record", return_value={"implementation_commit": "1" * 40}),
                    patch.object(experiment, "_record_introduced_by_commit", return_value=True),
                    patch.object(experiment, "_git", return_value="a" * 40),
                ):
                    return experiment._validate_sealed_launch(
                        root, candidate_records, result.evaluation_id, candidate_result,
                        candidate_plan, candidate_authorization, development, digest,
                    )

            validate(plan, authorization)
            plan_fields = (
                "Accepted development result commit", "Development result record SHA-256",
                "Development evidence SHA-256", "Frozen authority",
                "Frozen evaluation configuration", "Frozen qualitative matrix",
                "Access marker policy", "Sealed evidence policy", "Interpretation policy",
                "Sealed-test access", "Next gate",
            )
            for field in plan_fields:
                values = copy.deepcopy(dict(plan.values))
                values[field] = None
                candidate = experiment._Record(
                    plan.evaluation_id, plan.kind, values, plan.raw, plan.sha256
                )
                with self.subTest(plan_field=field), self.assertRaises((Phase9GovernanceError, Phase9ContractError)):
                    validate(candidate, authorization)
            authorization_fields = (
                "Sealed plan record SHA-256", "Sealed pre-registration commit",
                "Development evidence SHA-256", "Phase 9 implementation commit",
                "Checkpoint SHA-256", "Authorization", "Next gate",
            )
            for field in authorization_fields:
                values = copy.deepcopy(dict(authorization.values))
                values[field] = None
                candidate = experiment._Record(
                    authorization.evaluation_id, authorization.kind, values,
                    authorization.raw, authorization.sha256,
                )
                with self.subTest(authorization_field=field), self.assertRaises((Phase9GovernanceError, Phase9ContractError)):
                    validate(plan, candidate)
            result_mutations = (
                ("failed_status", lambda value: value.__setitem__("Status", "failed")),
                ("uncertain_status", lambda value: value.__setitem__("Status", "uncertain")),
                ("wrong_evidence_kind", lambda value: value["Development evidence"].update({"kind": "phase9_development_failure", "status": "failed", "relative_path": f"experiments/{result.evaluation_id}/artifacts/phase9-development-failure.json"})),
                ("nonnull_failure", lambda value: value.__setitem__("Failure", {"stage": "x"})),
                ("missing_metric", lambda value: value.__setitem__("Model validation", None)),
                ("missing_baseline", lambda value: value.__setitem__("Laplace unigram baseline", None)),
                ("missing_sample", lambda value: value.__setitem__("Qualitative samples", value["Qualitative samples"][:-1])),
            )
            for name, mutate in result_mutations:
                values = copy.deepcopy(dict(result.values))
                mutate(values)
                candidate = experiment._Record(
                    result.evaluation_id, result.kind, values, result.raw, result.sha256
                )
                with self.subTest(result_mutation=name), self.assertRaises((Phase9GovernanceError, Phase9ContractError)):
                    validate(plan, authorization, candidate)
            with (
                patch.object(experiment, "_validate_plan_record", return_value={"implementation_commit": "1" * 40}),
                patch.object(experiment, "_record_introduced_by_commit", return_value=False),
                patch.object(experiment, "_git", return_value="a" * 40),
            ):
                with self.assertRaises(Phase9GovernanceError):
                    experiment._validate_sealed_launch(
                        root, records, result.evaluation_id, result, plan,
                        authorization, development, digest,
                    )
            def broken_ancestry(_root, *args, **_kwargs):
                if args[:2] == ("merge-base", "--is-ancestor"):
                    raise Phase9GovernanceError(
                        "phase9.governance.repository", field="ancestor"
                    )
                return "a" * 40
            with (
                patch.object(experiment, "_validate_plan_record", return_value={"implementation_commit": "1" * 40}),
                patch.object(experiment, "_record_introduced_by_commit", return_value=True),
                patch.object(experiment, "_git", side_effect=broken_ancestry),
            ):
                with self.assertRaises(Phase9GovernanceError):
                    experiment._validate_sealed_launch(
                        root, records, result.evaluation_id, result, plan,
                        authorization, development, digest,
                    )

    def test_sealed_preflight_failure_never_publishes_marker_or_touches_test(self) -> None:
        records, result, plan, authorization, _, digest = self._sealed_launch_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "EXPERIMENT_LOG.md").write_bytes(b"synthetic")
            with (
                patch.object(experiment, "_validate_repository", return_value=root),
                patch.object(experiment, "_parse_selected_chain", return_value=records),
                patch.object(experiment, "_validate_predecessor_chain"),
                patch.object(experiment, "_read_evidence_artifact", return_value={}),
                patch.object(experiment, "_validate_sealed_launch", side_effect=Phase9GovernanceError("phase9.governance.sealed_test", field="record")),
                patch.object(experiment, "_serialized_sealed_access_publication") as publish,
                patch.object(experiment, "_sealed_test_metadata") as metadata,
                patch.object(experiment, "load_phase9_inference_bundle") as load,
                patch.object(experiment, "_read_sealed_test_once") as sealed_read,
            ):
                with self.assertRaises(Phase9GovernanceError):
                    inference.run_fixed_phase9_sealed_test_evaluation(
                        root, evaluation_id=result.evaluation_id
                    )
            publish.assert_not_called()
            metadata.assert_not_called()
            load.assert_not_called()
            sealed_read.assert_not_called()

    def test_selected_development_plan_requires_complete_semantics(self) -> None:
        evaluation_id = "EXP-20260913-01"
        values = _values("development_plan", evaluation_id)
        values.update({
            "Question": experiment.DEVELOPMENT_QUESTION,
            "Predecessor evaluation": None,
            "Phase 8 authority": copy.deepcopy(experiment._PHASE8_AUTHORITY),
            "Checkpoint authority": copy.deepcopy(experiment._CHECKPOINT_AUTHORITY),
            "Phase 9 contract authority": copy.deepcopy(experiment._CONTRACT_AUTHORITY),
            "Phase 9 implementation authority": {"synthetic": True},
            "Runtime identity": {"runtime": True},
            "Dataset authority": copy.deepcopy(experiment._DATASET_AUTHORITY),
            "Tokenizer authority": copy.deepcopy(experiment._TOKENIZER_AUTHORITY),
            "Model authority": copy.deepcopy(experiment._MODEL_AUTHORITY),
            "Evaluation configuration": experiment._configuration_mapping(),
            "Qualitative matrix": experiment._configuration_mapping()["qualitative_matrix"],
            "Attempt marker policy": {
                "schema_version": 1,
                "relative_path": f"experiments/{evaluation_id}/artifacts/phase9-development-attempt.json",
                "maximum_bytes": 16_384,
                "no_clobber": True,
                "commit_point": "known_present",
            },
            "Evidence policy": {
                "schema_version": 1,
                "completed_relative_path": f"experiments/{evaluation_id}/artifacts/phase9-development-evidence.json",
                "failure_relative_path": f"experiments/{evaluation_id}/artifacts/phase9-development-failure.json",
                "maximum_bytes": 1_048_576,
                "canonical_json": "utf8_ensure_ascii_sorted_keys_indent_2_allow_nan_false_final_lf",
                "generated_text_storage": "ignored_immutable_machine_readable_evidence",
                "tracked_record_storage": "references_and_hashes_without_generated_or_sealed_text",
            },
            "Sealed-test access": "none",
            "Success policy": experiment.DEVELOPMENT_SUCCESS_POLICY,
            "Next gate": "independent_pre_registration_review",
        })
        record = experiment._Record(
            evaluation_id, "development_plan", values, b"plan", "a" * 64
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(experiment, "_live_runtime_mapping", return_value={"runtime": True}),
            patch.object(experiment, "_validate_implementation_authority", return_value={"implementation_commit": "b" * 40}),
        ):
            root = Path(directory).resolve()
            experiment._validate_plan_record(root, record)
            for field in (
                "Phase 8 authority", "Checkpoint authority", "Phase 9 contract authority",
                "Runtime identity", "Dataset authority", "Tokenizer authority",
                "Model authority", "Evaluation configuration", "Qualitative matrix",
                "Attempt marker policy", "Evidence policy", "Success policy",
            ):
                candidate_values = copy.deepcopy(values)
                candidate_values[field] = None
                candidate = experiment._Record(
                    evaluation_id, "development_plan", candidate_values, b"plan", "a" * 64
                )
                with self.subTest(field=field), self.assertRaises((Phase9GovernanceError, Phase9ContractError)):
                    experiment._validate_plan_record(root, candidate)

    def test_post_load_model_mutation_is_rejected_and_rng_restored(self) -> None:
        source_model, _ = _model()
        payload = _epoch10_payload(source_model)
        reference = {
            "run_id": checkpoint.RUN_ID,
            "role": "best_validation",
            "logical_id": "epoch-0010",
            "epoch": 10,
            "validation_loss_hex": checkpoint.VALIDATION_LOSS_HEX,
            "sha256": checkpoint.CHECKPOINT_SHA256,
            "relative_path": f"objects/{checkpoint.CHECKPOINT_SHA256}.pt",
        }
        real_load = MiniGPT.load_state_dict

        def mutating_load(model, state, strict=True):
            result = real_load(model, state, strict=strict)
            with torch.no_grad():
                next(model.parameters()).add_(1.0)
            return result

        before = torch.get_rng_state().clone()
        with (
            patch.object(checkpoint, "_validate_repository", return_value=REPOSITORY_ROOT),
            patch.object(checkpoint, "_load_validated_artifacts", return_value=(reference, payload)),
            patch.object(MiniGPT, "load_state_dict", new=mutating_load),
        ):
            with self.assertRaises(Phase9ContractError):
                inference.load_phase9_inference_bundle(REPOSITORY_ROOT)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_all_evidence_artifact_validators_accept_exact_and_reject_mutation(self) -> None:
        evidence, marker, authority, configuration = self._valid_development_evidence()
        failure = copy.deepcopy(evidence)
        failure.update({
            "kind": "phase9_development_failure",
            "status": "failed",
            "started_at_utc": "2026-09-13T12:00:00Z",
            "finished_at_utc": "2026-09-13T12:00:01Z",
            "failure": {
                "stage": "sample_to_be_full",
                "exception_type": "Phase9ContractError",
                "invariant": "phase9.generation.output",
                "safe_facts": {
                    "sample_id": "to-be-full",
                    "field": "output",
                    "completed_token_count": 0,
                },
            },
        })
        experiment._validate_development_evidence_mapping(
            failure,
            evaluation_id=evidence["evaluation_id"],
            marker_sha256=marker.sha256,
            authority=authority,
            configuration=configuration,
            completed=False,
        )
        bad_failure = copy.deepcopy(failure)
        bad_failure["failure"]["safe_facts"]["secret"] = "forbidden"
        with self.assertRaises(Phase9ContractError):
            experiment._validate_development_evidence_mapping(
                bad_failure,
                evaluation_id=evidence["evaluation_id"],
                marker_sha256=marker.sha256,
                authority=authority,
                configuration=configuration,
                completed=False,
            )

        digest = hashlib.sha256(experiment._canonical_json(evidence)).hexdigest()
        evidence_reference = Phase9EvidenceReference(
            evidence["evaluation_id"],
            "phase9_development",
            "completed",
            f"experiments/{evidence['evaluation_id']}/artifacts/phase9-development-evidence.json",
            len(experiment._canonical_json(evidence)),
            digest,
        )
        baseline_reference = experiment._expected_baseline_result_reference(
            evidence["laplace_unigram_baseline"], evidence_reference
        )
        experiment._validate_laplace_result_reference(
            baseline_reference, evidence["evaluation_id"]
        )
        sample_reference = experiment._expected_sample_reference(
            evidence["samples"][0], 0, evidence_reference
        )
        experiment._validate_sample_reference(
            sample_reference, evidence["evaluation_id"], 0
        )

        test_metrics = {
            "model_test": dict(evidence["model_validation"], split="test"),
            "uniform_test": dict(evidence["uniform_validation"], split="test"),
            "laplace_unigram_test": dict(evidence["laplace_unigram_validation"], split="test"),
        }
        sealed_authority = {
            **authority,
            "development_result_record_sha256": "4" * 64,
            "development_result_commit": "4" * 40,
            "development_evidence_sha256": digest,
            "sealed_plan_record_sha256": "5" * 64,
            "sealed_pre_registration_commit": "5" * 40,
            "sealed_authorization_record_sha256": "6" * 64,
            "sealed_authorization_commit": "6" * 40,
        }
        sealed = {
            "schema_version": 1,
            "evaluation_id": evidence["evaluation_id"],
            "kind": "phase9_sealed_test",
            "status": "completed",
            "recorded_at_utc": "2026-09-13T12:00:02Z",
            "authority": sealed_authority,
            "access_marker_sha256": "d" * 64,
            "development_evidence_sha256": digest,
            **test_metrics,
            "sealed_test_access": "one_shot_completed",
            "failure": None,
            "limitations": list(experiment.LIMITATIONS),
        }
        experiment._validate_sealed_evidence_mapping(
            sealed,
            evaluation_id=evidence["evaluation_id"],
            marker_sha256="d" * 64,
            development_evidence_sha256=digest,
            authority=sealed_authority,
            completed=True,
        )
        sealed_failure = copy.deepcopy(sealed)
        sealed_failure.update({
            "kind": "phase9_sealed_test_failure",
            "status": "failed",
            "started_at_utc": "2026-09-13T12:00:00Z",
            "finished_at_utc": "2026-09-13T12:00:02Z",
            "model_test": None,
            "uniform_test": None,
            "laplace_unigram_test": None,
            "sealed_test_access": "one_shot_consumed_failed",
            "failure": {
                "stage": "sealed_corpus_open",
                "exception_type": "Phase9GovernanceError",
                "invariant": "phase9.governance.dataset",
                "safe_facts": {"field": "path"},
            },
        })
        experiment._validate_sealed_evidence_mapping(
            sealed_failure,
            evaluation_id=evidence["evaluation_id"],
            marker_sha256="d" * 64,
            development_evidence_sha256=digest,
            authority=sealed_authority,
            completed=False,
        )
        sealed_failure["sealed_test_access"] = "one_shot_completed"
        with self.assertRaises(Phase9ContractError):
            experiment._validate_sealed_evidence_mapping(
                sealed_failure,
                evaluation_id=evidence["evaluation_id"],
                marker_sha256="d" * 64,
                development_evidence_sha256=digest,
                authority=sealed_authority,
                completed=False,
            )

    def test_generation_accepts_one_and_1024_without_early_stop(self) -> None:
        bundle = _bundle()

        def fake_forward(_model, token_ids):
            logits = torch.zeros((len(token_ids), 81), dtype=torch.float32)
            logits[-1, 0] = 1.0
            return logits

        with patch.object(MiniGPT, "forward", new=fake_forward):
            for count in (1, 1024):
                result = inference.generate_phase9_text(
                    bundle,
                    "A",
                    generated_token_count=count,
                    mode="greedy",
                    temperature=None,
                    top_k=None,
                    seed=None,
                )
                self.assertEqual(len(result.generated_token_ids), count)
                self.assertEqual(len(result.generated_text), count)

    def test_complete_categorical_setting_boundaries(self) -> None:
        bundle = _bundle()
        logits = torch.zeros(81, dtype=torch.float32)
        for top_k in (1, 80, 81):
            probabilities = inference.build_categorical_probabilities(
                logits, temperature=1.0, top_k=top_k
            )
            self.assertEqual(int(torch.count_nonzero(probabilities)), top_k)
        for temperature in (float("nan"), float("inf"), float("-inf"), 0.0, -1.0):
            with self.assertRaises(Phase9ContractError):
                inference.generate_phase9_text(
                    bundle, "A", generated_token_count=0, mode="categorical",
                    temperature=temperature, top_k=81, seed=0,
                )
        for seed in (0, 9_223_372_036_854_775_807):
            result = inference.generate_phase9_text(
                bundle, "A", generated_token_count=0, mode="categorical",
                temperature=1.0, top_k=81, seed=seed,
            )
            self.assertEqual(result.seed, seed)

    def test_sealed_result_record_validator_covers_terminal_states(self) -> None:
        evaluation_id = "EXP-20260913-01"
        marker = {
            "evaluation_id": evaluation_id,
            "kind": "phase9_sealed_test_access",
            "status": "consumed",
            "relative_path": f"experiments/{evaluation_id}/artifacts/phase9-sealed-test-access.json",
            "byte_count": 100,
            "sha256": "a" * 64,
        }
        evidence = {
            "evaluation_id": evaluation_id,
            "kind": "phase9_sealed_test",
            "status": "completed",
            "relative_path": f"experiments/{evaluation_id}/artifacts/phase9-sealed-test-evidence.json",
            "byte_count": 100,
            "sha256": "b" * 64,
        }
        metric = {
            "predictor": "mini_gpt_best_validation",
            "split": "test", "window_sha256": "c" * 64,
            "nll": 1.0, "nll_hex": (1.0).hex(),
            "perplexity": math.e, "perplexity_hex": math.e.hex(),
            "correct_count": 1, "target_count": 2,
            "top1_accuracy": 0.5, "top1_accuracy_hex": (0.5).hex(),
            "window_count": 1,
        }
        values = _values("sealed_test_result", evaluation_id)
        values.update({
            "Scope": "sealed_test", "Status": "completed",
            "Recorded at UTC": "2026-09-13T12:00:02Z",
            "Started at UTC": "2026-09-13T12:00:00Z",
            "Finished at UTC": "2026-09-13T12:00:01Z",
            "Sealed plan record SHA-256": "d" * 64,
            "Sealed pre-registration commit": "e" * 40,
            "Sealed authorization record SHA-256": "f" * 64,
            "Sealed authorization commit": "1" * 40,
            "Access marker": marker, "Sealed evidence": evidence,
            "Model test": metric,
            "Uniform test": dict(metric, predictor="uniform_81"),
            "Laplace unigram test": dict(metric, predictor="training_laplace_unigram_add_one"),
            "Sealed-test access": "one_shot_completed", "Failure": None,
            "Observations": [], "Conclusions": [],
            "Limitations": list(experiment.LIMITATIONS),
            "Next gate": "independent_sealed_result_review",
        })
        record = experiment._Record(
            evaluation_id, "sealed_test_result", values, b"result", "2" * 64
        )
        experiment._validate_sealed_result_record(record)
        failed = copy.deepcopy(values)
        failed.update({
            "Status": "failed",
            "Sealed evidence": dict(
                evidence,
                kind="phase9_sealed_test_failure",
                status="failed",
                relative_path=f"experiments/{evaluation_id}/artifacts/phase9-sealed-test-failure.json",
            ),
            "Model test": None, "Uniform test": None, "Laplace unigram test": None,
            "Sealed-test access": "one_shot_consumed_failed",
            "Failure": {
                "stage": "sealed_corpus_open",
                "exception_type": "Phase9GovernanceError",
                "invariant": "phase9.governance.dataset",
                "safe_facts": {"field": "path"},
            },
        })
        experiment._validate_sealed_result_record(
            experiment._Record(
                evaluation_id, "sealed_test_result", failed, b"failed", "3" * 64
            )
        )
        uncertain = copy.deepcopy(failed)
        uncertain.update({
            "Status": "uncertain", "Access marker": None, "Sealed evidence": None,
            "Sealed-test access": "one_shot_consumed_uncertain",
            "Failure": {
                "stage": "attempt_uncertain", "exception_type": "ProcessInterruption",
                "invariant": "phase9.evidence.amendment",
                "safe_facts": {"attempt_marker_present": False, "final_evidence_present": False},
            },
        })
        experiment._validate_sealed_result_record(
            experiment._Record(
                evaluation_id, "sealed_test_result", uncertain, b"uncertain", "4" * 64
            )
        )
        broken = copy.deepcopy(values)
        broken["Sealed-test access"] = "one_shot_consumed_failed"
        with self.assertRaises(Phase9GovernanceError):
            experiment._validate_sealed_result_record(
                experiment._Record(
                    evaluation_id, "sealed_test_result", broken, b"result", "2" * 64
                )
            )

    def test_guarded_synthetic_sealed_reader_opens_once_and_rejects_symlink(self) -> None:
        content = b"A\n"
        expected = {
            "relative_path": "data/processed/shakespeare-eight-play/test/twelfth-night.txt",
            "byte_count": len(content),
            "code_point_count": 2,
            "line_count": 1,
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            target = root / expected["relative_path"]
            target.parent.mkdir(parents=True)
            target.write_bytes(content)
            real_open = experiment.os.open
            file_opens = 0

            def traced_open(path, *args, **kwargs):
                nonlocal file_opens
                if path == "twelfth-night.txt":
                    file_opens += 1
                return real_open(path, *args, **kwargs)

            with patch.object(experiment.os, "open", side_effect=traced_open):
                self.assertEqual(experiment._read_sealed_test_once(root, expected), "A\n")
            self.assertEqual(file_opens, 1)
            target.rename(target.with_name("real.txt"))
            target.symlink_to(target.with_name("real.txt"))
            with self.assertRaises(Phase9GovernanceError):
                experiment._read_sealed_test_once(root, expected)

    def test_complete_runtime_envelope_fails_before_checkpoint_access(self) -> None:
        for field, accepted in checkpoint.ACCEPTED_RUNTIME_IDENTITY.items():
            observed = copy.deepcopy(checkpoint.ACCEPTED_RUNTIME_IDENTITY)
            if type(accepted) is bool:
                observed[field] = not accepted
            elif type(accepted) is int:
                observed[field] = accepted + 1
            else:
                observed[field] = f"{accepted}-changed"
            with (
                self.subTest(field=field),
                patch.object(checkpoint, "_live_runtime_mapping", return_value=observed),
            ):
                with self.assertRaises(Phase9GovernanceError):
                    checkpoint._validate_runtime(REPOSITORY_ROOT)
        with (
            patch.object(
                checkpoint,
                "_validate_repository",
                side_effect=Phase9GovernanceError(
                    "phase9.governance.runtime", field="runtime"
                ),
            ),
            patch.object(checkpoint, "_load_validated_artifacts") as artifacts,
        ):
            with self.assertRaises(Phase9GovernanceError):
                inference.load_phase9_inference_bundle(REPOSITORY_ROOT)
        artifacts.assert_not_called()

    def test_synthetic_loader_uses_real_descriptor_artifact_validation(self) -> None:
        model, _ = _model()
        payload = _payload(model)
        payload_bytes = _serialized_payload(payload)
        digest = hashlib.sha256(payload_bytes).hexdigest()
        reference = {
            "run_id": checkpoint.RUN_ID,
            "logical_id": "epoch-0001",
            "epoch": 1,
            "validation_loss_hex": (1.0).hex(),
            "sha256": digest,
            "relative_path": f"objects/{digest}.pt",
        }
        catalog = {
            "schema_version": 1,
            "run_id": checkpoint.RUN_ID,
            "latest": {**reference, "role": "latest"},
            "best_validation": {**reference, "role": "best_validation"},
        }
        catalog_bytes = checkpoint._canonical_catalog(catalog)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            run = root / f"checkpoints/phase8/{checkpoint.RUN_ID}"
            objects = run / "objects"
            objects.mkdir(parents=True)
            (run / "checkpoint-catalog.json").write_bytes(catalog_bytes)
            (objects / f"{digest}.pt").write_bytes(payload_bytes)
            selected, loaded = checkpoint._load_validated_artifacts(
                root,
                expected_catalog_sha256=hashlib.sha256(catalog_bytes).hexdigest(),
                expected_checkpoint_sha256=digest,
                expected_runtime=payload["configuration"]["runtime"],
            )
        self.assertEqual(selected["role"], "best_validation")
        self.assertEqual(loaded["run"]["logical_id"], "epoch-0001")

    def test_unmocked_synthetic_git_record_introduction_and_separation(self) -> None:
        evaluation_id = "EXP-20260913-01"
        plan_values = _values("development_plan", evaluation_id)
        plan_raw = _render_record(evaluation_id, "development_plan", plan_values)
        plan_record = experiment._parse_records(plan_raw)[0]
        authorization_values = _values("development_authorization", evaluation_id)
        authorization_values["Development plan record SHA-256"] = plan_record.sha256
        authorization_raw = _render_record(
            evaluation_id, "development_authorization", authorization_values
        )
        authorization_record = experiment._parse_records(authorization_raw)[0]
        result_values = _values("development_result", evaluation_id)
        result_raw = _render_record(evaluation_id, "development_result", result_values)
        result_record = experiment._parse_records(result_raw)[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            subprocess.run(("git", "init", "-q"), cwd=root, check=True)
            subprocess.run(("git", "config", "user.email", "synthetic@example.invalid"), cwd=root, check=True)
            subprocess.run(("git", "config", "user.name", "Synthetic"), cwd=root, check=True)
            log = root / "EXPERIMENT_LOG.md"
            log.write_text("# Synthetic\n\n", encoding="utf-8")
            subprocess.run(("git", "add", "EXPERIMENT_LOG.md"), cwd=root, check=True)
            subprocess.run(("git", "commit", "-q", "-m", "base"), cwd=root, check=True)
            commits = []
            for name, raw in (
                ("plan", plan_raw),
                ("authorization", authorization_raw),
                ("result", result_raw),
            ):
                with log.open("ab") as stream:
                    stream.write(raw)
                subprocess.run(("git", "add", "EXPERIMENT_LOG.md"), cwd=root, check=True)
                subprocess.run(("git", "commit", "-q", "-m", name), cwd=root, check=True)
                commits.append(
                    subprocess.run(
                        ("git", "rev-parse", "HEAD"), cwd=root, check=True,
                        capture_output=True, text=True,
                    ).stdout.strip()
                )
            for record, commit in zip(
                (plan_record, authorization_record, result_record), commits, strict=True
            ):
                self.assertTrue(
                    experiment._record_introduced_by_commit(root, record, commit)
                )
            self.assertFalse(
                experiment._record_introduced_by_commit(root, plan_record, commits[-1])
            )
            for earlier, later in zip(commits, commits[1:]):
                subprocess.run(
                    ("git", "merge-base", "--is-ancestor", earlier, later),
                    cwd=root, check=True,
                )

    def test_payload_bool_and_metric_count_mutations_fail_independently(self) -> None:
        model, _ = _model()
        mutations = (
            lambda value: value["metric_state"].__setitem__("best_validation_epoch", True),
            lambda value: value["metric_state"].__setitem__(
                "epoch_validation",
                (dict(value["metric_state"]["epoch_validation"][0], target_count=9),),
            ),
            lambda value: value["metric_state"].__setitem__(
                "current_validation",
                dict(value["metric_state"]["current_validation"], window_count=5),
            ),
            lambda value: value["progress"].__setitem__("completed_epochs", True),
            lambda value: value["progress"].__setitem__("targets_processed", 15),
            lambda value: value["optimizer_state"]["param_groups"][0].__setitem__("amsgrad", 0),
            lambda value: value["optimizer_state"]["state"][0].__setitem__("exp_avg", torch.zeros(1)),
            lambda value: value["random_state"].__setitem__("order_generator", torch.zeros(4, dtype=torch.float32)),
            lambda value: value["mode_state"].__setitem__("named_modules", (("", True),)),
            lambda value: value["resume_lineage"].__setitem__("resume_count", False),
        )
        for mutate in mutations:
            candidate = copy.deepcopy(_payload(model))
            mutate(candidate)
            with self.subTest(mutation=mutate), self.assertRaises(inference.Phase9CheckpointError):
                checkpoint._validate_payload(candidate)

    def test_metric_validator_rejects_plausible_cross_field_inconsistency(self) -> None:
        metric = {
            "predictor": "uniform_81", "split": "validation",
            "window_sha256": "a" * 64, "nll": math.log(81),
            "nll_hex": math.log(81).hex(),
            "perplexity": math.exp(math.log(81)),
            "perplexity_hex": math.exp(math.log(81)).hex(),
            "correct_count": 1, "target_count": 4,
            "top1_accuracy": 0.25, "top1_accuracy_hex": (0.25).hex(),
            "window_count": 1,
        }
        experiment._validate_metric_mapping(
            metric, split="validation", predictor="uniform_81"
        )
        for field, replacement in (
            ("perplexity", 81.0),
            ("top1_accuracy", 0.5),
            ("correct_count", 2),
            ("nll_hex", (1.0).hex()),
        ):
            candidate = copy.deepcopy(metric)
            candidate[field] = replacement
            if field in ("perplexity", "top1_accuracy"):
                candidate[f"{field}_hex"] = replacement.hex()
            with self.subTest(field=field), self.assertRaises(Phase9ContractError):
                experiment._validate_metric_mapping(
                    candidate, split="validation", predictor="uniform_81"
                )

    def test_failure_validator_closed_table_rejects_every_adversarial_dimension(self) -> None:
        valid = {
            "stage": "sample_romeo_focused",
            "exception_type": "Phase9ContractError",
            "invariant": "phase9.generation.output",
            "safe_facts": {
                "sample_id": "romeo-focused", "field": "output",
                "completed_token_count": 10,
            },
        }
        experiment._validate_failure_mapping(valid, sealed=False)
        mutations = (
            lambda value: value.__setitem__("stage", "arbitrary_stage"),
            lambda value: value.__setitem__("exception_type", "RuntimeError"),
            lambda value: value.__setitem__("invariant", "phase9.arbitrary"),
            lambda value: value["safe_facts"].__setitem__("secret", "x"),
            lambda value: value["safe_facts"].__setitem__("field", "unknown"),
            lambda value: value["safe_facts"].__setitem__("sample_id", "unknown"),
            lambda value: value["safe_facts"].__setitem__("completed_token_count", True),
            lambda value: value["safe_facts"].__setitem__("completed_token_count", 257),
        )
        for mutate in mutations:
            candidate = copy.deepcopy(valid)
            mutate(candidate)
            with self.subTest(mutation=mutate), self.assertRaises(Phase9ContractError):
                experiment._validate_failure_mapping(candidate, sealed=False)
        publication = {
            "stage": "sealed_evidence_publication",
            "exception_type": "Phase9PublicationError",
            "invariant": "phase9.evidence.publication",
            "safe_facts": {
                "operation": "fsync_parent", "publication_state": "unknown"
            },
        }
        experiment._validate_failure_mapping(publication, sealed=True)
        for field, replacement in (
            ("operation", "rename"),
            ("publication_state", "known_present"),
        ):
            candidate = copy.deepcopy(publication)
            candidate["safe_facts"][field] = replacement
            with self.assertRaises(Phase9ContractError):
                experiment._validate_failure_mapping(candidate, sealed=True)
        corpus_failure = {
            "stage": "development_corpus",
            "exception_type": "Phase9GovernanceError",
            "invariant": "phase9.governance.dataset",
            "safe_facts": {"field": "manifest", "position": None},
        }
        experiment._validate_failure_mapping(corpus_failure, sealed=False)
        for position in (True, -1, "0"):
            candidate = copy.deepcopy(corpus_failure)
            candidate["safe_facts"]["position"] = position
            with self.assertRaises(Phase9ContractError):
                experiment._validate_failure_mapping(candidate, sealed=False)
        metric_failure = {
            "stage": "model_validation",
            "exception_type": "Phase9NumericalError",
            "invariant": "phase9.evaluation.metric",
            "safe_facts": {"field": "evidence", "split": "validation"},
        }
        experiment._validate_failure_mapping(metric_failure, sealed=False)
        for split in ("train", "test", 1):
            candidate = copy.deepcopy(metric_failure)
            candidate["safe_facts"]["split"] = split
            with self.assertRaises(Phase9ContractError):
                experiment._validate_failure_mapping(candidate, sealed=False)
        from sebgpt.tokenization.code_point import UnknownCodePointError

        tokenizer_failure = experiment._failure(
            "sample_romeo_greedy", UnknownCodePointError(ord("☃"), 0)
        )
        experiment._validate_failure_mapping(tokenizer_failure, sealed=False)

    def test_created_artifact_ancestors_fsync_parent_then_revalidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            events: list[tuple[str, str]] = []
            pending: list[str] = []
            real_mkdir = experiment.os.mkdir
            real_fsync = experiment.os.fsync
            real_stat = experiment.os.stat

            def traced_mkdir(name, *args, **kwargs):
                events.append(("mkdir", name))
                pending.append(name)
                return real_mkdir(name, *args, **kwargs)

            def traced_fsync(fd):
                if pending:
                    events.append(("fsync_parent", pending[-1]))
                return real_fsync(fd)

            def traced_stat(name, *args, **kwargs):
                if pending and name == pending[-1]:
                    events.append(("revalidate", name))
                    pending.pop()
                return real_stat(name, *args, **kwargs)

            with (
                patch.object(experiment.os, "mkdir", side_effect=traced_mkdir),
                patch.object(experiment.os, "fsync", side_effect=traced_fsync),
                patch.object(experiment.os, "stat", side_effect=traced_stat),
            ):
                fds, _ = experiment._open_artifact_directory(
                    root, "EXP-20260913-01", create=True
                )
            for fd in reversed(fds):
                experiment.os.close(fd)
            self.assertEqual(
                events,
                [
                    ("mkdir", "experiments"), ("fsync_parent", "experiments"),
                    ("revalidate", "experiments"),
                    ("mkdir", "EXP-20260913-01"),
                    ("fsync_parent", "EXP-20260913-01"),
                    ("revalidate", "EXP-20260913-01"),
                    ("mkdir", "artifacts"), ("fsync_parent", "artifacts"),
                    ("revalidate", "artifacts"),
                ],
            )
        content = experiment._canonical_json({"ok": True})
        for operation in ("fsync", "stat"):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                with patch.object(experiment.os, operation, side_effect=OSError):
                    publication = experiment._publish(
                        root, "EXP-20260913-01",
                        filename="phase9-development-attempt.json",
                        kind="phase9_development_attempt", status="consumed",
                        content=content,
                    )
                self.assertEqual(publication.state, experiment.UNKNOWN)
                self.assertFalse(
                    (root / "experiments/EXP-20260913-01/artifacts/phase9-development-attempt.json").exists()
                )

    def test_zero_token_generation_rejects_every_model_structure_corruption(self) -> None:
        def transposed(model):
            model.head.output_weight = torch.nn.Parameter(
                model.head.output_weight.detach().t().contiguous()
            )

        def missing(model):
            del model.head._parameters["output_bias"]

        def extra(model):
            model.head.register_parameter("extra", torch.nn.Parameter(torch.zeros(1)))

        def wrong_name(model):
            parameter = model.head._parameters.pop("output_bias")
            model.head.register_parameter("renamed_bias", parameter)

        def wrong_shape(model):
            model.head.output_bias = torch.nn.Parameter(torch.zeros(82))

        def alias(model):
            model.blocks[0].norm1.beta = model.blocks[0].norm1.gamma

        def wrong_dtype(model):
            model.head.output_bias = torch.nn.Parameter(
                model.head.output_bias.detach().to(torch.float64)
            )

        def wrong_device(model):
            model.head.output_bias = torch.nn.Parameter(
                torch.empty(81, device="meta")
            )

        def wrong_mode(model):
            model.blocks[0].attention_dropout.train()

        def wrong_gradient(model):
            model.head.output_bias.grad = torch.zeros_like(model.head.output_bias)

        def wrong_dropout_state(model):
            model.blocks[0].attention_dropout._generator.manual_seed(99)

        for corrupt in (
            transposed, missing, extra, wrong_name, wrong_shape, alias, wrong_dtype,
            wrong_device, wrong_mode, wrong_gradient, wrong_dropout_state,
        ):
            bundle = _bundle()
            corrupt(bundle.model)
            with (
                self.subTest(corruption=corrupt.__name__),
                patch.object(CodePointTokenizer, "encode", side_effect=AssertionError("prompt processed")),
            ):
                with self.assertRaises((Phase9TypeError, Phase9ContractError)):
                    inference.generate_phase9_text(
                        bundle, "A", generated_token_count=0, mode="greedy",
                        temperature=None, top_k=None, seed=None,
                    )

    def test_evaluation_uses_shared_structure_validation_before_windows(self) -> None:
        model, _ = _model()
        model.head.output_weight = torch.nn.Parameter(
            model.head.output_weight.detach().t().contiguous()
        )
        with patch(
            "sebgpt.inference.phase9_evaluation._validate_windows",
            side_effect=AssertionError("windows processed"),
        ):
            with self.assertRaises(Phase9ContractError):
                inference.evaluate_phase9_model(
                    model,
                    (inference.Phase9Window("v", 0, "validation", 0, (0,), (1,)),),
                    predictor="synthetic",
                    split="validation",
                )

    def test_sealed_read_failure_stage_distinguishes_open_from_validation(self) -> None:
        records, result, _plan, _authorization, development, digest = self._sealed_launch_fixture()
        marker_reference = Phase9EvidenceReference(
            result.evaluation_id,
            "phase9_sealed_test_access",
            "consumed",
            f"experiments/{result.evaluation_id}/artifacts/phase9-sealed-test-access.json",
            100,
            "a" * 64,
        )
        failure_reference = Phase9EvidenceReference(
            result.evaluation_id,
            "phase9_sealed_test_failure",
            "failed",
            f"experiments/{result.evaluation_id}/artifacts/phase9-sealed-test-failure.json",
            100,
            "b" * 64,
        )
        for field, expected_stage in (
            ("path", "sealed_corpus_open"),
            ("bytes", "sealed_corpus_validation"),
            ("utf8", "sealed_corpus_validation"),
            ("normalization", "sealed_corpus_validation"),
        ):
            captured: list[dict[str, object]] = []

            def terminal(*_args, **kwargs):
                captured.append(json.loads(kwargs["content"].decode("utf-8")))
                return experiment._Publication(
                    experiment.KNOWN_PRESENT, failure_reference
                )

            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                (root / "EXPERIMENT_LOG.md").write_bytes(b"synthetic")
                with (
                    patch.object(experiment, "_validate_repository", return_value=root),
                    patch.object(experiment, "_parse_selected_chain", return_value=records),
                    patch.object(experiment, "_validate_predecessor_chain"),
                    patch.object(experiment, "_read_evidence_artifact", return_value=development),
                    patch.object(experiment, "_validate_sealed_launch", return_value=("c" * 40, development["authority"])),
                    patch.object(experiment, "_sealed_test_metadata", return_value={}),
                    patch.object(experiment, "_validate_baseline_mapping"),
                    patch.object(experiment, "_make_laplace_unigram", return_value=object()),
                    patch.object(experiment, "load_phase9_inference_bundle", return_value=_bundle()),
                    patch.object(experiment, "_require_artifacts_absent"),
                    patch.object(experiment, "_validate_access_marker"),
                    patch.object(experiment, "_serialized_sealed_access_publication", return_value=experiment._Publication(experiment.KNOWN_PRESENT, marker_reference)),
                    patch.object(experiment, "_read_sealed_test_once", side_effect=Phase9GovernanceError("phase9.governance.dataset", field=field)),
                    patch.object(experiment, "_publish_terminal", side_effect=terminal),
                ):
                    returned = inference.run_fixed_phase9_sealed_test_evaluation(
                        root, evaluation_id=result.evaluation_id
                    )
            self.assertIs(returned, failure_reference)
            self.assertEqual(captured[0]["failure"]["stage"], expected_stage)

    def test_failure_invariants_are_closed_exact_sets_without_prefix_acceptance(self) -> None:
        accepted = (
            ({"stage": "inference_load", "exception_type": "Phase9CheckpointError", "invariant": "phase9.checkpoint.schema", "safe_facts": {"field": "payload"}}, False),
            ({"stage": "development_corpus", "exception_type": "Phase9GovernanceError", "invariant": "phase9.governance.dataset", "safe_facts": {"field": "manifest", "position": None}}, False),
            ({"stage": "training_window_build", "exception_type": "Phase9ContractError", "invariant": "phase9.contract.window", "safe_facts": {"field": "documents"}}, False),
            ({"stage": "laplace_unigram_fit", "exception_type": "Phase9NumericalError", "invariant": "phase9.baseline.probabilities", "safe_facts": {"field": "probabilities"}}, False),
            ({"stage": "model_validation", "exception_type": "Phase4ContractError", "invariant": "loss.logits.rank", "safe_facts": {"field": "logits", "split": "validation"}}, False),
            ({"stage": "model_validation", "exception_type": "MiniGPTNumericalError", "invariant": "mini_gpt.output.finite", "safe_facts": {"field": "output", "split": "validation"}}, False),
            ({"stage": "sample_romeo_greedy", "exception_type": "Phase9ContractError", "invariant": "phase9.generation.output", "safe_facts": {"sample_id": "romeo-greedy", "field": "output", "completed_token_count": 0}}, False),
            ({"stage": "sample_romeo_greedy", "exception_type": "Phase9NumericalError", "invariant": "phase9.sampling.probabilities", "safe_facts": {"sample_id": "romeo-greedy", "field": "probabilities", "completed_token_count": 0}}, False),
            ({"stage": "sample_romeo_greedy", "exception_type": "UnknownCodePointError", "invariant": "encode.unknown_code_point", "safe_facts": {"sample_id": "romeo-greedy", "field": "prompt", "completed_token_count": 0}}, False),
            ({"stage": "sealed_evidence_publication", "exception_type": "Phase9PublicationError", "invariant": "phase9.evidence.publication", "safe_facts": {"operation": "fsync_parent", "publication_state": "unknown"}}, True),
        )
        for failure, sealed in accepted:
            experiment._validate_failure_mapping(failure, sealed=sealed)
        base = copy.deepcopy(accepted[6][0])
        for invariant in (
            "phase9.generation.not_real",
            "phase9.generation.output.extra",
            "Phase9.generation.output",
            "phase9.generation",
            "",
            "mini_gpt.not_real",
            "mini_gpt.output.finite.extra",
            "loss.not_real",
            "encode.not_real",
        ):
            candidate = copy.deepcopy(base)
            candidate["invariant"] = invariant
            with self.subTest(invariant=invariant), self.assertRaises(Phase9ContractError):
                experiment._validate_failure_mapping(candidate, sealed=False)
        wrong_stage = copy.deepcopy(base)
        wrong_stage.update({
            "stage": "cross_predictor_comparison",
            "safe_facts": {"field": "output", "split": "validation"},
        })
        with self.assertRaises(Phase9ContractError):
            experiment._validate_failure_mapping(wrong_stage, sealed=False)
        invented_children = (
            {
                "stage": "model_validation",
                "exception_type": "MiniGPTNumericalError",
                "invariant": "mini_gpt.output.finite.extra",
                "safe_facts": {"field": "output", "split": "validation"},
            },
            {
                "stage": "model_validation",
                "exception_type": "Phase4ContractError",
                "invariant": "loss.logits.rank.extra",
                "safe_facts": {"field": "logits", "split": "validation"},
            },
            {
                "stage": "sample_romeo_greedy",
                "exception_type": "UnknownCodePointError",
                "invariant": "encode.unknown_code_point.extra",
                "safe_facts": {"sample_id": "romeo-greedy", "field": "prompt", "completed_token_count": 0},
            },
        )
        for candidate in invented_children:
            with self.assertRaises(Phase9ContractError):
                experiment._validate_failure_mapping(candidate, sealed=False)

    def test_sample_failure_stages_and_sample_ids_are_one_closed_pair_mapping(self) -> None:
        expected = {
            "sample_romeo_greedy": "romeo-greedy",
            "sample_romeo_focused": "romeo-focused",
            "sample_romeo_full": "romeo-full",
            "sample_to_be_greedy": "to-be-greedy",
            "sample_to_be_focused": "to-be-focused",
            "sample_to_be_full": "to-be-full",
        }
        self.assertEqual(experiment._SAMPLE_STAGE_TO_ID, expected)
        for stage, sample_id in expected.items():
            failure = {
                "stage": stage,
                "exception_type": "Phase9ContractError",
                "invariant": "phase9.generation.output",
                "safe_facts": {
                    "sample_id": sample_id,
                    "field": "output",
                    "completed_token_count": 0,
                },
            }
            experiment._validate_failure_mapping(failure, sealed=False)
            normalized = experiment._failure(
                stage,
                Phase9ContractError("phase9.generation.output", field="output"),
            )
            self.assertEqual(normalized["safe_facts"]["sample_id"], sample_id)

        wrong_pair = {
            "stage": "sample_romeo_greedy",
            "exception_type": "Phase9ContractError",
            "invariant": "phase9.generation.output",
            "safe_facts": {
                "sample_id": "romeo-focused",
                "field": "output",
                "completed_token_count": 0,
            },
        }
        with self.assertRaises(Phase9ContractError):
            experiment._validate_failure_mapping(wrong_pair, sealed=False)

        for stage in (
            "sample_not_real",
            "sample_romeo_greedy_extra",
            "sample_romeo",
            "Sample_romeo_greedy",
            "",
            "model_validation",
        ):
            candidate = copy.deepcopy(wrong_pair)
            candidate["stage"] = stage
            candidate["safe_facts"]["sample_id"] = "romeo-greedy"
            with self.subTest(stage=stage), self.assertRaises(Phase9ContractError):
                experiment._validate_failure_mapping(candidate, sealed=False)

    def test_reused_publication_ancestors_always_reestablish_durability(self) -> None:
        content = experiment._canonical_json({"ok": True})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            real_fsync = experiment.os.fsync
            calls = 0

            def fail_third(fd):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise OSError
                return real_fsync(fd)

            with patch.object(experiment.os, "fsync", side_effect=fail_third):
                first = experiment._publish(
                    root, "EXP-20260913-01",
                    filename="phase9-development-attempt.json",
                    kind="phase9_development_attempt", status="consumed", content=content,
                )
            self.assertEqual(first.state, experiment.UNKNOWN)
            artifacts = root / "experiments/EXP-20260913-01/artifacts"
            self.assertTrue(artifacts.is_dir())
            reuse_events: list[str] = []

            def trace_fsync(fd):
                reuse_events.append("fsync_parent")
                return real_fsync(fd)

            with patch.object(experiment.os, "fsync", side_effect=trace_fsync):
                second = experiment._publish(
                    root, "EXP-20260913-01",
                    filename="phase9-development-attempt.json",
                    kind="phase9_development_attempt", status="consumed", content=content,
                )
            self.assertEqual(second.state, experiment.KNOWN_PRESENT)
            self.assertGreaterEqual(len(reuse_events), 5)

        for operation in ("fsync", "stat"):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                (root / "experiments/EXP-20260913-01/artifacts").mkdir(parents=True)
                with patch.object(experiment.os, operation, side_effect=OSError):
                    result = experiment._publish(
                        root, "EXP-20260913-01",
                        filename="phase9-development-attempt.json",
                        kind="phase9_development_attempt", status="consumed", content=content,
                    )
                self.assertEqual(result.state, experiment.UNKNOWN)
                self.assertFalse(
                    (root / "experiments/EXP-20260913-01/artifacts/phase9-development-attempt.json").exists()
                )

    def test_reused_ancestor_order_and_swap_detection_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "experiments/EXP-20260913-01/artifacts").mkdir(parents=True)
            events: list[str] = []
            current: list[str] = []
            real_open = experiment.os.open
            real_fsync = experiment.os.fsync
            real_stat = experiment.os.stat

            def traced_open(path, *args, **kwargs):
                if isinstance(path, str) and path in (
                    "experiments", "EXP-20260913-01", "artifacts"
                ):
                    current[:] = [path]
                    events.append(f"open:{path}")
                return real_open(path, *args, **kwargs)

            def traced_fsync(fd):
                if current:
                    events.append(f"fsync:{current[0]}")
                return real_fsync(fd)

            def traced_stat(path, *args, **kwargs):
                if current and path == current[0]:
                    events.append(f"revalidate:{path}")
                    current.clear()
                return real_stat(path, *args, **kwargs)

            with (
                patch.object(experiment.os, "open", side_effect=traced_open),
                patch.object(experiment.os, "fsync", side_effect=traced_fsync),
                patch.object(experiment.os, "stat", side_effect=traced_stat),
            ):
                fds, _ = experiment._open_artifact_directory(
                    root, "EXP-20260913-01", create=True
                )
            for fd in reversed(fds):
                experiment.os.close(fd)
            self.assertEqual(
                events,
                [
                    "open:experiments", "fsync:experiments", "revalidate:experiments",
                    "open:EXP-20260913-01", "fsync:EXP-20260913-01",
                    "revalidate:EXP-20260913-01", "open:artifacts",
                    "fsync:artifacts", "revalidate:artifacts",
                ],
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            child = root / "experiments"
            child.mkdir()
            root_fd = experiment.os.open(
                root, experiment.os.O_RDONLY | experiment.os.O_DIRECTORY
            )
            real_stat = experiment.os.stat
            swapped = False

            def swap_before_revalidation(path, *args, **kwargs):
                nonlocal swapped
                if path == "experiments" and not swapped:
                    swapped = True
                    child.rename(root / "old-experiments")
                    child.mkdir()
                return real_stat(path, *args, **kwargs)

            try:
                with patch.object(
                    experiment.os, "stat", side_effect=swap_before_revalidation
                ):
                    with self.assertRaises(OSError):
                        experiment._open_child_directory(
                            root_fd, "experiments", create=True
                        )
            finally:
                experiment.os.close(root_fd)

    def test_exact_module_class_oracle_rejects_behavior_substitution(self) -> None:
        selectors = (
            ("root", lambda model: model),
            ("embedding", lambda model: model.representation),
            ("head", lambda model: model.head),
            ("block", lambda model: model.blocks[0]),
            ("attention", lambda model: model.blocks[0].attention),
            ("attention_head", lambda model: model.blocks[0].attention.heads[0]),
            ("normalization", lambda model: model.blocks[0].norm1),
            ("feed_forward", lambda model: model.blocks[0].feed_forward),
            ("dropout", lambda model: model.blocks[0].attention_dropout),
        )
        for name, selector in selectors:
            bundle = _bundle()
            module = selector(bundle.model)
            replacement_type = type(
                f"Synthetic{name.title()}Replacement", (type(module),), {}
            )
            module.__class__ = replacement_type
            with (
                self.subTest(module=name),
                patch.object(CodePointTokenizer, "encode", side_effect=AssertionError("prompt processed")),
            ):
                with self.assertRaises((Phase9TypeError, Phase9ContractError)):
                    inference.generate_phase9_text(
                        bundle, "A", generated_token_count=0, mode="greedy",
                        temperature=None, top_k=None, seed=None,
                    )
        model, _ = _model()
        model.head.__class__ = type(
            "SyntheticEvaluationHead", (type(model.head),), {}
        )
        with patch(
            "sebgpt.inference.phase9_evaluation._validate_windows",
            side_effect=AssertionError("arithmetic reached"),
        ):
            with self.assertRaises(Phase9ContractError):
                inference.evaluate_phase9_model(
                    model,
                    (inference.Phase9Window("v", 0, "validation", 0, (0,), (1,)),),
                    predictor="synthetic", split="validation",
                )

    def _sealed_serialization_fixture(self, root: Path):
        evaluation_id = "EXP-20260913-01"
        plan = experiment._Record(
            evaluation_id, "sealed_test_plan", {}, b"plan", "a" * 64
        )
        authorization = experiment._Record(
            evaluation_id,
            "sealed_test_authorization",
            {
                "Sealed pre-registration commit": "b" * 40,
                "Phase 9 implementation commit": "c" * 40,
            },
            b"authorization",
            "d" * 64,
        )
        marker = {
            "schema_version": 1,
            "evaluation_id": evaluation_id,
            "kind": "phase9_sealed_test_access",
            "sealed_plan_record_sha256": plan.sha256,
            "sealed_pre_registration_commit": "b" * 40,
            "sealed_authorization_record_sha256": authorization.sha256,
            "sealed_authorization_commit": "e" * 40,
            "phase9_contract_sha256": checkpoint.PHASE9_SPEC_SHA256,
            "phase9_implementation_commit": "c" * 40,
            "checkpoint_sha256": checkpoint.CHECKPOINT_SHA256,
            "development_evidence_sha256": "f" * 64,
            "created_at_utc": "2026-09-13T12:00:00Z",
            "consumed": True,
        }
        (root / "EXPERIMENT_LOG.md").write_bytes(b"# Synthetic\n\n")
        return evaluation_id, plan, authorization, marker

    def test_unknown_marker_journal_survives_every_uncertain_record_failure(self) -> None:
        real_append = experiment._append_sealed_uncertain_record
        for failure_mode in (
            "open", "write", "partial_write", "fsync_file", "fsync_parent"
        ):
            with self.subTest(failure_mode=failure_mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                evaluation_id, plan, authorization, marker = self._sealed_serialization_fixture(root)

                def failing_append(*args, **kwargs):
                    if failure_mode == "open":
                        real_open = experiment.os.open

                        def fail_log_open(path, *open_args, **open_kwargs):
                            if path == "EXPERIMENT_LOG.md":
                                raise OSError
                            return real_open(path, *open_args, **open_kwargs)

                        with patch.object(experiment.os, "open", side_effect=fail_log_open):
                            return real_append(*args, **kwargs)
                    if failure_mode == "write":
                        with patch.object(experiment.os, "write", side_effect=OSError):
                            return real_append(*args, **kwargs)
                    if failure_mode == "partial_write":
                        with patch.object(experiment.os, "write", side_effect=(1, OSError())):
                            return real_append(*args, **kwargs)
                    real_fsync = experiment.os.fsync
                    calls = 0

                    def fail_fsync(fd):
                        nonlocal calls
                        calls += 1
                        if (failure_mode == "fsync_file" and calls == 1) or (
                            failure_mode == "fsync_parent" and calls == 2
                        ):
                            raise OSError
                        return real_fsync(fd)

                    with patch.object(experiment.os, "fsync", side_effect=fail_fsync):
                        return real_append(*args, **kwargs)

                with (
                    patch.object(experiment.os, "link", side_effect=OSError),
                    patch.object(experiment, "_append_sealed_uncertain_record", side_effect=failing_append),
                ):
                    with self.assertRaises(Phase9PublicationError):
                        experiment._serialized_sealed_access_publication(
                            root, evaluation_id, plan, authorization,
                            authorization_commit="e" * 40, marker=marker,
                        )
                artifacts = root / f"experiments/{evaluation_id}/artifacts"
                journal = artifacts / f".phase9-sealed-test-access.json.{authorization.sha256}.tmp"
                self.assertTrue(journal.exists())
                self.assertFalse((artifacts / "phase9-sealed-test-access.json").exists())
                if failure_mode in ("open", "write", "partial_write", "fsync_file"):
                    self.assertEqual(
                        (root / "EXPERIMENT_LOG.md").read_bytes(),
                        b"# Synthetic\n\n",
                    )
                with patch.object(experiment.os, "link", side_effect=AssertionError("second marker publication")):
                    try:
                        second = experiment._serialized_sealed_access_publication(
                            root, evaluation_id, plan, authorization,
                            authorization_commit="e" * 40, marker=marker,
                        )
                    except Phase9GovernanceError:
                        second = None
                if second is not None:
                    self.assertEqual(second.state, experiment.UNKNOWN)
                self.assertFalse((artifacts / "phase9-sealed-test-access.json").exists())

    def test_concurrent_same_id_unknown_marker_has_one_publication_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            evaluation_id, plan, authorization, marker = self._sealed_serialization_fixture(root)
            link_calls = 0

            def fail_link(*_args, **_kwargs):
                nonlocal link_calls
                link_calls += 1
                raise OSError

            def invoke():
                try:
                    return experiment._serialized_sealed_access_publication(
                        root, evaluation_id, plan, authorization,
                        authorization_commit="e" * 40, marker=marker,
                    )
                except Phase9GovernanceError as error:
                    return error

            with (
                patch.object(experiment.os, "link", side_effect=fail_link),
                ThreadPoolExecutor(max_workers=2) as pool,
            ):
                futures = (pool.submit(invoke), pool.submit(invoke))
                results = tuple(future.result(timeout=5) for future in futures)
            self.assertEqual(link_calls, 1)
            self.assertEqual(
                sum(
                    isinstance(result, experiment._Publication)
                    and result.state == experiment.UNKNOWN
                    for result in results
                ),
                1,
            )
            self.assertEqual(
                sum(isinstance(result, Phase9GovernanceError) for result in results), 1
            )

    def test_model_evaluation_failure_restores_parameters_modes_dropout_and_rng(self) -> None:
        model, _ = _model()
        window = (
            inference.Phase9Window("v", 0, "validation", 0, (0,), (1,)),
        )
        parameter = next(model.parameters())
        parameter_before = parameter.detach().clone()
        dropout_before = tuple(
            generator.get_state().clone()
            for block in model.blocks
            for generator in (
                block.attention_dropout._generator,
                block.feed_forward_dropout._generator,
            )
        )
        rng_before = torch.get_rng_state().clone()

        def fail(_model, _tokens):
            torch.rand(1)
            with torch.no_grad():
                parameter.add_(1.0)
            parameter.requires_grad_(False)
            model.train()
            model.blocks[0].attention_dropout._generator.manual_seed(99)
            raise RuntimeError("synthetic")

        with patch.object(MiniGPT, "forward", new=fail):
            with self.assertRaises(RuntimeError):
                inference.evaluate_phase9_model(
                    model, window, predictor="synthetic", split="validation"
                )
        self.assertTrue(torch.equal(parameter, parameter_before))
        self.assertTrue(parameter.requires_grad)
        self.assertTrue(all(not module.training for _, module in model.named_modules()))
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        dropout_after = tuple(
            generator.get_state()
            for block in model.blocks
            for generator in (
                block.attention_dropout._generator,
                block.feed_forward_dropout._generator,
            )
        )
        self.assertTrue(
            all(torch.equal(left, right) for left, right in zip(dropout_before, dropout_after, strict=True))
        )

    def test_concurrent_publication_has_one_commit_and_no_clobber(self) -> None:
        content = experiment._canonical_json({"ok": True})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()

            def publish():
                return experiment._publish(
                    root,
                    "EXP-20260913-01",
                    filename="phase9-development-attempt.json",
                    kind="phase9_development_attempt",
                    status="consumed",
                    content=content,
                )

            with ThreadPoolExecutor(max_workers=2) as pool:
                results = tuple(pool.map(lambda _: publish(), range(2)))
            self.assertEqual(
                sum(result.state == experiment.KNOWN_PRESENT for result in results), 1
            )
            self.assertEqual(
                (root / "experiments/EXP-20260913-01/artifacts/phase9-development-attempt.json").read_bytes(),
                content,
            )

    def test_concurrent_terminal_publications_cannot_create_opposites(self) -> None:
        content = experiment._canonical_json({"ok": True})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()

            def publish_success():
                return experiment._publish_terminal(
                    root, "EXP-20260913-01",
                    filename="phase9-development-evidence.json",
                    opposite_filename="phase9-development-failure.json",
                    kind="phase9_development", status="completed", content=content,
                )

            def publish_failure():
                return experiment._publish_terminal(
                    root, "EXP-20260913-01",
                    filename="phase9-development-failure.json",
                    opposite_filename="phase9-development-evidence.json",
                    kind="phase9_development_failure", status="failed", content=content,
                )

            with ThreadPoolExecutor(max_workers=2) as pool:
                results = (pool.submit(publish_success), pool.submit(publish_failure))
                resolved = tuple(result.result(timeout=5) for result in results)
            artifacts = root / "experiments/EXP-20260913-01/artifacts"
            present = tuple(
                (artifacts / name).exists()
                for name in (
                    "phase9-development-evidence.json",
                    "phase9-development-failure.json",
                )
            )
            self.assertEqual(sum(present), 1)
            self.assertEqual(
                sum(result.state == experiment.KNOWN_PRESENT for result in resolved), 1
            )

    def test_absent_or_unknown_sealed_marker_never_allows_test_access(self) -> None:
        records, result, _plan, _authorization, _, _digest = self._sealed_launch_fixture()
        for state in (experiment.KNOWN_ABSENT, experiment.UNKNOWN):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                (root / "EXPERIMENT_LOG.md").write_bytes(b"synthetic")
                with (
                    patch.object(experiment, "_validate_repository", return_value=root),
                    patch.object(experiment, "_parse_selected_chain", return_value=records),
                    patch.object(experiment, "_validate_predecessor_chain"),
                    patch.object(experiment, "_read_evidence_artifact", return_value={"laplace_unigram_baseline": {
                        "vocabulary_size": 81,
                        "training_target_count": 1,
                        "raw_counts": [],
                        "smoothed_counts": [],
                        "probabilities": [],
                        "top_token_id": 0,
                    }}),
                    patch.object(experiment, "_validate_sealed_launch", return_value=("a" * 40, {})),
                    patch.object(experiment, "_sealed_test_metadata", return_value={}),
                    patch.object(experiment, "_validate_baseline_mapping"),
                    patch.object(experiment, "_make_laplace_unigram", return_value=object()),
                    patch.object(experiment, "load_phase9_inference_bundle", return_value=_bundle()),
                    patch.object(experiment, "_require_artifacts_absent"),
                    patch.object(experiment, "_validate_access_marker"),
                    patch.object(experiment, "_serialized_sealed_access_publication", return_value=experiment._Publication(state, None)),
                    patch.object(experiment, "_read_sealed_test_once") as sealed_read,
                ):
                    with self.assertRaises(Phase9PublicationError):
                        inference.run_fixed_phase9_sealed_test_evaluation(
                            root, evaluation_id=result.evaluation_id
                        )
                sealed_read.assert_not_called()

    def test_unknown_sealed_marker_is_consumed_across_restart_style_invocation(self) -> None:
        records, result, _plan, _authorization, development, _digest = self._sealed_launch_fixture()
        baseline = development["laplace_unigram_baseline"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            log_path = root / "EXPERIMENT_LOG.md"
            log_path.write_bytes(b"# Synthetic log\n\n")
            with (
                patch.object(experiment, "_validate_repository", return_value=root),
                patch.object(experiment, "_validate_predecessor_chain"),
                patch.object(experiment, "_parse_selected_chain", return_value=records),
                patch.object(experiment, "_read_evidence_artifact", return_value=development),
                patch.object(experiment, "_validate_sealed_launch", return_value=("a" * 40, development["authority"])),
                patch.object(experiment, "_sealed_test_metadata", return_value={}),
                patch.object(experiment, "_validate_baseline_mapping"),
                patch.object(experiment, "_make_laplace_unigram", return_value=object()),
                patch.object(experiment, "load_phase9_inference_bundle", return_value=_bundle()),
                patch.object(experiment, "_require_artifacts_absent"),
                patch.object(experiment, "_validate_access_marker"),
                patch.object(experiment, "_publish_sealed_access_marker", return_value=experiment._Publication(experiment.UNKNOWN, None)),
                patch.object(experiment, "_read_sealed_test_once") as sealed_read,
            ):
                with self.assertRaises(Phase9PublicationError):
                    inference.run_fixed_phase9_sealed_test_evaluation(
                        root, evaluation_id=result.evaluation_id
                    )
            sealed_read.assert_not_called()
            appended = experiment._parse_records(log_path.read_bytes())
            uncertain = experiment._one(
                appended, result.evaluation_id, "sealed_test_result"
            )
            experiment._validate_sealed_result_record(uncertain)
            with (
                patch.object(experiment, "_validate_repository", return_value=root),
                patch.object(experiment, "_parse_selected_chain", return_value=(*records, uncertain)),
                patch.object(experiment, "_validate_predecessor_chain"),
                patch.object(experiment, "_serialized_sealed_access_publication") as second_publish,
                patch.object(experiment, "_read_evidence_artifact") as second_evidence,
                patch.object(experiment, "_read_sealed_test_once") as second_sealed_read,
            ):
                with self.assertRaises(Phase9GovernanceError):
                    inference.run_fixed_phase9_sealed_test_evaluation(
                        root, evaluation_id=result.evaluation_id
                    )
            second_publish.assert_not_called()
            second_evidence.assert_not_called()
            second_sealed_read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
