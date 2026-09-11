"""Independent contract tests for the Phase 7 Complete Mini-GPT."""

from __future__ import annotations

import inspect
import json
import math
import sys
import unittest
from contextlib import ExitStack
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import sebgpt.model as model_package  # noqa: E402
import sebgpt.model.mini_gpt as mini_gpt_module  # noqa: E402
from sebgpt.model import (  # noqa: E402
    LanguageModelHead,
    MiniGPT,
    MiniGPTContractError,
    MiniGPTInspection,
    MiniGPTNumericalError,
    MiniGPTTypeError,
    explicit_cross_entropy,
)
from sebgpt.model.embeddings import TokenPositionEmbedding  # noqa: E402
from sebgpt.model.transformer_block import (  # noqa: E402
    ExplicitLayerNorm,
    TransformerBlock,
    TransformerBlockInspection,
)
from sebgpt.tokenization.vocabulary_artifact import (  # noqa: E402
    VocabularyBinding,
    load_accepted_vocabulary_binding,
)


TEST_BLOCK_SEEDS = (7001, 7002, 7003, 7004)
TEST_DEVICE = torch.device("cpu")
TEST_DTYPE = torch.float32

TEST_BLOCK_SUFFIXES = (
    "norm1.gamma",
    "norm1.beta",
    "attention.output_weight",
    "attention.heads.0.query_weight",
    "attention.heads.0.key_weight",
    "attention.heads.0.value_weight",
    "attention.heads.1.query_weight",
    "attention.heads.1.key_weight",
    "attention.heads.1.value_weight",
    "attention.heads.2.query_weight",
    "attention.heads.2.key_weight",
    "attention.heads.2.value_weight",
    "attention.heads.3.query_weight",
    "attention.heads.3.key_weight",
    "attention.heads.3.value_weight",
    "norm2.gamma",
    "norm2.beta",
    "feed_forward.first_weight",
    "feed_forward.first_bias",
    "feed_forward.second_weight",
    "feed_forward.second_bias",
)

TEST_BLOCK_SHAPES = (
    (32,),
    (32,),
    (32, 32),
    (32, 8),
    (32, 8),
    (32, 8),
    (32, 8),
    (32, 8),
    (32, 8),
    (32, 8),
    (32, 8),
    (32, 8),
    (32, 8),
    (32, 8),
    (32, 8),
    (32,),
    (32,),
    (32, 128),
    (128,),
    (128, 32),
    (32,),
)


class LongTensorSubclass(torch.Tensor):
    pass


class SentinelFailure(RuntimeError):
    pass


class DeviceBomb:
    @property
    def device(self) -> object:
        raise AssertionError("device was accessed before parameter type validation")


def _tokens(length: int = 4) -> torch.Tensor:
    return torch.arange(length, dtype=torch.long) % 81


def _block_random_parameters(block: TransformerBlock) -> tuple[nn.Parameter, ...]:
    parameters: list[nn.Parameter] = []
    for head in block.attention.heads:
        parameters.extend(
            (head.query_weight, head.key_weight, head.value_weight)
        )
    parameters.extend(
        (
            block.attention.output_weight,
            block.feed_forward.first_weight,
            block.feed_forward.second_weight,
        )
    )
    return tuple(parameters)


def _generator_states(
    model: MiniGPT,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    return tuple(
        (
            block.attention_dropout._generator.get_state().clone(),
            block.feed_forward_dropout._generator.get_state().clone(),
        )
        for block in model.blocks
    )


def _assert_generator_states_equal(
    test: unittest.TestCase,
    observed: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    expected: tuple[tuple[torch.Tensor, torch.Tensor], ...],
) -> None:
    test.assertEqual(len(observed), len(expected))
    for observed_pair, expected_pair in zip(observed, expected, strict=True):
        test.assertTrue(torch.equal(observed_pair[0], expected_pair[0]))
        test.assertTrue(torch.equal(observed_pair[1], expected_pair[1]))


def _parameter_snapshot(
    model: MiniGPT,
) -> tuple[tuple[int, torch.Tensor, tuple[int, ...], torch.device, torch.dtype], ...]:
    return tuple(
        (
            id(parameter),
            parameter.detach().clone(),
            tuple(parameter.shape),
            parameter.device,
            parameter.dtype,
        )
        for parameter in model.parameters()
    )


class MiniGPTTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vocabulary = load_accepted_vocabulary_binding(REPOSITORY_ROOT)

    def model(self, **overrides: object) -> MiniGPT:
        arguments: dict[str, object] = {
            "model_width": 32,
            "max_sequence_length": 256,
            "number_of_blocks": 4,
            "number_of_heads": 4,
            "head_width": 8,
            "hidden_width": 128,
            "layer_norm_epsilon": 1e-5,
            "dropout_probability": 0.1,
            "embedding_seed": 1337,
            "block_parameter_seeds": TEST_BLOCK_SEEDS,
            "output_head_seed": 7005,
            "device": TEST_DEVICE,
            "dtype": TEST_DTYPE,
        }
        arguments.update(overrides)
        return MiniGPT(
            self.vocabulary,
            **arguments,  # type: ignore[arg-type]
        )

    def head(self, **overrides: object) -> LanguageModelHead:
        arguments: dict[str, object] = {
            "model_width": 32,
            "vocabulary_size": 81,
            "seed": 7005,
            "device": TEST_DEVICE,
            "dtype": TEST_DTYPE,
        }
        arguments.update(overrides)
        return LanguageModelHead(**arguments)  # type: ignore[arg-type]


class PublicSurfaceAndConstructorTests(MiniGPTTestCase):
    def test_public_signatures_match_literal_specification_oracle(self) -> None:
        positional = inspect.Parameter.POSITIONAL_OR_KEYWORD
        keyword_only = inspect.Parameter.KEYWORD_ONLY

        cases = (
            (
                LanguageModelHead,
                (
                    ("model_width", keyword_only),
                    ("vocabulary_size", keyword_only),
                    ("seed", keyword_only),
                    ("device", keyword_only),
                    ("dtype", keyword_only),
                ),
                None,
            ),
            (
                MiniGPT,
                (
                    ("vocabulary", positional),
                    ("model_width", keyword_only),
                    ("max_sequence_length", keyword_only),
                    ("number_of_blocks", keyword_only),
                    ("number_of_heads", keyword_only),
                    ("head_width", keyword_only),
                    ("hidden_width", keyword_only),
                    ("layer_norm_epsilon", keyword_only),
                    ("dropout_probability", keyword_only),
                    ("embedding_seed", keyword_only),
                    ("block_parameter_seeds", keyword_only),
                    ("output_head_seed", keyword_only),
                    ("device", keyword_only),
                    ("dtype", keyword_only),
                ),
                None,
            ),
            (
                LanguageModelHead.forward,
                (("self", positional), ("representations", positional)),
                "torch.Tensor",
            ),
            (
                MiniGPT.forward,
                (("self", positional), ("token_ids", positional)),
                "torch.Tensor",
            ),
            (
                MiniGPT.inspect,
                (("self", positional), ("token_ids", positional)),
                "MiniGPTInspection",
            ),
        )
        for callable_object, expected_parameters, expected_return in cases:
            with self.subTest(callable=callable_object.__qualname__):
                signature = inspect.signature(callable_object)
                self.assertEqual(
                    tuple(
                        (parameter.name, parameter.kind)
                        for parameter in signature.parameters.values()
                    ),
                    expected_parameters,
                )
                self.assertTrue(
                    all(
                        parameter.default is inspect.Parameter.empty
                        for parameter in signature.parameters.values()
                    )
                )
                if expected_return is not None:
                    self.assertEqual(signature.return_annotation, expected_return)

    def test_exact_module_and_package_exports(self) -> None:
        self.assertEqual(
            tuple(mini_gpt_module.__all__),
            (
                "LanguageModelHead",
                "MiniGPT",
                "MiniGPTContractError",
                "MiniGPTInspection",
                "MiniGPTNumericalError",
                "MiniGPTTypeError",
            ),
        )
        self.assertEqual(
            tuple(model_package.__all__),
            (
                "AttentionContractError",
                "AttentionInspection",
                "AttentionNumericalError",
                "AttentionTypeError",
                "EmbeddingContractError",
                "EmbeddingTypeError",
                "DropoutInspection",
                "ExplicitDropout",
                "ExplicitLayerNorm",
                "FeedForwardInspection",
                "LayerNormInspection",
                "AggregateMeasurement",
                "Phase4ContractError",
                "Phase4ExperimentConfig",
                "Phase4ExperimentResult",
                "Phase4GovernanceError",
                "Phase4PermittedCorpus",
                "Phase4TypeError",
                "PositionwiseFeedForward",
                "MultiHeadCausalSelfAttention",
                "SimpleNeuralLanguageModel",
                "SingleHeadCausalSelfAttention",
                "TokenPositionEmbedding",
                "TrainingPassResult",
                "TransformerBlock",
                "TransformerBlockContractError",
                "TransformerBlockInspection",
                "TransformerBlockNumericalError",
                "TransformerBlockTransactionError",
                "TransformerBlockTypeError",
                "clear_gradients",
                "explicit_cross_entropy",
                "explicit_gelu",
                "load_phase4_experiment_config",
                "load_phase4_permitted_corpus",
                "manual_sgd_step",
                "measure_example_loss",
                "measure_aggregate_loss",
                "model_parameter_digest",
                "probabilities_from_logits",
                "run_fixed_phase4_experiment",
                "run_training_pass",
                "scale_backward_loss",
                "validate_phase4_experiment_preflight",
                "LanguageModelHead",
                "MiniGPT",
                "MiniGPTContractError",
                "MiniGPTInspection",
                "MiniGPTNumericalError",
                "MiniGPTTypeError",
            ),
        )

    def test_model_constructor_type_failures_precede_allocation(self) -> None:
        cases = (
            ("model_width", True, "mini_gpt.constructor.model_width.exact_int"),
            (
                "max_sequence_length",
                256.0,
                "mini_gpt.constructor.max_sequence_length.exact_int",
            ),
            (
                "number_of_blocks",
                False,
                "mini_gpt.constructor.number_of_blocks.exact_int",
            ),
            (
                "number_of_heads",
                "4",
                "mini_gpt.constructor.number_of_heads.exact_int",
            ),
            ("head_width", 8.0, "mini_gpt.constructor.head_width.exact_int"),
            ("hidden_width", None, "mini_gpt.constructor.hidden_width.exact_int"),
            ("embedding_seed", True, "mini_gpt.constructor.embedding_seed.exact_int"),
            (
                "layer_norm_epsilon",
                1,
                "mini_gpt.constructor.layer_norm_epsilon.exact_float",
            ),
            (
                "dropout_probability",
                1,
                "mini_gpt.constructor.dropout_probability.exact_float",
            ),
            (
                "block_parameter_seeds",
                [7001, 7002, 7003, 7004],
                "mini_gpt.constructor.block_parameter_seeds.exact_tuple",
            ),
            (
                "output_head_seed",
                7005.0,
                "mini_gpt.constructor.output_head_seed.exact_int",
            ),
            (
                "device",
                "cpu",
                "mini_gpt.constructor.device.exact_torch_device",
            ),
            ("dtype", "float32", "mini_gpt.constructor.dtype.torch_dtype"),
        )
        for name, value, invariant in cases:
            with self.subTest(name=name):
                with patch.object(mini_gpt_module, "TokenPositionEmbedding") as embedding:
                    with self.assertRaises(MiniGPTTypeError) as raised:
                        self.model(**{name: value})
                embedding.assert_not_called()
                self.assertEqual(raised.exception.details["invariant"], invariant)

    def test_vocabulary_type_length_and_seed_element_validation(self) -> None:
        with patch.object(mini_gpt_module, "TokenPositionEmbedding") as embedding:
            with self.assertRaises(MiniGPTTypeError) as wrong_type:
                MiniGPT(  # type: ignore[arg-type]
                    object(),
                    model_width=32,
                    max_sequence_length=256,
                    number_of_blocks=4,
                    number_of_heads=4,
                    head_width=8,
                    hidden_width=128,
                    layer_norm_epsilon=1e-5,
                    dropout_probability=0.1,
                    embedding_seed=1337,
                    block_parameter_seeds=TEST_BLOCK_SEEDS,
                    output_head_seed=7005,
                    device=TEST_DEVICE,
                    dtype=TEST_DTYPE,
                )
        embedding.assert_not_called()
        self.assertEqual(
            wrong_type.exception.details["invariant"],
            "mini_gpt.constructor.vocabulary.exact_type",
        )

        with self.assertRaises(MiniGPTContractError) as length:
            self.model(block_parameter_seeds=(7001, 7002, 7003))
        self.assertEqual(
            length.exception.details["invariant"],
            "mini_gpt.constructor.block_parameter_seeds.length",
        )
        with self.assertRaises(MiniGPTTypeError) as element:
            self.model(block_parameter_seeds=(7001, 7002, False, 7004))
        self.assertEqual(
            element.exception.details["invariant"],
            "mini_gpt.constructor.block_parameter_seeds.2.exact_int",
        )

    def test_model_constructor_value_failures_precede_allocation(self) -> None:
        cases = (
            ("model_width", 31, "mini_gpt.constructor.model_width.value"),
            (
                "max_sequence_length",
                255,
                "mini_gpt.constructor.max_sequence_length.value",
            ),
            (
                "number_of_blocks",
                3,
                "mini_gpt.constructor.number_of_blocks.value",
            ),
            (
                "number_of_heads",
                2,
                "mini_gpt.constructor.number_of_heads.value",
            ),
            ("head_width", 4, "mini_gpt.constructor.head_width.value"),
            ("hidden_width", 64, "mini_gpt.constructor.hidden_width.value"),
            (
                "layer_norm_epsilon",
                1e-4,
                "mini_gpt.constructor.layer_norm_epsilon.value",
            ),
            (
                "dropout_probability",
                0.0,
                "mini_gpt.constructor.dropout_probability.value",
            ),
            ("embedding_seed", 1, "mini_gpt.constructor.embedding_seed.value"),
            (
                "block_parameter_seeds",
                (7001, 7002, 7003, 7005),
                "mini_gpt.constructor.block_parameter_seeds.value",
            ),
            (
                "output_head_seed",
                1,
                "mini_gpt.constructor.output_head_seed.value",
            ),
            ("device", torch.device("meta"), "mini_gpt.constructor.device.value"),
            ("dtype", torch.float64, "mini_gpt.constructor.dtype.value"),
        )
        for name, value, invariant in cases:
            with self.subTest(name=name):
                with patch.object(mini_gpt_module, "TokenPositionEmbedding") as embedding:
                    with self.assertRaises(MiniGPTContractError) as raised:
                        self.model(**{name: value})
                embedding.assert_not_called()
                self.assertEqual(raised.exception.details["invariant"], invariant)

    def test_unverified_binding_and_seed_relationship_invariants(self) -> None:
        unverified = object.__new__(VocabularyBinding)
        with patch.object(mini_gpt_module, "TokenPositionEmbedding") as embedding:
            with self.assertRaises(MiniGPTContractError) as binding:
                MiniGPT(
                    unverified,
                    model_width=32,
                    max_sequence_length=256,
                    number_of_blocks=4,
                    number_of_heads=4,
                    head_width=8,
                    hidden_width=128,
                    layer_norm_epsilon=1e-5,
                    dropout_probability=0.1,
                    embedding_seed=1337,
                    block_parameter_seeds=TEST_BLOCK_SEEDS,
                    output_head_seed=7005,
                    device=TEST_DEVICE,
                    dtype=TEST_DTYPE,
                )
        embedding.assert_not_called()
        self.assertEqual(
            binding.exception.details["invariant"],
            "mini_gpt.constructor.vocabulary.verified",
        )

        duplicate = (7001, 7001, 7003, 7004)
        with patch.object(
            mini_gpt_module,
            "BLOCK_PARAMETER_INITIALIZATION_SEEDS",
            duplicate,
        ):
            with self.assertRaises(MiniGPTContractError) as distinct:
                self.model(block_parameter_seeds=duplicate)
        self.assertEqual(
            distinct.exception.details["invariant"],
            "mini_gpt.constructor.block_parameter_seeds.distinct",
        )

        overlapping = (1337, 7002, 7003, 7004)
        with patch.object(
            mini_gpt_module,
            "BLOCK_PARAMETER_INITIALIZATION_SEEDS",
            overlapping,
        ):
            with self.assertRaises(MiniGPTContractError) as disjoint:
                self.model(block_parameter_seeds=overlapping)
        self.assertEqual(
            disjoint.exception.details["invariant"],
            "mini_gpt.constructor.block_parameter_seeds.disjoint",
        )

    def test_head_constructor_type_and_value_oracle(self) -> None:
        type_cases = (
            ("model_width", True, "lm_head.constructor.model_width.exact_int"),
            (
                "vocabulary_size",
                81.0,
                "lm_head.constructor.vocabulary_size.exact_int",
            ),
            ("seed", False, "lm_head.constructor.seed.exact_int"),
            ("device", "cpu", "lm_head.constructor.device.exact_torch_device"),
            ("dtype", "float32", "lm_head.constructor.dtype.torch_dtype"),
        )
        for name, value, invariant in type_cases:
            with self.subTest(kind="type", name=name):
                with patch.object(mini_gpt_module.torch, "empty") as empty:
                    with self.assertRaises(MiniGPTTypeError) as raised:
                        self.head(**{name: value})
                empty.assert_not_called()
                self.assertEqual(raised.exception.details["invariant"], invariant)

        value_cases = (
            ("model_width", 31, "lm_head.constructor.model_width.value"),
            ("vocabulary_size", 80, "lm_head.constructor.vocabulary_size.value"),
            ("seed", 7004, "lm_head.constructor.seed.value"),
            ("device", torch.device("meta"), "lm_head.constructor.device.value"),
            ("dtype", torch.float64, "lm_head.constructor.dtype.value"),
        )
        for name, value, invariant in value_cases:
            with self.subTest(kind="value", name=name):
                with patch.object(mini_gpt_module.torch, "empty") as empty:
                    with self.assertRaises(MiniGPTContractError) as raised:
                        self.head(**{name: value})
                empty.assert_not_called()
                self.assertEqual(raised.exception.details["invariant"], invariant)


class StructureAndInitializationTests(MiniGPTTestCase):
    def test_exact_children_modules_parameters_and_counts(self) -> None:
        model = self.model()
        self.assertEqual(
            tuple(name for name, _ in model.named_children()),
            ("representation", "blocks", "final_norm", "head"),
        )
        self.assertIs(type(model.representation), TokenPositionEmbedding)
        self.assertIs(type(model.blocks), nn.ModuleList)
        self.assertIs(type(model.final_norm), ExplicitLayerNorm)
        self.assertIs(type(model.head), LanguageModelHead)
        self.assertEqual(len(model.blocks), 4)
        self.assertTrue(all(type(block) is TransformerBlock for block in model.blocks))
        self.assertEqual(len({id(block) for block in model.blocks}), 4)

        expected_names = [
            "representation.token_embeddings",
            "representation.position_embeddings",
        ]
        expected_shapes = [(81, 32), (256, 32)]
        for block_index in range(4):
            expected_names.extend(
                f"blocks.{block_index}.{suffix}" for suffix in TEST_BLOCK_SUFFIXES
            )
            expected_shapes.extend(TEST_BLOCK_SHAPES)
        expected_names.extend(
            ("final_norm.gamma", "final_norm.beta", "head.output_weight", "head.output_bias")
        )
        expected_shapes.extend(((32,), (32,), (32, 81), (81,)))
        items = tuple(model.named_parameters())
        self.assertEqual(tuple(name for name, _ in items), tuple(expected_names))
        self.assertEqual(
            tuple(tuple(parameter.shape) for _, parameter in items),
            tuple(expected_shapes),
        )
        self.assertEqual(len(items), 90)
        self.assertEqual(sum(parameter.numel() for _, parameter in items), 63825)
        self.assertEqual(tuple(model.named_buffers()), ())
        self.assertTrue(all(parameter.requires_grad for _, parameter in items))
        self.assertTrue(all(parameter.device == TEST_DEVICE for _, parameter in items))
        self.assertTrue(all(parameter.dtype is TEST_DTYPE for _, parameter in items))
        self.assertEqual(len({id(parameter) for _, parameter in items}), 90)

    def test_final_norm_is_independent_and_exactly_initialized(self) -> None:
        model = self.model()
        self.assertTrue(torch.equal(model.final_norm.gamma, torch.ones(32)))
        self.assertTrue(torch.equal(model.final_norm.beta, torch.zeros(32)))
        final_ids = {id(model.final_norm.gamma), id(model.final_norm.beta)}
        block_norm_ids = {
            id(parameter)
            for block in model.blocks
            for parameter in (
                block.norm1.gamma,
                block.norm1.beta,
                block.norm2.gamma,
                block.norm2.beta,
            )
        }
        self.assertTrue(final_ids.isdisjoint(block_norm_ids))

    def test_configuration_metadata_is_exact_and_immutable(self) -> None:
        model = self.model()
        self.assertEqual(
            dict(model._configuration),
            {
                "model_width": 32,
                "max_sequence_length": 256,
                "number_of_blocks": 4,
                "number_of_heads": 4,
                "head_width": 8,
                "hidden_width": 128,
                "layer_norm_epsilon": 1e-5,
                "dropout_probability": 0.1,
                "embedding_seed": 1337,
                "block_parameter_seeds": TEST_BLOCK_SEEDS,
                "output_head_seed": 7005,
                "device": TEST_DEVICE,
                "dtype": TEST_DTYPE,
            },
        )
        with self.assertRaises(TypeError):
            model._configuration["model_width"] = 64  # type: ignore[index]

    def test_head_initialization_is_seed_7005_and_untied(self) -> None:
        model = self.model()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(7005)
        expected = torch.empty((32, 81), dtype=torch.float32)
        torch.nn.init.normal_(
            expected,
            mean=0.0,
            std=1.0 / math.sqrt(32),
            generator=generator,
        )
        self.assertTrue(torch.equal(model.head.output_weight, expected))
        self.assertTrue(torch.equal(model.head.output_bias, torch.zeros(81)))
        self.assertIsNot(
            model.head.output_weight,
            model.representation.token_embeddings,
        )
        self.assertNotEqual(
            model.head.output_weight.untyped_storage().data_ptr(),
            model.representation.token_embeddings.untyped_storage().data_ptr(),
        )

    def test_each_block_matches_independent_phase7_generator_stream(self) -> None:
        model = self.model()
        random_shapes = (
            *((32, 8) for _ in range(12)),
            (32, 32),
            (32, 128),
            (128, 32),
        )
        for block_index, (block, seed) in enumerate(
            zip(model.blocks, TEST_BLOCK_SEEDS, strict=True)
        ):
            with self.subTest(block=block_index):
                generator = torch.Generator(device="cpu")
                generator.manual_seed(seed)
                expected: list[torch.Tensor] = []
                for position, shape in enumerate(random_shapes):
                    tensor = torch.empty(shape, dtype=torch.float32)
                    standard_deviation = (
                        1.0 / math.sqrt(128)
                        if position == len(random_shapes) - 1
                        else 1.0 / math.sqrt(32)
                    )
                    torch.nn.init.normal_(
                        tensor,
                        mean=0.0,
                        std=standard_deviation,
                        generator=generator,
                    )
                    expected.append(tensor)
                observed = _block_random_parameters(block)
                self.assertEqual(len(observed), 15)
                self.assertTrue(
                    all(
                        torch.equal(parameter, expected_tensor)
                        for parameter, expected_tensor in zip(
                            observed, expected, strict=True
                        )
                    )
                )

    def test_blocks_are_pairwise_nonidentical(self) -> None:
        model = self.model()
        for left in range(4):
            for right in range(left + 1, 4):
                self.assertFalse(
                    all(
                        torch.equal(left_parameter, right_parameter)
                        for left_parameter, right_parameter in zip(
                            _block_random_parameters(model.blocks[left]),
                            _block_random_parameters(model.blocks[right]),
                            strict=True,
                        )
                    )
                )

    def test_in_place_reinitialization_preserves_every_required_identity(self) -> None:
        original = mini_gpt_module._initialize_block_for_phase7
        observations: list[dict[str, object]] = []

        def delegated(
            block: TransformerBlock,
            *,
            block_index: int,
            seed: int,
        ) -> None:
            before = {
                "children": tuple(child for _, child in block.named_children()),
                "heads": tuple(block.attention.heads),
                "parameters": tuple(block.parameters()),
                "dropouts": (block.attention_dropout, block.feed_forward_dropout),
                "generators": (
                    block.attention_dropout._generator,
                    block.feed_forward_dropout._generator,
                ),
                "states": (
                    block.attention_dropout._generator.get_state().clone(),
                    block.feed_forward_dropout._generator.get_state().clone(),
                ),
                "index": block_index,
                "seed": seed,
            }
            original(block, block_index=block_index, seed=seed)
            before["after_children"] = tuple(
                child for _, child in block.named_children()
            )
            before["after_heads"] = tuple(block.attention.heads)
            before["after_parameters"] = tuple(block.parameters())
            before["after_dropouts"] = (
                block.attention_dropout,
                block.feed_forward_dropout,
            )
            before["after_generators"] = (
                block.attention_dropout._generator,
                block.feed_forward_dropout._generator,
            )
            before["after_states"] = (
                block.attention_dropout._generator.get_state().clone(),
                block.feed_forward_dropout._generator.get_state().clone(),
            )
            observations.append(before)

        with patch.object(
            mini_gpt_module,
            "_initialize_block_for_phase7",
            side_effect=delegated,
        ) as initializer:
            self.model()
        self.assertEqual(initializer.call_count, 4)
        self.assertEqual(tuple(item["seed"] for item in observations), TEST_BLOCK_SEEDS)
        for item in observations:
            self.assertTrue(
                all(
                    before is after
                    for before, after in zip(
                        item["children"], item["after_children"], strict=True
                    )
                )
            )
            self.assertTrue(
                all(
                    before is after
                    for before, after in zip(
                        item["heads"], item["after_heads"], strict=True
                    )
                )
            )
            self.assertTrue(
                all(
                    before is after
                    for before, after in zip(
                        item["parameters"], item["after_parameters"], strict=True
                    )
                )
            )
            self.assertTrue(
                all(
                    before is after
                    for before, after in zip(
                        item["dropouts"], item["after_dropouts"], strict=True
                    )
                )
            )
            self.assertTrue(
                all(
                    before is after
                    for before, after in zip(
                        item["generators"], item["after_generators"], strict=True
                    )
                )
            )
            self.assertTrue(
                all(
                    torch.equal(before, after)
                    for before, after in zip(
                        item["states"], item["after_states"], strict=True
                    )
                )
            )

    def test_embedding_bytes_remain_the_accepted_phase3_initialization(self) -> None:
        model = self.model()
        standalone = TokenPositionEmbedding(
            self.vocabulary,
            embedding_dim=32,
            max_positions=256,
            seed=1337,
            device=TEST_DEVICE,
            dtype=TEST_DTYPE,
        )
        self.assertTrue(
            torch.equal(
                model.representation.token_embeddings,
                standalone.token_embeddings,
            )
        )
        self.assertTrue(
            torch.equal(
                model.representation.position_embeddings,
                standalone.position_embeddings,
            )
        )

    def test_complete_construction_is_repeatable_and_global_rng_isolated(self) -> None:
        global_before = torch.get_rng_state().clone()
        first = self.model()
        global_after_first = torch.get_rng_state().clone()
        torch.rand(257)
        second = self.model()
        global_after_second = torch.get_rng_state().clone()
        try:
            self.assertTrue(torch.equal(global_before, global_after_first))
            self.assertTrue(
                all(
                    torch.equal(left, right)
                    for left, right in zip(
                        first.parameters(), second.parameters(), strict=True
                    )
                )
            )
        finally:
            torch.set_rng_state(global_before)
        self.assertFalse(torch.equal(global_after_first, global_after_second))

    def test_initialization_nonidentity_and_global_rng_postcondition_invariants(self) -> None:
        with patch.object(
            mini_gpt_module,
            "_blocks_are_pairwise_nonidentical",
            return_value=False,
        ):
            with self.assertRaises(MiniGPTContractError) as nonidentical:
                self.model()
        self.assertEqual(
            nonidentical.exception.details["invariant"],
            "mini_gpt.initialization.blocks.nonidentical",
        )

        real_state = torch.get_rng_state().clone()
        changed_state = real_state.clone()
        changed_state[0] = (changed_state[0] + 1) % 255
        with patch.object(
            mini_gpt_module.torch,
            "get_rng_state",
            side_effect=(real_state, changed_state),
        ):
            with self.assertRaises(MiniGPTContractError) as global_rng:
                self.model()
        self.assertEqual(
            global_rng.exception.details["invariant"],
            "mini_gpt.initialization.global_rng",
        )
        self.assertTrue(torch.equal(torch.get_rng_state(), real_state))


class HeadAndModelValidationTests(MiniGPTTestCase):
    def test_head_forward_matches_exact_affine_arithmetic(self) -> None:
        head = self.head()
        representations = torch.arange(96, dtype=torch.float32).reshape(3, 32) / 31.0
        snapshot = tuple(parameter.detach().clone() for parameter in head.parameters())
        logits = head(representations)
        expected = representations @ head.output_weight + head.output_bias
        self.assertTrue(torch.equal(logits, expected))
        self.assertEqual(tuple(logits.shape), (3, 81))
        self.assertTrue(
            all(
                torch.equal(parameter, before)
                for parameter, before in zip(head.parameters(), snapshot, strict=True)
            )
        )

    def test_head_input_validation_and_nonfinite_oracle(self) -> None:
        head = self.head()
        cases = (
            (object(), MiniGPTTypeError, "lm_head.input.tensor"),
            (torch.zeros(32), MiniGPTContractError, "lm_head.input.rank"),
            (
                torch.zeros((1, 32), dtype=torch.float64),
                MiniGPTContractError,
                "lm_head.input.dtype",
            ),
            (
                torch.empty((1, 32), device="meta"),
                MiniGPTContractError,
                "lm_head.input.device",
            ),
            (
                torch.zeros((1, 31)),
                MiniGPTContractError,
                "lm_head.input.model_width",
            ),
            (
                torch.zeros((0, 32)),
                MiniGPTContractError,
                "lm_head.input.sequence_length",
            ),
            (
                torch.full((1, 32), torch.nan),
                MiniGPTNumericalError,
                "lm_head.input.finite",
            ),
        )
        for value, exception, invariant in cases:
            with self.subTest(invariant=invariant):
                with self.assertRaises(exception) as raised:
                    head(value)  # type: ignore[arg-type]
                self.assertEqual(raised.exception.details["invariant"], invariant)

        with torch.no_grad():
            head.output_weight.fill_(torch.finfo(torch.float32).max)
        finite_input = torch.full((1, 32), torch.finfo(torch.float32).max)
        with self.assertRaises(MiniGPTNumericalError) as output:
            head(finite_input)
        self.assertEqual(output.exception.details["invariant"], "lm_head.output.finite")

    def test_head_parameter_validation_order_and_invariants(self) -> None:
        head = self.head()
        head._parameters["output_weight"] = DeviceBomb()  # type: ignore[assignment]
        with self.assertRaises(MiniGPTTypeError) as wrong_type:
            head(torch.zeros((1, 32)))
        self.assertEqual(
            wrong_type.exception.details["invariant"],
            "lm_head.parameter.output_weight.type",
        )

        head = self.head()
        head.output_weight = nn.Parameter(torch.empty((1,), device="meta"))
        with self.assertRaises(MiniGPTContractError) as shape:
            head(torch.zeros((1, 32)))
        self.assertEqual(
            shape.exception.details["invariant"],
            "lm_head.parameter.output_weight.shape",
        )

        head = self.head()
        head.output_weight = nn.Parameter(torch.empty((32, 81), device="meta"))
        with self.assertRaises(MiniGPTContractError) as device:
            head(torch.zeros((1, 32)))
        self.assertEqual(
            device.exception.details["invariant"],
            "lm_head.parameter.output_weight.device",
        )

        head = self.head()
        head.output_weight = nn.Parameter(torch.zeros((32, 81), dtype=torch.float64))
        with self.assertRaises(MiniGPTContractError) as dtype:
            head(torch.zeros((1, 32)))
        self.assertEqual(
            dtype.exception.details["invariant"],
            "lm_head.parameter.output_weight.dtype",
        )

        head = self.head()
        with torch.no_grad():
            head.output_weight[0, 0] = torch.nan
        with self.assertRaises(MiniGPTNumericalError) as finite:
            head(torch.zeros((1, 32)))
        self.assertEqual(
            finite.exception.details["invariant"],
            "lm_head.parameter.output_weight.finite",
        )

    def test_model_input_validation_order_and_boundaries(self) -> None:
        model = self.model().eval()
        cases = (
            (object(), MiniGPTTypeError, "mini_gpt.input.tensor"),
            (torch.zeros((1, 1), dtype=torch.long), MiniGPTContractError, "mini_gpt.input.rank"),
            (torch.zeros(1), MiniGPTContractError, "mini_gpt.input.dtype"),
            (
                torch.empty((1,), dtype=torch.long, device="meta"),
                MiniGPTContractError,
                "mini_gpt.input.device",
            ),
            (torch.empty((0,), dtype=torch.long), MiniGPTContractError, "mini_gpt.input.sequence_length"),
            (torch.zeros(257, dtype=torch.long), MiniGPTContractError, "mini_gpt.input.sequence_length"),
            (torch.tensor([-1]), MiniGPTContractError, "mini_gpt.input.token_id_range"),
            (torch.tensor([81]), MiniGPTContractError, "mini_gpt.input.token_id_range"),
        )
        for value, exception, invariant in cases:
            with self.subTest(invariant=invariant, shape=getattr(value, "shape", None)):
                with self.assertRaises(exception) as raised:
                    model(value)  # type: ignore[arg-type]
                self.assertEqual(raised.exception.details["invariant"], invariant)

        self.assertEqual(tuple(model(_tokens(1)).shape), (1, 81))
        self.assertEqual(tuple(model(_tokens(256)).shape), (256, 81))

    def test_tensor_subclass_and_noncontiguous_rank_one_input_are_accepted(self) -> None:
        model = self.model().eval()
        subclass = torch.tensor([1, 2, 3], dtype=torch.long).as_subclass(
            LongTensorSubclass
        )
        noncontiguous = torch.arange(8, dtype=torch.long)[::2]
        self.assertFalse(noncontiguous.is_contiguous())
        self.assertEqual(tuple(model(subclass).shape), (3, 81))
        self.assertEqual(tuple(model(noncontiguous).shape), (4, 81))

    def test_model_structure_mode_and_parameter_failure_oracle(self) -> None:
        model = self.model()
        model._modules["final_norm"] = nn.Identity()
        with self.assertRaises(MiniGPTTypeError) as child:
            model(_tokens())
        self.assertEqual(child.exception.details["invariant"], "mini_gpt.submodules.structure")

        model = self.model()
        del model.blocks[-1]
        with self.assertRaises(MiniGPTContractError) as blocks:
            model(_tokens())
        self.assertEqual(blocks.exception.details["invariant"], "mini_gpt.blocks.structure")

        model = self.model()
        model.blocks[0].training = 1  # type: ignore[assignment]
        with self.assertRaises(MiniGPTTypeError) as mode_type:
            model(_tokens())
        self.assertEqual(mode_type.exception.details["invariant"], "mini_gpt.modes.type")

        model = self.model()
        model.blocks[0].eval()
        with self.assertRaises(MiniGPTContractError) as mode:
            model(_tokens())
        self.assertEqual(mode.exception.details["invariant"], "mini_gpt.modes.consistent")

    def test_child_structure_and_exact_type_failures_are_separated(self) -> None:
        model = self.model()
        del model._modules["final_norm"]
        with self.assertRaises(MiniGPTContractError) as direct_structure:
            model(_tokens())
        self.assertEqual(
            direct_structure.exception.details["invariant"],
            "mini_gpt.submodules.structure",
        )

        model = self.model()
        model.register_buffer("extra", torch.zeros(1))
        with self.assertRaises(MiniGPTContractError) as buffer_structure:
            model(_tokens())
        self.assertEqual(
            buffer_structure.exception.details["invariant"],
            "mini_gpt.submodules.structure",
        )

        model = self.model()
        model._modules["final_norm"] = nn.Identity()
        with self.assertRaises(MiniGPTTypeError) as final_norm_type:
            model(_tokens())
        self.assertEqual(
            final_norm_type.exception.details["invariant"],
            "mini_gpt.submodules.structure",
        )
        self.assertEqual(final_norm_type.exception.details["child"], "final_norm")

        model = self.model()
        model.blocks[1] = nn.Identity()
        with self.assertRaises(MiniGPTTypeError) as block_type:
            model(_tokens())
        self.assertEqual(
            block_type.exception.details["invariant"],
            "mini_gpt.blocks.structure",
        )
        self.assertEqual(block_type.exception.details["block_index"], 1)

        model = self.model()
        model.blocks[2]._modules["norm2"] = nn.Identity()
        with self.assertRaises(MiniGPTTypeError) as nested_type:
            model(_tokens())
        self.assertEqual(
            nested_type.exception.details["invariant"],
            "mini_gpt.blocks.structure",
        )
        self.assertEqual(nested_type.exception.details["block_index"], 2)
        self.assertEqual(nested_type.exception.details["child"], "norm2")

        model = self.model()
        model.blocks[0].attention.heads[3] = nn.Identity()
        with self.assertRaises(MiniGPTTypeError) as head_type:
            model(_tokens())
        self.assertEqual(
            head_type.exception.details["invariant"],
            "mini_gpt.blocks.structure",
        )
        self.assertEqual(head_type.exception.details["block_index"], 0)
        self.assertEqual(head_type.exception.details["child"], "attention.heads.3")

    def test_parameter_type_and_shape_precede_parameter_device_access(self) -> None:
        model = self.model()
        model.head._parameters["output_weight"] = DeviceBomb()  # type: ignore[assignment]
        with self.assertRaises(MiniGPTTypeError) as wrong_type:
            model(_tokens())
        self.assertEqual(
            wrong_type.exception.details["invariant"],
            "mini_gpt.parameter.head.output_weight.type",
        )

        model = self.model()
        model.head.output_weight = nn.Parameter(torch.empty((1,), device="meta"))
        with self.assertRaises(MiniGPTContractError) as shape:
            model(_tokens())
        self.assertEqual(
            shape.exception.details["invariant"],
            "mini_gpt.parameter.head.output_weight.shape",
        )

        model = self.model()
        model.head.output_weight = nn.Parameter(
            torch.empty((32, 81), device="meta")
        )
        with self.assertRaises(MiniGPTContractError) as device:
            model(_tokens())
        self.assertEqual(
            device.exception.details["invariant"],
            "mini_gpt.parameter.head.output_weight.device",
        )

    def test_parameter_dtype_finiteness_unique_and_untied_invariants(self) -> None:
        model = self.model()
        model.head.output_weight = nn.Parameter(torch.zeros((32, 81), dtype=torch.float64))
        with self.assertRaises(MiniGPTContractError) as dtype:
            model(_tokens())
        self.assertEqual(
            dtype.exception.details["invariant"],
            "mini_gpt.parameter.head.output_weight.dtype",
        )

        model = self.model()
        with torch.no_grad():
            model.head.output_weight[0, 0] = torch.nan
        with self.assertRaises(MiniGPTNumericalError) as finite:
            model(_tokens())
        self.assertEqual(
            finite.exception.details["invariant"],
            "mini_gpt.parameter.head.output_weight.finite",
        )

        model = self.model()
        model.final_norm.beta = model.final_norm.gamma
        with self.assertRaises(MiniGPTContractError) as unique:
            model(_tokens())
        self.assertEqual(unique.exception.details["invariant"], "mini_gpt.parameters.unique")

        model = self.model()
        tied_view = model.representation.token_embeddings.transpose(0, 1)
        model.head.output_weight = nn.Parameter(tied_view)
        with self.assertRaises(MiniGPTContractError) as untied:
            model(_tokens())
        self.assertEqual(untied.exception.details["invariant"], "mini_gpt.head.untied")

    def test_parameter_enumeration_ownership_and_head_structure_invariants(self) -> None:
        model = self.model()
        model.register_parameter("extra", nn.Parameter(torch.zeros(1)))
        with self.assertRaises(MiniGPTContractError) as enumeration:
            model(_tokens())
        self.assertEqual(
            enumeration.exception.details["invariant"],
            "mini_gpt.parameters.enumeration",
        )

        model = self.model()
        expected = list(model._expected_parameter_specs())
        name, _, shape = expected[0]
        expected[0] = (name, object(), shape)
        with patch.object(
            model,
            "_expected_parameter_specs",
            return_value=tuple(expected),
        ):
            with self.assertRaises(MiniGPTContractError) as ownership:
                model(_tokens())
        self.assertEqual(
            ownership.exception.details["invariant"],
            "mini_gpt.parameter.representation.token_embeddings.ownership",
        )

        head = self.head()
        head.register_parameter("extra", nn.Parameter(torch.zeros(1)))
        with self.assertRaises(MiniGPTContractError) as structure:
            head(torch.zeros((1, 32)))
        self.assertEqual(
            structure.exception.details["invariant"],
            "lm_head.parameters.structure",
        )

    def test_rejected_call_is_nonmutating_and_content_safe(self) -> None:
        model = self.model()
        parameters = _parameter_snapshot(model)
        states = _generator_states(model)
        global_rng = torch.get_rng_state().clone()
        invalid = torch.tensor([7, 9999], dtype=torch.long)
        with self.assertRaises(MiniGPTContractError) as raised:
            model(invalid)
        self.assertEqual(raised.exception.details["invariant"], "mini_gpt.input.token_id_range")
        self.assertNotIn("9999", str(raised.exception))
        expected_json = json.loads(json.dumps(dict(raised.exception.details)))
        self.assertEqual(json.loads(str(raised.exception)), expected_json)
        with self.assertRaises(TypeError):
            raised.exception.details["extra"] = 1  # type: ignore[index]
        after = _parameter_snapshot(model)
        for before_item, after_item in zip(parameters, after, strict=True):
            self.assertEqual(before_item[0], after_item[0])
            self.assertTrue(torch.equal(before_item[1], after_item[1]))
            self.assertEqual(before_item[2:], after_item[2:])
        _assert_generator_states_equal(self, _generator_states(model), states)
        self.assertTrue(torch.equal(torch.get_rng_state(), global_rng))


class ForwardLossInspectionAndGradientTests(MiniGPTTestCase):
    def test_forward_exact_topology_shape_and_handoffs(self) -> None:
        model = self.model().eval()
        token_ids = torch.tensor([1, 2, 3], dtype=torch.long)
        captured_representation: list[torch.Tensor] = []
        captured_block_outputs: list[torch.Tensor] = []
        captured_final: list[torch.Tensor] = []
        captured_logits: list[torch.Tensor] = []

        original_representation = model.representation.forward

        def representation_delegate(value: torch.Tensor) -> torch.Tensor:
            output = original_representation(value)
            captured_representation.append(output)
            return output

        def block_delegate(original):
            def delegated(value: torch.Tensor) -> torch.Tensor:
                output = original(value)
                captured_block_outputs.append(output)
                return output

            return delegated

        original_final = model.final_norm.forward

        def final_delegate(value: torch.Tensor) -> torch.Tensor:
            output = original_final(value)
            captured_final.append(output)
            return output

        original_head = model.head.forward

        def head_delegate(value: torch.Tensor) -> torch.Tensor:
            output = original_head(value)
            captured_logits.append(output)
            return output

        with ExitStack() as stack:
            representation = stack.enter_context(
                patch.object(
                    model.representation,
                    "forward",
                    side_effect=representation_delegate,
                )
            )
            blocks = tuple(
                stack.enter_context(
                    patch.object(
                        block,
                        "forward",
                        side_effect=block_delegate(block.forward),
                    )
                )
                for block in model.blocks
            )
            final_norm = stack.enter_context(
                patch.object(model.final_norm, "forward", side_effect=final_delegate)
            )
            head = stack.enter_context(
                patch.object(model.head, "forward", side_effect=head_delegate)
            )
            logits = model(token_ids)

        self.assertIs(type(logits), torch.Tensor)
        self.assertEqual(tuple(logits.shape), (3, 81))
        representation.assert_called_once()
        self.assertIs(representation.call_args.args[0], token_ids)
        previous = captured_representation[0]
        for index, call in enumerate(blocks):
            call.assert_called_once()
            self.assertIs(call.call_args.args[0], previous)
            previous = captured_block_outputs[index]
        final_norm.assert_called_once()
        self.assertIs(final_norm.call_args.args[0], previous)
        head.assert_called_once()
        self.assertIs(head.call_args.args[0], captured_final[0])
        self.assertIs(logits, captured_logits[0])

    def test_forward_matches_independent_eval_composition(self) -> None:
        model = self.model().eval()
        token_ids = torch.tensor([3, 1, 4, 1], dtype=torch.long)
        logits = model(token_ids)
        hidden = model.representation(token_ids)
        for block in model.blocks:
            hidden = block(hidden)
        hidden = model.final_norm(hidden)
        expected = hidden @ model.head.output_weight + model.head.output_bias
        self.assertTrue(torch.equal(logits, expected))

    def test_separate_next_token_loss_alignment_matches_reference(self) -> None:
        model = self.model().eval()
        source = torch.tensor([12, 4, 9, 4, 7], dtype=torch.long)
        inputs = source[:-1]
        targets = source[1:]
        logits = model(inputs)
        loss = explicit_cross_entropy(logits, targets)
        reference = F.cross_entropy(logits, targets)
        self.assertEqual(tuple(inputs.shape), tuple(targets.shape))
        self.assertEqual(tuple(logits.shape), (4, 81))
        self.assertEqual(tuple(loss.shape), ())
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(torch.allclose(loss, reference, rtol=1e-6, atol=1e-6))

    def test_inspection_is_frozen_hidden_and_same_call(self) -> None:
        model = self.model()
        token_ids = torch.tensor([1, 2, 3], dtype=torch.long)
        captured_representation: list[torch.Tensor] = []
        captured_blocks: list[TransformerBlockInspection] = []
        captured_final = []
        captured_logits: list[torch.Tensor] = []

        original_representation = model.representation.forward

        def representation_delegate(value: torch.Tensor) -> torch.Tensor:
            output = original_representation(value)
            captured_representation.append(output)
            return output

        def block_delegate(original):
            def delegated(value: torch.Tensor) -> TransformerBlockInspection:
                output = original(value)
                captured_blocks.append(output)
                return output

            return delegated

        original_final = model.final_norm.inspect

        def final_delegate(value: torch.Tensor):
            output = original_final(value)
            captured_final.append(output)
            return output

        original_head = model.head.forward

        def head_delegate(value: torch.Tensor) -> torch.Tensor:
            output = original_head(value)
            captured_logits.append(output)
            return output

        with ExitStack() as stack:
            representation = stack.enter_context(
                patch.object(
                    model.representation,
                    "forward",
                    side_effect=representation_delegate,
                )
            )
            blocks = tuple(
                stack.enter_context(
                    patch.object(
                        block,
                        "inspect",
                        side_effect=block_delegate(block.inspect),
                    )
                )
                for block in model.blocks
            )
            final_norm = stack.enter_context(
                patch.object(model.final_norm, "inspect", side_effect=final_delegate)
            )
            head = stack.enter_context(
                patch.object(model.head, "forward", side_effect=head_delegate)
            )
            forward_trap = stack.enter_context(patch.object(model, "forward"))
            inspection = model.inspect(token_ids)

        forward_trap.assert_not_called()
        representation.assert_called_once()
        self.assertIs(inspection.embedding_representation, captured_representation[0])
        previous = inspection.embedding_representation
        self.assertEqual(len(inspection.block_inspections), 4)
        for index, call in enumerate(blocks):
            call.assert_called_once()
            self.assertIs(call.call_args.args[0], previous)
            self.assertIs(inspection.block_inspections[index], captured_blocks[index])
            previous = inspection.block_inspections[index].output
        final_norm.assert_called_once()
        self.assertIs(final_norm.call_args.args[0], previous)
        self.assertIs(inspection.final_normalization, captured_final[0])
        head.assert_called_once()
        self.assertIs(
            head.call_args.args[0],
            inspection.final_normalization.output,
        )
        self.assertIs(inspection.logits, captured_logits[0])
        self.assertNotIn("tensor", repr(inspection))
        with self.assertRaises(FrozenInstanceError):
            inspection.logits = torch.zeros_like(inspection.logits)  # type: ignore[misc]

    def test_inspection_packaging_validation_invariant_oracle(self) -> None:
        token_ids = _tokens(3)

        def malformed_factory(field: str):
            def factory(**values):
                if field == "block_count":
                    values["block_inspections"] = ()
                elif field == "handoff":
                    values["embedding_representation"] = object()
                elif field == "final":
                    values["final_normalization"] = object()
                elif field == "logits":
                    values["logits"] = values["logits"].clone()
                return SimpleNamespace(**values)

            return factory

        cases = (
            ("block_count", "mini_gpt.inspection.block_count"),
            ("handoff", "mini_gpt.inspection.handoff_identity"),
            ("final", "mini_gpt.inspection.final_normalization_identity"),
            ("logits", "mini_gpt.inspection.logits_identity"),
        )
        for field, invariant in cases:
            with self.subTest(field=field):
                model = self.model().eval()
                with patch.object(
                    mini_gpt_module,
                    "MiniGPTInspection",
                    side_effect=malformed_factory(field),
                ):
                    with self.assertRaises(MiniGPTContractError) as raised:
                        model.inspect(token_ids)
                self.assertEqual(raised.exception.details["invariant"], invariant)

    def test_final_normalization_matches_independent_population_reference(self) -> None:
        model = self.model().eval()
        inspection = model.inspect(torch.tensor([2, 7, 1], dtype=torch.long))
        incoming = inspection.block_inspections[-1].output
        mean = incoming.mean(dim=-1, keepdim=True)
        variance = ((incoming - mean) * (incoming - mean)).mean(
            dim=-1, keepdim=True
        )
        normalized = (incoming - mean) * torch.rsqrt(variance + 1e-5)
        expected = normalized * model.final_norm.gamma + model.final_norm.beta
        self.assertTrue(torch.equal(inspection.final_normalization.mean, mean))
        self.assertTrue(torch.equal(inspection.final_normalization.variance, variance))
        self.assertTrue(torch.equal(inspection.final_normalization.normalized, normalized))
        self.assertTrue(torch.equal(inspection.final_normalization.output, expected))

    def test_loss_backward_reaches_embedding_representation_and_all_90_parameters(self) -> None:
        model = self.model().eval()
        token_ids = torch.tensor([1, 5, 2, 8, 3], dtype=torch.long)
        targets = torch.tensor([5, 2, 8, 3, 13], dtype=torch.long)
        snapshot = _parameter_snapshot(model)
        inspection = model.inspect(token_ids)
        inspection.embedding_representation.retain_grad()
        loss = explicit_cross_entropy(inspection.logits, targets)
        loss.backward()
        self.assertIsNotNone(inspection.embedding_representation.grad)
        assert inspection.embedding_representation.grad is not None
        self.assertTrue(torch.isfinite(inspection.embedding_representation.grad).all())
        self.assertTrue(torch.any(inspection.embedding_representation.grad != 0).item())
        parameters = tuple(model.parameters())
        self.assertEqual(len(parameters), 90)
        for position, parameter in enumerate(parameters):
            with self.subTest(position=position):
                self.assertIsNotNone(parameter.grad)
                assert parameter.grad is not None
                self.assertEqual(tuple(parameter.grad.shape), tuple(parameter.shape))
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertTrue(torch.any(parameter.grad != 0).item())
        after = _parameter_snapshot(model)
        for before_item, after_item in zip(snapshot, after, strict=True):
            self.assertEqual(before_item[0], after_item[0])
            self.assertTrue(torch.equal(before_item[1], after_item[1]))
            self.assertEqual(before_item[2:], after_item[2:])

    def test_phase7_numerical_boundary_invariants(self) -> None:
        token_ids = _tokens(2)
        model = self.model().eval()
        with patch.object(
            model.representation,
            "forward",
            return_value=torch.full((2, 32), torch.nan),
        ):
            with self.assertRaises(MiniGPTNumericalError) as embedding:
                model(token_ids)
        self.assertEqual(embedding.exception.details["invariant"], "mini_gpt.embedding_output.finite")

        model = self.model().eval()
        with patch.object(
            model.blocks[1],
            "forward",
            return_value=torch.full((2, 32), torch.nan),
        ):
            with self.assertRaises(MiniGPTNumericalError) as block:
                model(token_ids)
        self.assertEqual(block.exception.details["invariant"], "mini_gpt.block_output.1.finite")

        model = self.model().eval()
        with patch.object(
            model.final_norm,
            "forward",
            return_value=torch.full((2, 32), torch.nan),
        ):
            with self.assertRaises(MiniGPTNumericalError) as final_norm:
                model(token_ids)
        self.assertEqual(
            final_norm.exception.details["invariant"],
            "mini_gpt.final_normalization_output.finite",
        )

        model = self.model().eval()
        with patch.object(
            model.head,
            "forward",
            return_value=torch.full((2, 81), torch.nan),
        ):
            with self.assertRaises(MiniGPTNumericalError) as output:
                model(token_ids)
        self.assertEqual(output.exception.details["invariant"], "mini_gpt.output.finite")


class CausalityTests(MiniGPTTestCase):
    def test_evaluation_causality_uses_equal_total_lengths(self) -> None:
        model = self.model().eval()
        original = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        future_changed = torch.tensor([1, 2, 30, 40, 50], dtype=torch.long)
        original_logits = model(original)
        changed_logits = model(future_changed)
        self.assertTrue(torch.equal(original_logits[:2], changed_logits[:2]))

        original_embeddings = model.representation(original)
        changed_embeddings = model.representation(future_changed)
        self.assertFalse(
            torch.equal(
                original_embeddings.sum(dim=0),
                changed_embeddings.sum(dim=0),
            )
        )

    def test_paired_training_causality_uses_equal_states_and_histories(self) -> None:
        first = self.model().train()
        second = self.model().train()
        original = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        future_changed = torch.tensor([1, 2, 30, 40, 50], dtype=torch.long)
        self.assertTrue(
            all(
                torch.equal(left, right)
                for left, right in zip(
                    first.parameters(), second.parameters(), strict=True
                )
            )
        )
        _assert_generator_states_equal(
            self,
            _generator_states(first),
            _generator_states(second),
        )
        first_logits = first(original)
        second_logits = second(future_changed)
        self.assertTrue(torch.equal(first_logits[:2], second_logits[:2]))

    def test_permitted_earlier_token_can_change_later_logits(self) -> None:
        model = self.model().eval()
        original = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        history_changed = torch.tensor([17, 2, 3, 4], dtype=torch.long)
        self.assertFalse(torch.equal(model(original)[3], model(history_changed)[3]))


class StackFailureSemanticsTests(MiniGPTTestCase):
    def _advance_control(
        self,
        model: MiniGPT,
        token_ids: torch.Tensor,
        count: int,
        *,
        inspect_blocks: bool,
    ) -> None:
        hidden = model.representation(token_ids)
        for block in tuple(model.blocks)[:count]:
            if inspect_blocks:
                hidden = block.inspect(hidden).output
            else:
                hidden = block(hidden)

    def _assert_failure_state_partition(
        self,
        observed: MiniGPT,
        control: MiniGPT,
        before: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        failing_index: int,
    ) -> None:
        observed_states = _generator_states(observed)
        control_states = _generator_states(control)
        for index in range(4):
            expected = control_states[index] if index < failing_index else before[index]
            self.assertTrue(torch.equal(observed_states[index][0], expected[0]))
            self.assertTrue(torch.equal(observed_states[index][1], expected[1]))

    def test_forward_block_failures_commit_earlier_rollback_failing_and_skip_later(self) -> None:
        token_ids = _tokens(4)
        for failing_index in (0, 2, 3):
            with self.subTest(failing_index=failing_index):
                model = self.model().train()
                control = self.model().train()
                before = _generator_states(model)
                self._advance_control(
                    control,
                    token_ids,
                    failing_index,
                    inspect_blocks=False,
                )
                target = model.blocks[failing_index]
                original_compute = target._compute_forward

                def fail_after_compute(value: torch.Tensor) -> torch.Tensor:
                    original_compute(value)
                    raise SentinelFailure("forward block failure")

                global_before = torch.get_rng_state().clone()
                with ExitStack() as stack:
                    calls = tuple(
                        stack.enter_context(
                            patch.object(block, "forward", wraps=block.forward)
                        )
                        for block in model.blocks
                    )
                    stack.enter_context(
                        patch.object(
                            target,
                            "_compute_forward",
                            side_effect=fail_after_compute,
                        )
                    )
                    with self.assertRaises(SentinelFailure):
                        model(token_ids)
                for index, call in enumerate(calls):
                    self.assertEqual(call.call_count, 1 if index <= failing_index else 0)
                self._assert_failure_state_partition(
                    model, control, before, failing_index
                )
                self.assertTrue(torch.equal(torch.get_rng_state(), global_before))

    def test_inspect_block_failures_commit_earlier_rollback_failing_and_skip_later(self) -> None:
        token_ids = _tokens(4)
        for failing_index in (0, 2, 3):
            with self.subTest(failing_index=failing_index):
                model = self.model().train()
                control = self.model().train()
                before = _generator_states(model)
                self._advance_control(
                    control,
                    token_ids,
                    failing_index,
                    inspect_blocks=True,
                )
                target = model.blocks[failing_index]
                original_compute = target._compute_inspection

                def fail_after_compute(value: torch.Tensor) -> TransformerBlockInspection:
                    original_compute(value)
                    raise SentinelFailure("inspect block failure")

                global_before = torch.get_rng_state().clone()
                with ExitStack() as stack:
                    calls = tuple(
                        stack.enter_context(
                            patch.object(block, "inspect", wraps=block.inspect)
                        )
                        for block in model.blocks
                    )
                    stack.enter_context(
                        patch.object(
                            target,
                            "_compute_inspection",
                            side_effect=fail_after_compute,
                        )
                    )
                    with self.assertRaises(SentinelFailure):
                        model.inspect(token_ids)
                for index, call in enumerate(calls):
                    self.assertEqual(call.call_count, 1 if index <= failing_index else 0)
                self._assert_failure_state_partition(
                    model, control, before, failing_index
                )
                self.assertTrue(torch.equal(torch.get_rng_state(), global_before))

    def test_forward_post_stack_failures_leave_all_block_advancement_committed(self) -> None:
        token_ids = _tokens(4)
        for stage in ("final_norm", "head"):
            with self.subTest(stage=stage):
                model = self.model().train()
                control = self.model().train()
                self._advance_control(control, token_ids, 4, inspect_blocks=False)
                global_before = torch.get_rng_state().clone()
                with ExitStack() as stack:
                    calls = tuple(
                        stack.enter_context(
                            patch.object(block, "forward", wraps=block.forward)
                        )
                        for block in model.blocks
                    )
                    final_call = stack.enter_context(
                        patch.object(
                            model.final_norm,
                            "forward",
                            wraps=model.final_norm.forward,
                        )
                    )
                    head_call = stack.enter_context(
                        patch.object(model.head, "forward", wraps=model.head.forward)
                    )
                    if stage == "final_norm":
                        final_call.side_effect = SentinelFailure(stage)
                    else:
                        head_call.side_effect = SentinelFailure(stage)
                    with self.assertRaises(SentinelFailure):
                        model(token_ids)
                self.assertEqual(final_call.call_count, 1)
                self.assertEqual(head_call.call_count, 0 if stage == "final_norm" else 1)
                self.assertTrue(all(call.call_count == 1 for call in calls))
                _assert_generator_states_equal(
                    self,
                    _generator_states(model),
                    _generator_states(control),
                )
                self.assertTrue(torch.equal(torch.get_rng_state(), global_before))

    def test_inspect_post_stack_failures_and_packaging_keep_committed_states(self) -> None:
        token_ids = _tokens(4)
        for stage in ("final_norm", "head", "packaging"):
            with self.subTest(stage=stage):
                model = self.model().train()
                control = self.model().train()
                self._advance_control(control, token_ids, 4, inspect_blocks=True)
                global_before = torch.get_rng_state().clone()
                with ExitStack() as stack:
                    block_calls = tuple(
                        stack.enter_context(
                            patch.object(block, "inspect", wraps=block.inspect)
                        )
                        for block in model.blocks
                    )
                    final_call = stack.enter_context(
                        patch.object(
                            model.final_norm,
                            "inspect",
                            wraps=model.final_norm.inspect,
                        )
                    )
                    head_call = stack.enter_context(
                        patch.object(model.head, "forward", wraps=model.head.forward)
                    )
                    if stage == "final_norm":
                        final_call.side_effect = SentinelFailure(stage)
                    elif stage == "head":
                        head_call.side_effect = SentinelFailure(stage)
                    else:
                        stack.enter_context(
                            patch.object(
                                mini_gpt_module,
                                "MiniGPTInspection",
                                side_effect=SentinelFailure(stage),
                            )
                        )
                    with self.assertRaises(SentinelFailure):
                        model.inspect(token_ids)
                self.assertTrue(all(call.call_count == 1 for call in block_calls))
                self.assertEqual(final_call.call_count, 1)
                self.assertEqual(head_call.call_count, 0 if stage == "final_norm" else 1)
                _assert_generator_states_equal(
                    self,
                    _generator_states(model),
                    _generator_states(control),
                )
                self.assertTrue(torch.equal(torch.get_rng_state(), global_before))

    def test_evaluation_forward_and_inspect_advance_no_dropout_stream(self) -> None:
        model = self.model().eval()
        before = _generator_states(model)
        model(_tokens())
        model.inspect(_tokens())
        _assert_generator_states_equal(self, _generator_states(model), before)

    def test_forward_failure_paths_preserve_exact_stage_handoff_identities(self) -> None:
        token_ids = _tokens(4)
        for stage in ("block_2", "final_norm", "head"):
            with self.subTest(stage=stage):
                model = self.model().train()
                control = self.model().train()
                before = _generator_states(model)
                completed_blocks = 2 if stage == "block_2" else 4
                self._advance_control(
                    control,
                    token_ids,
                    completed_blocks,
                    inspect_blocks=False,
                )
                global_before = torch.get_rng_state().clone()
                representation_outputs: list[torch.Tensor] = []
                block_inputs: list[torch.Tensor | None] = [None] * 4
                block_outputs: list[torch.Tensor | None] = [None] * 4
                final_inputs: list[torch.Tensor] = []
                final_outputs: list[torch.Tensor] = []
                head_inputs: list[torch.Tensor] = []

                original_representation = model.representation.forward

                def representation_delegate(value: torch.Tensor) -> torch.Tensor:
                    output = original_representation(value)
                    representation_outputs.append(output)
                    return output

                def block_delegate(index: int, original):
                    def delegated(value: torch.Tensor) -> torch.Tensor:
                        block_inputs[index] = value
                        output = original(value)
                        block_outputs[index] = output
                        return output

                    return delegated

                original_final = model.final_norm.forward

                def final_delegate(value: torch.Tensor) -> torch.Tensor:
                    final_inputs.append(value)
                    if stage == "final_norm":
                        raise SentinelFailure(stage)
                    output = original_final(value)
                    final_outputs.append(output)
                    return output

                original_head = model.head.forward

                def head_delegate(value: torch.Tensor) -> torch.Tensor:
                    head_inputs.append(value)
                    if stage == "head":
                        raise SentinelFailure(stage)
                    return original_head(value)

                original_block_forwards = tuple(block.forward for block in model.blocks)
                with ExitStack() as stack:
                    representation_call = stack.enter_context(
                        patch.object(
                            model.representation,
                            "forward",
                            side_effect=representation_delegate,
                        )
                    )
                    block_calls = tuple(
                        stack.enter_context(
                            patch.object(
                                block,
                                "forward",
                                side_effect=block_delegate(index, original),
                            )
                        )
                        for index, (block, original) in enumerate(
                            zip(
                                model.blocks,
                                original_block_forwards,
                                strict=True,
                            )
                        )
                    )
                    final_call = stack.enter_context(
                        patch.object(
                            model.final_norm,
                            "forward",
                            side_effect=final_delegate,
                        )
                    )
                    head_call = stack.enter_context(
                        patch.object(
                            model.head,
                            "forward",
                            side_effect=head_delegate,
                        )
                    )
                    if stage == "block_2":
                        target = model.blocks[2]
                        original_compute = target._compute_forward

                        def fail_after_compute(value: torch.Tensor) -> torch.Tensor:
                            original_compute(value)
                            raise SentinelFailure(stage)

                        stack.enter_context(
                            patch.object(
                                target,
                                "_compute_forward",
                                side_effect=fail_after_compute,
                            )
                        )
                    with self.assertRaises(SentinelFailure):
                        model(token_ids)

                representation_call.assert_called_once()
                self.assertIs(representation_call.call_args.args[0], token_ids)
                self.assertIs(block_inputs[0], representation_outputs[0])
                attempted_blocks = 3 if stage == "block_2" else 4
                for index, call in enumerate(block_calls):
                    self.assertEqual(
                        call.call_count,
                        1 if index < attempted_blocks else 0,
                    )
                successful_blocks = 2 if stage == "block_2" else 4
                for index in range(successful_blocks - 1):
                    self.assertIs(block_inputs[index + 1], block_outputs[index])
                if stage == "block_2":
                    self.assertIs(block_inputs[2], block_outputs[1])
                    self.assertEqual(final_call.call_count, 0)
                    self.assertEqual(head_call.call_count, 0)
                    self._assert_failure_state_partition(model, control, before, 2)
                else:
                    self.assertIs(final_inputs[0], block_outputs[3])
                    self.assertEqual(final_call.call_count, 1)
                    if stage == "final_norm":
                        self.assertEqual(head_call.call_count, 0)
                    else:
                        self.assertEqual(head_call.call_count, 1)
                        self.assertIs(head_inputs[0], final_outputs[0])
                    _assert_generator_states_equal(
                        self,
                        _generator_states(model),
                        _generator_states(control),
                    )
                self.assertTrue(torch.equal(torch.get_rng_state(), global_before))

    def test_inspect_failure_paths_preserve_exact_stage_handoff_identities(self) -> None:
        token_ids = _tokens(4)
        for stage in ("block_2", "final_norm", "head", "packaging"):
            with self.subTest(stage=stage):
                model = self.model().train()
                control = self.model().train()
                before = _generator_states(model)
                completed_blocks = 2 if stage == "block_2" else 4
                self._advance_control(
                    control,
                    token_ids,
                    completed_blocks,
                    inspect_blocks=True,
                )
                global_before = torch.get_rng_state().clone()
                representation_outputs: list[torch.Tensor] = []
                block_inputs: list[torch.Tensor | None] = [None] * 4
                block_outputs: list[TransformerBlockInspection | None] = [None] * 4
                final_inputs: list[torch.Tensor] = []
                final_outputs = []
                head_inputs: list[torch.Tensor] = []
                head_outputs: list[torch.Tensor] = []
                packaging_values: list[dict[str, object]] = []

                original_representation = model.representation.forward

                def representation_delegate(value: torch.Tensor) -> torch.Tensor:
                    output = original_representation(value)
                    representation_outputs.append(output)
                    return output

                def block_delegate(index: int, original):
                    def delegated(value: torch.Tensor) -> TransformerBlockInspection:
                        block_inputs[index] = value
                        output = original(value)
                        block_outputs[index] = output
                        return output

                    return delegated

                original_final = model.final_norm.inspect

                def final_delegate(value: torch.Tensor):
                    final_inputs.append(value)
                    if stage == "final_norm":
                        raise SentinelFailure(stage)
                    output = original_final(value)
                    final_outputs.append(output)
                    return output

                original_head = model.head.forward

                def head_delegate(value: torch.Tensor) -> torch.Tensor:
                    head_inputs.append(value)
                    if stage == "head":
                        raise SentinelFailure(stage)
                    output = original_head(value)
                    head_outputs.append(output)
                    return output

                def packaging_delegate(**values):
                    packaging_values.append(values)
                    raise SentinelFailure(stage)

                original_block_inspects = tuple(block.inspect for block in model.blocks)
                with ExitStack() as stack:
                    representation_call = stack.enter_context(
                        patch.object(
                            model.representation,
                            "forward",
                            side_effect=representation_delegate,
                        )
                    )
                    block_calls = tuple(
                        stack.enter_context(
                            patch.object(
                                block,
                                "inspect",
                                side_effect=block_delegate(index, original),
                            )
                        )
                        for index, (block, original) in enumerate(
                            zip(
                                model.blocks,
                                original_block_inspects,
                                strict=True,
                            )
                        )
                    )
                    final_call = stack.enter_context(
                        patch.object(
                            model.final_norm,
                            "inspect",
                            side_effect=final_delegate,
                        )
                    )
                    head_call = stack.enter_context(
                        patch.object(
                            model.head,
                            "forward",
                            side_effect=head_delegate,
                        )
                    )
                    if stage == "block_2":
                        target = model.blocks[2]
                        original_compute = target._compute_inspection

                        def fail_after_compute(
                            value: torch.Tensor,
                        ) -> TransformerBlockInspection:
                            original_compute(value)
                            raise SentinelFailure(stage)

                        stack.enter_context(
                            patch.object(
                                target,
                                "_compute_inspection",
                                side_effect=fail_after_compute,
                            )
                        )
                    elif stage == "packaging":
                        stack.enter_context(
                            patch.object(
                                mini_gpt_module,
                                "MiniGPTInspection",
                                side_effect=packaging_delegate,
                            )
                        )
                    with self.assertRaises(SentinelFailure):
                        model.inspect(token_ids)

                representation_call.assert_called_once()
                self.assertIs(representation_call.call_args.args[0], token_ids)
                self.assertIs(block_inputs[0], representation_outputs[0])
                attempted_blocks = 3 if stage == "block_2" else 4
                for index, call in enumerate(block_calls):
                    self.assertEqual(
                        call.call_count,
                        1 if index < attempted_blocks else 0,
                    )
                successful_blocks = 2 if stage == "block_2" else 4
                for index in range(successful_blocks - 1):
                    assert block_outputs[index] is not None
                    self.assertIs(
                        block_inputs[index + 1],
                        block_outputs[index].output,
                    )
                if stage == "block_2":
                    assert block_outputs[1] is not None
                    self.assertIs(block_inputs[2], block_outputs[1].output)
                    self.assertEqual(final_call.call_count, 0)
                    self.assertEqual(head_call.call_count, 0)
                    self._assert_failure_state_partition(model, control, before, 2)
                else:
                    assert block_outputs[3] is not None
                    self.assertIs(final_inputs[0], block_outputs[3].output)
                    self.assertEqual(final_call.call_count, 1)
                    if stage == "final_norm":
                        self.assertEqual(head_call.call_count, 0)
                    else:
                        self.assertEqual(head_call.call_count, 1)
                        self.assertIs(head_inputs[0], final_outputs[0].output)
                    if stage == "packaging":
                        self.assertEqual(len(packaging_values), 1)
                        self.assertIs(
                            packaging_values[0]["embedding_representation"],
                            representation_outputs[0],
                        )
                        self.assertTrue(
                            all(
                                observed is expected
                                for observed, expected in zip(
                                    packaging_values[0]["block_inspections"],
                                    block_outputs,
                                    strict=True,
                                )
                            )
                        )
                        self.assertIs(
                            packaging_values[0]["final_normalization"],
                            final_outputs[0],
                        )
                        self.assertIs(
                            packaging_values[0]["logits"],
                            head_outputs[0],
                        )
                    _assert_generator_states_equal(
                        self,
                        _generator_states(model),
                        _generator_states(control),
                    )
                self.assertTrue(torch.equal(torch.get_rng_state(), global_before))


class PhaseBoundaryTests(MiniGPTTestCase):
    def test_source_contains_no_forbidden_later_phase_or_opaque_model_path(self) -> None:
        source = (REPOSITORY_ROOT / "src/sebgpt/model/mini_gpt.py").read_text(
            encoding="utf-8"
        )
        forbidden = (
            "SimpleNeuralLanguageModel",
            "torch.optim",
            "nn.Transformer",
            "torch.nn.Transformer",
            "sebgpt.data",
            "checkpoint",
            "generate_text",
            "sampling_temperature",
        )
        for value in forbidden:
            with self.subTest(value=value):
                self.assertNotIn(value, source)
        self.assertNotIn("explicit_cross_entropy", source)


if __name__ == "__main__":
    unittest.main()
