"""Synthetic isolated tests for Phase 8 checkpoint durability and restore."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import stat
import shutil
import sys
import tempfile
import unittest
from collections import OrderedDict
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import sebgpt.training.phase8_checkpoint as checkpoint  # noqa: E402
import sebgpt.training.phase8_optimization as optimization  # noqa: E402
from sebgpt.model.mini_gpt import MiniGPT  # noqa: E402
from sebgpt.tokenization.vocabulary_artifact import (  # noqa: E402
    load_accepted_vocabulary_binding,
)
from sebgpt.training import (  # noqa: E402
    Phase8CheckpointError,
    Phase8Evaluation,
    Phase8GovernanceError,
    Phase8LogicalBatch,
    Phase8Progress,
    Phase8PublicationError,
    Phase8Window,
    configure_phase8_runtime,
    create_phase8_epoch_order,
    create_phase8_optimizer,
    evaluate_phase8_split,
    iter_phase8_logical_batches,
    load_phase8_checkpoint,
    load_phase8_configuration,
    run_phase8_logical_batch,
    save_phase8_checkpoint,
)
from sebgpt.training.phase8_types import (  # noqa: E402
    _make_metric_state,
    _make_training_state,
)


CODE_COMMIT = "809834323d53407cb4a54ae539585bb3d78856eb"
PRE_REGISTRATION_COMMIT = "b" * 40
RUN_ID = "EXP-20260911-01"
EXPECTED_MODE_NAMES = (
    "",
    "representation",
    "blocks",
    "blocks.0",
    "blocks.0.norm1",
    "blocks.0.attention",
    "blocks.0.attention.heads",
    "blocks.0.attention.heads.0",
    "blocks.0.attention.heads.1",
    "blocks.0.attention.heads.2",
    "blocks.0.attention.heads.3",
    "blocks.0.attention_dropout",
    "blocks.0.norm2",
    "blocks.0.feed_forward",
    "blocks.0.feed_forward_dropout",
    "blocks.1",
    "blocks.1.norm1",
    "blocks.1.attention",
    "blocks.1.attention.heads",
    "blocks.1.attention.heads.0",
    "blocks.1.attention.heads.1",
    "blocks.1.attention.heads.2",
    "blocks.1.attention.heads.3",
    "blocks.1.attention_dropout",
    "blocks.1.norm2",
    "blocks.1.feed_forward",
    "blocks.1.feed_forward_dropout",
    "blocks.2",
    "blocks.2.norm1",
    "blocks.2.attention",
    "blocks.2.attention.heads",
    "blocks.2.attention.heads.0",
    "blocks.2.attention.heads.1",
    "blocks.2.attention.heads.2",
    "blocks.2.attention.heads.3",
    "blocks.2.attention_dropout",
    "blocks.2.norm2",
    "blocks.2.feed_forward",
    "blocks.2.feed_forward_dropout",
    "blocks.3",
    "blocks.3.norm1",
    "blocks.3.attention",
    "blocks.3.attention.heads",
    "blocks.3.attention.heads.0",
    "blocks.3.attention.heads.1",
    "blocks.3.attention.heads.2",
    "blocks.3.attention.heads.3",
    "blocks.3.attention_dropout",
    "blocks.3.norm2",
    "blocks.3.feed_forward",
    "blocks.3.feed_forward_dropout",
    "final_norm",
    "head",
)
EXPECTED_MODE_STATE = tuple((name, True) for name in EXPECTED_MODE_NAMES)


def _model(repository_root: Path) -> MiniGPT:
    vocabulary = load_accepted_vocabulary_binding(repository_root)
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


def _copy_authority(repository_root: Path) -> None:
    paths = (
        "requirements.lock",
        "artifacts/tokenizers/shakespeare-code-point-v1/vocabulary.json",
        "docs/data/shakespeare-eight-play-manifest.json",
        "data/processed/shakespeare-eight-play/processing-manifest.json",
    )
    for relative in paths:
        target = repository_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPOSITORY_ROOT / relative, target)
    (repository_root / "checkpoints").mkdir()


def _windows() -> tuple[Phase8Window, ...]:
    return (
        Phase8Window("hamlet", 1, "train", 0, (1, 2, 3), (2, 3, 4)),
        Phase8Window("hamlet", 1, "train", 256, (5,), (6,)),
    )


def _state(repository_root: Path):
    model = _model(repository_root)
    optimizer = create_phase8_optimizer(model)
    windows = _windows()
    run_phase8_logical_batch(
        model,
        optimizer,
        windows,
        Phase8LogicalBatch(1, 0, (0, 1), 4),
    )
    training = Phase8Evaluation("train", 1.25, 4, 2)
    validation = Phase8Evaluation("validation", 1.5, 2, 1)
    metrics = _make_metric_state(
        initialized_training=Phase8Evaluation("train", 2.0, 4, 2),
        initialized_validation=Phase8Evaluation("validation", 2.1, 2, 1),
        epoch_training=(training,),
        epoch_validation=(validation,),
        current_training=training,
        current_validation=validation,
        best_validation_loss=validation.loss,
        best_validation_epoch=1,
        best_logical_id="epoch-0001",
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(8001)
    return _make_training_state(
        run_id=RUN_ID,
        code_commit=CODE_COMMIT,
        configuration=load_phase8_configuration(),
        model=model,
        optimizer=optimizer,
        order_generator=generator,
        progress=Phase8Progress(1, 2, 0, 1, 2, 4),
        metrics=metrics,
        global_cpu_rng_state=torch.get_rng_state().clone(),
        resumed_from_checkpoint_sha256=None,
        resume_checkpoint_sha256s=(),
    )


def _epoch_two_state(repository_root: Path):
    prior = _state(repository_root)
    windows = _windows()
    run_phase8_logical_batch(
        prior.model,
        prior.optimizer,
        windows,
        Phase8LogicalBatch(2, 0, (1, 0), 4),
    )
    training = Phase8Evaluation("train", 1.1, 4, 2)
    validation = Phase8Evaluation("validation", 1.6, 2, 1)
    metrics = _make_metric_state(
        initialized_training=prior.metrics.initialized_training,
        initialized_validation=prior.metrics.initialized_validation,
        epoch_training=(*prior.metrics.epoch_training, training),
        epoch_validation=(*prior.metrics.epoch_validation, validation),
        current_training=training,
        current_validation=validation,
        best_validation_loss=prior.metrics.best_validation_loss,
        best_validation_epoch=1,
        best_logical_id="epoch-0001",
    )
    return _make_training_state(
        run_id=prior.run_id,
        code_commit=prior.code_commit,
        configuration=prior.configuration,
        model=prior.model,
        optimizer=prior.optimizer,
        order_generator=prior.order_generator,
        progress=Phase8Progress(2, 3, 0, 2, 4, 8),
        metrics=metrics,
        global_cpu_rng_state=torch.get_rng_state().clone(),
        resumed_from_checkpoint_sha256=None,
        resume_checkpoint_sha256s=(),
    )


def _state_dict_equal(left: dict, right: dict) -> bool:
    if tuple(left) != tuple(right):
        return False
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, torch.Tensor):
            if not isinstance(b, torch.Tensor) or not torch.equal(a, b):
                return False
        elif type(a) is dict:
            if type(b) is not dict or not _state_dict_equal(a, b):
                return False
        elif type(a) is list:
            if type(b) is not list or len(a) != len(b):
                return False
            for x, y in zip(a, b, strict=True):
                if isinstance(x, dict):
                    if not isinstance(y, dict) or not _state_dict_equal(x, y):
                        return False
                elif x != y:
                    return False
        elif a != b:
            return False
    return True


def _mode_state_corruptions():
    def replace_entry(modes, position, replacement):
        values = list(modes)
        values[position] = replacement
        return tuple(values)

    def reorder(modes):
        values = list(modes)
        values[3], values[4] = values[4], values[3]
        return tuple(values)

    return (
        ("empty", lambda modes: ()),
        ("missing", lambda modes: modes[:-1]),
        ("duplicate", lambda modes: (*modes[:-1], modes[0])),
        ("reordered", reorder),
        ("extra", lambda modes: (*modes, ("unexpected", True))),
        (
            "wrong_name",
            lambda modes: replace_entry(modes, 4, ("blocks.0.wrong", True)),
        ),
        (
            "wrong_mode",
            lambda modes: replace_entry(modes, 4, (modes[4][0], False)),
        ),
        (
            "wrong_name_type",
            lambda modes: replace_entry(modes, 4, (4, True)),
        ),
        (
            "wrong_mode_type",
            lambda modes: replace_entry(modes, 4, (modes[4][0], 1)),
        ),
    )


class Phase8CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        configure_phase8_runtime()

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.repository_root = Path(self.temporary.name).resolve()
        _copy_authority(self.repository_root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _patch_authority(self):
        return (
            patch.object(
                checkpoint,
                "_validate_live_repository",
                return_value=PRE_REGISTRATION_COMMIT,
            ),
            patch.object(checkpoint, "_git", return_value=CODE_COMMIT),
        )

    def _save(self):
        first, second = self._patch_authority()
        with first, second:
            return save_phase8_checkpoint(
                _state(self.repository_root),
                self.repository_root,
            )

    def _load(self):
        first, second = self._patch_authority()
        with first, second:
            return load_phase8_checkpoint(
                self.repository_root,
                run_id=RUN_ID,
                role="latest",
            )

    def _corrupt_role_payload(self, role, mutate) -> None:
        run_root = self.repository_root / "checkpoints/phase8" / RUN_ID
        catalog_path = run_root / "checkpoint-catalog.json"
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        reference = catalog[role]
        prior_digest = reference["sha256"]
        object_path = run_root / reference["relative_path"]
        payload = torch.load(
            io.BytesIO(object_path.read_bytes()),
            map_location="cpu",
            weights_only=True,
        )
        mutate(payload)
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        content = buffer.getvalue()
        digest = hashlib.sha256(content).hexdigest()
        (run_root / "objects" / f"{digest}.pt").write_bytes(content)
        for catalog_role in ("latest", "best_validation"):
            if catalog[catalog_role]["sha256"] == prior_digest:
                catalog[catalog_role]["sha256"] = digest
                catalog[catalog_role]["relative_path"] = f"objects/{digest}.pt"
        catalog_path.write_text(
            json.dumps(
                catalog,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )

    def test_checkpoint_save_and_load_validate_corrected_provenance(self) -> None:
        state = _state(self.repository_root)
        with patch.object(
            checkpoint,
            "_validate_live_repository",
            return_value=PRE_REGISTRATION_COMMIT,
        ) as validate:
            save_phase8_checkpoint(state, self.repository_root)
            restored = load_phase8_checkpoint(
                self.repository_root,
                run_id=RUN_ID,
                role="latest",
            )
        self.assertEqual(restored.code_commit, CODE_COMMIT)
        self.assertEqual(validate.call_count, 2)
        for call in validate.call_args_list:
            self.assertEqual(call.args, (self.repository_root, CODE_COMMIT))
            self.assertEqual(call.kwargs["run_id"], RUN_ID)
            self.assertEqual(call.kwargs["configuration"], state.configuration)

    def test_resume_under_different_pre_registration_authority_fails(self) -> None:
        state = _state(self.repository_root)
        with patch.object(
            checkpoint,
            "_validate_live_repository",
            return_value=PRE_REGISTRATION_COMMIT,
        ):
            save_phase8_checkpoint(state, self.repository_root)
        mismatch = Phase8GovernanceError(
            "phase8.governance.repository",
            field="pre_registration_commit",
        )
        with (
            patch.object(
                checkpoint,
                "_validate_live_repository",
                side_effect=mismatch,
            ),
            self.assertRaises(Phase8GovernanceError) as caught,
        ):
            load_phase8_checkpoint(
                self.repository_root,
                run_id=RUN_ID,
                role="latest",
            )
        self.assertEqual(caught.exception.details["field"], "pre_registration_commit")

    def test_save_bootstraps_content_addressed_object_and_catalog(self) -> None:
        reference = self._save()
        run_root = self.repository_root / "checkpoints/phase8" / RUN_ID
        object_path = run_root / reference.relative_path
        catalog_path = run_root / "checkpoint-catalog.json"
        self.assertTrue(object_path.is_file())
        self.assertTrue(catalog_path.is_file())
        self.assertEqual(hashlib.sha256(object_path.read_bytes()).hexdigest(), reference.sha256)
        catalog_bytes = catalog_path.read_bytes()
        catalog = json.loads(catalog_bytes)
        self.assertEqual(
            catalog_bytes,
            (
                json.dumps(
                    catalog,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8"),
        )
        self.assertEqual(catalog["latest"]["relative_path"], f"objects/{reference.sha256}.pt")
        self.assertEqual(catalog["best_validation"]["sha256"], reference.sha256)

    def test_bootstrap_reuse_partial_and_unsafe_entry_matrix(self) -> None:
        cases = (
            "valid_eexist_reuse",
            "safe_partial_completion",
            "symlink_directory",
            "wrong_type_directory",
            "symlink_catalog_file",
            "uncertain_child_fsync",
        )
        for case in cases:
            with self.subTest(case=case):
                self.tearDown()
                self.setUp()
                phase8_root = self.repository_root / "checkpoints/phase8"
                run_root = phase8_root / RUN_ID
                outside = self.repository_root / "outside"
                if case == "valid_eexist_reuse":
                    (run_root / "objects").mkdir(parents=True)
                elif case == "safe_partial_completion":
                    run_root.mkdir(parents=True)
                elif case == "symlink_directory":
                    outside.mkdir()
                    phase8_root.symlink_to(outside, target_is_directory=True)
                elif case == "wrong_type_directory":
                    phase8_root.write_bytes(b"not a directory")
                elif case == "symlink_catalog_file":
                    (run_root / "objects").mkdir(parents=True)
                    outside.write_bytes(b"not a catalog")
                    (run_root / "checkpoint-catalog.json").symlink_to(outside)
                first, second = self._patch_authority()
                if case == "uncertain_child_fsync":
                    with first, second, patch.object(
                        checkpoint.os,
                        "fsync",
                        side_effect=OSError("forced bootstrap fsync"),
                    ):
                        with self.assertRaises(Phase8PublicationError):
                            save_phase8_checkpoint(
                                _state(self.repository_root),
                                self.repository_root,
                            )
                    self.assertTrue(phase8_root.is_dir())
                    self.assertFalse((run_root / "checkpoint-catalog.json").exists())
                elif case in ("valid_eexist_reuse", "safe_partial_completion"):
                    with first, second:
                        reference = save_phase8_checkpoint(
                            _state(self.repository_root),
                            self.repository_root,
                        )
                    self.assertTrue((run_root / reference.relative_path).is_file())
                    self.assertTrue((run_root / "checkpoint-catalog.json").is_file())
                else:
                    with first, second, self.assertRaises(Phase8CheckpointError):
                        save_phase8_checkpoint(
                            _state(self.repository_root),
                            self.repository_root,
                        )
                    self.assertFalse((run_root / "objects").joinpath("unexpected.pt").exists())

    def test_payload_schema_optimizer_steps_and_configuration_version(self) -> None:
        reference = self._save()
        path = self.repository_root / "checkpoints/phase8" / RUN_ID / reference.relative_path
        payload = torch.load(io.BytesIO(path.read_bytes()), map_location="cpu", weights_only=True)
        self.assertEqual(
            tuple(payload),
            (
                "schema_version",
                "run",
                "authority",
                "configuration",
                "model_state",
                "optimizer_state",
                "progress",
                "random_state",
                "mode_state",
                "metric_state",
                "resume_lineage",
            ),
        )
        self.assertEqual(next(iter(payload["configuration"])), "schema_version")
        self.assertEqual(payload["configuration"]["schema_version"], 1)
        group = payload["optimizer_state"]["param_groups"][0]
        self.assertEqual(
            tuple(group),
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
        self.assertIs(group["decoupled_weight_decay"], True)
        for state in payload["optimizer_state"]["state"].values():
            self.assertEqual(state["step"].shape, torch.Size([]))
            self.assertIs(state["step"].dtype, torch.float32)
            self.assertEqual(float(state["step"].item()), 1.0)
        self.assertEqual(
            payload["mode_state"],
            {
                "training": True,
                "named_modules": EXPECTED_MODE_STATE,
            },
        )
        self.assertEqual(len(EXPECTED_MODE_STATE), 53)
        self.assertEqual(
            len({name for name, _ in EXPECTED_MODE_STATE}),
            len(EXPECTED_MODE_STATE),
        )
        self.assertTrue(
            all(type(name) is str and type(mode) is bool and mode is True for name, mode in EXPECTED_MODE_STATE)
        )

    def test_malformed_latest_mode_maps_are_rejected_before_restoration(self) -> None:
        for label, transform in _mode_state_corruptions():
            with self.subTest(case=label):
                self.tearDown()
                self.setUp()
                self._save()
                self._corrupt_role_payload(
                    "latest",
                    lambda payload, transform=transform: payload["mode_state"].__setitem__(
                        "named_modules",
                        transform(payload["mode_state"]["named_modules"]),
                    ),
                )
                with patch.object(checkpoint, "_construct_model") as construct:
                    with self.assertRaises(Phase8CheckpointError):
                        self._load()
                construct.assert_not_called()

    def test_malformed_historical_best_mode_maps_are_rejected_before_cross_validation(self) -> None:
        for label, transform in _mode_state_corruptions():
            with self.subTest(case=label):
                self.tearDown()
                self.setUp()
                first, second = self._patch_authority()
                with first, second:
                    save_phase8_checkpoint(_state(self.repository_root), self.repository_root)
                    save_phase8_checkpoint(
                        _epoch_two_state(self.repository_root),
                        self.repository_root,
                    )
                self._corrupt_role_payload(
                    "best_validation",
                    lambda payload, transform=transform: payload["mode_state"].__setitem__(
                        "named_modules",
                        transform(payload["mode_state"]["named_modules"]),
                    ),
                )
                with patch.object(checkpoint, "_cross_validate_catalog_graph") as cross:
                    with self.assertRaises(Phase8CheckpointError):
                        self._load()
                cross.assert_not_called()

    def test_checkpoint_and_catalog_schemas_directly_exclude_forbidden_content(self) -> None:
        reference = self._save()
        run_root = self.repository_root / "checkpoints/phase8" / RUN_ID
        payload = torch.load(
            io.BytesIO((run_root / reference.relative_path).read_bytes()),
            map_location="cpu",
            weights_only=True,
        )
        catalog = json.loads(
            (run_root / "checkpoint-catalog.json").read_text(encoding="utf-8")
        )
        forbidden_keys = {
            "input_ids",
            "target_ids",
            "token_ids",
            "tokens",
            "text",
            "processed_text",
            "windows",
            "activations",
            "logits",
            "probabilities",
            "gradients",
            "generated_text",
            "samples",
            "test_loss",
            "test_metrics",
        }

        def collect_keys(value):
            if type(value) is dict or isinstance(value, OrderedDict):
                return set(value).union(
                    *(collect_keys(item) for item in value.values())
                )
            if type(value) in (tuple, list):
                return set().union(*(collect_keys(item) for item in value))
            return set()

        self.assertTrue(forbidden_keys.isdisjoint(collect_keys(payload)))
        self.assertTrue(forbidden_keys.isdisjoint(collect_keys(catalog)))
        self.assertEqual(payload["authority"]["sealed_test_access"], "none")
        self.assertEqual(
            tuple(payload),
            (
                "schema_version",
                "run",
                "authority",
                "configuration",
                "model_state",
                "optimizer_state",
                "progress",
                "random_state",
                "mode_state",
                "metric_state",
                "resume_lineage",
            ),
        )
        self.assertEqual(
            set(catalog),
            {"schema_version", "run_id", "latest", "best_validation"},
        )

    def test_missing_and_unknown_configuration_schema_versions_are_rejected(self) -> None:
        for replacement in (None, 2):
            with self.subTest(replacement=replacement):
                self.tearDown()
                self.setUp()
                reference = self._save()
                run_root = self.repository_root / "checkpoints/phase8" / RUN_ID
                object_path = run_root / reference.relative_path
                payload = torch.load(
                    io.BytesIO(object_path.read_bytes()),
                    map_location="cpu",
                    weights_only=True,
                )
                if replacement is None:
                    payload["configuration"].pop("schema_version")
                else:
                    payload["configuration"]["schema_version"] = replacement
                buffer = io.BytesIO()
                torch.save(payload, buffer)
                content = buffer.getvalue()
                digest = hashlib.sha256(content).hexdigest()
                (run_root / "objects" / f"{digest}.pt").write_bytes(content)
                catalog = json.loads(
                    (run_root / "checkpoint-catalog.json").read_text(encoding="utf-8")
                )
                for role in ("latest", "best_validation"):
                    catalog[role]["sha256"] = digest
                    catalog[role]["relative_path"] = f"objects/{digest}.pt"
                (run_root / "checkpoint-catalog.json").write_text(
                    json.dumps(
                        catalog,
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n",
                    encoding="utf-8",
                )
                with self.assertRaises(Phase8CheckpointError):
                    self._load()

    def test_model_state_dict_alone_is_rejected_as_incomplete_checkpoint(self) -> None:
        state = _state(self.repository_root)
        with self.assertRaises(Phase8CheckpointError):
            checkpoint._validate_payload(state.model.state_dict())

    def test_round_trip_restores_model_optimizer_progress_rng_and_lineage(self) -> None:
        original = _state(self.repository_root)
        first, second = self._patch_authority()
        with first, second:
            reference = save_phase8_checkpoint(original, self.repository_root)
            restored = load_phase8_checkpoint(
                self.repository_root,
                run_id=RUN_ID,
                role="latest",
            )
        for left, right in zip(
            original.model.state_dict().values(),
            restored.model.state_dict().values(),
            strict=True,
        ):
            self.assertTrue(torch.equal(left, right))
        self.assertTrue(
            _state_dict_equal(
                original.optimizer.state_dict(),
                restored.optimizer.state_dict(),
            )
        )
        self.assertEqual(restored.progress, original.progress)
        self.assertEqual(restored.metrics, original.metrics)
        self.assertTrue(
            torch.equal(
                restored.order_generator.get_state(),
                original.order_generator.get_state(),
            )
        )
        self.assertEqual(restored.resumed_from_checkpoint_sha256, reference.sha256)
        self.assertEqual(restored.resume_checkpoint_sha256s, (reference.sha256,))

    def test_checkpointed_order_state_reproduces_the_next_epoch_permutation(self) -> None:
        original = _state(self.repository_root)
        create_phase8_epoch_order(
            2,
            epoch=1,
            generator=original.order_generator,
        )
        original = _make_training_state(
            run_id=original.run_id,
            code_commit=original.code_commit,
            configuration=original.configuration,
            model=original.model,
            optimizer=original.optimizer,
            order_generator=original.order_generator,
            progress=original.progress,
            metrics=original.metrics,
            global_cpu_rng_state=torch.get_rng_state().clone(),
            resumed_from_checkpoint_sha256=None,
            resume_checkpoint_sha256s=(),
        )
        checkpointed_order = original.order_generator.get_state().clone()
        checkpointed_dropout = tuple(
            value
            for block in original.model.blocks
            for value in (
                block.attention_dropout._generator.get_state().clone(),
                block.feed_forward_dropout._generator.get_state().clone(),
            )
        )
        first, second = self._patch_authority()
        with first, second:
            save_phase8_checkpoint(original, self.repository_root)
            restored = load_phase8_checkpoint(
                self.repository_root,
                run_id=RUN_ID,
                role="latest",
            )
        self.assertTrue(
            torch.equal(restored.order_generator.get_state(), checkpointed_order)
        )
        restored_checkpointed_dropout = tuple(
            value
            for block in restored.model.blocks
            for value in (
                block.attention_dropout._generator.get_state(),
                block.feed_forward_dropout._generator.get_state(),
            )
        )
        self.assertTrue(
            all(
                torch.equal(expected, observed)
                for expected, observed in zip(
                    checkpointed_dropout,
                    restored_checkpointed_dropout,
                    strict=True,
                )
            )
        )
        self.assertEqual(
            (
                restored.progress.next_example_offset,
                restored.progress.next_epoch,
            ),
            (0, restored.progress.completed_epochs + 1),
        )
        self.assertTrue(
            torch.equal(
                original.order_generator.get_state(),
                restored.order_generator.get_state(),
            )
        )
        global_before_perturbation = torch.get_rng_state().clone()
        control_dropout_before = tuple(
            value.clone()
            for block in original.model.blocks
            for value in (
                block.attention_dropout._generator.get_state(),
                block.feed_forward_dropout._generator.get_state(),
            )
        )
        restored_dropout_before = tuple(
            value.clone()
            for block in restored.model.blocks
            for value in (
                block.attention_dropout._generator.get_state(),
                block.feed_forward_dropout._generator.get_state(),
            )
        )
        torch.rand(13, device="cpu")
        torch.rand(
            3,
            generator=original.model.blocks[0].attention_dropout._generator,
        )
        torch.rand(
            5,
            generator=original.model.blocks[1].feed_forward_dropout._generator,
        )
        torch.rand(
            7,
            generator=restored.model.blocks[0].attention_dropout._generator,
        )
        torch.rand(
            11,
            generator=restored.model.blocks[2].feed_forward_dropout._generator,
        )
        perturbed_global = torch.get_rng_state().clone()
        perturbed_control_dropout = tuple(
            value.clone()
            for block in original.model.blocks
            for value in (
                block.attention_dropout._generator.get_state(),
                block.feed_forward_dropout._generator.get_state(),
            )
        )
        perturbed_restored_dropout = tuple(
            value.clone()
            for block in restored.model.blocks
            for value in (
                block.attention_dropout._generator.get_state(),
                block.feed_forward_dropout._generator.get_state(),
            )
        )
        self.assertFalse(torch.equal(global_before_perturbation, perturbed_global))
        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(
                    control_dropout_before,
                    perturbed_control_dropout,
                    strict=True,
                )
            )
        )
        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(
                    restored_dropout_before,
                    perturbed_restored_dropout,
                    strict=True,
                )
            )
        )
        control_order_before = original.order_generator.get_state().clone()
        restored_order_before = restored.order_generator.get_state().clone()
        self.assertTrue(torch.equal(control_order_before, checkpointed_order))
        self.assertTrue(torch.equal(restored_order_before, checkpointed_order))
        control_next = create_phase8_epoch_order(
            9,
            epoch=2,
            generator=original.order_generator,
        )
        restored_next = create_phase8_epoch_order(
            9,
            epoch=2,
            generator=restored.order_generator,
        )
        self.assertEqual(control_next, (5, 1, 7, 2, 3, 6, 0, 8, 4))
        self.assertEqual(restored_next, control_next)
        self.assertTrue(torch.equal(control_order_before, restored_order_before))
        self.assertFalse(
            torch.equal(control_order_before, original.order_generator.get_state())
        )
        self.assertTrue(
            torch.equal(
                original.order_generator.get_state(),
                restored.order_generator.get_state(),
            )
        )
        self.assertEqual(
            hashlib.sha256(
                bytes(original.order_generator.get_state().tolist())
            ).hexdigest(),
            "81ec8da2828dc8222b423104eca85e9b2203bf06d1abfa3d2a9d550e40b27799",
        )
        self.assertTrue(torch.equal(perturbed_global, torch.get_rng_state()))
        control_dropout_after_order = tuple(
            value
            for block in original.model.blocks
            for value in (
                block.attention_dropout._generator.get_state(),
                block.feed_forward_dropout._generator.get_state(),
            )
        )
        restored_dropout_after_order = tuple(
            value
            for block in restored.model.blocks
            for value in (
                block.attention_dropout._generator.get_state(),
                block.feed_forward_dropout._generator.get_state(),
            )
        )
        self.assertTrue(
            all(
                torch.equal(expected, observed)
                for expected, observed in zip(
                    perturbed_control_dropout,
                    control_dropout_after_order,
                    strict=True,
                )
            )
        )
        self.assertTrue(
            all(
                torch.equal(expected, observed)
                for expected, observed in zip(
                    perturbed_restored_dropout,
                    restored_dropout_after_order,
                    strict=True,
                )
            )
        )

    def test_best_validation_is_not_a_continuation_role(self) -> None:
        with self.assertRaises(Phase8CheckpointError) as caught:
            load_phase8_checkpoint(
                self.repository_root,
                run_id=RUN_ID,
                role="best_validation",  # type: ignore[arg-type]
            )
        self.assertEqual(
            caught.exception.details["invariant"],
            "phase8.checkpoint.role.continuation",
        )

    def test_run_id_traversal_is_rejected_before_path_access(self) -> None:
        with self.assertRaises(Phase8CheckpointError) as caught:
            load_phase8_checkpoint(
                self.repository_root,
                run_id="../EXP-20260911-01",
                role="latest",
            )
        self.assertEqual(
            caught.exception.details["invariant"],
            "phase8.checkpoint.path",
        )

    def test_literal_reference_grammar_rejects_alternate_spellings(self) -> None:
        base = {
            "run_id": RUN_ID,
            "role": "latest",
            "logical_id": "epoch-0001",
            "epoch": 1,
            "validation_loss_hex": float(1.5).hex(),
            "sha256": "a" * 64,
            "relative_path": f"objects/{'a' * 64}.pt",
        }
        mutations = (
            ("run_id", RUN_ID.lower()),
            ("role", "best_validation"),
            ("logical_id", "epoch-0011"),
            ("epoch", True),
            ("validation_loss_hex", "0X1.8000000000000P+0"),
            ("sha256", "A" * 64),
            ("relative_path", f"./objects/{'a' * 64}.pt"),
            ("relative_path", f"/objects/{'a' * 64}.pt"),
        )
        for field, replacement in mutations:
            with self.subTest(field=field, replacement=replacement):
                malformed = dict(base)
                malformed[field] = replacement
                with self.assertRaises(Phase8CheckpointError):
                    checkpoint._validate_reference(malformed, "latest", RUN_ID)

    def test_catalog_relabel_is_rejected_without_global_rng_leak(self) -> None:
        self._save()
        catalog_path = (
            self.repository_root
            / "checkpoints/phase8"
            / RUN_ID
            / "checkpoint-catalog.json"
        )
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        catalog["latest"]["logical_id"] = "epoch-0002"
        catalog_path.write_text(
            json.dumps(catalog, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        before = torch.get_rng_state().clone()
        with self.assertRaises(Phase8CheckpointError):
            self._load()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_catalog_boolean_schema_version_is_rejected(self) -> None:
        self._save()
        path = (
            self.repository_root
            / "checkpoints/phase8"
            / RUN_ID
            / "checkpoint-catalog.json"
        )
        catalog = json.loads(path.read_text(encoding="utf-8"))
        catalog["schema_version"] = True
        path.write_text(
            json.dumps(catalog, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(Phase8CheckpointError):
            self._load()

    def test_progress_schema_is_literal_and_cross_validated(self) -> None:
        mutations = (
            lambda payload: payload["progress"].__setitem__("completed_epochs", True),
            lambda payload: payload["progress"].__setitem__(
                "per_epoch_optimizer_updates", (2,)
            ),
            lambda payload: payload["progress"].__setitem__(
                "per_epoch_example_counts", (3,)
            ),
            lambda payload: payload["progress"].__setitem__(
                "per_epoch_target_counts", (5,)
            ),
            lambda payload: payload["progress"].__setitem__(
                "next_ordering_action", "reuse_previous_permutation"
            ),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                self.tearDown()
                self.setUp()
                self._save()
                self._corrupt_role_payload("latest", mutate)
                with self.assertRaises(Phase8CheckpointError):
                    self._load()

    def test_every_nested_schema_category_is_noncoercing(self) -> None:
        mutations = (
            lambda payload: payload.__setitem__("schema_version", True),
            lambda payload: payload["metric_state"]["current_training"].__setitem__(
                "target_count", True
            ),
            lambda payload: payload["metric_state"]["current_validation"].__setitem__(
                "loss_hex", float(9.0).hex()
            ),
            lambda payload: payload["random_state"].__setitem__(
                "order_generator", (1, 2, 3)
            ),
            lambda payload: payload["random_state"]["dropout_generators"].pop(
                "blocks.3.feed_forward_dropout"
            ),
            lambda payload: payload["mode_state"].__setitem__("training", 1),
            lambda payload: payload["mode_state"].__setitem__(
                "named_modules",
                (("", 1),),
            ),
            lambda payload: payload["resume_lineage"].__setitem__(
                "resume_count", True
            ),
            lambda payload: payload["optimizer_state"]["param_groups"][0].__setitem__(
                "amsgrad", 0
            ),
            lambda payload: payload["model_state"].__setitem__(
                next(iter(payload["model_state"])),
                next(iter(payload["model_state"].values())).to(torch.float64),
            ),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                self.tearDown()
                self.setUp()
                self._save()
                self._corrupt_role_payload("latest", mutate)
                with self.assertRaises(Phase8CheckpointError):
                    self._load()

    def test_latest_best_catalog_semantics_are_bound_to_latest_payload(self) -> None:
        first, second = self._patch_authority()
        with first, second:
            epoch_one = _state(self.repository_root)
            save_phase8_checkpoint(epoch_one, self.repository_root)
            save_phase8_checkpoint(_epoch_two_state(self.repository_root), self.repository_root)
        catalog_path = (
            self.repository_root
            / "checkpoints/phase8"
            / RUN_ID
            / "checkpoint-catalog.json"
        )
        original = json.loads(catalog_path.read_text(encoding="utf-8"))
        cases = []
        wrong_epoch = copy.deepcopy(original)
        wrong_epoch["best_validation"] = dict(wrong_epoch["latest"])
        wrong_epoch["best_validation"]["role"] = "best_validation"
        cases.append(wrong_epoch)
        wrong_logical = copy.deepcopy(original)
        wrong_logical["best_validation"]["logical_id"] = "epoch-0002"
        cases.append(wrong_logical)
        wrong_loss = copy.deepcopy(original)
        wrong_loss["best_validation"]["validation_loss_hex"] = float(1.6).hex()
        cases.append(wrong_loss)
        for malformed in cases:
            with self.subTest(best=malformed["best_validation"]):
                catalog_path.write_text(
                    json.dumps(
                        malformed,
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n",
                    encoding="utf-8",
                )
                with self.assertRaises(Phase8CheckpointError):
                    self._load()

    def test_historical_best_payload_is_fully_validated(self) -> None:
        mutations = (
            lambda payload: payload["authority"].__setitem__(
                "phase7_closure_commit", "b" * 40
            ),
            lambda payload: payload["configuration"].__setitem__(
                "context_length", 64
            ),
            lambda payload: next(iter(payload["model_state"].values())).fill_(
                float("nan")
            ),
            lambda payload: payload["optimizer_state"]["state"][0][
                "exp_avg"
            ].fill_(float("nan")),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                self.tearDown()
                self.setUp()
                first, second = self._patch_authority()
                with first, second:
                    save_phase8_checkpoint(_state(self.repository_root), self.repository_root)
                    save_phase8_checkpoint(
                        _epoch_two_state(self.repository_root),
                        self.repository_root,
                    )
                self._corrupt_role_payload("best_validation", mutate)
                with self.assertRaises(Phase8CheckpointError):
                    self._load()

    def test_late_load_failure_rolls_back_global_rng(self) -> None:
        self._save()
        before = torch.get_rng_state().clone()
        first, second = self._patch_authority()
        with first, second, patch.object(
            checkpoint,
            "_make_training_state",
            side_effect=Phase8CheckpointError("phase8.checkpoint.restore"),
        ):
            with self.assertRaises(Phase8CheckpointError):
                load_phase8_checkpoint(
                    self.repository_root,
                    run_id=RUN_ID,
                    role="latest",
                )
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_global_rng_install_failure_rolls_back_once(self) -> None:
        self._save()
        before = torch.get_rng_state().clone()
        real_set = checkpoint.torch.set_rng_state
        calls = 0

        def fail_install_then_restore(value):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("install")
            return real_set(value)

        first, second = self._patch_authority()
        with first, second, patch.object(
            checkpoint.torch,
            "set_rng_state",
            side_effect=fail_install_then_restore,
        ):
            with self.assertRaises(Phase8CheckpointError) as caught:
                load_phase8_checkpoint(
                    self.repository_root,
                    run_id=RUN_ID,
                    role="latest",
                )
        self.assertEqual(calls, 2)
        self.assertNotIn("state_guarantee", caught.exception.details)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_global_rng_install_and_rollback_failure_reports_unknown_once(self) -> None:
        self._save()
        calls = 0

        def fail_both(_value):
            nonlocal calls
            calls += 1
            raise RuntimeError("forced")

        first, second = self._patch_authority()
        with first, second, patch.object(
            checkpoint.torch,
            "set_rng_state",
            side_effect=fail_both,
        ):
            with self.assertRaises(Phase8CheckpointError) as caught:
                load_phase8_checkpoint(
                    self.repository_root,
                    run_id=RUN_ID,
                    role="latest",
                )
        self.assertEqual(calls, 2)
        self.assertEqual(caught.exception.details["state_guarantee"], "unknown")

    def test_restoration_order_commits_global_rng_as_final_semantic_action(self) -> None:
        reference = self._save()
        run_root = self.repository_root / "checkpoints/phase8" / RUN_ID
        payload = torch.load(
            io.BytesIO((run_root / reference.relative_path).read_bytes()),
            map_location="cpu",
            weights_only=True,
        )
        events: list[str] = []
        after_global_install = False
        progress_calls = 0
        metric_calls = 0
        real_read_catalog = checkpoint._read_catalog
        real_validate_payload = checkpoint._validate_payload
        real_cross = checkpoint._cross_validate_catalog_graph
        real_construct = checkpoint._construct_model
        real_model_load = MiniGPT.load_state_dict
        real_create_optimizer = checkpoint.create_phase8_optimizer
        real_optimizer_load = torch.optim.AdamW.load_state_dict
        real_progress = checkpoint._progress_from_mapping
        real_metrics = checkpoint._metric_from_mapping
        real_make_state = checkpoint._make_training_state
        real_validate_state = checkpoint._validate_training_state
        real_set_rng = checkpoint.torch.set_rng_state
        real_get_rng = checkpoint.torch.get_rng_state

        def read_catalog(*args, **kwargs):
            events.append("catalog_parse")
            return real_read_catalog(*args, **kwargs)

        def validate_payload(*args, **kwargs):
            events.append("payload_validation")
            return real_validate_payload(*args, **kwargs)

        def cross_validate(*args, **kwargs):
            events.append("catalog_cross_validation")
            return real_cross(*args, **kwargs)

        def construct(*args, **kwargs):
            events.append("fresh_model_construction")
            return real_construct(*args, **kwargs)

        def model_load(model, *args, **kwargs):
            events.append("model_restoration")
            return real_model_load(model, *args, **kwargs)

        def create_optimizer(*args, **kwargs):
            events.append("optimizer_construction")
            return real_create_optimizer(*args, **kwargs)

        def optimizer_load(optimizer, *args, **kwargs):
            events.append("optimizer_restoration")
            return real_optimizer_load(optimizer, *args, **kwargs)

        def progress(*args, **kwargs):
            nonlocal progress_calls
            progress_calls += 1
            events.append(
                "progress_schema_validation"
                if progress_calls == 1
                else "progress_restoration"
            )
            return real_progress(*args, **kwargs)

        def metrics(*args, **kwargs):
            nonlocal metric_calls
            metric_calls += 1
            events.append(
                "metric_schema_validation"
                if metric_calls == 1
                else "metric_restoration"
            )
            return real_metrics(*args, **kwargs)

        def make_state(**kwargs):
            events.append("local_rng_mode_gradient_and_result_packaging")
            self.assertTrue(
                torch.equal(
                    kwargs["order_generator"].get_state(),
                    payload["random_state"]["order_generator"],
                )
            )
            observed_dropout = tuple(
                value
                for block in kwargs["model"].blocks
                for value in (
                    block.attention_dropout._generator.get_state(),
                    block.feed_forward_dropout._generator.get_state(),
                )
            )
            self.assertTrue(
                all(
                    torch.equal(expected, observed)
                    for expected, observed in zip(
                        payload["random_state"]["dropout_generators"].values(),
                        observed_dropout,
                        strict=True,
                    )
                )
            )
            self.assertTrue(
                all(module.training for _, module in kwargs["model"].named_modules())
            )
            self.assertTrue(
                all(parameter.grad is None for parameter in kwargs["model"].parameters())
            )
            return real_make_state(**kwargs)

        def validate_state(*args, **kwargs):
            events.append("final_state_validation")
            return real_validate_state(*args, **kwargs)

        def set_rng(value):
            nonlocal after_global_install
            events.append("global_rng_install")
            result = real_set_rng(value)
            after_global_install = True
            return result

        def get_rng():
            if after_global_install:
                events.append("global_rng_immediate_verification")
            return real_get_rng()

        first, second = self._patch_authority()
        with (
            first,
            second,
            patch.object(checkpoint, "_read_catalog", side_effect=read_catalog),
            patch.object(checkpoint, "_validate_payload", side_effect=validate_payload),
            patch.object(checkpoint, "_cross_validate_catalog_graph", side_effect=cross_validate),
            patch.object(checkpoint, "_construct_model", side_effect=construct),
            patch.object(MiniGPT, "load_state_dict", new=model_load),
            patch.object(checkpoint, "create_phase8_optimizer", side_effect=create_optimizer),
            patch.object(torch.optim.AdamW, "load_state_dict", new=optimizer_load),
            patch.object(checkpoint, "_progress_from_mapping", side_effect=progress),
            patch.object(checkpoint, "_metric_from_mapping", side_effect=metrics),
            patch.object(checkpoint, "_make_training_state", side_effect=make_state),
            patch.object(checkpoint, "_validate_training_state", side_effect=validate_state),
            patch.object(checkpoint.torch, "set_rng_state", side_effect=set_rng),
            patch.object(checkpoint.torch, "get_rng_state", side_effect=get_rng),
        ):
            restored = load_phase8_checkpoint(
                self.repository_root,
                run_id=RUN_ID,
                role="latest",
            )
        expected_order = (
            "catalog_parse",
            "payload_validation",
            "progress_schema_validation",
            "metric_schema_validation",
            "catalog_cross_validation",
            "fresh_model_construction",
            "model_restoration",
            "optimizer_construction",
            "optimizer_restoration",
            "progress_restoration",
            "metric_restoration",
            "local_rng_mode_gradient_and_result_packaging",
            "final_state_validation",
            "global_rng_install",
            "global_rng_immediate_verification",
        )
        self.assertEqual(tuple(events), expected_order)
        self.assertEqual(restored.progress.completed_epochs, 1)

    def test_cleanup_refuses_substituted_temporary_entry(self) -> None:
        directory_fd = os.open(self.repository_root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            name = ".owned-temporary"
            (self.repository_root / name).write_bytes(b"original")
            identity = checkpoint._entry_identity(directory_fd, name)
            (self.repository_root / name).unlink()
            (self.repository_root / name).write_bytes(b"substitute")
            self.assertFalse(
                checkpoint._unlink_if_identity(directory_fd, name, identity)
            )
            self.assertEqual((self.repository_root / name).read_bytes(), b"substitute")
        finally:
            os.close(directory_fd)

    def test_post_commit_diagnostic_failure_reports_committed_reference(self) -> None:
        first, second = self._patch_authority()
        with first, second, patch.object(
            checkpoint,
            "_read_catalog",
            side_effect=(None, Phase8CheckpointError("phase8.checkpoint.catalog")),
        ):
            with self.assertRaises(Phase8PublicationError) as caught:
                save_phase8_checkpoint(
                    _state(self.repository_root),
                    self.repository_root,
                )
        self.assertEqual(caught.exception.details["state"], "committed")
        self.assertEqual(caught.exception.details["run_id"], RUN_ID)
        self.assertTrue(
            (
                self.repository_root
                / "checkpoints/phase8"
                / RUN_ID
                / "checkpoint-catalog.json"
            ).is_file()
        )

    def test_hash_and_deserialization_use_the_same_bounded_bytes(self) -> None:
        reference = self._save()
        object_path = (
            self.repository_root
            / "checkpoints/phase8"
            / RUN_ID
            / reference.relative_path
        )
        expected = object_path.read_bytes()
        real_load = checkpoint.torch.load
        observed: list[bytes] = []
        observed_options: list[dict[str, object]] = []

        def load(stream, *args, **kwargs):
            observed.append(stream.getvalue())
            observed_options.append(dict(kwargs))
            return real_load(stream, *args, **kwargs)

        first, second = self._patch_authority()
        with first, second, patch.object(checkpoint.torch, "load", side_effect=load):
            load_phase8_checkpoint(
                self.repository_root,
                run_id=RUN_ID,
                role="latest",
            )
        self.assertEqual(observed, [expected])
        self.assertEqual(
            observed_options,
            [{"map_location": "cpu", "weights_only": True}],
        )

    def test_object_byte_limit_is_checked_before_deserialization(self) -> None:
        self._save()
        first, second = self._patch_authority()
        with first, second, patch.object(
            checkpoint,
            "MAXIMUM_OBJECT_BYTES",
            4,
        ), patch.object(checkpoint.torch, "load") as load:
            with self.assertRaises(Phase8CheckpointError):
                load_phase8_checkpoint(
                    self.repository_root,
                    run_id=RUN_ID,
                    role="latest",
                )
        load.assert_not_called()

    def test_bounded_read_zero_limit_growth_shrink_short_and_truncated_cases(self) -> None:
        path = self.repository_root / "bounded.bin"

        def opened():
            return os.open(path, os.O_RDONLY | os.O_NOFOLLOW)

        path.write_bytes(b"")
        fd = opened()
        try:
            with self.assertRaises(Phase8CheckpointError) as caught:
                checkpoint._read_bounded_fd(fd, 4)
        finally:
            os.close(fd)
        self.assertEqual(caught.exception.details["invariant"], "phase8.checkpoint.size")

        path.write_bytes(b"1234")
        fd = opened()
        try:
            self.assertEqual(checkpoint._read_bounded_fd(fd, 4), b"1234")
        finally:
            os.close(fd)

        path.write_bytes(b"12345")
        fd = opened()
        try:
            with self.assertRaises(Phase8CheckpointError) as caught:
                checkpoint._read_bounded_fd(fd, 4)
        finally:
            os.close(fd)
        self.assertEqual(caught.exception.details["invariant"], "phase8.checkpoint.size")

        path.write_bytes(b"1234")
        fd = opened()
        real_read = checkpoint.os.read
        read_calls = 0

        def grow_after_first_read(read_fd, size):
            nonlocal read_calls
            read_calls += 1
            value = real_read(read_fd, size)
            if read_calls == 1:
                with path.open("ab") as stream:
                    stream.write(b"5")
            return value

        try:
            with patch.object(checkpoint.os, "read", side_effect=grow_after_first_read):
                with self.assertRaises(Phase8CheckpointError) as caught:
                    checkpoint._read_bounded_fd(fd, 4)
        finally:
            os.close(fd)
        self.assertEqual(caught.exception.details["field"], "changing")

        path.write_bytes(b"1234")
        fd = opened()
        read_calls = 0

        def shrink_before_first_read(read_fd, size):
            nonlocal read_calls
            read_calls += 1
            if read_calls == 1:
                os.truncate(path, 2)
            return real_read(read_fd, size)

        try:
            with patch.object(checkpoint.os, "read", side_effect=shrink_before_first_read):
                with self.assertRaises(Phase8CheckpointError) as caught:
                    checkpoint._read_bounded_fd(fd, 4)
        finally:
            os.close(fd)
        self.assertEqual(caught.exception.details["field"], "short_read")

        path.write_bytes(b"1234")
        fd = opened()
        read_calls = 0

        def forced_short_read(read_fd, size):
            nonlocal read_calls
            read_calls += 1
            return real_read(read_fd, 1) if read_calls == 1 else b""

        try:
            with patch.object(checkpoint.os, "read", side_effect=forced_short_read):
                with self.assertRaises(Phase8CheckpointError) as caught:
                    checkpoint._read_bounded_fd(fd, 4)
        finally:
            os.close(fd)
        self.assertEqual(caught.exception.details["field"], "short_read")

        run_root = self.repository_root / "truncated-catalog"
        run_root.mkdir()
        (run_root / "checkpoint-catalog.json").write_bytes(b"{")
        run_fd = os.open(run_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            with self.assertRaises(Phase8CheckpointError) as caught:
                checkpoint._read_catalog(run_fd)
        finally:
            os.close(run_fd)
        self.assertEqual(caught.exception.details["invariant"], "phase8.checkpoint.catalog")

    def test_fsync_order_places_objects_before_catalog_and_run_commit(self) -> None:
        events: list[str] = []
        real_fsync = checkpoint.os.fsync
        real_replace = checkpoint.os.replace
        checkpoints_root = self.repository_root / "checkpoints"
        phase8_root = checkpoints_root / "phase8"
        run_root = phase8_root / RUN_ID
        objects_root = run_root / "objects"

        def identity(path):
            info = path.stat(follow_symlinks=False)
            return info.st_dev, info.st_ino

        def descriptor_label(fd):
            info = os.fstat(fd)
            observed = (info.st_dev, info.st_ino)
            stable_paths = (
                ("checkpoints_directory", checkpoints_root),
                ("phase8_directory", phase8_root),
                ("run_directory", run_root),
                ("objects_directory", objects_root),
            )
            for label, path in stable_paths:
                if path.exists() and identity(path) == observed:
                    return label
            for path in run_root.glob(".checkpoint-*.tmp"):
                if identity(path) == observed:
                    return "temporary_object_file"
            for path in run_root.glob(".catalog-*.tmp"):
                if identity(path) == observed:
                    return "temporary_catalog_file"
            return "unexpected_descriptor"

        def fsync(fd):
            events.append(descriptor_label(fd))
            return real_fsync(fd)

        def replace(*args, **kwargs):
            events.append("catalog_replace")
            return real_replace(*args, **kwargs)

        first, second = self._patch_authority()
        with first, second, patch.object(
            checkpoint.os,
            "fsync",
            side_effect=fsync,
        ), patch.object(checkpoint.os, "replace", side_effect=replace):
            save_phase8_checkpoint(_state(self.repository_root), self.repository_root)
        self.assertEqual(
            events,
            [
                "phase8_directory",
                "checkpoints_directory",
                "run_directory",
                "phase8_directory",
                "objects_directory",
                "run_directory",
                "temporary_object_file",
                "objects_directory",
                "temporary_catalog_file",
                "catalog_replace",
                "run_directory",
            ],
        )

    def test_checkpoint_filesystem_calls_are_descriptor_relative_no_follow(self) -> None:
        open_calls = []
        stat_calls = []
        link_calls = []
        replace_calls = []
        real_open = checkpoint.os.open
        real_stat = checkpoint.os.stat
        real_link = checkpoint.os.link
        real_replace = checkpoint.os.replace

        def open_file(path, flags, *args, **kwargs):
            open_calls.append((path, flags, kwargs.get("dir_fd")))
            return real_open(path, flags, *args, **kwargs)

        def stat_file(path, *args, **kwargs):
            stat_calls.append(
                (path, kwargs.get("dir_fd"), kwargs.get("follow_symlinks"))
            )
            return real_stat(path, *args, **kwargs)

        def link_file(*args, **kwargs):
            link_calls.append((args, dict(kwargs)))
            return real_link(*args, **kwargs)

        def replace_file(*args, **kwargs):
            replace_calls.append((args, dict(kwargs)))
            return real_replace(*args, **kwargs)

        state = _state(self.repository_root)
        first, second = self._patch_authority()
        with (
            first,
            second,
            patch.object(checkpoint.os, "open", side_effect=open_file),
            patch.object(checkpoint.os, "stat", side_effect=stat_file),
            patch.object(checkpoint.os, "link", side_effect=link_file),
            patch.object(checkpoint.os, "replace", side_effect=replace_file),
        ):
            save_phase8_checkpoint(state, self.repository_root)
            load_phase8_checkpoint(
                self.repository_root,
                run_id=RUN_ID,
                role="latest",
            )
        root_calls = [call for call in open_calls if call[2] is None]
        self.assertEqual(len(root_calls), 2)
        self.assertTrue(
            all(
                path == self.repository_root
                and flags & os.O_DIRECTORY
                and flags & os.O_NOFOLLOW
                for path, flags, _ in root_calls
            )
        )
        relative_calls = [call for call in open_calls if call[2] is not None]
        self.assertTrue(relative_calls)
        self.assertTrue(
            all(
                type(path) is str
                and "/" not in path
                and flags & os.O_NOFOLLOW
                and type(dir_fd) is int
                for path, flags, dir_fd in relative_calls
            )
        )
        temporary_object_calls = [
            call for call in relative_calls if call[0].startswith(".checkpoint-")
        ]
        temporary_catalog_calls = [
            call for call in relative_calls if call[0].startswith(".catalog-")
        ]
        self.assertEqual(len(temporary_object_calls), 1)
        self.assertEqual(len(temporary_catalog_calls), 1)
        for _, flags, _ in (*temporary_object_calls, *temporary_catalog_calls):
            self.assertTrue(flags & os.O_CREAT)
            self.assertTrue(flags & os.O_EXCL)
            self.assertTrue(flags & os.O_NOFOLLOW)
        self.assertEqual(len(link_calls), 1)
        _, link_options = link_calls[0]
        self.assertIs(link_options["follow_symlinks"], False)
        self.assertIsInstance(link_options["src_dir_fd"], int)
        self.assertIsInstance(link_options["dst_dir_fd"], int)
        self.assertEqual(len(replace_calls), 1)
        _, replace_options = replace_calls[0]
        self.assertIsInstance(replace_options["src_dir_fd"], int)
        self.assertIsInstance(replace_options["dst_dir_fd"], int)
        descriptor_stat_calls = [call for call in stat_calls if call[1] is not None]
        self.assertTrue(descriptor_stat_calls)
        self.assertTrue(
            all(
                type(path) is str
                and "/" not in path
                and type(dir_fd) is int
                and follow is False
                for path, dir_fd, follow in descriptor_stat_calls
            )
        )

    def test_catalog_replace_failure_leaves_only_non_authoritative_object(self) -> None:
        first, second = self._patch_authority()
        with first, second, patch.object(
            checkpoint.os,
            "replace",
            side_effect=OSError("forced"),
        ):
            with self.assertRaises(Phase8PublicationError):
                save_phase8_checkpoint(_state(self.repository_root), self.repository_root)
        run_root = self.repository_root / "checkpoints/phase8" / RUN_ID
        self.assertFalse((run_root / "checkpoint-catalog.json").exists())
        self.assertEqual(len(tuple((run_root / "objects").glob("*.pt"))), 1)

    def test_publication_failure_matrix_preserves_exact_authority_boundaries(self) -> None:
        cases = (
            "before_object_promotion",
            "at_object_promotion",
            "before_objects_durability",
            "after_objects_durability",
            "before_catalog_replace",
            "during_catalog_replace",
            "after_catalog_replace_before_run_fsync",
            "after_final_run_fsync",
        )
        for case in cases:
            with self.subTest(stage=case):
                self.tearDown()
                self.setUp()
                first, second = self._patch_authority()
                with first, second:
                    save_phase8_checkpoint(
                        _state(self.repository_root),
                        self.repository_root,
                    )
                run_root = self.repository_root / "checkpoints/phase8" / RUN_ID
                catalog_path = run_root / "checkpoint-catalog.json"
                prior_catalog = catalog_path.read_bytes()
                prior_objects = set((run_root / "objects").glob("*.pt"))
                real_fsync = checkpoint.os.fsync
                real_catalog_bytes = checkpoint._canonical_catalog_bytes
                real_read_catalog = checkpoint._read_catalog
                catalog_bytes_calls = 0
                read_catalog_calls = 0

                def fsync(fd):
                    info = os.fstat(fd)
                    observed = (info.st_dev, info.st_ino)
                    objects_info = (run_root / "objects").stat()
                    run_info = run_root.stat()
                    if (
                        case == "before_objects_durability"
                        and observed == (objects_info.st_dev, objects_info.st_ino)
                    ):
                        raise OSError("objects fsync")
                    if (
                        case == "before_catalog_replace"
                        and not stat.S_ISDIR(info.st_mode)
                        and tuple(run_root.glob(".catalog-*.tmp"))
                    ):
                        raise OSError("catalog file fsync")
                    if (
                        case == "after_catalog_replace_before_run_fsync"
                        and observed == (run_info.st_dev, run_info.st_ino)
                    ):
                        raise OSError("run fsync")
                    return real_fsync(fd)

                def catalog_bytes(value):
                    nonlocal catalog_bytes_calls
                    catalog_bytes_calls += 1
                    if case == "after_objects_durability" and catalog_bytes_calls == 2:
                        raise Phase8CheckpointError("phase8.checkpoint.catalog")
                    return real_catalog_bytes(value)

                def read_catalog(*args, **kwargs):
                    nonlocal read_catalog_calls
                    read_catalog_calls += 1
                    if case == "after_final_run_fsync" and read_catalog_calls == 2:
                        raise Phase8CheckpointError("phase8.checkpoint.catalog")
                    return real_read_catalog(*args, **kwargs)

                with ExitStack() as stack:
                    first, second = self._patch_authority()
                    stack.enter_context(first)
                    stack.enter_context(second)
                    if case == "before_object_promotion":
                        stack.enter_context(
                            patch.object(
                                checkpoint.torch,
                                "save",
                                side_effect=RuntimeError("serialize"),
                            )
                        )
                    if case == "at_object_promotion":
                        stack.enter_context(
                            patch.object(
                                checkpoint.os,
                                "link",
                                side_effect=OSError("link"),
                            )
                        )
                    if case in (
                        "before_objects_durability",
                        "before_catalog_replace",
                        "after_catalog_replace_before_run_fsync",
                    ):
                        stack.enter_context(
                            patch.object(checkpoint.os, "fsync", side_effect=fsync)
                        )
                    if case == "after_objects_durability":
                        stack.enter_context(
                            patch.object(
                                checkpoint,
                                "_canonical_catalog_bytes",
                                side_effect=catalog_bytes,
                            )
                        )
                    if case == "during_catalog_replace":
                        stack.enter_context(
                            patch.object(
                                checkpoint.os,
                                "replace",
                                side_effect=OSError("replace"),
                            )
                        )
                    if case == "after_final_run_fsync":
                        stack.enter_context(
                            patch.object(
                                checkpoint,
                                "_read_catalog",
                                side_effect=read_catalog,
                            )
                        )
                    with self.assertRaises(Exception) as caught:
                        save_phase8_checkpoint(
                            _epoch_two_state(self.repository_root),
                            self.repository_root,
                        )
                new_objects = set((run_root / "objects").glob("*.pt")) - prior_objects
                self.assertEqual(tuple(run_root.glob(".*.tmp")), ())
                if case in (
                    "before_object_promotion",
                    "at_object_promotion",
                ):
                    self.assertFalse(new_objects)
                    self.assertEqual(catalog_path.read_bytes(), prior_catalog)
                elif case in (
                    "before_objects_durability",
                    "after_objects_durability",
                    "before_catalog_replace",
                    "during_catalog_replace",
                ):
                    self.assertEqual(len(new_objects), 1)
                    self.assertEqual(catalog_path.read_bytes(), prior_catalog)
                elif case == "after_catalog_replace_before_run_fsync":
                    self.assertEqual(len(new_objects), 1)
                    self.assertNotEqual(catalog_path.read_bytes(), prior_catalog)
                    self.assertEqual(
                        caught.exception.details["invariant"],
                        "phase8.publication.uncertain",
                    )
                else:
                    self.assertEqual(len(new_objects), 1)
                    self.assertNotEqual(catalog_path.read_bytes(), prior_catalog)
                    self.assertEqual(caught.exception.details["state"], "committed")

    def test_orphan_object_never_changes_catalog_authority(self) -> None:
        reference = self._save()
        run_root = self.repository_root / "checkpoints/phase8" / RUN_ID
        catalog_before = (run_root / "checkpoint-catalog.json").read_bytes()
        orphan = run_root / "objects" / ("f" * 64 + ".pt")
        orphan.write_bytes(b"orphan")
        restored = self._load()
        self.assertEqual(restored.progress.completed_epochs, 1)
        self.assertEqual((run_root / "checkpoint-catalog.json").read_bytes(), catalog_before)
        self.assertNotEqual(orphan.name, Path(reference.relative_path).name)

    def test_symlinked_objects_directory_is_rejected(self) -> None:
        phase8 = self.repository_root / "checkpoints/phase8"
        phase8.mkdir()
        outside = self.repository_root / "outside"
        outside.mkdir()
        (phase8 / RUN_ID).mkdir()
        (phase8 / RUN_ID / "objects").symlink_to(outside, target_is_directory=True)
        first, second = self._patch_authority()
        with first, second, self.assertRaises(Phase8CheckpointError):
            load_phase8_checkpoint(
                self.repository_root,
                run_id=RUN_ID,
                role="latest",
            )

    def test_synthetic_resume_complete_epoch_and_payload_are_equal(self) -> None:
        original = _state(self.repository_root)
        first, second = self._patch_authority()
        with first, second:
            save_phase8_checkpoint(original, self.repository_root)
            restored = load_phase8_checkpoint(
                self.repository_root,
                run_id=RUN_ID,
                role="latest",
            )
        windows = _windows()
        original_order = create_phase8_epoch_order(
            len(windows),
            epoch=2,
            generator=original.order_generator,
        )
        restored_order = create_phase8_epoch_order(
            len(windows),
            epoch=2,
            generator=restored.order_generator,
        )
        self.assertEqual(original_order, restored_order)
        original_batches = tuple(
            iter_phase8_logical_batches(windows, original_order, epoch=2)
        )
        restored_batches = tuple(
            iter_phase8_logical_batches(windows, restored_order, epoch=2)
        )
        self.assertEqual(original_batches, restored_batches)
        for original_batch, restored_batch in zip(
            original_batches,
            restored_batches,
            strict=True,
        ):
            original_result = run_phase8_logical_batch(
                original.model,
                original.optimizer,
                windows,
                original_batch,
            )
            restored_result = run_phase8_logical_batch(
                restored.model,
                restored.optimizer,
                windows,
                restored_batch,
            )
            self.assertEqual(original_result, restored_result)
        progress = Phase8Progress(2, 3, 0, 2, 4, 8)
        original_training = evaluate_phase8_split(
            original.model,
            original.optimizer,
            windows,
            original.order_generator,
            progress,
            split="train",
        )
        restored_training = evaluate_phase8_split(
            restored.model,
            restored.optimizer,
            windows,
            restored.order_generator,
            progress,
            split="train",
        )
        validation_windows = (
            Phase8Window("the-tempest", 7, "validation", 0, (10, 11), (11, 12)),
        )
        original_validation = evaluate_phase8_split(
            original.model,
            original.optimizer,
            validation_windows,
            original.order_generator,
            progress,
            split="validation",
        )
        restored_validation = evaluate_phase8_split(
            restored.model,
            restored.optimizer,
            validation_windows,
            restored.order_generator,
            progress,
            split="validation",
        )
        self.assertEqual(original_training, restored_training)
        self.assertEqual(original_validation, restored_validation)
        best_loss = min(original.metrics.best_validation_loss, original_validation.loss)
        best_epoch = 2 if original_validation.loss < original.metrics.best_validation_loss else 1
        metrics = _make_metric_state(
            initialized_training=original.metrics.initialized_training,
            initialized_validation=original.metrics.initialized_validation,
            epoch_training=(*original.metrics.epoch_training, original_training),
            epoch_validation=(*original.metrics.epoch_validation, original_validation),
            current_training=original_training,
            current_validation=original_validation,
            best_validation_loss=best_loss,
            best_validation_epoch=best_epoch,
            best_logical_id=f"epoch-{best_epoch:04d}",
        )
        original = _make_training_state(
            run_id=original.run_id,
            code_commit=original.code_commit,
            configuration=original.configuration,
            model=original.model,
            optimizer=original.optimizer,
            order_generator=original.order_generator,
            progress=progress,
            metrics=metrics,
            global_cpu_rng_state=torch.get_rng_state().clone(),
            resumed_from_checkpoint_sha256=None,
            resume_checkpoint_sha256s=(),
        )
        restored_metrics = _make_metric_state(
            initialized_training=restored.metrics.initialized_training,
            initialized_validation=restored.metrics.initialized_validation,
            epoch_training=(*restored.metrics.epoch_training, restored_training),
            epoch_validation=(*restored.metrics.epoch_validation, restored_validation),
            current_training=restored_training,
            current_validation=restored_validation,
            best_validation_loss=best_loss,
            best_validation_epoch=best_epoch,
            best_logical_id=f"epoch-{best_epoch:04d}",
        )
        restored = _make_training_state(
            run_id=restored.run_id,
            code_commit=restored.code_commit,
            configuration=restored.configuration,
            model=restored.model,
            optimizer=restored.optimizer,
            order_generator=restored.order_generator,
            progress=progress,
            metrics=restored_metrics,
            global_cpu_rng_state=torch.get_rng_state().clone(),
            resumed_from_checkpoint_sha256=restored.resumed_from_checkpoint_sha256,
            resume_checkpoint_sha256s=restored.resume_checkpoint_sha256s,
        )
        for left, right in zip(
            original.model.state_dict().values(),
            restored.model.state_dict().values(),
            strict=True,
        ):
            self.assertTrue(torch.equal(left, right))
        self.assertTrue(
            _state_dict_equal(
                original.optimizer.state_dict(),
                restored.optimizer.state_dict(),
            )
        )
        original_payload = checkpoint._payload(original, self.repository_root)
        restored_payload = checkpoint._payload(restored, self.repository_root)
        original_payload.pop("resume_lineage")
        restored_payload.pop("resume_lineage")
        self.assertTrue(optimization._objects_equal(original_payload, restored_payload))


if __name__ == "__main__":
    unittest.main()
