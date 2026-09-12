"""Synthetic lifecycle and provenance tests for Phase 8."""

from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import sebgpt.training.phase8_checkpoint as checkpoint  # noqa: E402
import sebgpt.training.phase8_experiment as experiment  # noqa: E402
from sebgpt.model.phase4_experiment import Phase4PermittedCorpus  # noqa: E402
from sebgpt.tokenization.vocabulary_artifact import (  # noqa: E402
    load_accepted_vocabulary_binding,
)
from sebgpt.training import (  # noqa: E402
    Phase8GovernanceError,
    Phase8Evaluation,
    Phase8Window,
    configure_phase8_runtime,
    load_phase8_configuration,
    run_fixed_phase8_experiment,
    validate_phase8_runtime,
)


class Phase8ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        configure_phase8_runtime()

    def test_runtime_and_configuration_are_repeatable(self) -> None:
        first = validate_phase8_runtime()
        second = validate_phase8_runtime()
        self.assertEqual(first, second)
        config = load_phase8_configuration()
        self.assertEqual(config.runtime, first)
        self.assertEqual(config.schema_version, 1)
        self.assertEqual(config.evaluation_split_order, ("train", "validation"))
        self.assertIsNone(config.scheduler_name)
        self.assertEqual(config.warmup_steps, 0)

    def test_exact_experiment_record_oracles(self) -> None:
        self.assertEqual(
            experiment._PLANNED_RECORD_FIELDS,
            (
                "Experiment ID", "Entry kind", "Recorded at UTC", "Status",
                "Question", "Authorization", "Code commit", "Phase 7 authority",
                "Phase 8 contract authority", "Runtime identity", "Dataset authority",
                "Tokenizer authority", "Model configuration", "Training configuration",
                "Checkpoint configuration", "Feasibility evidence", "Success predicate",
                "Sealed-test policy", "Generation policy", "Next gate",
            ),
        )
        self.assertEqual(len(experiment._RESULT_RECORD_FIELDS), 38)
        self.assertEqual(experiment._RESULT_RECORD_FIELDS[0], "Experiment ID")
        self.assertEqual(experiment._RESULT_RECORD_FIELDS[-1], "Next gate")
        self.assertEqual(
            experiment._float_record(1.5),
            {"decimal": "1.5", "hex": "0x1.8000000000000p+0"},
        )

    def test_planned_run_id_requires_one_exact_phase8_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = {name: "recorded" for name in experiment._PLANNED_RECORD_FIELDS}
            values.update(
                {
                    "Experiment ID": "EXP-20260911-03",
                    "Entry kind": "planned",
                    "Recorded at UTC": "2026-09-11T12:00:00Z",
                    "Status": "planned",
                    "Success predicate": "min(training_loss at completed epochs 1..10) < initialized training_loss",
                }
            )
            (root / "EXPERIMENT_LOG.md").write_text(
                "### EXP-20260911-03 — Phase 8 fixed training\n\n"
                + "\n".join(
                    f"- **{label}:** `{values[label]}`"
                    for label in experiment._PLANNED_RECORD_FIELDS
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(experiment._planned_run_id(root), "EXP-20260911-03")
            (root / "EXPERIMENT_LOG.md").write_text("no planned run\n", encoding="utf-8")
            with self.assertRaises(Exception):
                experiment._planned_run_id(root)

    def test_complete_pre_registration_binds_every_authority_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for relative in (
                "requirements.lock",
                "artifacts/tokenizers/shakespeare-code-point-v1/vocabulary.json",
                "docs/data/shakespeare-eight-play-manifest.json",
                "data/processed/shakespeare-eight-play/processing-manifest.json",
            ):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(REPOSITORY_ROOT / relative, target)
            config = load_phase8_configuration()
            authority = checkpoint._authority_mapping(config, root)
            values = {name: "recorded" for name in experiment._PLANNED_RECORD_FIELDS}
            values.update(
                {
                    "Experiment ID": "EXP-20260911-03",
                    "Entry kind": "planned",
                    "Recorded at UTC": "2026-09-11T12:00:00Z",
                    "Status": "planned",
                    "Code commit": "a" * 40,
                    "Phase 7 authority": {
                        key: authority[key]
                        for key in (
                            "phase7_closure_commit",
                            "phase7_contract_commit",
                            "phase7_implementation_commit",
                            "mini_gpt_spec_sha256",
                            "mini_gpt_source_sha256",
                            "mini_gpt_test_sha256",
                        )
                    },
                    "Phase 8 contract authority": {
                        "phase8_contract_commit": authority["phase8_contract_commit"],
                        "phase8_spec_sha256": authority["phase8_spec_sha256"],
                    },
                    "Runtime identity": authority["runtime"],
                    "Dataset authority": authority["dataset"],
                    "Tokenizer authority": authority["tokenizer"],
                    "Model configuration": authority["model"],
                    "Training configuration": checkpoint._configuration_mapping(config),
                    "Checkpoint configuration": {
                        "checkpoint_schema_version": 1,
                        "catalog_schema_version": 1,
                        "maximum_catalog_bytes": 16_384,
                        "maximum_checkpoint_object_bytes": 67_108_864,
                    },
                    "Sealed-test policy": "none",
                    "Generation policy": "none",
                    "Success predicate": "min(training_loss at completed epochs 1..10) < initialized training_loss",
                }
            )

            def render(value):
                if type(value) is dict:
                    return json.dumps(
                        value,
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                return f"`{value}`"

            log_path = root / "EXPERIMENT_LOG.md"
            log_path.write_text(
                "### EXP-20260911-03 — Phase 8 fixed training\n\n"
                + "\n".join(
                    f"- **{label}:** {render(values[label])}"
                    for label in experiment._PLANNED_RECORD_FIELDS
                )
                + "\n",
                encoding="utf-8",
            )
            run_id, record = experiment._planned_run_record(
                root,
                code_commit="a" * 40,
                configuration=config,
            )
            self.assertEqual(run_id, "EXP-20260911-03")
            self.assertEqual(tuple(record), experiment._PLANNED_RECORD_FIELDS)
            values["Runtime identity"] = {
                **authority["runtime"],
                "processor": "mismatch",
            }
            log_path.write_text(
                "### EXP-20260911-03 — Phase 8 fixed training\n\n"
                + "\n".join(
                    f"- **{label}:** {render(values[label])}"
                    for label in experiment._PLANNED_RECORD_FIELDS
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Exception):
                experiment._planned_run_record(
                    root,
                    code_commit="a" * 40,
                    configuration=config,
                )

    def test_live_repository_preflight_rejects_caller_commit_mismatch_first(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()

            def fake_git(_root, *arguments):
                if arguments == ("rev-parse", "--show-toplevel"):
                    return str(root)
                if arguments == ("branch", "--show-current"):
                    return "main"
                if arguments == ("rev-parse", "HEAD"):
                    return "b" * 40
                if arguments == ("rev-parse", "origin/main"):
                    return "b" * 40
                if arguments[:2] == ("status", "--porcelain=v2"):
                    return ""
                raise AssertionError(arguments)

            with patch.object(checkpoint, "_git", side_effect=fake_git):
                with self.assertRaises(Phase8GovernanceError) as caught:
                    checkpoint._validate_live_repository(root, "a" * 40)
        self.assertEqual(caught.exception.details["field"], "head")
        self.assertEqual(
            caught.exception.details["invariant"],
            "phase8.governance.repository",
        )

    def test_sealed_supplier_is_rejected_before_configuration_or_repository(self) -> None:
        corpus = Phase4PermittedCorpus(
            phase_1_manifest={},
            phase_1_manifest_sha256="x",
            training_works=(),
            validation_works=(),
            training_source_tokens=0,
            training_targets=0,
            training_examples=0,
            validation_source_tokens=0,
            validation_targets=0,
            validation_examples=0,
            sealed_test_supplier=object(),
        )
        vocabulary = load_accepted_vocabulary_binding(REPOSITORY_ROOT)
        with self.assertRaises(Phase8GovernanceError) as caught:
            run_fixed_phase8_experiment(
                corpus,
                vocabulary,
                load_phase8_configuration(),
                code_commit="a" * 40,
            )
        self.assertEqual(
            caught.exception.details["invariant"],
            "phase8.governance.sealed_test",
        )

    def test_runner_has_no_epoch_zero_state_and_executes_durable_dual_branch_audit(self) -> None:
        from tests.test_phase8_checkpoint import _copy_authority
        from sebgpt.training.phase8_types import _make_metric_state as real_make_metric

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            _copy_authority(root)
            (root / "experiments").mkdir()
            corpus = Phase4PermittedCorpus(
                phase_1_manifest={},
                phase_1_manifest_sha256="x",
                training_works=(),
                validation_works=(),
                training_source_tokens=0,
                training_targets=0,
                training_examples=0,
                validation_source_tokens=0,
                validation_targets=0,
                validation_examples=0,
                sealed_test_supplier=None,
            )
            training_windows = (
                Phase8Window("hamlet", 1, "train", 0, (1, 2), (2, 3)),
            )
            validation_windows = (
                Phase8Window(
                    "the-tempest", 7, "validation", 0, (1, 2), (2, 3)
                ),
            )
            metric_calls = []
            audit_branches = []
            audit_publications = []
            audit_reloads = []
            semantic_comparisons = []
            real_audit = experiment._run_exact_resume_audit
            real_publish = experiment._publish_payload
            real_private_load = experiment._load_phase8_checkpoint
            real_public_load = experiment.load_phase8_checkpoint
            real_semantic_equal = experiment._semantic_equal

            def capture_metric(**values):
                metric_calls.append(values)
                return real_make_metric(**values)

            def evaluate(_model, _optimizer, windows, _generator, progress, *, split):
                base = 2.0 if split == "train" else 2.1
                direction = -0.1 if split == "train" else 0.01
                return Phase8Evaluation(
                    split,
                    base + direction * progress.completed_epochs,
                    2,
                    len(windows),
                )

            def audit(*args, **kwargs):
                result = real_audit(*args, **kwargs)
                audit_branches.append(result[:2])
                return result

            def publish(*args, **kwargs):
                result = real_publish(*args, **kwargs)
                audit_publications.append((kwargs.get("components"), result))
                return result

            def private_load(*args, **kwargs):
                result = real_private_load(*args, **kwargs)
                audit_reloads.append((kwargs.get("components"), result))
                return result

            def public_load(*args, **kwargs):
                result = real_public_load(*args, **kwargs)
                audit_reloads.append((None, result))
                return result

            def semantic_equal(left, right):
                result = real_semantic_equal(left, right)
                semantic_comparisons.append(
                    (left.progress.completed_epochs, right.progress.completed_epochs, result)
                )
                return result

            vocabulary = load_accepted_vocabulary_binding(REPOSITORY_ROOT)
            with (
                patch.object(experiment, "REPOSITORY_ROOT", root),
                patch.object(experiment, "_validate_live_repository", return_value=None),
                patch.object(checkpoint, "_validate_live_repository", return_value=None),
                patch.object(checkpoint, "_git", return_value="a" * 40),
                patch.object(
                    experiment,
                    "_planned_run_record",
                    return_value=("EXP-20260911-03", {}),
                ),
                patch.object(
                    experiment,
                    "build_phase8_windows",
                    side_effect=lambda *_args, split: (
                        training_windows if split == "train" else validation_windows
                    ),
                ),
                patch.object(experiment, "evaluate_phase8_split", side_effect=evaluate),
                patch.object(experiment, "_make_metric_state", side_effect=capture_metric),
                patch.object(experiment, "_run_exact_resume_audit", side_effect=audit),
                patch.object(experiment, "_publish_payload", side_effect=publish),
                patch.object(experiment, "_load_phase8_checkpoint", side_effect=private_load),
                patch.object(experiment, "load_phase8_checkpoint", side_effect=public_load),
                patch.object(experiment, "_semantic_equal", side_effect=semantic_equal),
            ):
                result = run_fixed_phase8_experiment(
                    corpus,
                    vocabulary,
                    load_phase8_configuration(),
                    code_commit="a" * 40,
                )
            self.assertEqual(result.progress.completed_epochs, 10)
            self.assertTrue(result.exact_resume_passed)
            self.assertEqual(len(audit_branches), 1)
            control, resumed = audit_branches[0]
            self.assertIsNot(control, resumed)
            self.assertIsNot(control.model, resumed.model)
            self.assertTrue(
                set(map(id, control.model.parameters())).isdisjoint(
                    set(map(id, resumed.model.parameters()))
                )
            )
            self.assertEqual(
                (control.progress.completed_epochs, resumed.progress.completed_epochs),
                (2, 2),
            )
            control_payload = checkpoint._payload(control, root)
            resumed_payload = checkpoint._payload(resumed, root)
            control_lineage = control_payload.pop("resume_lineage")
            resumed_lineage = resumed_payload.pop("resume_lineage")
            self.assertNotEqual(control_lineage, resumed_lineage)
            self.assertTrue(
                experiment._objects_equal(control_payload, resumed_payload)
            )
            audit_components = (
                "experiments",
                "EXP-20260911-03",
                "artifacts",
                "resume-audit-control",
                "objects",
            )
            self.assertTrue(
                any(components == audit_components for components, _ in audit_publications)
            )
            self.assertTrue(
                any(components == audit_components for components, _ in audit_reloads)
            )
            self.assertTrue(
                any(
                    components is None and state.progress.completed_epochs == 2
                    for components, state in audit_reloads
                )
            )
            self.assertIn((2, 2, True), semantic_comparisons)
            control_root = root / "experiments/EXP-20260911-03/artifacts/resume-audit-control"
            canonical_root = root / "checkpoints/phase8/EXP-20260911-03"
            self.assertTrue((control_root / "checkpoint-catalog.json").is_file())
            self.assertTrue((canonical_root / "checkpoint-catalog.json").is_file())
            self.assertGreaterEqual(len(tuple((control_root / "objects").glob("*.pt"))), 1)
            self.assertGreaterEqual(len(tuple((canonical_root / "objects").glob("*.pt"))), 10)
            evidence = json.loads(
                (
                    root
                    / "experiments/EXP-20260911-03/artifacts/phase8-training-evidence.json"
                ).read_text(encoding="utf-8")
            )
            audit_record = next(
                item
                for item in evidence["batch_and_resume_evidence"]
                if item.get("both_catalog_graphs_verified") is True
            )
            self.assertIs(audit_record["both_public_load_paths_reloaded"], True)
            self.assertIs(
                audit_record["semantic_state_equal_except_lineage_hash_location"],
                True,
            )
            self.assertNotEqual(
                audit_record["control_checkpoint_sha256"],
                audit_record["resumed_checkpoint_sha256"],
            )
            self.assertTrue(metric_calls)
            self.assertTrue(
                all(
                    math.isfinite(values["best_validation_loss"])
                    and values["best_validation_epoch"] >= 1
                    and values["best_logical_id"] != "epoch-0000"
                    for values in metric_calls
                )
            )


if __name__ == "__main__":
    unittest.main()
