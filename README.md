# SebGPT

SebGPT is a learning-first project for understanding modern language models by progressively building a small GPT-style language model from first principles in Python and PyTorch.

The repository is the authoritative source of truth across working sessions. Begin with [PROJECT_STATE.md](PROJECT_STATE.md) to see the current milestone and next action, then consult [ROADMAP.md](ROADMAP.md) and the relevant entries in [DECISIONS.md](DECISIONS.md).

## Current status

Phases 0–7 are complete and remotely closed. Phase 7 closure commit
`33d4510421107848c4aa8a6014f4a7b1e391065a`, `Complete Phase 7 Mini-GPT`, is
pushed and synchronized; its accepted contract and implementation checkpoints
remain `60b2a9cce55da79ccc9fbd03fad014cb2a939290` and
`3139b1736f005fe903e2ea111d91934478b5a683`.

Phase 8 — Training and Checkpointing has an accepted contract and authorized
local implementation work.
DEC-0020 records the accepted consolidated policy: unchanged accepted
data/tokenizer/model authority; CPU float32; context 256; rank-one logical
batches of eight; deterministic epoch shuffling; constant AdamW at `3e-4` with
betas `(0.9, 0.999)`, epsilon `1e-8`, and weight decay `0.01`; global-norm
clipping at `1.0`; ten epochs without validation early stopping; full
initialized and end-epoch training/validation evaluation; latest and
best-validation checkpoints; complete runtime-state restoration; and exact
supported-environment resume.

Documentation-only proposed
[docs/TRAINING_CHECKPOINTING_SPEC.md](docs/TRAINING_CHECKPOINTING_SPEC.md)
defines the detailed window, ordering, accumulation, optimizer, evaluation,
checkpoint, restoration, evidence, failure, test, feasibility, and gate
contracts. Independent review returned CORRECT BEFORE ACCEPTANCE with three
BLOCKER, eight IMPORTANT, and one MINOR finding. The authorized targeted
documentation corrections are complete without changing DEC-0020: durability,
transactional RNG restoration, runtime identity, clipping, exact schemas,
catalog agreement, filesystem trust, live provenance, latest-only continuation,
extended resume evidence, gates, and permutation tests are corrected. The
focused re-review passed those items but found one remaining AdamW-state BLOCKER
and two public-ownership/configuration-schema IMPORTANT findings. Their final
targeted documentation corrections passed final focused re-review with no new
or remaining finding. Master Chat formally accepted the corrected contract at
SHA-256 `0b3e2a79de7038e2233560c0836101d2f5757f9a5e3c3e89e6ccb62e7cc6fffd`
without changing DEC-0020. The accepted documentation remains unstaged,
uncommitted, and unpushed pending separate checkpoint-commit authorization.
At the contract-acceptance checkpoint, Phase 8 source/tests and all later work
remained unauthorized; the subsequent Gate 24 authorization below supersedes
only the implementation portion of that historical boundary.

The accepted contract checkpoint `09b2c2e` is pushed and remotely verified.
Master Chat then authorized Phase 8 implementation. Gate 25 is complete locally.
Fresh independent review found five BLOCKER, six IMPORTANT, and one MINOR issue;
their authorized targeted correction is complete without changing DEC-0020 or
the accepted contract. Focused re-review found one remaining historical-best
mode-map BLOCKER and two IMPORTANT evidence/batching gaps; their final targeted
correction is complete. The focused suite now maps all 48 obligations to
categorized substantive evidence across 77 passing tests; relevant Phase
3/5/6/7 regressions pass 174/174; and the complete 520-test suite has 519 passes,
one expected MPS skip, and zero failures. Final focused re-review accepted the
production behavior but found one remaining evidence-quality gap. Its
evidence-only correction changed no Phase 8 source and expanded the focused
suite to 85/85 passes with direct matrices for ordering restoration, evaluation,
publication, restoration order, durable audit roots, bootstrap, descriptor
safety, and bounded reads. Relevant regressions remain 174/174; the complete
528-test suite has 527 passes, one expected MPS skip, and zero failures. The
checkpoint is unaccepted, unstaged, uncommitted, and unpushed pending one final
evidence-focused re-review. Feasibility measurement, real training, generation,
sampling, sealed-test access, and Phase 9 remain unauthorized.

That re-review left only static-boundary and unrelated-RNG ordering evidence
incomplete. Their final test-only correction changed no production source: a
mutation-tested AST oracle now covers all prohibited Phase 8 policy families,
and a real checkpoint round trip deliberately perturbs global/dropout RNG before
proving exact local order-state independence. Focused tests pass 86/86, the
174-test prior-phase subset passes, and the complete 529-test suite has 528
passes, one expected MPS skip, and zero failures. Final two-obligation review is
next; all implementation and execution boundaries remain unchanged.

Final verification accepted obligation 9 and left only static-boundary bypass
coverage incomplete. Its final test-only correction adds alias-aware padding,
special-token IDs/literals, prohibited precision strings, bounded test/sealed
alias propagation, expanded mutations across every Phase 8 module, and safe
negative controls. No production source changed. Focused tests pass 87/87; the
174-test prior-phase subset passes; and the complete 530-test suite has 529
passes, one expected MPS skip, and zero failures. Obligation-4-only verification
is next.

Final obligation-4 verification returned PASS with no remaining implementation
finding, all 48 focused obligations satisfied, and exact recommendation
`ACCEPT CORRECTED IMPLEMENTATION`. Master Chat formally accepts the corrected
Phase 8 implementation. The accepted local checkpoint retains the verified
87/87 focused, 174/174 prior-phase regression, and 529-pass/one-skip full-suite
evidence. It remains unstaged, uncommitted, and unpushed pending separate
implementation-checkpoint commit authorization. Feasibility, real training,
Phase 9, generation, sampling, and sealed-test access remain unauthorized.

Run the read-only Phase 1 preflight from the repository root with:

```sh
PYTHONPATH=src .venv/bin/python -m sebgpt.data.shakespeare_preflight
```

The command reports only approved structural and numeric facts. It does not extract plays, transform text, or create processed output.

## Guiding principle

Core machine-learning concepts should be understood before they are abstracted or automated. PyTorch provides tensors, automatic differentiation, low-level neural-network primitives, optimization, and CPU/MPS execution. SebGPT will still implement and explain its educational data flow, tokenization, model architecture, training logic, generation, evaluation, and interpretability work phase by phase.

## Verified development environment

- macOS 26.6.2 on Apple Silicon (`arm64`)
- Python.org CPython 3.14.4
- PyTorch 2.14.0
- Project-local virtual environment at `.venv/`
- CPU and Apple Metal Performance Shaders (MPS) execution verified

Python 3.14.4 must already be available as `python3`. From the repository root, create and verify the environment with:

```sh
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip check
python -m unittest discover -s tests -v
```

The expected Python version is `3.14.4`. On a normal session on this Mac, all three environment tests pass. A restricted execution environment without GPU access may skip the MPS test. Phase 8 training remains CPU-only regardless of MPS availability.

`requirements.in` contains the dependency intentionally chosen by the project. `requirements.lock` pins the complete resolved environment and is the reproducible installation source. Change or regenerate either file only as part of an approved dependency decision.

Deactivate the virtual environment when finished:

```sh
deactivate
```

See [PROJECT_CHARTER.md](PROJECT_CHARTER.md) for the stable project purpose and [docs/SESSION_PROTOCOL.md](docs/SESSION_PROTOCOL.md) for the session workflow.
