# SebGPT

SebGPT is a learning-first project for understanding modern language models by progressively building a small GPT-style language model from first principles in Python and PyTorch.

The repository is the authoritative source of truth across working sessions. Begin with [PROJECT_STATE.md](PROJECT_STATE.md) to see the current milestone and next action, then consult [ROADMAP.md](ROADMAP.md) and the relevant entries in [DECISIONS.md](DECISIONS.md).

## Current status

Phases 0–7 are complete and remotely closed. Phase 7 closure commit
`33d4510421107848c4aa8a6014f4a7b1e391065a`, `Complete Phase 7 Mini-GPT`, is
pushed and synchronized; its accepted contract and implementation checkpoints
remain `60b2a9cce55da79ccc9fbd03fad014cb2a939290` and
`3139b1736f005fe903e2ea111d91934478b5a683`.

Phase 8 — Training and Checkpointing is technically complete, its result and
exit state are accepted, and it is ready for formal closure. It is not yet
formally or remotely closed. DEC-0020 remains the accepted conceptual policy;
the corrected detailed contract is fixed at commit `c50d77a` and SHA-256
`1630f9c7a113ff4af6db709ba2c356e8dab450eb4d918a54d64b734084e70efd`;
the accepted implementation and corrected provenance implementation are
`8098343` and `3e0b9c6`.

The fixed real run `EXP-20260912-01` completed all ten epochs and `3880`
optimizer updates. Training loss improved from `4.750465878532` to
`2.4611534265660575`; validation loss improved from `4.754711149949085` to its
epoch-10 best/final value `2.4967258539791177`. The training-improvement
predicate, checkpoint/catalog validation, and complete exact-resume audit all
passed. Latest and best-validation are distinct roles for immutable epoch-10
checkpoint `6990c89166d48732b0601aeae1edd57a9f0e22eec1c7c1160d5be1b8281c9148`;
catalog SHA-256 is
`6d590f0355ec950d770e77b4c542d65cda15012fbea56349b97b154a0d614241`.

Gate 78 independent review returned PASS / NO ISSUE with no BLOCKER, IMPORTANT,
or MINOR finding and recommended `ACCEPT PHASE 8 RUN RESULT`. Sebastien accepted
the experiment, checkpoint state, technical exit, and closure-ready state at
Gates 80 and 86. The one-file result commit
`334119e63c716e4922cd0b67da4f5fc221faa886`, `Record Phase 8 training result`,
is pushed and remotely verified.

Gate 88 documentation-only closure bookkeeping is complete locally. Gate 89
separate closure-commit authorization is next. No generation or sampling
occurred, *Twelfth Night* and the sealed test were not accessed, and Phase 9
remains unauthorized until Gate 93 independently verifies formal remote Phase 8
closure.

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

Deactivate the environment when finished:

```sh
deactivate
```

See [PROJECT_CHARTER.md](PROJECT_CHARTER.md) for the stable project purpose and [docs/SESSION_PROTOCOL.md](docs/SESSION_PROTOCOL.md) for the session workflow.
