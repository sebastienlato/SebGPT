# SebGPT

SebGPT is a learning-first project for understanding modern language models by progressively building a small GPT-style language model from first principles in Python and PyTorch.

The repository is the authoritative source of truth across working sessions. Begin with [PROJECT_STATE.md](PROJECT_STATE.md) to see the current milestone and next action, then consult [ROADMAP.md](ROADMAP.md) and the relevant entries in [DECISIONS.md](DECISIONS.md).

## Current status

Phases 0–3 are complete and remotely verified. Phase 4 — Simple Neural Language
Model is accepted, closed, pushed, and independently remote-verified at
`e8b5c55fb2f2f03b155c1a8b4308dd9df20d9e1c`. Its sole fixed experiment,
`EXP-20260909-01`, passed its pre-registered training-loss predicate; this does
not establish generalization or test performance. Gate 30 authorized Phase 5 —
Self-Attention learning/design. Sebastien approved the conceptual architecture,
and [docs/SELF_ATTENTION_SPEC.md](docs/SELF_ATTENTION_SPEC.md) now contains a
documentation-only detailed contract proposal. Gate 6 Master Chat sanity
review passed; Gate 7 fresh independent review failed with five MUST-FIX and
three SHOULD-FIX findings; and all eight accepted Gate 8 documentation
corrections are resolved. Gate 9 focused re-review and Master Chat adjudication
returned PASS, and Sebastien explicitly accepted the corrected contract without
changing DEC-0017. Its detailed mechanics are authoritative for later
implementation. The contract checkpoint is pushed and remotely verified at
`9443dec240ccdb5fdc1ebfcbe0f7b334893dbfff`; Gate 12 authorized implementation;
and Gate 13 implementation is complete locally with 40 focused tests passing.
The full 344-test suite passes with 343 passes and one expected MPS skip. The
Gate 14 independent review found two test-only evidence gaps and no production
or contract defect. Both Gate 15 corrections are complete locally; focused
tests pass 41/41 and the full 345-test suite passes with 344 passes and one
expected MPS skip. Focused independent re-review and Master Chat adjudication
returned PASS, and Sebastien explicitly accepted the implementation and
corrected evidence suite. Production remained byte-identical, the accepted
contract and DEC-0017 remain unchanged, and the dedicated implementation
checkpoint commit is authorized. Push, Phase 5 exit review/closure, training,
experiments, sealed-test access, and Phase 6 remain unauthorized.

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

The expected Python version is `3.14.4`. On a normal session on this Mac, all three environment tests pass. A restricted execution environment without GPU access may skip the MPS test.

`requirements.in` contains the dependency intentionally chosen by the project. `requirements.lock` pins the complete resolved environment and is the reproducible installation source. Change or regenerate either file only as part of an approved dependency decision.

Deactivate the virtual environment when finished:

```sh
deactivate
```

See [PROJECT_CHARTER.md](PROJECT_CHARTER.md) for the stable project purpose and [docs/SESSION_PROTOCOL.md](docs/SESSION_PROTOCOL.md) for the session workflow.
