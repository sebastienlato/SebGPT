# SebGPT

SebGPT is a learning-first project for understanding modern language models by progressively building a small GPT-style language model from first principles in Python and PyTorch.

The repository is the authoritative source of truth across working sessions. Begin with [PROJECT_STATE.md](PROJECT_STATE.md) to see the current milestone and next action, then consult [ROADMAP.md](ROADMAP.md) and the relevant entries in [DECISIONS.md](DECISIONS.md).

## Current status

Phase 0 — Project and Environment, Phase 1 — Dataset, Phase 2 — Tokenization,
and Phase 3 — Embeddings are complete and remotely verified through Phase 3
closure commit `a66d169e76f06bee18d4f32b6340206f9f45ee63`. Phase 4 — Simple
Neural Language Model is in experiment-orchestration prerequisite work. Its
twelve conceptual decisions
and corrected detailed contract are accepted after a seven-finding correction
cycle and passing focused re-review. The contract is remotely verified at
`3fcf007ce819ca1a45aa75b48fa19f10311f2819`. Its deterministic shifted-example
and Shakespeare-governance production slice passed independent review; two
test-only coverage blockers were corrected, and 44 focused tests now pass with
production source unchanged. Focused re-review and adjudication passed, and the
slice is accepted and remotely verified at
`266797f891e9980b57bb35a633c91bef838d4111`. The positionwise model,
probability/loss mechanics, manual update, and no-update measurement slice is
implemented and accepted. Gate 19 found only a gradient-clearing validation-order
defect and its missing simultaneous-defect test; both are corrected with 48
focused tests passing. Focused Gate 20 re-review and master-chat adjudication
passed, and Sebastien explicitly accepted the model/loss/update slice. Gates
21–22 committed, pushed, and remotely verified it at `497ecde3577677903f14669722d61dcdf8caa1d6`.
An attempted experiment authorization stopped before model construction because
the accepted runner prerequisite was absent. That orchestration layer and 27
synthetic tests were reviewed; four focused corrections were required. The
corrections are complete with 41 synthetic tests passing and await focused
re-review. That review passed production and found one final test-only lifecycle
role blocker; the strengthened sentinel-identity test passed final focused
re-review and master-chat adjudication, and Sebastien explicitly accepted the
runner. Its dedicated commit remains separately gated. No experiment has run,
and the bounded experiment remains unauthorized.

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
