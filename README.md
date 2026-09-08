# SebGPT

SebGPT is a learning-first project for understanding modern language models by progressively building a small GPT-style language model from first principles in Python and PyTorch.

The repository is the authoritative source of truth across working sessions. Begin with [PROJECT_STATE.md](PROJECT_STATE.md) to see the current milestone and next action, then consult [ROADMAP.md](ROADMAP.md) and the relevant entries in [DECISIONS.md](DECISIONS.md).

## Current status

Phase 0 — Project and Environment, Phase 1 — Dataset, and Phase 2 — Tokenization are complete and verified. SebGPT's first tokenizer uses one Python Unicode code point per token, a frozen training-derived 81-entry vocabulary, numeric code-point ID order, no special tokens or normalization, strict unknown rejection, and exact supported-text round trips. The sole Phase 2 production artifact is `artifacts/tokenizers/shakespeare-code-point-v1/vocabulary.json`, with SHA-256 `9f4235f9dab3e0361221a90c5ae7ca7fee540760fcc1708964368e6f17d6d70e`. Training and validation statistics are accepted; the sealed test remained untouched by tokenizer use, and no persistent token streams exist. Phase 3 learning/design and its corrected detailed contract are accepted after independent review and passing focused re-review. The contract uses explicit 32-dimensional token embeddings, learned absolute positional embeddings with capacity 256, deterministic CPU initialization, and exact vocabulary-artifact binding. The dedicated accepted-contract commit completes Gate 6; no embedding or model implementation exists, and push and implementation are not authorized.

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
