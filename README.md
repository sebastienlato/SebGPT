# SebGPT

SebGPT is a learning-first project for understanding modern language models by progressively building a small GPT-style language model from first principles in Python and PyTorch.

The repository is the authoritative source of truth across working sessions. Begin with [PROJECT_STATE.md](PROJECT_STATE.md) to see the current milestone and next action, then consult [ROADMAP.md](ROADMAP.md) and the relevant entries in [DECISIONS.md](DECISIONS.md).

## Current status

Phases 0–6 are complete and remotely closed. Phase 6 closure commit
`e84b364a7955ddba86b30938babd5dae9e829935`, `Complete Phase 6 transformer
block`, is pushed and independently remote-verified; its accepted implementation
checkpoint remains `c2624078eae5947e505d7f8093869e32c58e521b`.

Sebastien authorized Phase 7 learning/design only. The continuity inspection is
complete, and DEC-0019 records the accepted consolidated Complete Mini-GPT
architecture: the accepted Phase 3 representation, exactly four distinct
accepted Phase 6 block structures, final explicit width-32 normalization, an
untied biased 81-class head, single-sequence length `1..256`, logits-only
forward, separate next-token loss, same-call inspection, and 63,825 parameters.
Documentation-only proposed
[docs/MINI_GPT_SPEC.md](docs/MINI_GPT_SPEC.md) defines detailed interfaces,
deterministic distinct-block initialization, parameter accounting, validation,
inspection, evidence, exclusions, and gates. It remained unaccepted pending
focused independent re-review after an initial CORRECT BEFORE ACCEPTANCE result
with five IMPORTANT findings and no BLOCKER. All five targeted documentation
corrections were completed without changing DEC-0019 or the initialization
design. Focused re-review returned PASS with all five findings resolved and no
remaining BLOCKER, IMPORTANT, or MINOR finding. Codex recommended `ACCEPT
CORRECTED CONTRACT`, and Master Chat formally accepted the corrected detailed
contract. Separate documentation-checkpoint commit authorization is next and
remains unauthorized. Phase 7 implementation, source, tests, parameter
construction, training, experiments, sealed-test access, staging, commit, push,
Phase 8, and Phase 9 remain unauthorized.

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
