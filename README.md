# SebGPT

SebGPT is a learning-first project for understanding modern language models by progressively building a small GPT-style language model from first principles in Python and PyTorch.

The repository is the authoritative source of truth across working sessions. Begin with [PROJECT_STATE.md](PROJECT_STATE.md) to see the current milestone and next action, then consult [ROADMAP.md](ROADMAP.md) and the relevant entries in [DECISIONS.md](DECISIONS.md).

## Current status

Phases 0–4 are complete and remotely closed. Phase 5 — Self-Attention is also
formally remotely closed at `6519d8c813b7e5bc899b73d6b1fa8c38a6819750`,
`Complete Phase 5 self-attention`. Local `HEAD`, `origin/main`, and actual remote
`main` were independently verified at that exact commit with ahead/behind `0/0`
and a clean worktree/index. All four Phase 5 exit criteria remain satisfied, and
the accepted implementation checkpoint remains
`f4b5a1d8d29ab7ee6eb5b987fe150fb040c4544e`.

Sebastien subsequently explicitly authorized learning/design only for Phase 6
— Transformer Block, whose exact goal is to combine attention and feed-forward
computation into a stable reusable block. The read-only continuity inspection
is complete and accepted. Sebastien then accepted the consolidated DEC-0018
conceptual architecture and authorized documentation-only detailed-contract
drafting. Proposed [docs/TRANSFORMER_BLOCK_SPEC.md](docs/TRANSFORMER_BLOCK_SPEC.md)
is complete locally. Master Chat sanity review returned PASS; fresh Codex
independent review returned FAIL with four MUST-FIX and one SHOULD-FIX finding;
and Master Chat accepted all five without changing DEC-0018. The authorized
documentation-only corrections completed Gate 9. Gate 10 focused independent
re-review returned PASS with all findings resolved; Master Chat adjudicated
PASS; and Sebastien explicitly accepted the corrected detailed contract. Its
mechanics are authoritative for later separately authorized implementation.
The dedicated six-file accepted-contract checkpoint commit is authorized. Push,
Phase 6 implementation, source, tests, parameter construction, and experiments
remain unauthorized. That checkpoint was subsequently committed as `2177c7e`,
pushed, and independently remote-verified with ahead/behind `0/0`. Sebastien then
explicitly authorized Phase 6 implementation. Gate 16 is complete locally with
the explicit Transformer block source, public exports, and 50 focused synthetic
tests. Relevant Phase 3–5 regressions pass 223/223, and the complete 395-test
suite has 394 passes, one expected MPS skip, and zero failures. Implementation
acceptance, staging, commit, and push remain unauthorized pending review.
Independent review returned FAIL with three MUST-FIX findings accepted by
Master Chat: one narrow standalone-dropout validation defect and two evidence
gaps. Focused source/test corrections are complete without changing the accepted
contract or DEC-0018. Focused tests pass 51/51, relevant Phase 3–5 regressions
pass 223/223, and the complete 396-test suite has 395 passes, one expected MPS
skip, and zero failures. Focused independent implementation re-review
subsequently returned PASS with all three findings
resolved. Master Chat adjudicated PASS, and Sebastien explicitly accepted the
corrected implementation and evidence suite. The accepted contract and
DEC-0018 remain unchanged. The dedicated eight-file accepted-implementation
checkpoint commit is authorized; push and Phase 6 exit review/closure remain
separately unauthorized. The implementation checkpoint was subsequently
committed as `c2624078eae5947e505d7f8093869e32c58e521b`, pushed, and
independently remote-verified. Educational inspection and independent exit
review passed all three exact Phase 6 criteria with no MUST-FIX findings. Master
Chat adjudicated PASS, and Sebastien accepted Phase 6 as technically complete.
No training experiment was required or performed. Formal remote closure still
requires a separately authorized closure commit, push, and independent remote
verification. Phase 7 remains unauthorized.

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
