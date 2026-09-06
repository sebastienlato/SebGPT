# Project State

> Primary session anchor. Keep this document concise and current; it describes now, not project history.

## Current phase

Phase 0 — Project and Environment

## Current milestone

Phase 0 environment complete; awaiting explicit approval to begin Phase 1.

## Completed work

- SebGPT project concept agreed.
- Learning-first approach agreed.
- Python and PyTorch direction agreed.
- Work will be used for primary project collaboration.
- Codex will be used as an engineering assistant.
- Repository files will be the authoritative source of truth.
- Initial continuity documents and project directories created.
- Phase 0.1 continuity infrastructure reviewed and approved.
- Mac environment audited: macOS 26.6.2, Apple M4 Max (`arm64`), Python.org CPython 3.14.4, pip 26.0.1, and Git 2.54.0.
- Project-local `.venv` created and excluded from Git.
- Stable PyTorch 2.14.0 installed from its native macOS ARM64 wheel.
- Direct and fully resolved dependency manifests created.
- CPU tensor operations, autograd, MPS availability, and an MPS tensor operation verified.
- Environment test suite added and verified with all three tests passing on the host Mac.
- Git repository initialized on `main` with a clean Phase 0 baseline commit.

## Current work

None. Phase 0 is complete and Phase 1 has not been authorized.

## Current model status

None. No model code exists.

## Last verified working state

The reproducible Python 3.14.4 environment installs from `requirements.lock`; PyTorch 2.14.0 imports successfully; package consistency checks pass; CPU tensor/autograd and Apple MPS tensor execution pass; the standard-library environment suite passes all three tests. No dataset or model implementation exists.

## Next exact step

Await Sebastien's explicit approval to begin Phase 1. Once approved, the first Phase 1 action is to define dataset learning objectives and selection criteria before selecting or downloading data.

## Known issues

- PyTorch emits a warning that optional NumPy interoperability is unavailable. NumPy is intentionally not installed because no current milestone requires it.
- MPS is unavailable inside some restricted execution contexts, but direct host verification and the unrestricted test suite both pass.

## Important constraints

- Do not proceed to Phase 1 without Sebastien's explicit approval.
- Do not add dependencies without an approved, recorded reason and lock-file update.
- Do not download a dataset yet.
- Do not implement tokenization, neural networks, Transformers, training, or inference yet.
- Do not use a pretrained language model or external model API as the model implementation.
- Do not skip educational phases or silently change the model architecture.
- Preserve reproducibility and keep core machine-learning code understandable.

## Open questions

- None for Phase 0.

## Session handoff

Phase 0 is complete and verified on the host Mac. Resume by reading this state and awaiting explicit Phase 1 approval. Do not select or download data before defining the Phase 1 learning objectives and dataset selection criteria with Sebastien.
