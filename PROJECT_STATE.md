# Project State

> Primary session anchor. Keep this document concise and current; it describes now, not project history.

## Current phase

Phase 2 — Tokenization (contract design only)

## Current milestone

Gate 6 — await separate explicit tokenizer implementation authorization after the accepted contract checkpoint is committed, pushed, and remotely verified in this task. Tokenizer implementation and production vocabulary construction are not authorized.

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
- Phase 1 explicitly authorized.
- Dataset suitability, scope, provenance, and leakage tradeoffs discussed.
- Project Gutenberg eBook #100 selected as the sole source for an approximately 170,000-word eight-play corpus.
- Whole-work manifest approved: six training plays, *The Tempest* for validation, and *Twelfth Night* for test.
- Training-only Unicode character inventory and non-mutating validation/test unseen-character audit approved without selecting a tokenizer.
- Dataset specification and gated acquisition contract documented in `docs/DATASET_SPEC.md`.
- Exact raw-source Git preservation exception and authoritative tracked manifest contract documented.
- Corrected dataset specification and acquisition contract reviewed and approved.
- Dataset-design specification committed as `a3821da6331c8e885f6935c442e896105c9a9340`.
- The catalog-linked Project Gutenberg eBook #100 UTF-8 artifact acquired once to the approved raw path without transformation.
- Raw source verified at 5,638,480 bytes with SHA-256 `3cf4b3d44ee14cff4e14e78e2ad3318eff76f3f7f2afc3cee6bb925879110a37`.
- Strict UTF-8 decoding passed; no BOM or NUL bytes were found; all 196,398 observed line endings are CRLF.
- Source provenance and response metadata recorded in the authoritative tracked JSON manifest.
- Read-only wrapper, global contents, selected-work marker, and formatting inspection recorded without exposing *Twelfth Night* prose.
- Raw-source acquisition and provenance milestone reviewed and approved for its dedicated checkpoint.
- Raw-source acquisition checkpoint committed as `8ffba6e005ac26e1b3f81c1d914df97e2acd7b25`.
- Deterministic eight-play extraction policy approved and recorded as DEC-0009.
- Exact marker-based boundaries, content inclusions/exclusions, minimal normalization, split paths, verification invariants, metadata, and sealed-test rules documented.
- Exact Gutenberg wrapper anchors, global contents contract, ordered 44-entry list, and eight successor-separator assertions approved under DEC-0010.
- Exact-preflight documentation checkpoint committed as `6efb00a5cd8e9831f83846fc30acf152bc27e1e9`.
- Final authoritative-status consistency verification completed with no remaining technical blockers.
- Read-only preflight implementation explicitly authorized.
- Standard-library preflight and focused tests implemented without extraction or output-writing behavior.
- Production preflight passed against the pinned source; all 16 focused tests passed, and the full 19-test suite passed with the expected restricted-context MPS skip.
- Independent review passed and the read-only preflight implementation was accepted for its dedicated checkpoint.
- Source-faithful per-work Dramatis indentation and three distinct provenance hash stages approved under DEC-0011.
- Deterministic in-memory extraction explicitly authorized and implemented through a separate module.
- Immutable validated-source handoff added so extraction consumes the exact byte object that passed preflight.
- Eighteen focused extraction tests passed; the 17-test preflight suite passed; the full 38-test suite passed with the expected restricted-context MPS skip.
- Production extraction returned eight separate works deterministically and computed three-stage provenance without creating processed output.
- Independent review passed and the deterministic in-memory extraction implementation was accepted for its dedicated checkpoint.
- In-memory Unicode character inventory explicitly authorized and implemented without rereading, re-extracting, or publishing text.
- Ten focused inventory tests passed; extraction and preflight regressions passed; the full 48-test suite passed with the expected restricted-context MPS skip.
- Production inventory counted eight independent works deterministically under Python 3.14.4 without creating `data/processed/`.
- Independent review passed with no must-fix issues, and Sebastien accepted the in-memory Unicode character-inventory checkpoint.
- The unseen-character comparison found zero validation/test code points absent from training; its candidate and occurrence-entry collections are empty, so occurrence-location reporting is not applicable for the pinned corpus.
- The transactional publisher design was accepted and recorded as DEC-0012.
- A standard-library publisher and 26 focused tests were implemented using temporary repositories only; the full 74-test suite passed with the expected restricted-context MPS skip.
- Production `ExtractedWork` compatibility passed through a temporary repository without creating the real `data/processed/` tree.
- Publisher reviews reproduced target-promotion and cleanup-ownership races; exclusive no-clobber promotion and preservation of failed staging state now pass deterministic regression tests for target, top-level staging, nested-entry, and final-root replacement.
- Descriptor-relative exclusive/no-follow directory and file creation prevents staged-path symlinks or substituted intermediate directories from redirecting writes; both construction races now have deterministic regression coverage.
- Darwin feasibility review established that the directory-tree publisher cannot guarantee safety against an unrelated same-user namespace adversary; Sebastien accepted an append-only DEC-0012 amendment limiting guarantees to ordinary workspaces and cooperating locked publishers.
- An empty atomic `data/.publish-lock` presence-file protocol now serializes cooperating publishers, releases after success or clean pre-publication refusal, and is preserved with stale or uncertain state; four focused lock-lifecycle tests pass.
- Final independent review passed with no must-fix issues, and Sebastien accepted the complete publisher implementation and DEC-0012 amendment.
- The authoritative manifest now records publisher revision `cd4cd8159b420920aa66755630fe26f9633a5373`, the accepted environment, exact exclusions, per-work provenance and processed results, split/global aggregates, character-inventory results, duplicate/boundary conclusions, and expected generated-manifest identity.
- A production ledger-verification test and temporary-root publication/rerun verification pass; the publisher suite has 27 tests and the full 75-test suite passes with the expected restricted-context MPS skip.
- Independent review passed with no must-fix issues, and Sebastien accepted the final expected-result ledger checkpoint.
- Production publication passed every precondition and returned `created`; all eight processed documents and the generated processing manifest match the authoritative ledger exactly.
- The generated processing manifest has the expected SHA-256 `bbf938e565022dde72f26470e2fa7214f2fe1735afbc62ace320f1e6ebf372cc`.
- A second production publication returned `already_current` without changing any processed file hash, size, or modification time; the publication lock and staging tree are absent.
- All processed artifacts remain ignored, reproducible derivatives. The read-only preflight, extraction, and inventory production tests were updated to prove the published tree remains unchanged; all 75 tests pass with the expected restricted-context MPS skip.
- Final closure review confirmed every Phase 1 exit criterion is technically satisfied; the stale broad publication prohibition was corrected to preserve the accepted publisher-only permission boundary.
- Phase 1 — Dataset is complete.
- The accepted Phase 1 completion commit `16fd4a6e27e44c3b705b0050090e2c1d9e515032` is pushed to and independently verified on `origin/main`.
- Phase 2 documentation and contract design are explicitly authorized.
- DEC-0013's scope correction limits the accepted conceptual strategy to one Python Unicode code point per token, training-only vocabulary membership, numeric code-point ordering determining IDs, no Unicode normalization, no special tokens initially, strict unsupported-code-point rejection, exact supported-input round trips, independent documents, and no Phase 2 context-window construction.
- The first independent DEC-0014 review returned FAIL with six required contract corrections, and Sebastien accepted every finding.
- Corrected DEC-0014 and `docs/TOKENIZER_SPEC.md` define the accepted full Python integer domain including surrogates, first-failure encode/decode rules, `Sequence[int]` decoding, immutable canonical state, two-layer vocabulary construction, orchestration-level sealed-test enforcement, an exact JSON artifact schema and provenance chain, ephemeral statistics, proposed later-phase allocation, and nineteen separate Phase 2 gates.
- Focused independent re-review passed with no must-fix issues after two non-semantic wording refinements, and Sebastien explicitly accepted corrected DEC-0014.
- Gate 3 is complete. The Gate 4 contract checkpoint commit and Gate 5 push/remote verification are explicitly authorized in this task.

## Current work

The accepted Phase 2 contract/documentation checkpoint is being committed and remotely verified under explicit authorization. No tokenizer implementation, production vocabulary, real token IDs, corpus encoding, or tokenizer statistics exist.

## Current model status

None. No model code exists.

## Last verified working state

The accepted Phase 1 pipeline and authoritative ledger remain intact at the pushed completion commit. The raw source and ignored processed dataset retain their accepted hashes. The full 75-test suite passes with the expected restricted-context MPS skip after final contract acceptance. Only Phase 2 Markdown contract documentation is included in the contract checkpoint; no tokenizer, vocabulary artifact, token IDs, dataset loader, context windows, or model implementation exists.

## Next exact step

Obtain separate explicit Gate 6 authorization before implementing any part of the accepted tokenizer contract.

## Known issues

- PyTorch emits a warning that optional NumPy interoperability is unavailable. NumPy is intentionally not installed because no current milestone requires it.
- MPS is unavailable inside some restricted execution contexts, but direct host verification and the unrestricted test suite both pass.
- The catalog reports a 2025-08-24 update date, while the artifact response reports `Last-Modified: Tue, 01 Sep 2026 07:55:49 GMT`; both are recorded without assuming they describe the same revision mechanism.
- Processed files are intentionally ignored derivatives; a fresh checkout regenerates them through the accepted publisher and authoritative ledger.
- The Phase 1 dataset specification and manifest retain their Phase 1-closure statements that tokenization was unauthorized; DEC-0013 and this current-state file supersede only that historical authorization status without modifying either Phase 1 artifact.

## Important constraints

- Publication or regeneration is authorized only through the accepted deterministic Shakespeare publisher using the authoritative manifest/ledger under DEC-0012's Models A/B threat model, with validation, cooperative locking, idempotence, no-clobber promotion, and mismatch refusal intact.
- Do not add dependencies without an approved, recorded reason and lock-file update.
- Do not manually edit, copy, or write processed prose or metadata; use alternate or unledgered publication mechanisms; bypass publisher validation; introduce force, overwrite, or repair behavior; or apply tokenizer-derived or other Phase 2 transformations.
- Do not implement occurrence-location lookup for the pinned corpus; the accepted audit has zero candidates and requires no location records.
- Derive the Phase 1 Unicode character inventory from training data only; do not define a token vocabulary in Phase 1.
- Audit and report validation/test-only Unicode code points under the sealed-test diagnostic contract without discarding, replacing, normalizing, or adding them.
- Do not implement the tokenizer or construct the production vocabulary until their separate gates are explicitly authorized.
- Do not tokenize or inspect the sealed test work during Phase 2.
- Do not create context windows, neural networks, Transformers, training, or inference yet.
- Do not use a pretrained language model or external model API as the model implementation.
- Do not skip educational phases or silently change the model architecture.
- Preserve reproducibility and keep core machine-learning code understandable.

## Open questions

- No known contract ambiguity or review blocker remains.
- Tokenizer implementation still requires separate explicit Gate 6 authorization.

## Session handoff

Phase 1 is complete at pushed and remotely verified commit `16fd4a6e27e44c3b705b0050090e2c1d9e515032`. Corrected DEC-0013 contains the accepted conceptual boundary. Corrected DEC-0014 passed focused independent re-review with no must-fix issues and is accepted. The documentation-only contract checkpoint is committed and remotely verified by the Gate 4/5 operation associated with this state. The next exact gate is separate explicit Gate 6 implementation authorization. No tokenizer, production vocabulary, real token IDs, corpus encoding, test tokenization, context windows, or later-phase implementation exists.
