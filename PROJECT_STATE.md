# Project State

> Primary session anchor. Keep this document concise and current; it describes now, not project history.

## Current phase

Phase 4 — Simple Neural Language Model (contract design)

## Current milestone

Phase 4 Gate 14 — Both Gate 13 test-only findings are resolved, production
remains byte-identical, focused independent re-review returned PASS,
master-chat adjudication returned PASS, and Sebastien explicitly accepted the
shifted-example/governance slice. Gate 15, its dedicated implementation commit,
requires separate authorization. Model, loss, update, and experiment work
remain unauthorized.

## Completed work

- Phase 0 — Project and Environment is complete with a reproducible Python
  3.14.4/PyTorch 2.14.0 environment and verified CPU/MPS fundamentals.
- Phase 1 — Dataset is complete. The pinned Gutenberg source, deterministic
  eight-play extraction, whole-work train/validation/test split, provenance,
  publication, and sealed-test governance are accepted and verified.
- Phase 2 — Tokenization is complete. The accepted code-point tokenizer has a
  frozen training-derived 81-entry vocabulary bound to artifact SHA-256
  `9f4235f9dab3e0361221a90c5ae7ca7fee540760fcc1708964368e6f17d6d70e`.
  Training and validation tokenization statistics were verified; the sealed
  test remained untokenized.
- Phase 3 — Embeddings is complete. The accepted implementation has explicit
  `(81, 32)` token and `(256, 32)` learned absolute-position tables, direct
  addition, deterministic local initialization, CPU float32 parameters,
  `torch.long` IDs, and exact vocabulary binding. All three exit criteria
  passed independent review.
- Phase 3 closure commit `a66d169e76f06bee18d4f32b6340206f9f45ee63`
  is pushed and independently verified on remote `main`; local `HEAD` and
  `origin/main` matched with ahead/behind `0/0` before the current edits.
- Phase 4 learning, repository orientation, and conceptual design are complete.
  Sebastien accepted the twelve decisions recorded in DEC-0016: a positionwise
  linear/log-bilinear baseline, shifted examples, context/stride 64, variable
  tails without padding, on-demand immutable examples, one-example runtime,
  an untied `(32, 81)` output weight plus `(81,)` bias, transparent stable
  cross-entropy, manual SGD, train/validation/test governance, and `ln(81)` as
  the primary conceptual baseline.
- The corrected Phase 4 detailed contract passed focused review, was explicitly
  accepted, committed, pushed, and independently verified at
  `3fcf007ce819ca1a45aa75b48fa19f10311f2819`.

## Current work

`src/sebgpt/data/shifted_examples.py` implements the corpus-neutral frozen
`ShiftedTokenExample`, three Phase 4 content-safe exceptions, and an eager
ordinary `iter_shifted_examples` function returning a private on-demand
iterator. It covers every within-document adjacent transition exactly once at
context/stride 64, preserves variable tails, and creates no padding or output.

`src/sebgpt/data/shakespeare_examples.py` separately enforces accepted
vocabulary binding, manifest identity, exact ordered train/validation work
membership, complete metadata-before-text validation, strict recomputed
provenance, per-document encoding, and test refusal before content access. It
contains no corpus loader and cannot implement an all-eight-then-filter path.

All original 27 focused tests remain. One incremental-construction test and 16
governance-matrix tests bring the focused suite to 44: 15 neutral and 29
governance tests. Production source is unchanged from Gate 13 review. The
implementation is accepted, unstaged, uncommitted, and unpushed pending
separate Gate 15 commit authorization.

## Current model status

The accepted Phase 3 representation remains the only model code. Each
construction has exactly 10,784 learnable CPU float32 parameters. Phase 4 now
has tuple-based document-local example construction and governance code, but no
language-model output head, logits, probability helper, loss, backward demo,
manual update utility, training/evaluation loop, checkpoint, or generation path.

The accepted Phase 4 contract would add 2,592 output-weight and 81 bias
parameters for 13,457 total, but no such parameters have been instantiated or
implemented.

## Last verified working state

The corrected focused suite passes 44/44: 15 corpus-neutral shifted-example
tests and 29 Shakespeare-governance tests. The complete suite ran 201 tests:
200 passed, one restricted-context MPS test was skipped as expected, and zero
failed. The accepted vocabulary artifact remains 9,182 bytes with the pinned
SHA-256 above. Production-source identity, diff, whitespace, index, and
unstaged-state checks pass.

## Next exact step

Obtain explicit authorization for the dedicated Phase 4 Gate 15 accepted
example/governance implementation commit.

## Known issues

- PyTorch warns that optional NumPy interoperability is unavailable. NumPy is
  intentionally absent because no accepted milestone requires it.
- MPS may be unavailable in restricted execution contexts; CPU float32 remains
  the accepted semantic baseline.
- Processed corpus files are ignored reproducible derivatives and must be
  regenerated only through the accepted publisher.
- The catalog and artifact report different update/Last-Modified dates; both
  remain recorded without assuming they describe the same revision mechanism.

## Important constraints

- Do not expand the Gate 12 example/governance implementation into model,
  loss, update, training, or experiment work before their separate gates.
- Do not instantiate the accepted output head, construct persistent
  token/window/target artifacts, or run the experiment.
- Never access, tokenize, window, score, inspect, or derive Phase 4 statistics
  from the sealed test work.
- Preserve independent documents and the accepted complete-work split. No
  example may cross a document or split boundary.
- Preserve the accepted tokenizer, exact vocabulary identity, Phase 3
  representation, CPU float32 baseline, and `torch.long` IDs.
- Do not introduce attention, recurrence, convolution, Transformer components,
  padding/special tokens, runtime batching/sampling, `torch.optim`, checkpoints,
  generation, MPS/mixed precision, or pretrained/hosted models in Phase 4.
- Do not add dependencies without an accepted recorded decision and lock-file
  update.
- Do not commit or push the current contract checkpoint without separate
  explicit authorization.

## Open questions

- No known detailed-contract ambiguity remains after Gate 8 PASS and explicit
  acceptance.
- No known shifted-example/governance implementation or focused-test issue
  remains after Gate 14 PASS and explicit acceptance.
- The accepted output-head seed 4004, learning rate 0.05, two passes, gradient
  scaling, and primary success predicate have not been empirically tried and
  must not be tuned before separate experiment authorization.
- Validation direction is deliberately observational rather than part of the
  accepted Phase 4 PASS condition.

## Session handoff

Phases 0–3 are complete. The accepted Phase 4 contract is remotely verified at
`3fcf007ce819ca1a45aa75b48fa19f10311f2819`. Gate 13 passed production and
found two test-only blockers. Gate 14 retained production bytes, expanded
focused coverage to 44 tests, passed focused re-review and adjudication, and
was explicitly accepted. Obtain separate Gate 15 commit authorization only; do
not stage, commit, push, expand implementation, run the experiment, or access
the sealed test.
