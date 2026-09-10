# Project State

> Primary session anchor. Keep this document concise and current; it describes now, not project history.

## Current phase

Phase 4 — Simple Neural Language Model (safe corpus-factory prerequisite)

## Current milestone

Phase 4 safe permitted-corpus factory prerequisite — Final focused independent
re-review returned PASS, master-chat adjudication returned PASS, and Sebastien
explicitly accepted the safe seven-work factory and its 14-test suite. The
factory remains unstaged, uncommitted, and unpushed. Its dedicated commit
requires separate authorization. No Gate 24 run or experiment result exists.

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
- The accepted shifted-example/governance slice was committed, pushed, and
  independently verified at `266797f891e9980b57bb35a633c91bef838d4111`.
- The accepted model/loss/update slice was committed, pushed, and independently
  verified at `497ecde3577677903f14669722d61dcdf8caa1d6`.
- The accepted fixed-experiment runner was committed, pushed, and independently
  verified at `a4da629a59584a0185a2ec286b0e770b7940940e`.

## Current work

`src/sebgpt/model/simple_language_model.py` implements the accepted positionwise
`SimpleNeuralLanguageModel`, exact four-parameter structure and initialization,
rank-one logits-only forward, separate stable probability inspection,
cancellation-safer representability-aware cross-entropy, atomic manual SGD,
category-major parameter validation for gradient clearing, tail-loss scaling,
canonical parameter-byte hashing, and one-example no-update measurement.

`src/sebgpt/model/__init__.py` exports this public slice and the three accepted
Phase 4 exception classes. `tests/test_simple_language_model.py` contains 48
focused independent checks across structure, initialization, forward isolation,
loss numerics, gradient routing, update atomicity, tail scaling, and measurement
invariants, including the corrected `clear_gradients()` simultaneous-defect
priority. The implementation is accepted, committed, pushed, and remotely
verified at `497ecde3577677903f14669722d61dcdf8caa1d6`.

`src/sebgpt/model/phase4_experiment.py` implements the missing accepted runner
prerequisite: a factory-created frozen configuration, fifteen-stage preflight,
target-token-weighted aggregate measurement, deterministic one-pass update
operation, fixed one-model/two-pass lifecycle, and immutable result records.
It separately preserves the accepted `497ecde3577677903f14669722d61dcdf8caa1d6`
model authority from the future dynamic runner commit, validates training
governance before tensorization, and produces a complete evidence-limited
durable result. `tests/test_phase4_experiment.py` contains 41 synthetic checks
and now derives externally observed measurement roles from the actual split
sentinel received rather than call position. It never invokes the runner on the
real Phase 4 corpus. This slice is accepted, committed, pushed, and remotely
verified at `a4da629a59584a0185a2ec286b0e770b7940940e`.

`src/sebgpt/model/phase4_corpus.py` implements the safe public seven-work
factory. It validates both accepted manifests and the exact eight-work metadata
shape before any text access, rejects symlinked roots/components and physical
escape before reads, discards the sealed metadata record before work
construction, reads only six train files plus *The Tempest*, preserves every
Phase 1 `ExtractedWork` field, and returns `Phase4PermittedCorpus` with accepted
counts. `tests/test_phase4_corpus.py` contains 14 focused checks. This factory
slice is accepted, unstaged, uncommitted, and unpushed.

## Current model status

The Phase 4 positionwise model now exists locally. Each construction owns the
accepted 10,784 representation parameters plus a 2,592-element output weight
and 81-element bias for exactly 13,457 trainable CPU float32 parameters. Tests
instantiate only synthetic models. No corpus training/evaluation loop,
aggregate experiment measurement, checkpoint, generation, or attention path
exists in the model-only checkpoint. The accepted runner can compose those
operations after the accepted safe factory is committed, pushed, remotely
verified, and granted a fresh experiment authorization, but it has not been
invoked on the real corpus and persists no checkpoint.

## Last verified working state

The corrected safe-corpus factory suite passes 14/14. Orchestration remains
41/41, model/loss/update 48/48, and shifted-example/governance 44/44. The
complete 304-test suite passes with 303 passes, one expected restricted-context
MPS skip, and zero failures. Phase 1 source/artifact identity, diff,
phase-boundary, and unstaged-state checks pass.

## Next exact step

Obtain explicit authorization for the dedicated accepted safe-corpus factory
commit.

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

- Do not invoke the fixed experiment entry point before the accepted safe corpus
  factory is committed, pushed, remotely verified, and followed by a fresh
  experiment authorization.
- Do not persist model parameters, gradients, logits, probabilities, losses,
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
- Do not stage, commit, or push the safe corpus factory without separate
  explicit authorization after acceptance.

## Open questions

- No known detailed-contract ambiguity remains after Gate 8 PASS and explicit
  acceptance.
- No known shifted-example/governance implementation or focused-test issue
  remains after Gate 14 PASS and explicit acceptance.
- No model/loss/update implementation-review correction remains after Gate 20
  focused re-review, adjudication, and explicit acceptance.
- No runner implementation or test correction remains after final focused
  re-review, master-chat adjudication, and explicit acceptance.
- No safe corpus-factory implementation or test correction remains after final
  focused re-review, master-chat adjudication, and explicit acceptance.
- The accepted output-head seed 4004, learning rate 0.05, two passes, gradient
  scaling, and primary success predicate have not been empirically tried and
  must not be tuned before separate experiment authorization.
- Validation direction is deliberately observational rather than part of the
  accepted Phase 4 PASS condition.

## Session handoff

Phases 0–3 are complete. The accepted Phase 4 contract is remotely verified at
`3fcf007ce819ca1a45aa75b48fa19f10311f2819`; accepted examples/governance are
remotely verified at `266797f891e9980b57bb35a633c91bef838d4111`; and accepted
model/loss/update mechanics are remotely verified at `497ecde3577677903f14669722d61dcdf8caa1d6`.
The attempted Gate 23 run authorization stopped before experiment work because
the runner was absent; the accepted runner is now remotely verified at
`a4da629a59584a0185a2ec286b0e770b7940940e`. A fresh authorization then stopped
before Gate 24 because a safe seven-work handoff factory was missing. That
factory's independent review found five focused issues, all now corrected with
14/14 tests passing; final focused re-review and adjudication passed, and the
factory is explicitly accepted. Obtain dedicated factory commit authorization
only; do not stage, commit, push, run the experiment, or begin Phase 5.
