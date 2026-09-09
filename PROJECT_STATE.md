# Project State

> Primary session anchor. Keep this document concise and current; it describes now, not project history.

## Current phase

Phase 4 — Simple Neural Language Model (contract design)

## Current milestone

Phase 4 Gate 8 — Focused independent re-review returned PASS with all seven
Gate 6 findings resolved, master-chat adjudication returned PASS, and Sebastien
explicitly accepted the corrected detailed contract. Gate 9, the dedicated
accepted-contract commit, requires separate authorization. No Phase 4
implementation or experiment is authorized.

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

## Current work

The accepted corrected Phase 4 contract is documented at
`docs/SIMPLE_LANGUAGE_MODEL_SPEC.md`. Gate 7 added an independent probability
contract, cancellation-safe representability-aware cross-entropy, exact
exception ownership and first-failure order, one uninterrupted model lifecycle,
controlled failed-run analysis/re-registration, complete inherited experiment
identities, and enforceable tests for all corrections.

The passing Gate 6 mechanics remain unchanged: corpus-neutral transition
coverage, Shakespeare governance, rank-one model input, 13,457 trainable
parameters, head seed 4004, deterministic manual SGD, tail scaling `L / 64`,
token-weighted reporting, learning rate 0.05, two passes, 12,389 examples per
pass, 24,778 updates, the primary success predicate, and the technical phase
boundary.

Gate 8 accepted all corrected detailed mechanics. The repository contains no
Phase 4 source, tests, parameters, examples, or experiment record.

## Current model status

Only the accepted Phase 3 representation exists. Each construction has exactly
10,784 learnable CPU float32 parameters. No language-model output head, logits,
probability helper, language-model loss, manual update utility, production
context-window implementation, training loop, checkpoint, or generation path
exists.

The accepted Phase 4 contract would add 2,592 output-weight and 81 bias
parameters for 13,457 total, but no such parameters have been instantiated or
implemented.

## Last verified working state

After the current documentation-only edits, the complete existing suite ran
157 tests in the project environment: 156 passed, one restricted-context MPS
test was skipped as expected, and zero failed. This includes 7/7 accepted
vocabulary-runtime tests, 28/28 embedding tests, and all 35 automated Phase 3
checks. The accepted vocabulary artifact remains 9,182 bytes with the pinned
SHA-256 above. Documentation structure, arithmetic, tracked-file boundaries,
unstaged status, and whitespace checks pass.

## Next exact step

Obtain explicit authorization for the dedicated Phase 4 Gate 9
accepted-contract checkpoint commit without implementing or running Phase 4.

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

- The accepted detailed contract does not itself authorize implementation or
  an experiment.
- Do not create Phase 4 source or tests, instantiate the accepted output head,
  construct persistent token/window/target artifacts, or run the experiment
  before their separate gates.
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
- The accepted output-head seed 4004, learning rate 0.05, two passes, gradient
  scaling, and primary success predicate have not been empirically tried and
  must not be tuned before separate experiment authorization.
- Validation direction is deliberately observational rather than part of the
  accepted Phase 4 PASS condition.

## Session handoff

Phases 0–3 are complete and remotely verified through closure commit
`a66d169e76f06bee18d4f32b6340206f9f45ee63`. Phase 4 learning/design and its
twelve conceptual decisions are accepted in DEC-0016. Gate 6 returned FAIL with
seven accepted findings; Gate 7 corrected them in documentation while
preserving all accepted and passing architecture. Gate 8 focused re-review and
master-chat adjudication returned PASS, and Sebastien accepted the corrected
contract. The next step is separate Gate 9 commit authorization; do not
implement, experiment, stage, commit, push, or access the sealed test.
