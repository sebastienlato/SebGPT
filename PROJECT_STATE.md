# Project State

> Primary session anchor. Keep this document concise and current; it describes now, not project history.

## Current phase

Phase 8 — Training and Checkpointing (detailed contract accepted; uncommitted)

## Current milestone

Phase 7 — Complete Mini-GPT is formally remotely closed at
`33d4510421107848c4aa8a6014f4a7b1e391065a`, `Complete Phase 7 Mini-GPT`.
Its accepted detailed-contract checkpoint is
`60b2a9cce55da79ccc9fbd03fad014cb2a939290`, and its accepted implementation
checkpoint is `3139b1736f005fe903e2ea111d91934478b5a683`.

Sebastien explicitly authorized Phase 8 learning/design. The read-only
continuity inspection is complete, and Sebastien accepted the consolidated
Phase 8 conceptual training/checkpointing policy as DEC-0020. Documentation-only
detailed-contract drafting and stale Phase 7 closure reconciliation were then
authorized.

`docs/TRAINING_CHECKPOINTING_SPEC.md` makes DEC-0020 mechanical. Its accepted
SHA-256 is
`0b3e2a79de7038e2233560c0836101d2f5757f9a5e3c3e89e6ccb62e7cc6fffd`.
Fresh independent review returned CORRECT BEFORE
ACCEPTANCE with three BLOCKER, eight IMPORTANT, and one MINOR finding. Master
Chat accepted all twelve without changing DEC-0020 and authorized only their
targeted documentation correction. Focused re-review passed those corrections
but returned CORRECT AGAIN BEFORE ACCEPTANCE with one remaining BLOCKER, two
IMPORTANT schema/API findings, and no MINOR finding. Master Chat accepted all
three without changing DEC-0020 and authorized this final targeted correction.
Final focused independent re-review returned PASS with all three resolved, no
new or remaining finding, and exact recommendation `ACCEPT CORRECTED CONTRACT`.
Master Chat formally accepted the corrected detailed contract without changing
DEC-0020. No implementation, feasibility measurement, or training is authorized.

## Completed work

- Phases 0–7 are formally remotely closed.
- Phase 7 closure commit `33d4510`, accepted contract `60b2a9c`, and accepted
  implementation `3139b17` are recorded and synchronized.
- The Phase 8 continuity inspection established inherited authority, open
  choices, checkpoint gaps, and the Phase 8/9 boundary without accessing the
  sealed test.
- DEC-0020 accepts unchanged dataset/tokenizer/MiniGPT authority; CPU float32;
  context 256; logical rank-one batches of eight; deterministic epoch shuffle;
  constant AdamW with learning rate `3e-4`, betas `(0.9, 0.999)`, epsilon
  `1e-8`, and weight decay `0.01`; global-norm clipping at `1.0`; ten epochs
  without validation early stopping; complete initialized/end-epoch evaluation;
  latest and best-validation checkpoints; complete stochastic/progress state;
  exact resume; durable experiment evidence; and no Phase 9 or sealed-test use.
- Accepted `docs/TRAINING_CHECKPOINTING_SPEC.md` defines stride-256 exact-once
  document-local windows with retained unpadded tails, order seed 8001,
  target-token-weighted gradient accumulation, final partial logical batches,
  exact AdamW/update/clipping/evaluation order, ten runtime RNG states, complete
  versioned checkpoint payloads, content-addressed immutable objects, one
  atomic catalog commit point, fail-closed restoration, exact-resume equality,
  experiment records, fitting evidence, numerical/failure behavior, focused
  tests, feasibility boundary, and separate Phase 8 gates.
- Stale Phase 7 remote-closure and Phase 8 authorization wording is reconciled
  in current continuity documentation.
- The authorized detailed-contract correction resolves checkpoint-directory
  durability, caller-global-RNG load transactionality, the frozen runtime
  envelope, exact PyTorch clipping, complete API/record/error/checkpoint/log
  schemas, catalog/payload agreement, the local filesystem trust boundary,
  live repository provenance, latest-only continuation, a complete-epoch real
  resume audit, irreversible gate separation, and deterministic permutation
  evidence. DEC-0020 is unchanged.
- The final targeted correction adds intrinsic supported AdamW
  `decoupled_weight_decay=True` state, exact 90-parameter step-tensor/progress
  equality, literal owning modules and module/package exports for every public
  Phase 8 object, and an unambiguous first-key nested configuration schema
  version. All prior corrections remain unchanged.
- Final focused independent re-review passed. All Phase 8 contract-review
  findings are resolved; no BLOCKER, IMPORTANT, or MINOR finding remains. The
  corrected specification is formally accepted at its exact SHA-256.

## Current work

Detailed-contract acceptance bookkeeping is complete locally. The accepted
specification and continuity updates are unstaged, uncommitted, and unpushed.
They await separate accepted-contract documentation-checkpoint commit
authorization.

## Current model status

The accepted Phase 7 MiniGPT remains unchanged: one rank-one CPU-long sequence
of length `1..256` produces CPU-float32 logits `(T, 81)`. It contains 90 unique
trainable tensors and 63,825 parameters, four accepted blocks, eight local
dropout generators, final normalization, and an untied biased head.

No retained Phase 7 model instance, optimizer, checkpoint, Phase 8 window,
training tensor, gradient, logit, loss, metric, or generated sample exists. The
accepted Phase 4 trained instance was not retained.

## Last verified working state

At the start of this documentation-only work, local `HEAD`, `main`, and
`origin/main` matched Phase 7 closure commit `33d4510` at ahead/behind `0/0`
with a clean worktree/index. The inherited last accepted regression evidence is
the Phase 7 result: focused tests 47/47 and the complete 443-test suite with 442
passes, one expected restricted-context MPS skip, and zero failures.

This acceptance-bookkeeping stage did not rerun code tests. The accepted
specification remains byte-identical at its recorded SHA-256, has 74 sequential
gates and 48 sequential focused obligations, and preserves every fixed DEC-0020
value. Documentation consistency checks and `git diff --check` pass.

## Next exact step

Obtain separate Master Chat authorization for the accepted Phase 8
documentation-checkpoint commit.

## Known issues

- No runtime feasibility evidence exists for context 256, logical batch
  capacity 8, CPU float32, four blocks, and ten epochs.
- Optional NumPy interoperability is unavailable and intentionally unnecessary.
- MPS may be unavailable in restricted contexts; Phase 8 is CPU-only regardless.
- Processed corpus files are ignored reproducible derivatives.
- Checkpoint binaries are excluded from Git by accepted policy.
- Content-addressed checkpoint objects may be orphaned by a failure before the
  atomic catalog replacement; the accepted contract preserves them for diagnosis rather
  than deleting uncertain state.

## Important constraints

- DEC-0020 and the detailed specification at accepted SHA-256
  `0b3e2a79de7038e2233560c0836101d2f5757f9a5e3c3e89e6ccb62e7cc6fffd` are
  accepted authority and must not be changed without explicit adjudication.
- Do not change context 256, logical batch capacity 8, CPU float32, model
  architecture, optimizer, or training policy without a new explicit decision.
- Preserve the accepted tokenizer/vocabulary identity and all accepted Phase
  3, 5, 6, and 7 source, tests, contracts, and behavior.
- Never access, tokenize, window, score, inspect, or derive statistics from the
  sealed test work.
- Do not rerun or tune `EXP-20260909-01`.
- Do not implement Phase 8 source or tests; construct an optimizer/checkpoint;
  measure feasibility; train; generate; sample; stage; commit; or push without
  the applicable separate authorization.
- Do not begin Phase 9 generation or final evaluation.

## Open questions

- No Phase 8 detailed-contract review finding remains unresolved.
- No conceptual DEC-0020 question remains open.
- Runtime feasibility remains deliberately unknown until a later specifically
  authorized bounded measurement after accepted implementation.

## Session handoff

Phase 7 is formally remotely closed at `33d4510`; accepted Phase 7 contract and
implementation checkpoints remain `60b2a9c` and `3139b17`. Phase 8
learning/design is authorized, its continuity inspection is complete, and
DEC-0020 records the accepted consolidated policy. Accepted
`docs/TRAINING_CHECKPOINTING_SPEC.md` defines exact windows, deterministic
ordering, rank-one logical batches, token-weighted gradients, AdamW, clipping,
evaluation, progress, the frozen runtime, all RNG state, crash-durable atomic
checkpointing, transactional load, exact schemas, latest-only continuation,
complete-epoch exact resume, experiment evidence, failures, tests, feasibility
limits, and 74 gates. Independent review returned three BLOCKER, eight
IMPORTANT, and one MINOR finding; all targeted documentation corrections are
complete without changing DEC-0020. Focused re-review then found one remaining
AdamW-state BLOCKER and two ownership/configuration-schema IMPORTANT findings;
their final targeted corrections passed final focused re-review with no new or
remaining finding. Master Chat accepted the corrected contract at SHA-256
`0b3e2a79de7038e2233560c0836101d2f5757f9a5e3c3e89e6ccb62e7cc6fffd` without
changing DEC-0020. No source, tests, optimizer, checkpoint payload, benchmark,
training, generation, sealed test, staging, commit, or push occurred. The sole
next action is separate accepted-contract documentation-checkpoint commit
authorization.
