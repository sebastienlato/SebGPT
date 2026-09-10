# Project State

> Primary session anchor. Keep this document concise and current; it describes now, not project history.

## Current phase

Phase 4 — Simple Neural Language Model (technically COMPLETE; Gate 27 accepted)

## Current milestone

Gate 25 independently returned PASS with no MUST-FIX issues, master-chat
adjudication returned PASS, and Sebastien accepted the Gate 24 result, all four
Phase 4 exit criteria, and Phase 4 as technically complete. Gate 26 is not
applicable. Gate 27 closure bookkeeping is complete locally. Gate 28 closure
commit authorization is the next required action.

## Completed work

- Phases 0–3 are complete and remotely verified.
- The Phase 4 contract, shifted-example/governance slice, model/loss/update
  implementation, fixed runner, and safe seven-work corpus factory are accepted,
  committed, pushed, and remotely verified through
  `ba09d017e97a6d25317e160fc1a40a6304bcdd96`.
- The first authorized launcher stopped before imports and before Gate 24 because
  it omitted the documented `PYTHONPATH=src` convention. Master chat classified
  it as a pre-experiment launcher abort and authorized one corrected launch.
- The corrected one-shot launch used only accepted production APIs, passed the
  fifteen-stage preflight, constructed one model, completed two passes and
  exactly 24,778 manual-SGD updates, and returned one valid result.
- Training loss fell from `4.426503102003006` to `2.5551429421586387`, below
  exact `math.log(81) = 4.394449154672439`; both frozen predicate clauses are
  true. Validation loss improved observationally from `4.427434147633409` to
  `2.565088013405899`.
- Gate 25 independently verified the experiment evidence, source/test
  immutability, reproduced regressions, and all four Phase 4 exit criteria with
  no MUST-FIX issues. Gate 26 is not applicable because the experiment passed.
- Gate 27 master-chat adjudication and explicit Sebastien acceptance make Phase
  4 technically complete. No technical Phase 4 work remains.

## Phase 4 exit criteria

Gate 25 independently verified and Gate 27 accepted all four criteria as
satisfied:

1. Inputs, targets, logits, probabilities, and cross-entropy loss are understood.
2. A simple neural language model is implemented and tested.
3. Backpropagation and parameter updates are inspected on a small example.
4. A reproducible run shows loss improving over a baseline.

## Current work

The complete Gate 24 result remains recorded append-only in `EXPERIMENT_LOG.md`
as the sole completed Phase 4 experiment, `EXP-20260909-01`. The Gate 27 closure
documentation is unstaged and uncommitted pending separate Gate 28 authority.
*Twelfth Night* remains sealed. No checkpoint or model/data artifact persisted.

The result demonstrates only that this fixed positionwise baseline met its
pre-registered training-loss predicate. It does not establish generalization or
test performance. Phase 5 remains unauthorized.

## Current model status

The accepted model has exactly four trainable CPU float32 tensors and 13,457
parameters. The one Gate 24 model instance remained identity-stable throughout
initial measurement, two training passes, and final measurement. Initial and
final canonical parameter digests were
`969c9f4606ff823a027324cb5ba6bdb76e72da8f4f903c42958703e0a233db20` and
`c8ce1ae08c4c3466d74b30068fb6b77eaf2142c28a5b914d64bf0277d63bac72`.
All measurements preserved their boundary digest and left gradients `None`.
The model was not retained.

## Last verified working state

Gate 24 completed normally with PASS and no tuning or retry. After Gate 27
bookkeeping, safe corpus passes 14/14, orchestration 41/41, model/loss/update
48/48, and shifted-example/governance 44/44. The complete 304-test suite passes
with 303 passes, one expected restricted-context MPS skip, and zero failures.
Production source and tests remain byte-identical to the execution checkpoint.
Gate 25 independently reproduced the required evidence and returned PASS with
no MUST-FIX issues. Gate 27 is accepted.

## Next exact step

Obtain explicit Gate 28 authorization for the Phase 4 closure documentation
commit only.

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

- Do not rerun or tune `EXP-20260909-01`.
- Never access, tokenize, window, score, inspect, or derive Phase 4 statistics
  from the sealed test work.
- Do not persist the completed run's parameters, gradients, logits,
  probabilities, losses, token/window/target data, or a checkpoint.
- Preserve the accepted tokenizer, vocabulary identity, Phase 3 representation,
  model architecture, frozen experiment settings, and evidence-limited claim.
- Do not begin attention, generation, Phase 5, or any later-phase work.
- Leave the accepted Gate 24 result and Gate 27 closure documentation unstaged
  and uncommitted until explicit Gate 28 authorization.
- Gate 29 push requires separate authorization after a Gate 28 commit.
- Phase 5 may begin only after an explicit Gate 30 authorization.

## Open questions

- No technical Phase 4 questions or work remain after Gate 27 acceptance.
- Gate 28 commit, Gate 29 push, and Gate 30 Phase 5 authorization remain
  separate future gates.
- Validation improved, but remains observation only and is not evidence of
  generalization or test performance.

## Session handoff

The accepted Phase 4 implementation is remotely verified through
`ba09d017e97a6d25317e160fc1a40a6304bcdd96`. The corrected one-shot Gate 24 run
produced valid PASS result `EXP-20260909-01`; Gate 25 independently returned
PASS with no MUST-FIX issues; Gate 26 is N/A; and Gate 27 accepted all four exit
criteria and Phase 4 as technically complete. No technical Phase 4 work remains.
No sealed-test access, rerun, tuning, checkpoint, retained model, commit, or push
occurred. Proceed only with explicit Gate 28 closure commit authorization.
