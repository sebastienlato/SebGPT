# Project State

> Primary session anchor. Keep this document concise and current; it describes now, not project history.

## Current phase

Phase 5 — Self-Attention (corrected detailed contract accepted)

## Current milestone

Phase 4 closure commit `e8b5c55fb2f2f03b155c1a8b4308dd9df20d9e1c`
is pushed and independently remote-verified. Gate 30 authorized Phase 5
learning/design. Sebastien completed the learning discussion, accepted the
DEC-0017 conceptual architecture, and authorized a documentation-only detailed
contract. Gate 5 drafting completed, Gate 6 Master Chat sanity review passed,
and Gate 7 fresh independent review failed with five MUST-FIX and three
SHOULD-FIX findings. Gate 8 corrected all eight findings, Gate 9 focused
independent re-review returned PASS, Master Chat adjudicated PASS, and Sebastien
explicitly accepted the corrected detailed contract. The dedicated Gate 10
accepted-contract checkpoint commit is authorized.

## Completed work

- Phases 0–3 are complete and remotely verified.
- Phase 4 is accepted and remotely closed at `e8b5c55`; do not reopen it.
- `EXP-20260909-01` is the sole completed Phase 4 experiment and passed both
  frozen training-loss predicates. Validation improvement remains observation
  only and is not evidence of generalization or test performance.
- Gate 30 explicitly authorized Phase 5 learning/design only.
- Sebastien accepted standalone width-32, single-sequence causal attention with
  one bias-free full-width head and four separately visible bias-free width-8
  heads followed by a bias-free width-32 output projection.
- Causal visibility is exactly `j <= i`; normal forward use returns contextual
  representations, and separate inspection exposes the exact post-softmax
  weights used for the output.
- DEC-0017 records the accepted architecture and
  `docs/SELF_ATTENTION_SPEC.md` proposes detailed mechanics and gates.
- Gate 6 Master Chat sanity review returned PASS; Gate 7 independent contract
  review returned FAIL, with all eight findings accepted for correction.
- Gate 8 resolved all five MUST-FIX and all three SHOULD-FIX findings; Gate 9
  focused re-review and Master Chat adjudication returned PASS.
- Sebastien explicitly accepted the corrected contract without changing
  DEC-0017. Its detailed mechanics are authoritative for later separately
  authorized implementation.

## Phase 5 exit criteria

These exact roadmap criteria are not yet satisfied:

1. Queries, keys, values, scaling, masking, and attention weights are explainable.
2. Single-head causal self-attention is implemented transparently and tested.
3. Multi-head attention is built from understood components and tested.
4. Attention shapes and selected weights can be inspected.

## Current work

The accepted six-file documentation checkpoint is authorized for a dedicated
commit. No source or tests have been created. Push and Phase 5 implementation
remain unauthorized until this contract checkpoint is pushed, independently
remote-verified, and implementation is separately authorized.

## Current model status

The accepted Phase 4 model architecture remains unchanged; its trained Gate 24
instance was not retained. Phase 5 attention parameters are approved
conceptually but do not yet exist. The proposal would add exactly three tensors
and 3,072 parameters for standalone single-head attention, or thirteen tensors
and 4,096 parameters for standalone four-head attention. Construction remains
unauthorized until the contract is accepted, committed, pushed, remotely
verified, and separately authorized for implementation.

## Last verified working state

Before this documentation task, local `HEAD`, cached `origin/main`, and
independently queried actual remote `main` all matched `e8b5c55` with
ahead/behind `0/0` and a clean worktree/index. The prior accepted regression
record remains 304 tests total: 303 passes, one expected restricted-context MPS
skip, and zero failures. This task changes documentation only; no production
source, tests, parameters, training, experiment, or sealed-test state changed.

## Next exact step

Obtain separate authorization to push and independently remote-verify the accepted-contract checkpoint.

## Known issues

- PyTorch warns that optional NumPy interoperability is unavailable. NumPy is
  intentionally absent because no accepted milestone requires it.
- MPS may be unavailable in restricted execution contexts; CPU float32 remains
  the accepted semantic baseline.
- Processed corpus files are ignored reproducible derivatives and must be
  regenerated only through the accepted publisher.
- The catalog and artifact report different update/Last-Modified dates; both
  remain recorded without assuming they describe the same revision mechanism.
- No accepted Phase 5 contract issue remains open after Gate 9 PASS.

## Important constraints

- Do not rerun or tune `EXP-20260909-01`.
- Never access, tokenize, window, score, inspect, or derive Phase 4 statistics
  from the sealed test work.
- Do not persist the completed run's parameters, gradients, logits,
  probabilities, losses, token/window/target data, or a checkpoint.
- Preserve the accepted tokenizer, vocabulary identity, Phase 3 representation,
  model architecture, frozen experiment settings, and evidence-limited claim.
- Phase 5 work is limited to the current documentation-only proposal.
- Do not create Phase 5 source, tests, parameters, training, or experiments.
- Do not add batching, caller masks, padding, residual paths, normalization,
  feed-forward layers, dropout, Transformer blocks, stacking, checkpoints,
  generation, final evaluation, or Phase 6 work.
- Do not commit or push without separate authorization after contract
  acceptance.

## Open questions

- No accepted conceptual or detailed-contract question remains open.
- Push/remote verification and implementation authorization remain separate
  gates.

## Session handoff

Phase 4 is remotely closed at `e8b5c55`. Gate 30 Phase 5 learning/design,
conceptual approval, and documentation-only contract authorization are
complete. DEC-0017, the proposed self-attention specification, and minimum
continuity corrections completed Gate 8. Gate 9 focused re-review and Master
Chat adjudication returned PASS, and Sebastien explicitly accepted the
corrected detailed contract without changing DEC-0017. The dedicated Gate 10
checkpoint commit is authorized. No implementation, tests, Phase 5 parameter
construction, training, experiment, sealed-test access, push, or Phase 6 work
is authorized. Proceed only to the separately gated push/remote-verification
workflow after the checkpoint commit.
