# Project State

> Primary session anchor. Keep this document concise and current; it describes now, not project history.

## Current phase

Phase 7 — Complete Mini-GPT (detailed contract accepted; implementation unauthorized)

## Current milestone

Phase 6 is formally remotely closed at
`e84b364a7955ddba86b30938babd5dae9e829935`, `Complete Phase 6 transformer
block`. Its accepted implementation checkpoint is
`c2624078eae5947e505d7f8093869e32c58e521b`.

Sebastien authorized Phase 7 learning/design only. The read-only continuity
inspection is complete and accepted. Sebastien then accepted the consolidated
Phase 7 conceptual architecture recorded in DEC-0019 and authorized
documentation-only detailed-contract drafting and stale-closure reconciliation.
Fresh independent review returned CORRECT BEFORE ACCEPTANCE with five IMPORTANT
findings and no BLOCKER. Master Chat accepted all five without changing
DEC-0019 and authorized a targeted documentation-only correction pass. All five
corrections completed locally. Focused independent re-review returned PASS with
all five findings resolved and no BLOCKER, IMPORTANT, or MINOR findings. Codex
recommended `ACCEPT CORRECTED CONTRACT`, and Master Chat formally accepted the
corrected detailed contract.

## Completed work

- Phases 0–4 are complete and remotely closed.
- Phase 5 is formally remotely closed at `6519d8c`; its accepted implementation
  remains `f4b5a1d`.
- Phase 6 is formally remotely closed at `e84b364`; its accepted contract is
  `2177c7e`, and its accepted implementation is `c262407`.
- All accepted Phase 3 embedding, Phase 5 attention, and Phase 6 block
  interfaces remain unchanged.
- Phase 7 continuity inspection separated inherited authority, open conceptual
  choices, and later detailed mechanics.
- DEC-0019 accepts exactly four distinct non-shared blocks, width 32, four
  width-8 heads per block, sequence range `1..256`, the accepted Phase 3
  representation, final explicit normalization, an untied biased 81-class
  head, logits-only forward, separate next-token loss, same-call inspection,
  and 63,825 parameters.
- Proposed `docs/MINI_GPT_SPEC.md` resolves distinct-block initialization
  without changing accepted Phase 5/6 code or standalone behavior.
- Independent contract review confirmed the architecture, initialization
  mechanism, parameter count, and tensor count, while identifying five
  precision/evidence defects. The authorized corrections resolve validation
  order, the public/error oracle, equal-length causality evidence, stack-wide
  failure-state evidence, and irreversible-action gate separation.
- Focused independent re-review passed all five corrections with no remaining
  finding. The corrected `docs/MINI_GPT_SPEC.md` is formally accepted at SHA-256
  `3e1987658d4c9ece59bebbea7061f939c1138aaa73751a523f659f8b3217c022`.

## Current work

The Phase 7 detailed contract is accepted. Acceptance bookkeeping is complete
locally. Separate documentation-checkpoint commit authorization is the sole
next gate and remains unauthorized. No Phase 7 implementation is authorized.

## Current model status

No complete Mini-GPT model exists and no Phase 7 parameters have been
constructed. The accepted Phase 4 trained instance was not retained. Accepted
Phase 3, 5, and 6 components remain source authority for later composition.

The approved future Phase 7 architecture will contain 90 trainable tensors and
63,825 parameters: 10,784 in the accepted embedding representation, 50,304
across four accepted block structures, 64 in final normalization, and 2,673 in
the untied biased LM head. These are architectural counts only, not a retained
model artifact.

## Last verified working state

Before this documentation-only work, local `HEAD` and `origin/main` matched
Phase 6 closure commit `e84b364` at ahead/behind `0/0` with a clean worktree and
index. The accepted Phase 6 specification, implementation, focused tests, and
exports match their recorded SHA-256 identities. No code tests were rerun
because this stage authorizes no parameter construction and changes no source
or tests.

## Next exact step

Obtain separate Master Chat authorization for the accepted Phase 7
documentation-checkpoint commit.

## Known issues

- The checked-in Phase 6 closure commit contained stale current-status wording;
  this working documentation update supersedes it without rewriting historical
  gate facts.
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
- Never access, tokenize, window, score, inspect, or derive statistics from the
  sealed test work.
- Preserve the accepted tokenizer/vocabulary identity and accepted Phase 3,
  Phase 5, and Phase 6 source, tests, contracts, and mathematical behavior.
- DEC-0019 is the accepted Phase 7 conceptual architecture. Do not change it
  through detailed mechanics without Master Chat adjudication and Sebastien's
  explicit approval.
- Do not create or modify Phase 7 source or tests, construct model parameters,
  train, run experiments, stage, commit, or push.
- Do not begin Phase 8 training/checkpointing or Phase 9 generation/evaluation.

## Open questions

- No Phase 7 detailed-contract review finding remains unresolved.
- No conceptual Phase 7 architecture question remains open under DEC-0019.

## Session handoff

Phase 6 is formally remotely closed at `e84b364`. Phase 7 learning/design was
authorized, its continuity inspection was accepted, and DEC-0019 records the
accepted consolidated conceptual architecture. Documentation-only proposed
`docs/MINI_GPT_SPEC.md` defines exact interfaces, validation, conservative
in-place block-specific initialization with Phase 7 seeds 7001–7004, head seed
7005, final normalization, logits-only forward, separate inherited explicit
cross-entropy, same-call inspection, causality, gradients, 90 tensors, 63,825
parameters, tests, exclusions, and gates. Independent review returned CORRECT
BEFORE ACCEPTANCE with five IMPORTANT findings and no BLOCKER. The targeted
documentation corrections are complete without changing DEC-0019 or the
initialization design. Focused re-review returned PASS with all five findings
resolved and no remaining finding, and Master Chat formally accepted the
corrected detailed contract. No source, tests, parameters, training, experiment,
sealed-test access, staging, commit, push, Phase 8, or Phase 9 work occurred.
The sole next action is separate Master Chat authorization for the accepted
documentation-checkpoint commit.
