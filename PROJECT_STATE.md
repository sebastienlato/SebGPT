# Project State

> Primary session anchor. Keep this document concise and current; it describes now, not project history.

## Current phase

Phase 7 — Complete Mini-GPT (technically complete; closure bookkeeping local)

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

The accepted contract checkpoint was committed as
`60b2a9cce55da79ccc9fbd03fad014cb2a939290`, `Define Phase 7 Mini-GPT
contract`, pushed, and independently remote-verified at ahead/behind `0/0`.
Master Chat then explicitly authorized Phase 7 implementation. Gate 20
implementation and focused synthetic testing are complete locally and await
Master Chat sanity review and fresh independent review. Independent review
subsequently returned CORRECT BEFORE ACCEPTANCE with three IMPORTANT findings
and no BLOCKER. Master Chat accepted all three without changing DEC-0019 or the
accepted contract and authorized a targeted implementation/test correction
pass. All three corrections completed locally. Focused independent re-review
returned PASS with all three findings resolved and no BLOCKER, IMPORTANT, or
MINOR findings. Codex recommended `ACCEPT CORRECTED IMPLEMENTATION`, and Master
Chat formally accepted the corrected Phase 7 implementation checkpoint.

The accepted implementation checkpoint was committed as
`3139b1736f005fe903e2ea111d91934478b5a683`, `Implement Phase 7 Mini-GPT`,
pushed, and independently remote-verified at ahead/behind `0/0`. All four exact
Phase 7 exit criteria are satisfied. Phase 7 is technically complete and ready
for formal closure; closure bookkeeping is complete locally.

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
- Accepted contract checkpoint `60b2a9c` is pushed and independently
  remote-verified.
- The Phase 7 implementation composes four distinct accepted block structures,
  in-place block-specific parameter initialization, final explicit
  normalization, an untied biased head, logits-only forward, and same-call
  inspection without changing accepted Phase 3, 5, or 6 source.
- The targeted corrections separate child-type exceptions from structural
  contract failures, replace partial signature checks with an exact literal
  signature oracle, and add exact failure-path object-handoff evidence.
- Corrected focused Phase 7 tests pass 47/47. The complete 443-test suite has
  442 passes, one expected restricted-context MPS skip, and zero failures.
- Focused implementation re-review passed all three corrections with no
  remaining finding. Master Chat formally accepted the exact source, focused
  tests, exports, counts, and evidence.
- Accepted implementation checkpoint `3139b17` is pushed and independently
  remote-verified. All four Phase 7 exit criteria are satisfied, and Phase 7 is
  technically complete.

## Current work

This documentation-only closure bookkeeping records the accepted Phase 7 exit
result. It remains unstaged and uncommitted pending separate Phase 7
closure-commit authorization.

## Current model status

The Phase 7 model implementation exists locally. Tests construct only ephemeral
synthetic models; no parameter, activation, gradient, logit, loss, attention,
mask, or checkpoint artifact is retained. The accepted Phase 4 trained instance
was not retained. Accepted Phase 3, 5, and 6 components remain unchanged.

The accepted Phase 7 implementation contains 90 unique trainable tensors and
63,825 parameters: 10,784 in the accepted embedding representation, 50,304
across four accepted block structures, 64 in final normalization, and 2,673 in
the untied biased LM head. Tests construct this structure ephemerally; no
retained model artifact exists.

## Last verified working state

Local `HEAD`, `origin/main`, and actual remote `main` matched accepted Phase 7
implementation checkpoint `3139b17` with ahead/behind `0/0` and a clean
worktree/index before closure bookkeeping. Focused Phase 7 tests pass 47/47. Phase 3
vocabulary/embedding, Phase 5 attention, and Phase 6 block regressions pass
35/35, 41/41, and 51/51. The complete 443-test suite has 442 passes, one
expected restricted-context MPS skip, and zero failures. The accepted
specification and accepted Phase 3, 5, and 6 source remain byte-identical.

## Next exact step

Obtain separate Master Chat authorization for the Phase 7 closure commit.

## Known issues

- The checked-in Phase 6 closure commit contained stale current-status wording;
  the accepted Phase 7 contract checkpoint supersedes it without rewriting
  historical gate facts.
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
- Do not change DEC-0019, the accepted Phase 7 contract, source, focused tests,
  exports, or evidence suite.
- Do not stage, commit, or push this closure bookkeeping without separate
  authorization.
- Do not train, run experiments, create an optimizer/checkpoint, or retain
  model-derived artifacts.
- Do not begin Phase 8 training/checkpointing or Phase 9 generation/evaluation.

## Open questions

- No Phase 7 detailed-contract review finding remains unresolved.
- No conceptual Phase 7 architecture question remains open under DEC-0019.
- No Phase 7 implementation-review finding remains unresolved.
- No Phase 7 technical or exit-criteria question remains open.
- Formal closure commit authorization and later Phase 8 authorization remain
  separate future gates.

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
corrected detailed contract. Contract checkpoint `60b2a9c` is pushed and
remotely verified. Master Chat authorized implementation, and Gate 20 is
complete locally with exact source, exports, and focused tests. Independent
review returned CORRECT BEFORE ACCEPTANCE with three IMPORTANT findings and no
BLOCKER. The authorized corrections are complete: child types now have accepted
type-error ownership, public signatures have an exact literal oracle, and
failure paths prove exact object handoffs. Corrected focused tests pass 47/47;
the complete 443-test suite passes with one expected MPS skip. Focused re-review
returned PASS with all three findings resolved and no remaining finding, and
Master Chat formally accepted the corrected implementation. No training,
experiment, sealed-test access, staging, commit, push, Phase 8, or Phase 9 work
occurred. Accepted implementation checkpoint `3139b17` is pushed and remotely
verified. All four exact Phase 7 exit criteria are satisfied, and Phase 7 is
technically complete. Closure bookkeeping is complete locally. The sole next
action is separate Phase 7 closure-commit authorization; Phase 8 remains
unauthorized.
