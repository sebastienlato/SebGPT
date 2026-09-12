# Project State

> Primary session anchor. Keep this document concise and current; it describes now, not project history.

## Current phase

Phase 8 — Training and Checkpointing (technically complete, closure-state
accepted, and Gate 88 closure bookkeeping complete locally; formal remote
closure still requires Gates 89–93)

## Current milestone

Phases 0–7 are formally remotely closed. Phase 7 closure commit is
`33d4510421107848c4aa8a6014f4a7b1e391065a`; its accepted contract and
implementation commits are `60b2a9cce55da79ccc9fbd03fad014cb2a939290` and
`3139b1736f005fe903e2ea111d91934478b5a683`.

Phase 8 policy DEC-0020, the corrected detailed contract, and the corrected
implementation are accepted. Their current authority is:

- implementation `Code commit`:
  `809834323d53407cb4a54ae539585bb3d78856eb`;
- corrected provenance-contract commit:
  `c50d77ac935bdf924b9b429a5776c419982a5d11`;
- corrected provenance implementation commit:
  `3e0b9c69963702718446901b2fb85e4f5e38aea9`;
- corrected specification SHA-256:
  `1630f9c7a113ff4af6db709ba2c356e8dab450eb4d918a54d64b734084e70efd`;
- accepted external `pre_registration_commit`:
  `2871ebfa20a0b6fcd4f8f3519f7057aaf30a1a03`.

The fixed real run `EXP-20260912-01` completed successfully. Gate 78 fresh
independent experiment/checkpoint/exact-resume/provenance/exit review returned
PASS / NO ISSUE, with no BLOCKER, IMPORTANT, or MINOR finding and exact
recommendation `ACCEPT PHASE 8 RUN RESULT`. At Gate 79, Master Chat formally
accepted the run result and checkpoint authority locally. Sebastien then
explicitly completed Gate 80 by accepting the experiment result, retained
checkpoint/evidence state, and technical Phase 8 exit.

The accepted-result commit `334119e63c716e4922cd0b67da4f5fc221faa886`,
`Record Phase 8 training result`, was pushed and independently remote-verified
at Gate 85 with local `HEAD`, `main`, `origin/main`, and live remote `main`
synchronized at ahead/behind `0/0`. Sebastien explicitly completed Gate 86 by
accepting that Phase 8 is technically complete and ready for formal closure.
This is not yet formal remote closure.

Gate 87 authorized the exact documentation-only closure-bookkeeping scope, and
Gate 88 applied it. Current continuity now records all Phase 8 exit criteria as
verified and satisfied while preserving the distinction between technical
completion and future formal remote closure.

## Completed work

- The accepted run used the fixed ten-epoch CPU-float32 configuration under
  unchanged DEC-0020 and the accepted 94-gate specification.
- Initialized training and validation losses were `4.750465878532` and
  `4.754711149949085`.
- Final training and validation losses were `2.4611534265660575` and
  `2.4967258539791177`.
- Best validation loss was `2.4967258539791177` at epoch `10`; training and
  validation losses decreased at every completed epoch.
- All ten epochs completed with `3880` optimizer updates. Stopping reason was
  `maximum_epochs`.
- The exact predicate
  `min(completed-epoch training losses) < initialized training loss` passed.
- Checkpoint/catalog validation passed. Latest checkpoint is epoch-10 object
  `6990c89166d48732b0601aeae1edd57a9f0e22eec1c7c1160d5be1b8281c9148`;
  best-validation is the same immutable object under its distinct role; catalog
  SHA-256 is
  `6d590f0355ec950d770e77b4c542d65cda15012fbea56349b97b154a0d614241`.
- The exact-resume audit passed. Control checkpoint is
  `c406478feef00267120e0ee8b5f8188c803538a806e5321fc1ab4f91d1292210`;
  restored/resumed checkpoint is
  `a3a0c0ca98d8de70c5729a3eb433bac8dc091ee56f0de880bcee2b20765d3213`.
  Both durable graphs and public reload paths passed, with semantic equality
  except permitted lineage/hash/location differences.
- Actual wall time was `1763.81` seconds (`29m 23.81s`), `109.01` seconds or
  `6.59%` above the accepted `27.58`-minute estimate excluding checkpoint I/O.
- No generation or sampling occurred. *Twelfth Night* and the sealed test were
  not accessed. No Phase 9 work occurred.

## Current work

Gate 78–80 acceptance bookkeeping and Gate 86 closure-state acceptance are
recorded outside the byte-frozen result record. The completed 39-label record
in `EXPERIMENT_LOG.md` has SHA-256
`e7b40c888bc23c4e58d5abad10a766be16698d917ee8d8f54eaa6a1d199706d5`;
the committed file has SHA-256
`824f6897f048c3e27390421a054c7aea5ae60e89d6af6562873867aae2858fd4`;
and the accepted pre-registration log remains an exact byte prefix. The result
record is remotely authoritative in commit `334119e6`.

The complete closure-ready working set is exactly `DECISIONS.md`,
`LEARNING_NOTES.md`, `PROJECT_STATE.md`, `README.md`, and `ROADMAP.md`. It
includes the carried Gate 78–80 and Gate 86 records plus Gate 88 reconciliation.
Nothing is staged. Every changed path is documentation/continuity authority and
is necessary for the final closure-ready state.

## Current model status

The accepted Phase 7 MiniGPT architecture remains unchanged: one rank-one
CPU-long sequence of length `1..256` produces CPU-float32 logits `(T, 81)`. It
has 90 unique trainable tensors, 63,825 parameters, four accepted blocks, eight
local dropout generators, final normalization, and an untied biased head.

The accepted trained epoch-10 state is retained in the ignored canonical
checkpoint object named above. The canonical catalog gives the object distinct
`latest` and `best_validation` roles. The exact-resume control graph and run
authority/training evidence remain retained and ignored.

## Last verified working state

Gate 85 remote verification established:

- local `HEAD`, `main`, `origin/main`, and live remote `main` all equal
  `334119e63c716e4922cd0b67da4f5fc221faa886` at ahead/behind `0/0`;
- the accepted-result commit has one parent, exact title
  `Record Phase 8 training result`, and changes only `EXPERIMENT_LOG.md`;
- committed log, result-record, and pre-registration-prefix hashes are exact;
- all ten canonical objects and both audit-control objects match their
  content-addressed filenames, and both catalogs and retained evidence match
  their recorded hashes;
- the Phase 8 specification, DEC-0020, accepted Phase 8 code/tests, and Phase 7
  identities remain exact;
- before Gate 88, the only tracked modifications were the unstaged
  `DECISIONS.md` and `PROJECT_STATE.md` acceptance records; Gate 88 adds only
  necessary `LEARNING_NOTES.md`, `README.md`, and `ROADMAP.md` closure
  reconciliation; the index remains empty and accepted ignored artifacts are
  intact.

This bookkeeping session does not rerun training or tests and makes no claim of
new runtime verification beyond Gate 78's accepted evidence and the read-only
continuity checks.

## Next exact step

Obtain separate Gate 89 authorization for the exact Phase 8 closure commit.

## Remaining workflow

The accepted specification requires every remaining action separately:

1. Gate 89: authorize the exact closure commit.
2. Gate 90: create that commit without amendment or extra files.
3. Gates 91–93: separately authorize the closure push, push it, and independently
   verify formal remote Phase 8 closure.
4. Gate 94: only then consider separate Phase 9 learning/design authorization.

The proposed Gate 89 scope is exactly `DECISIONS.md`, `LEARNING_NOTES.md`,
`PROJECT_STATE.md`, `README.md`, and `ROADMAP.md`, with recommended title
`Complete Phase 8 training and checkpointing`. This matches the established
Phase 5–7 five-file closure convention. It remains a recommendation until Gate
89 explicitly authorizes the exact title and paths.

## Known issues

- Runtime remains a material cost: the accepted run took `29m 23.81s`.
- Optional NumPy interoperability is unavailable and intentionally unnecessary.
- MPS may be unavailable in restricted contexts; Phase 8 is CPU-only.
- Processed corpus derivatives, checkpoints, and experiment evidence are
  intentionally ignored by Git. They must be preserved.
- Content-addressed checkpoint objects may be orphaned by a publication failure
  before catalog replacement; accepted policy preserves them for diagnosis.

## Important constraints

- Do not modify `EXPERIMENT_LOG.md`; its planned and completed records, canonical
  JSON, whitespace, and full working-file bytes are accepted and frozen.
- Do not modify DEC-0020, `docs/TRAINING_CHECKPOINTING_SPEC.md`, Phase 8 source or
  tests, accepted Phase 7 files, or retained ignored run artifacts.
- Do not stage, commit, push, or perform later gates without their separate
  explicit authorizations.
- Do not rerun or tune `EXP-20260912-01`.
- Do not generate or sample, access the sealed test or *Twelfth Night*, or begin
  Phase 9. Phase 9 remains unauthorized through Gate 93.

## Open questions

- No Gate 78 review finding remains.
- No run-result, checkpoint, exact-resume, provenance, or Phase 8 exit evidence
  finding remains.
- Gate 80 explicit Sebastien acceptance is complete.
- Gates 81–85 accepted-result commit publication and remote verification are
  complete.
- Gate 86 closure-state acceptance is complete.
- Gates 87–88 closure-bookkeeping authorization and application are complete.
- Gate 89 closure-commit authorization is the sole current gate.

## Session handoff

Phase 8's fixed real run `EXP-20260912-01` passed its training-improvement
predicate, checkpoint/catalog validation, and exact-resume audit. Gate 78
returned PASS / NO ISSUE with no remaining finding and recommended `ACCEPT PHASE
8 RUN RESULT`; Master Chat completed Gate 79 formal local result acceptance,
and Sebastien explicitly accepted the experiment, checkpoints, and technical
Phase 8 exit at Gate 80. The frozen result is remotely authoritative at
`334119e63c716e4922cd0b67da4f5fc221faa886`, and Gate 85 remote verification is
complete. Sebastien explicitly accepts at Gate 86 that Phase 8 is technically
complete and ready for formal closure. Gate 88 closure bookkeeping is complete
locally across the exact five-file documentation set; nothing is staged. Gate
89 separate closure-commit authorization is next. Formal closure and Phase 9
remain unauthorized.
