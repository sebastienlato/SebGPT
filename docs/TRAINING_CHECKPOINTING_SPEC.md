# Phase 8 Training and Checkpointing Specification

## Status and authority

This is the documentation-only detailed-contract proposal for:

> Phase 8 — Training and Checkpointing
>
> Goal: Create a reproducible training loop and durable model state.

Sebastien has accepted DEC-0020 as one consolidated conceptual policy and has
authorized this proposal only. This document does not authorize source, tests,
optimizer construction, checkpoint construction, corpus traversal, feasibility
measurement, training, generation, sealed-test access, staging, commit, or
push. Every later action remains separately gated.

Fresh independent review returned `CORRECT BEFORE ACCEPTANCE` with three
BLOCKER, eight IMPORTANT, and one MINOR finding. Master Chat accepted all twelve
without changing DEC-0020 and authorized only their targeted documentation
correction. The corrected proposal resolves crash-durable object/catalog order,
transactional load RNG state, the full runtime envelope, exact PyTorch clipping,
the complete public/schema/error oracle, catalog/payload agreement, filesystem
trust and races, live repository authority, latest-only continuation, the
full-epoch real resume audit, irreversible gate separation, and deterministic
permutation evidence. It remains unaccepted pending focused independent
re-review. That focused re-review subsequently returned `CORRECT AGAIN BEFORE
ACCEPTANCE` with one remaining BLOCKER and two IMPORTANT schema/API findings,
and no MINOR finding. Master Chat accepted all three without changing DEC-0020
and authorized this final targeted documentation correction. The proposal now
adds intrinsic PyTorch-2.14.0 `decoupled_weight_decay=True` state and exact step
tensors, literal ownership/exports for every public Phase 8 object, and an
unambiguous first-key `configuration.schema_version`. It remains unaccepted
pending one final focused independent re-review.

The corrected contract and implementation were subsequently accepted, committed,
pushed, and remotely verified. During the separately authorized real-run
pre-registration gate, the original provenance wording was found to be
self-referential: it required a planned record to contain the SHA of the commit
that would later contain that same record. Master Chat accepted the stop and
authorized only the provenance/governance correction below. This corrected
documentation remains proposed pending focused independent review and acceptance;
it changes no DEC-0020 training policy or accepted training behavior.

DEC-0020 is the conceptual authority. This specification may make its approved
policy mechanical, but may not change the dataset, tokenizer, Phase 7 model,
CPU-float32 boundary, rank-one model call, context length 256, logical batch
capacity 8, deterministic epoch shuffle, AdamW settings, global-norm clipping,
ten-epoch budget, evaluation cadence, checkpoint policy, exact-resume
requirement, evidence standard, or Phase 8/9 boundary.

## Phase 8 exit contract

The exact roadmap exit criteria remain:

1. Batching, optimization, evaluation intervals, seeds, and device behavior are
   explicit.
2. Training and validation loss are recorded reproducibly.
3. Checkpoints save and restore model, optimizer, configuration, and progress.
4. At least one fully documented training run can be resumed successfully.

Phase 8 exit additionally requires every accepted DEC-0020 evidence obligation
defined below. Passing tests alone does not authorize or complete the training
run.

## Inherited authority

Phase 8 preserves the following accepted identities and behavior:

| Field | Accepted authority |
|---|---|
| Dataset | `shakespeare-eight-play` |
| Training works | exactly manifest orders 1 through 6 |
| Validation work | exactly manifest order 7, *The Tempest* |
| Sealed test | *Twelfth Night*; zero Phase 8 access |
| Vocabulary | `shakespeare-code-point-v1`, schema 1, size 81 |
| Vocabulary artifact | `artifacts/tokenizers/shakespeare-code-point-v1/vocabulary.json` |
| Vocabulary SHA-256 | `9f4235f9dab3e0361221a90c5ae7ca7fee540760fcc1708964368e6f17d6d70e` |
| Model contract | accepted `docs/MINI_GPT_SPEC.md` |
| Model implementation | `3139b1736f005fe903e2ea111d91934478b5a683` |
| Model source SHA-256 | `6b8db96577a0ad1f5daa4649d7ca9375e59873d4b635c3f76e75a6751f74f12d` |
| Model structure | 90 trainable tensors, 63,825 parameters |
| Model input/output | one rank-one `(T,)` input, logits `(T, 81)`, `1 <= T <= 256` |
| Loss | accepted standalone `explicit_cross_entropy` |
| Device/dtype | CPU float32 parameters/logits; CPU `torch.long` IDs |
| Dropout | accepted `p=0.1`, two model-local streams in each of four blocks |

The Phase 1 manifest and processing-manifest hashes and the six training and one
validation processed-work hashes remain part of the durable data authority.
Their exact values must be read from the accepted safe permitted-corpus
authority and recorded without opening, constructing, or exposing the sealed
work.

Phase 4's context/stride 64, one-example update rule, manual SGD, two-pass run,
and fixed unshuffled traversal are historical controls only. Phase 8 does not
modify those accepted files or route MiniGPT training through the Phase 4 model
or update helpers.

## Approved fixed configuration

```text
CONTEXT_LENGTH = 256
STRIDE = 256
LOGICAL_BATCH_CAPACITY = 8
MAXIMUM_EPOCHS = 10
ORDER_SEED = 8001

OPTIMIZER = AdamW
LEARNING_RATE = 0.0003
BETAS = (0.9, 0.999)
EPSILON = 0.00000001
WEIGHT_DECAY = 0.01
AMSGRAD = false
MAXIMIZE = false
FOREACH = false
CAPTURABLE = false
DIFFERENTIABLE = false
FUSED = false
ADAMW_STATE_DECOUPLED_WEIGHT_DECAY = true

GRADIENT_CLIP_NORM_TYPE = 2.0
GRADIENT_CLIP_MAX_NORM = 1.0

CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_CATALOG_SCHEMA_VERSION = 1
PHASE8_CONFIGURATION_SCHEMA_VERSION = 1
RUNTIME_IDENTITY_SCHEMA_VERSION = 1
MAXIMUM_CATALOG_BYTES = 16384
MAXIMUM_CHECKPOINT_OBJECT_BYTES = 67108864

TORCH_INTRA_OP_THREADS = 1
TORCH_INTER_OP_THREADS = 1
DETERMINISTIC_ALGORITHMS = true
DETERMINISTIC_WARN_ONLY = false
FLOAT32_MATMUL_PRECISION = highest
DEFAULT_DTYPE = torch.float32
DEFAULT_DEVICE = cpu
TRAINING_BOUNDARY_GRAD_MODE = enabled
CPU_AUTOCAST = disabled
INFERENCE_MODE = disabled
MKLDNN = disabled
```

`ORDER_SEED = 8001` is a proposed Phase 8 mechanics seed, chosen to be visible
and distinct from accepted Phase 3, 5, 6, and 7 seeds. It controls only epoch
ordering. It does not initialize or reseed model parameters or dropout streams.

No scheduler, warmup, random-replacement sampling, early stopping, mixed
precision, MPS/CUDA device, padding, special token, or rank-two model call is
permitted.

## Supported runtime envelope

The bitwise resume guarantee applies only inside one dedicated Phase 8 process
configured before corpus, model, optimizer, generator, or training-state
construction and before any parallel PyTorch work. The public
`configure_phase8_runtime()` operation performs the one-time configuration and
returns a frozen `Phase8RuntimeIdentity`. A separate
`validate_phase8_runtime()` operation is read-only and must pass at every
training, evaluation, save, and load boundary.

The required process settings are exact:

| Setting | Required value | Configuration and restoration rule |
|---|---|---|
| Intra-op threads | `torch.get_num_threads() == 1` | Set once to 1 before work; not restored inside the dedicated process |
| Inter-op threads | `torch.get_num_interop_threads() == 1` | Set once to 1 before inter-op work; any too-late/set-twice failure refuses the process |
| Deterministic algorithms | enabled | Set once with warn-only false; not restored inside the dedicated process |
| Deterministic warn-only | disabled | A true live value is a mismatch |
| Float32 matmul precision | `"highest"` | Set before tensor work; not restored inside the dedicated process |
| Default dtype | `torch.float32` | Set before construction; a later mismatch is refused |
| Default device | CPU | Set/verify before construction; no device override is accepted |
| MKLDNN | disabled | Disable before tensor work to freeze the CPU kernel path; not restored inside the process |
| Grad mode at training boundaries | enabled | Verified before/after every public training boundary; local no-grad evaluation restores it |
| CPU autocast | disabled | Never enabled; any enabled boundary is refused |
| Inference mode | disabled | Never enabled for training/load; any enabled boundary is refused |

The process is intentionally dedicated because inter-op thread count cannot be
safely changed after parallel work begins and global backend/determinism settings
must not be toggled around individual calls. Configuration is allowed only at
process startup. Later code never attempts to restore prior process settings;
it validates and refuses mismatch before returning or restoring training state.

`Phase8RuntimeIdentity` records, in this exact field order:

```text
schema_version: int
python_version: str
python_implementation: str
torch_version: str
operating_system: str
operating_system_release: str
machine: str
processor: str
device: str
parameter_dtype: str
token_dtype: str
torch_intra_op_threads: int
torch_inter_op_threads: int
deterministic_algorithms: bool
deterministic_warn_only: bool
float32_matmul_precision: str
default_dtype: str
default_device: str
grad_mode_enabled: bool
cpu_autocast_enabled: bool
inference_mode_enabled: bool
mkldnn_enabled: bool
torch_build_config_sha256: str
torch_parallel_info_sha256: str
requirements_lock_sha256: str
```

`schema_version` is exactly 1. The accepted base runtime identity begins with
CPython `3.14.4`, PyTorch `2.14.0`, Darwin/macOS arm64, and CPU float32/long;
the pre-registration freezes the full literal strings and digests observed from
the accepted clean run-authority commit. Checkpoint load requires exact equality
to those frozen literals, not merely compatible version ranges.

The two Torch digests hash the exact UTF-8 outputs of `torch.__config__.show()`
and `torch.__config__.parallel_info()`. The lock digest hashes exact tracked
`requirements.lock` bytes. The runtime identity is stored in configuration,
checkpoints, and experiment records. Live validation compares every field and
setting exactly. A mismatch raises the Phase 8 runtime contract invariant before
corpus/model/optimizer construction or before a loaded training state is
returned.

No broader portability is claimed across another runtime identity, thread
setting, PyTorch build, dependency lock, operating-system build, processor,
backend, device, or dtype.

## Live repository and run-authority preflight

An authorized feasibility measurement or training run may begin only in a
dedicated process after an independent live repository preflight. The planned
record's `Code commit` is the accepted Phase 8 implementation anchor, exactly
`809834323d53407cb4a54ae539585bb3d78856eb`; it is not the later commit that
contains the planned record. A caller-supplied `code_commit` is evidence to
verify against that immutable planned value, never authority by itself.

The future pre-registration commit is separate external repository authority.
It is not and cannot be embedded in the planned record that it creates. The
separate commit, push, and remote-verification gates establish its exact SHA as
`pre_registration_commit`. At run launch, live `HEAD` is that exact remotely
verified commit and the locally refreshed `origin/main` matches it. The runner
derives this value from live Git state and verifies the committed planned record;
it does not accept a caller substitute for it.

Before corpus, vocabulary binding, model, optimizer, window, generator, or
training-state construction, the preflight requires exactly:

1. the working directory is the physical repository root, not a symlink;
2. the checked-out branch is exactly `main`;
3. planned-record `Code commit` and caller `code_commit` both equal the accepted
   implementation anchor `809834323d53407cb4a54ae539585bb3d78856eb`;
4. that accepted implementation commit is an ancestor of live `HEAD`;
5. live `HEAD` is the exact accepted, remotely verified pre-registration commit;
6. live `HEAD` equals the locally refreshed `origin/main`, with ahead/behind
   `0/0`; the separate prior remote-verification gate supplies actual-remote
   authority and a run does not perform network access;
7. index, tracked worktree, and non-ignored untracked status are empty under
   `git status --porcelain=v2 --untracked-files=all`;
8. ignored processed corpus derivatives, checkpoint objects/catalogs, and
   ignored experiment artifacts are the only permitted generated filesystem
   state and do not relax tracked cleanliness;
9. accepted Phase 7 closure, contract, and implementation commits are ancestors
   of `HEAD` and equal the recorded identities;
10. live `docs/MINI_GPT_SPEC.md`, `src/sebgpt/model/mini_gpt.py`, and accepted
   Phase 7 focused test bytes equal their accepted SHA-256 identities;
11. the accepted Phase 8 contract commit and the accepted provenance-corrected
    contract commit are ancestors of `HEAD`, and live
    `docs/TRAINING_CHECKPOINTING_SPEC.md` equals the latest accepted corrected
    SHA-256;
12. the accepted Phase 8 implementation commit and any separately reviewed,
    accepted provenance-only implementation-correction commit are ancestors of
    `HEAD`; every Phase 8 source/test file equals its latest accepted exact hash;
    and the Git diff from that latest accepted implementation authority to the
    pre-registration commit contains no source, test, model, tokenizer, dataset,
    runtime-policy, or specification change;
13. live `EXPERIMENT_LOG.md` bytes equal the blob committed at live `HEAD`; that
    commit introduces exactly one accepted planned Phase 8 record whose run ID
    was absent from its first parent, and the record passes its complete schema,
    authority, and configuration validation;
14. `requirements.lock`, tokenizer artifact, dataset manifest, processing
    manifest, and the exact seven permitted work identities/hashes match their
    accepted authorities through existing content-safe verifiers; and
15. configured live runtime equals the complete recorded runtime envelope.

The future planned `EXPERIMENT_LOG.md` pre-registration must therefore be
reviewed, committed, pushed, and remotely verified before fixed-run
authorization. That commit becomes the external `pre_registration_commit`, not
the planned record's `Code commit`. The planned text is never rewritten. The
runner produces only ignored checkpoint and experiment artifacts while active,
so tracked state stays clean. Result text is appended to `EXPERIMENT_LOG.md`
only after the run stops and under its separate result-recording gate.

Wrong planned `Code commit` retains `Phase8ContractError` invariant
`phase8.contract.lifecycle` with field `planned_record_code_commit`. Missing,
wrong, uncommitted, unpushed, or nonmatching pre-registration authority raises
`Phase8GovernanceError` with invariant `phase8.governance.repository` and field
`pre_registration_commit`. An implementation ancestry or accepted-byte mismatch
raises the same governance invariant with field `implementation_authority`.
Dirty state retains field `clean`. Every mismatch occurs before corpus text
access or training-object construction. Tests mock repository facts; they do
not change live Git state.

### Required provenance-only implementation correction

A later separately authorized implementation pass must make only these
mechanical changes:

1. `_planned_run_record` validates planned `Code commit` against literal
   accepted implementation anchor
   `809834323d53407cb4a54ae539585bb3d78856eb`.
2. `_validate_live_repository` no longer requires `HEAD == code_commit`. It
   requires the implementation anchor to be an ancestor, derives
   `pre_registration_commit = HEAD`, requires `HEAD == origin/main`, proves the
   planned record is committed at that exact `HEAD`, and applies every accepted
   hash/cleanliness/data/runtime check above.
3. The preflight proves that only separately accepted provenance-governance
   source/test corrections and later pre-registration documentation changed
   after the implementation anchor; every executable Phase 8 file matches its
   latest independently accepted hash.
4. The existing `Phase 8 contract authority` mapping and checkpoint authority
   keys retain their schema but, after this correction is accepted and committed,
   their commit/hash values identify the latest accepted provenance-corrected
   contract. Original contract commit
   `09b2c2e0487a0b7a766655951422471085d943a8` remains a required ancestor.
5. `run_fixed_phase8_experiment` completes this preflight before corpus access,
   retains the derived pre-registration SHA as private run evidence, and keeps
   `Phase8TrainingState.code_commit` equal to the implementation anchor.
6. Save/load validate the same planned record, implementation anchor, live
   pre-registration commit, and accepted bytes at every checkpoint boundary.
7. Ignored experiment/audit evidence and the later result record include the
   external pre-registration SHA. Payload and resume-lineage schemas remain
   unchanged; exact-resume branches share this run-level external authority.

No caller value may replace live Git or committed-record evidence. No model,
optimizer, batching, RNG, evaluation, checkpoint durability, or Phase 8/9
behavior changes in this correction.

## Proposed file ownership and public surface

Later separately authorized implementation should add only the transparent
training package and focused tests required by this contract:

```text
src/sebgpt/training/__init__.py
src/sebgpt/training/phase8_types.py
src/sebgpt/training/phase8_data.py
src/sebgpt/training/phase8_optimization.py
src/sebgpt/training/phase8_checkpoint.py
src/sebgpt/training/phase8_experiment.py

tests/test_phase8_types.py
tests/test_phase8_data.py
tests/test_phase8_optimization.py
tests/test_phase8_checkpoint.py
tests/test_phase8_experiment.py
```

The proposed public records are frozen dataclasses. Tensor-, token-, model-,
optimizer-, and corpus-bearing fields use `repr=False`:

```text
Phase8Window(
    work_id: str,
    manifest_order: int,
    split: str,
    start_index: int,
    input_ids: tuple[int, ...],
    target_ids: tuple[int, ...],
)

Phase8LogicalBatch(
    epoch: int,
    batch_index: int,
    window_indices: tuple[int, ...],
    target_count: int,
)

Phase8BatchUpdate(
    batch_loss: float,
    target_count: int,
    sequence_count: int,
    pre_clip_global_norm: float,
)

Phase8Evaluation(
    split: str,
    loss: float,
    target_count: int,
    window_count: int,
)

Phase8Progress(
    completed_epochs: int,
    next_epoch: int,
    next_example_offset: int,
    optimizer_updates: int,
    examples_processed: int,
    targets_processed: int,
)

Phase8CheckpointReference(
    run_id: str,
    role: str,
    logical_id: str,
    epoch: int,
    validation_loss_hex: str,
    sha256: str,
    relative_path: str,
)

Phase8RunResult(
    run_id: str,
    status: str,
    stopping_reason: str,
    progress: Phase8Progress,
    initialized_training: Phase8Evaluation,
    initialized_validation: Phase8Evaluation,
    epoch_training: tuple[Phase8Evaluation, ...],
    epoch_validation: tuple[Phase8Evaluation, ...],
    latest_checkpoint: Phase8CheckpointReference,
    best_checkpoint: Phase8CheckpointReference,
    exact_resume_passed: bool,
)
```

`Phase8RuntimeIdentity` has the exact fields listed in the supported-runtime
section. `Phase8Configuration` is a factory-produced frozen record with public
construction blocked and these exact fields in order:

```text
schema_version: int
runtime: Phase8RuntimeIdentity
context_length: int
stride: int
retain_unpadded_tail: bool
logical_batch_capacity: int
allow_final_partial_batch: bool
maximum_epochs: int
order_seed: int
order_algorithm: str
optimizer_name: str
learning_rate: float
betas: tuple[float, float]
epsilon: float
weight_decay: float
amsgrad: bool
maximize: bool
foreach: bool
capturable: bool
differentiable: bool
fused: bool
scheduler_name: None
warmup_steps: int
gradient_clip_norm_type: float
gradient_clip_max_norm: float
gradient_clip_epsilon: float
evaluate_initialized_state: bool
evaluation_interval_epochs: int
evaluation_split_order: tuple[str, str]
best_comparison: str
validation_early_stopping: bool
parameter_device: str
parameter_dtype: str
token_dtype: str
checkpoint_schema_version: int
catalog_schema_version: int
maximum_catalog_bytes: int
maximum_checkpoint_object_bytes: int
```

Its exact values are the fixed constants in this specification;
`order_algorithm` is `torch_randperm_local_cpu_generator`, `scheduler_name` is
`None`, `warmup_steps` is 0, `gradient_clip_epsilon` is `1e-6`,
`evaluation_interval_epochs` is 1, `evaluation_split_order` is
`("train", "validation")`, `best_comparison` is `strict_lower`, and the three
device/dtype strings are `cpu`, `torch.float32`, and `torch.long`.

`Phase8MetricState` is a factory-produced frozen record with these exact fields
in order:

```text
initialized_training: Phase8Evaluation
initialized_validation: Phase8Evaluation
epoch_training: tuple[Phase8Evaluation, ...]
epoch_validation: tuple[Phase8Evaluation, ...]
current_training: Phase8Evaluation
current_validation: Phase8Evaluation
best_validation_loss: float
best_validation_epoch: int
best_logical_id: str
```

It exists only after at least one completed epoch. Before epoch 1, initialized
measurements are held by the runner and no checkpointable `Phase8MetricState`
exists. Thus checkpoint metric fields are never optional.

`Phase8TrainingState` is a factory-produced frozen handoff record with these
exact fields in order:

```text
run_id: str
code_commit: str
configuration: Phase8Configuration
model: MiniGPT
optimizer: torch.optim.AdamW
order_generator: torch.Generator
progress: Phase8Progress
metrics: Phase8MetricState
global_cpu_rng_state: torch.Tensor
resumed_from_checkpoint_sha256: str | None
resume_checkpoint_sha256s: tuple[str, ...]
```

`code_commit` is exactly the planned record's accepted implementation anchor
`809834323d53407cb4a54ae539585bb3d78856eb`. The separate live
`pre_registration_commit` is repository/run-launch authority validated before
state construction; it is not duplicated in this in-memory semantic record.

Model-, optimizer-, generator-, RNG-, and corpus-bearing fields are repr-hidden.
The optional resumed-from hash is `None` only before the first continuation;
each later checkpoint requires it to equal the last item of the nonempty lineage
tuple. Immutability of the handoff record does not pretend that its contained
training objects are immutable.

The exact proposed public operations are:

```text
load_phase8_configuration() -> Phase8Configuration

configure_phase8_runtime() -> Phase8RuntimeIdentity

validate_phase8_runtime() -> Phase8RuntimeIdentity

build_phase8_windows(
    works: tuple[ExtractedWork, ...],
    vocabulary: VocabularyBinding,
    phase_1_manifest: Mapping[str, object],
    *,
    split: str,
) -> tuple[Phase8Window, ...]

create_phase8_epoch_order(
    number_of_windows: int,
    *,
    epoch: int,
    generator: torch.Generator,
) -> tuple[int, ...]

iter_phase8_logical_batches(
    windows: tuple[Phase8Window, ...],
    order: tuple[int, ...],
    *,
    epoch: int,
) -> Iterator[Phase8LogicalBatch]

create_phase8_optimizer(model: MiniGPT) -> torch.optim.AdamW

run_phase8_logical_batch(
    model: MiniGPT,
    optimizer: torch.optim.AdamW,
    windows: tuple[Phase8Window, ...],
    batch: Phase8LogicalBatch,
) -> Phase8BatchUpdate

evaluate_phase8_split(
    model: MiniGPT,
    optimizer: torch.optim.AdamW,
    windows: tuple[Phase8Window, ...],
    order_generator: torch.Generator,
    progress: Phase8Progress,
    *,
    split: str,
) -> Phase8Evaluation

save_phase8_checkpoint(
    state: Phase8TrainingState,
    repository_root: Path,
) -> Phase8CheckpointReference

load_phase8_checkpoint(
    repository_root: Path,
    *,
    run_id: str,
    role: Literal["latest"],
) -> Phase8TrainingState

run_fixed_phase8_experiment(
    corpus: Phase4PermittedCorpus,
    vocabulary: VocabularyBinding,
    config: Phase8Configuration,
    *,
    code_commit: str,
) -> Phase8RunResult
```

For `run_fixed_phase8_experiment`, `code_commit` means the immutable accepted
implementation anchor recorded by the planned entry, not live launch `HEAD`.
The operation independently derives and validates live `HEAD` as the accepted
external pre-registration authority before corpus access.

All arguments are required and have no defaults except the keyword-only
boundaries shown. `role` accepts exactly `latest`; `best_validation` is not a
training-continuation role.
`split` accepts exactly `train` or `validation`; construction still validates
the supplied works against the one permitted identity sequence for that split.
No operation accepts an override for sealed test, model shape, device, dtype,
context, stride, batch capacity, optimizer, or approved numerical policy.

Exact implementation-module ownership and each literal module `__all__` are:

```text
sebgpt.training.phase8_types
    Phase8BatchUpdate
    Phase8CheckpointError
    Phase8CheckpointReference
    Phase8Configuration
    Phase8ContractError
    Phase8Evaluation
    Phase8GovernanceError
    Phase8LogicalBatch
    Phase8MetricState
    Phase8NumericalError
    Phase8Progress
    Phase8PublicationError
    Phase8RunResult
    Phase8RuntimeIdentity
    Phase8TrainingState
    Phase8TypeError
    Phase8Window

sebgpt.training.phase8_data
    build_phase8_windows
    create_phase8_epoch_order
    iter_phase8_logical_batches

sebgpt.training.phase8_optimization
    create_phase8_optimizer
    evaluate_phase8_split
    run_phase8_logical_batch

sebgpt.training.phase8_checkpoint
    load_phase8_checkpoint
    save_phase8_checkpoint

sebgpt.training.phase8_experiment
    configure_phase8_runtime
    load_phase8_configuration
    run_fixed_phase8_experiment
    validate_phase8_runtime
```

Each block is the exact `__all__` order for that exact module path. Every public
record's and exception's `__module__` is exactly
`sebgpt.training.phase8_types`; every public operation's `__module__` is its
owning module shown above. In particular, both runtime operations belong to
`phase8_experiment`, and all six Phase 8 exception classes belong only to
`phase8_types`. No implementation module may add another `__all__` name. Any
dependency or helper imported into a module is private, is underscore-aliased
when necessary, and is not a contractual public export.

`sebgpt.training.__init__` imports each name from its owning module and has this
exact package-level `__all__` order:

```text
Phase8BatchUpdate
Phase8CheckpointError
Phase8CheckpointReference
Phase8Configuration
Phase8ContractError
Phase8Evaluation
Phase8GovernanceError
Phase8LogicalBatch
Phase8MetricState
Phase8NumericalError
Phase8Progress
Phase8PublicationError
Phase8RunResult
Phase8RuntimeIdentity
Phase8TrainingState
Phase8TypeError
Phase8Window
build_phase8_windows
configure_phase8_runtime
create_phase8_epoch_order
create_phase8_optimizer
evaluate_phase8_split
iter_phase8_logical_batches
load_phase8_checkpoint
load_phase8_configuration
run_fixed_phase8_experiment
run_phase8_logical_batch
save_phase8_checkpoint
validate_phase8_runtime
```

The field and public-signature blocks above are the exact independent oracle:
names, order, annotation text, required/default status, and positional versus
keyword-only placement must match literally. No public record or operation may
have an extra field or parameter.

## Deterministic window contract

### Base stream

Every work is encoded independently with the accepted vocabulary binding. No
encoded work, window, or tensor is persisted. Training and validation windows
are constructed in memory only after all safe metadata, split, identity, hash,
count, and vocabulary checks pass.

For one permitted work with immutable token IDs `z` and `N = len(z)`, define
candidate starts:

```text
0, 256, 512, ... strictly below N - 1
```

For each start `s`:

```text
L = min(256, N - 1 - s)
input_ids  = z[s : s + L]
target_ids = z[s + 1 : s + L + 1]
```

Every produced length therefore satisfies `1 <= L <= 256`. Inputs and targets
have equal lengths. The target at each row is exactly the next token in the same
work. No window crosses a work or split boundary.

`STRIDE = 256` makes the windows non-overlapping in target-transition space.
Every within-work adjacent transition appears exactly once in the canonical
example set. The final short tail is retained whenever at least one transition
remains. It is not padded, repeated, merged with another work, or dropped.

The canonical unshuffled order is accepted manifest order followed by ascending
start within each work. A window's safe identity is exactly:

```text
(split, manifest_order, work_id, start_index, length)
```

Its repr and errors never expose token IDs or text.

Training constructs only the six training-work windows. Validation constructs
only *The Tempest* windows. No API accepts `test`, `all`, an arbitrary work list,
or a supplier that could expose the sealed work.

## Deterministic epoch ordering

Training uses one module-local CPU `torch.Generator` created with seed 8001.
The generator is distinct from global CPU RNG and all eight model-local dropout
generators.

At the start of each one-based epoch, call exactly one CPU
`torch.randperm(number_of_training_windows, generator=order_generator,
device="cpu")`. Convert the resulting exact integer permutation to immutable
indices and consume it left to right. No other operation draws from the order
generator.

An epoch is exactly one use of every canonical training window according to
that permutation. There is no replacement, omission, duplication, or reshuffle
inside the epoch. A fresh permutation is drawn between epochs. Evaluation
always uses canonical unshuffled order and consumes no ordering randomness.

The complete order-generator state after the current epoch permutation has
been consumed is checkpointed. Because durable checkpoints occur only after a
completed epoch, checkpoint progress records:

```text
completed_epochs = e
next_epoch = e + 1
next_example_offset = 0
active_epoch_permutation = none
```

The explicit zero cursor prevents an end-of-epoch checkpoint from being
misread as a mid-epoch checkpoint. Phase 8 does not publish mid-epoch
checkpoints. Interrupted partial-epoch work is discarded and exact continuation
restarts from the preceding completed-epoch checkpoint.

## Logical batches

The logical batch capacity is eight sequence windows. Permuted indices are
partitioned consecutively into groups of eight. If an epoch's number of windows
is not divisible by eight, the final group contains the remaining `1..7`
windows. The tail logical batch is retained so an epoch still visits every
training transition exactly once.

Each sequence is converted independently to rank-one CPU-long input and target
tensors and passed through the unchanged accepted rank-one MiniGPT interface.
No rank-two token tensor, padding, masking, or model-interface change is
introduced.

For a logical batch containing sequences `i = 1..B`, let `L_i` be sequence
length, let `m_i` be the accepted scalar mean cross-entropy for that sequence,
and let:

```text
M = sum_i L_i
batch_loss = sum_i (L_i / M) * m_i
```

The update gradient is:

```text
gradient(batch_loss)
```

Thus every target token contributes exactly `1 / M` to the logical-batch mean,
including tokens in a final short window. The implementation computes `M`
before any forward call, then calls `backward()` once per sequence on
`m_i * (L_i / M)`. Autograd accumulation is the transparent rank-one
implementation of the formula. It must not divide the accumulated gradient by
eight again.

Per-batch reporting uses the detached target-weighted mean above. Epoch
training loss used for evaluation is not the average of update-time batch
losses; it is a separate full evaluation of the post-epoch model.

## AdamW and update ordering

The optimizer is constructed only after the exact accepted model has passed
structural and parameter validation. It has one parameter group containing all
90 parameters in exact `model.named_parameters()` order. Weight decay 0.01
applies uniformly, including biases and normalization parameters; there is no
unapproved parameter-group exception.

The exact proposed construction is semantically:

```text
torch.optim.AdamW(
    parameters_in_accepted_model_order,
    lr=3e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
    amsgrad=False,
    maximize=False,
    foreach=False,
    capturable=False,
    differentiable=False,
    fused=False,
)
```

Supported PyTorch 2.14.0 AdamW internally constructs its Adam parameter group
with `decoupled_weight_decay=True`. This field is intrinsic supported optimizer
state, not a new DEC-0020 policy choice and not an additional constructor
argument. Immediately after construction, the one live parameter group must
contain `decoupled_weight_decay` in the supported order immediately after
`fused`, its exact type must be `bool`, and its value must be `True`.
Production must neither remove nor normalize this field before serialization.
Checkpoint validation and restoration require it, and exact resume compares it.

There is no scheduler and no call that changes the learning rate.

For each logical batch, perform exactly:

1. Validate model identity, training mode, configuration, progress, window
   identities, batch size, sequence lengths, and target-token total.
2. Require every model gradient to be `None`.
3. Call `optimizer.zero_grad(set_to_none=True)`; gradients remain `None`.
4. For each sequence in batch order, compute rank-one logits, accepted mean
   cross-entropy, the exact token-weighted scalar, and `backward()`.
5. Validate all 90 gradients for presence, identity ownership, shape, CPU
   device, float32 dtype, and finiteness before any clipping or parameter
   update.
6. Compute and apply global L2 gradient-norm clipping with maximum norm 1.0.
7. Validate the returned pre-clip norm and all post-clip gradients are finite.
8. Call `optimizer.step()` exactly once.
9. Call `optimizer.zero_grad(set_to_none=True)` exactly once and prove every
   gradient is `None`.
10. Validate all parameters and all tensor-valued AdamW state are finite.
11. Increment update, example, and target counters exactly once.

### Exact clipping semantics

Clipping uses `torch.nn.utils.clip_grad_norm_` over all 90 parameters in their
accepted order with:

```text
max_norm=1.0
norm_type=2.0
error_if_nonfinite=True
foreach=False
```

Under accepted PyTorch 2.14.0 with `foreach=False`, the exact oracle is:

```text
per_parameter_norms = [
    torch.linalg.vector_norm(gradient, 2.0)
    for gradient in the accepted 90-parameter order
]
total_norm = torch.linalg.vector_norm(
    torch.stack(per_parameter_norms),
    2.0,
)
clip_coefficient = 1.0 / (total_norm + 1e-6)
clamped_coefficient = torch.clamp(clip_coefficient, max=1.0)
for gradient in accepted parameter order:
    gradient *= clamped_coefficient
```

`clip_grad_norm_` returns the pre-clip `total_norm`. The documented `1e-6` is
part of the supported implementation semantics. Consequently, a norm clearly
below `1.0 - 1e-6` has clamped coefficient 1 and is unchanged; a norm in the
interval `(1.0 - 1e-6, 1.0]`, including exactly 1.0, may be scaled slightly;
and a norm above 1.0 is scaled. Tests derive expected tensors from this formula
with the supported float32 operation order and must not use the idealized rule
"norm at or below 1.0 is unchanged." Clipping occurs after complete
logical-batch accumulation and before AdamW, never per sequence.

## Evaluation protocol

A full training evaluation and full validation evaluation occur:

1. once at initialized epoch 0 before any optimizer update; and
2. after each completed epoch 1 through 10.

The initialized measurements establish the required training-loss improvement
baseline. Epoch-0 validation is observational and is not eligible for the
best-checkpoint role; the first completed epoch necessarily establishes the
first best-validation checkpoint.

For each evaluation event:

1. Require the model to be in training mode with all 90 gradients `None`.
2. Snapshot parameter identity and raw-byte digest, optimizer state, all eight
   dropout-generator states, order-generator state, global CPU RNG state, and
   progress.
3. Call `model.eval()` and verify consistent propagation to every block and
   dropout child.
4. Enter `torch.no_grad()`.
5. Traverse every window in canonical manifest/start order, one rank-one call
   at a time.
6. Aggregate with `math.fsum(float(mean_loss) * L)` and divide once by the
   exact total target-token count.
7. Exit no-grad and call `model.train()`.
8. Verify mode consistency and exact nonmutation of every snapshotted state.

Training is evaluated first and validation second. A failure uses a `finally`
boundary to attempt restoration of training mode, publishes no metric or
checkpoint, and terminates the run. Evaluation performs no backward call,
gradient clear, optimizer call, ordering draw, or checkpoint mutation.

Primary metrics are target-token-weighted training and validation
cross-entropy in natural-log units. `exp(loss)` perplexity and `ln(81)` may be
recorded as derived explanatory values only; neither changes updates, stopping,
best-checkpoint selection, or Phase 8 PASS/FAIL.

Best validation selection is strict numeric `<`. An equal validation loss keeps
the earlier best checkpoint. All comparisons use the exact finite Python float
recorded by the target-weighted aggregate.

## Training budget and stop behavior

The normal run completes exactly ten epochs. There is no validation-based early
stopping. Permitted stops are:

- an explicit contract, governance, type, numerical, checkpoint, integrity, or
  runtime failure;
- a separately specified nonfinite/safety stop reached by the checks in this
  document; or
- an explicit Master Chat termination.

No exception is caught and treated as a skipped sequence, batch, update,
evaluation, epoch, or retry. No setting is tuned after a failure. A stopped or
failed run receives an append-only record with the last valid checkpoint and
exact stopping reason before further work is considered.

If a failure occurs before `optimizer.step`, parameters are unchanged but
partial gradients and some dropout streams may have advanced. The process
stops without publishing a checkpoint. If a failure occurs after the step,
current in-memory state may be advanced; it still stops without publishing an
epoch checkpoint. In either case, only the previous atomically published
completed-epoch checkpoint is resumable. No outer in-memory rollback or silent
retry is promised.

## Runtime stochastic-state ownership

Phase 8 owns exactly these runtime stochastic states:

1. the Phase 8 order-generator state seeded initially with 8001;
2. `blocks.0.attention_dropout` generator state;
3. `blocks.0.feed_forward_dropout` generator state;
4. `blocks.1.attention_dropout` generator state;
5. `blocks.1.feed_forward_dropout` generator state;
6. `blocks.2.attention_dropout` generator state;
7. `blocks.2.feed_forward_dropout` generator state;
8. `blocks.3.attention_dropout` generator state;
9. `blocks.3.feed_forward_dropout` generator state; and
10. global CPU `torch` RNG state.

The eight dropout generators retain the accepted Phase 6 seeds and behavior;
Phase 8 neither reseeds nor replaces them. Their complete current state tensors,
not merely initial seeds, are checkpointed.

Phase 8 training must not use Python `random`, NumPy, global Torch sampling,
CUDA RNG, or MPS RNG. Python and NumPy state are therefore outside the semantic
checkpoint because those sources are prohibited, not omitted accidentally.
Global CPU Torch state is captured and restored even though approved operations
must leave it unchanged; this makes unexpected global consumption detectable
and exact continuation defensive.

Reproducibility is bitwise only on the accepted Python 3.14.4, PyTorch 2.14.0,
macOS arm64 CPU-float32 environment and matching accepted code. No cross-version,
cross-build, cross-platform, or cross-device bitwise promise is made.

## Model-mode ownership

Ordinary training and every durable checkpoint boundary require the complete
model and every block/dropout child to be in training mode. Evaluation is the
only normal transition to evaluation mode. It must restore training mode before
return or before reporting a failure.

The checkpoint records the complete expected mode as `training=true` plus a
literal ordered child-mode map. Load rejects inconsistent or non-training
durable state. It does not coerce a malformed mixed-mode checkpoint into a
valid model.

## Checkpoint trust, names, roles, and resource bounds

Phase 8 checkpoints are trusted local SebGPT project artifacts produced by the
accepted implementation under the approved run. They are not a safe interchange
format for arbitrary or adversarial checkpoint files. `weights_only=True`
restricts deserialization object construction but is not a complete hostile-file
sandbox and does not by itself prevent CPU, memory, storage, or decompression
resource exhaustion.

Within that local trust model, every path operation still protects repository
authority against mistakes, stale files, symlinks, and cooperating concurrent
substitution. The exact grammars are:

```text
run_id                  := EXP-[0-9]{8}-[0-9]{2}
logical_checkpoint_id   := epoch-[0-9]{4}
permitted logical IDs   := epoch-0001 through epoch-0010
sha256                  := [0-9a-f]{64}
object_relative_path    := objects/<sha256>.pt
catalog_filename        := checkpoint-catalog.json
```

No alternate slash, dot, case, Unicode-normalized, absolute, percent-encoded,
or path-equivalent spelling is accepted. Catalog bytes may not exceed 16,384
bytes. Checkpoint object bytes must be positive and may not exceed 67,108,864
bytes. File size is checked with `fstat` on an already opened descriptor before
allocation or reading; reads stop and fail if observed bytes exceed the recorded
size or limit.

`latest` is the only Phase 8 training-continuation role. It must reference the
highest durably completed epoch in the catalog. `best_validation` is an
immutable retained model/training-state artifact selected by strict validation
loss for later Phase 9 selection/evaluation. Phase 8 may verify its integrity
but `load_phase8_checkpoint(..., role="best_validation")` is rejected by
`phase8.checkpoint.role.continuation`; the public continuation signature accepts
only `Literal["latest"]`. Branching or continuing training from an older best
checkpoint requires a separately specified future new-run/fork policy.

## Complete checkpoint payload

### Storage layout

Generated files remain outside Git under:

```text
checkpoints/phase8/<run_id>/
    objects/<sha256>.pt
    checkpoint-catalog.json
```

Objects are immutable and content-addressed. The canonical UTF-8 catalog is the
single atomic publication point and contains references for `latest` and
`best_validation`. Old content-addressed epoch objects are retained for this
first run; Phase 8 performs no deletion or garbage collection.

### Serialization

The `.pt` object is one exact built-in `dict` serialized once with `torch.save`.
Its transitive values are restricted to exact built-in dictionaries, lists,
tuples, strings, integers, finite floats, booleans, `None`, CPU tensors, and the
single `collections.OrderedDict[str, torch.Tensor]` required for `model_state`.
No other mapping subclass or custom class is accepted. It contains no callable,
path object, open file, text, token sequence, window tensor, activation, logit,
gradient, or arbitrary pickle payload.

Loading first verifies the referenced object's SHA-256 and path safety, then
uses `torch.load(..., map_location="cpu", weights_only=True)`. Any unsupported
payload type, missing field, extra field, wrong tensor metadata, or authority
mismatch is refused before a returned training object exists.

### Schema version 1

Every `mapping` below means exact built-in `dict` and rejects missing and extra
keys; `model_state` is the one explicitly named `OrderedDict` exception. Exact
built-in types reject `bool` where `int` is required. Tuple means exact built-in
tuple. Mapping keys are exact ASCII strings.

The payload has these exact top-level keys in order:

```text
schema_version: int                       # exactly 1
run: mapping
authority: mapping
configuration: mapping
model_state: collections.OrderedDict
optimizer_state: mapping
progress: mapping
random_state: mapping
mode_state: mapping
metric_state: mapping
resume_lineage: mapping
```

`run` has exactly:

```text
run_id: str                              # exact run-ID grammar
logical_id: str                          # epoch-0001..epoch-0010
completed_epoch: int                     # 1..10 and matches logical ID
code_commit: str                         # accepted implementation anchor; exact 8098343...
```

Checkpoint `run.code_commit` remains the planned record's accepted
implementation anchor. The external `pre_registration_commit` is deliberately
not inserted into the semantic payload after pre-registration: doing so would
change the accepted planned record or require another self-referential value.
Every save and load instead re-runs the corrected live repository preflight,
which binds the payload run ID and implementation anchor to the exact committed
planned record at live `HEAD`. The final result record stores that externally
derived pre-registration SHA for durable review.

`authority` has exactly these keys in order:

```text
phase7_closure_commit: str
phase7_contract_commit: str
phase7_implementation_commit: str
mini_gpt_spec_sha256: str
mini_gpt_source_sha256: str
mini_gpt_test_sha256: str
phase8_contract_commit: str
phase8_spec_sha256: str
requirements_lock_sha256: str
tokenizer: mapping
dataset: mapping
model: mapping
runtime: mapping
sealed_test_access: str                  # exactly "none"
```

`tokenizer` has exactly:

```text
implementation_commit: str
tokenizer_id: str                        # shakespeare-code-point-v1
schema_version: int                      # 1
vocabulary_size: int                     # 81
artifact_path: str
artifact_sha256: str
```

`dataset` has exactly:

```text
dataset_id: str                          # shakespeare-eight-play
manifest_sha256: str
processing_manifest_sha256: str
training_works: tuple[mapping, ...]      # exactly six, manifest order
validation_works: tuple[mapping, ...]    # exactly one, The Tempest
```

Each work mapping has exactly `manifest_order: int`, `work_id: str`,
`split: str`, and `processed_sha256: str` in that order. No sealed-work record
is present.

`model` has exactly:

```text
model_width: int                         # 32
max_sequence_length: int                 # 256
number_of_blocks: int                    # 4
number_of_heads: int                     # 4
head_width: int                          # 8
hidden_width: int                        # 128
layer_norm_epsilon: float                # 1e-5
dropout_probability: float               # 0.1
embedding_seed: int                      # 1337
block_parameter_seeds: tuple[int, int, int, int]
output_head_seed: int                    # 7005
parameter_device: str                    # cpu
parameter_dtype: str                     # torch.float32
trainable_tensor_count: int              # 90
parameter_count: int                     # 63825
```

`runtime` is the exact ordered mapping serialization of every
`Phase8RuntimeIdentity` field, beginning with its own `schema_version: int == 1`.
`configuration` is an exact built-in dict whose first key is
`schema_version: int == 1`, the Phase8Configuration schema version. Every
remaining key then serializes every remaining `Phase8Configuration` field in
the already defined record order, beginning with `runtime` and ending with
`maximum_checkpoint_object_bytes`. Its nested `runtime` mapping must equal
`authority.runtime` exactly. Missing `configuration.schema_version`, a first key
other than `schema_version`, a non-exact-int value, any value other than 1, or
any missing/extra configuration key raises `Phase8CheckpointError` with
invariant `phase8.checkpoint.configuration` before restoration.

The version domains are distinct and are never substituted for one another:

```text
payload.schema_version                 # checkpoint payload schema, exact 1
payload.configuration.schema_version   # Phase8Configuration schema, exact 1
payload.authority.runtime.schema_version # Phase8RuntimeIdentity schema, exact 1
catalog.schema_version                 # checkpoint catalog schema, exact 1
tokenizer.schema_version               # inherited vocabulary schema, exact 1
```

Equal numeric values do not merge their ownership. A future change to one
schema does not authorize or imply a change to another.

`model_state` is an exact `collections.OrderedDict` containing the accepted
model `state_dict` with all 90 tensors and
no unexpected buffer, in the exact literal parameter-name order accepted by
`docs/MINI_GPT_SPEC.md`; every value is a finite CPU float32 tensor of its
accepted shape. That accepted Phase 7 parameter-name/shape oracle is incorporated
by reference and may not be inferred from production enumeration alone.

`optimizer_state` is an exact built-in dict with exactly top-level keys `state`
then `param_groups`.
`param_groups` is an exact one-element list. Its mapping has exactly, in order,
`lr`, `betas`, `eps`, `weight_decay`, `amsgrad`, `maximize`, `foreach`,
`capturable`, `differentiable`, `fused`, `decoupled_weight_decay`, and `params`.
The first ten policy values equal the accepted configuration;
`decoupled_weight_decay` is exact built-in `bool` `True`; and `params` is the
ordered list of integer IDs `0..89`. This is the literal supported PyTorch
2.14.0 AdamW state shape. The intrinsic field must not be stripped, renamed,
defaulted, or normalized before checkpoint serialization.

`state` has exactly integer keys `0..89`; every value mapping has exactly
`step`, `exp_avg`, and `exp_avg_sq`. `step` is exactly a zero-dimensional CPU
`torch.float32` tensor. It is finite, its numeric value is integral, and it
equals `progress.optimizer_updates`. Every one of the 90 optimizer-state
entries has the same accepted step value at an optimizer-step/epoch checkpoint
boundary. Both moment tensors are finite CPU float32 with the corresponding
parameter shape. No `max_exp_avg_sq` exists because `amsgrad` is false.

`progress` has exactly:

```text
completed_epochs: int
next_epoch: int
next_example_offset: int                 # exactly 0
optimizer_updates: int
examples_processed: int
targets_processed: int
per_epoch_optimizer_updates: tuple[int, ...]
per_epoch_example_counts: tuple[int, ...]
per_epoch_target_counts: tuple[int, ...]
next_ordering_action: str                # exactly "draw_next_epoch_permutation"
```

All three per-epoch tuples have length `completed_epochs`; their sums equal the
cumulative values, `next_epoch == completed_epochs + 1`, and the completed epoch
matches `run`.

`random_state` has exactly:

```text
order_generator: torch.Tensor
global_cpu: torch.Tensor
dropout_generators: mapping
```

The dropout mapping has exactly these eight keys in order, each mapped to a
one-dimensional CPU `torch.uint8` generator-state tensor:

```text
blocks.0.attention_dropout
blocks.0.feed_forward_dropout
blocks.1.attention_dropout
blocks.1.feed_forward_dropout
blocks.2.attention_dropout
blocks.2.feed_forward_dropout
blocks.3.attention_dropout
blocks.3.feed_forward_dropout
```

The order and global states are also one-dimensional CPU `torch.uint8` tensors.

`mode_state` has exactly `training: bool` and `named_modules: tuple`.
`training` is true. `named_modules` is the exact tuple of `(module_name: str,
training: bool)` pairs returned in accepted MiniGPT named-module order, and
every bool is true.

An evaluation mapping has exactly:

```text
split: str
loss: float
loss_hex: str
target_count: int
window_count: int
```

The finite float and lowercase `float.hex()` representation must agree exactly.
`metric_state` has exactly:

```text
initialized_training: evaluation mapping
initialized_validation: evaluation mapping
epoch_training: tuple[evaluation mapping, ...]
epoch_validation: tuple[evaluation mapping, ...]
current_training: evaluation mapping
current_validation: evaluation mapping
best_validation_loss: float
best_validation_loss_hex: str
best_validation_epoch: int
best_logical_id: str
best_comparison: str                     # exactly "strict_lower"
```

Epoch tuples have length `completed_epochs`; current mappings equal their final
items; best fields are the strict minimum over completed validation mappings.

`resume_lineage` has exactly:

```text
root_run_id: str
resumed_from_checkpoint_sha256: str | None
resume_count: int
checkpoint_sha256s: tuple[str, ...]
```

`root_run_id` equals `run.run_id`; tuple length equals `resume_count`; the
nullable field is `None` exactly when the tuple is empty and otherwise equals
its last item. Operational lineage is provenance, not training-semantic state.
Both exact-resume branches share the same externally validated
`pre_registration_commit`; it is run-level provenance rather than a branch or
training-semantic difference. Resume lineage therefore remains unchanged and
may differ only as already specified.

The canonical catalog has these exact top-level keys in order:

```text
schema_version: int                      # exactly 1
run_id: str
latest: checkpoint-reference mapping
best_validation: checkpoint-reference mapping
```

Each checkpoint-reference mapping has exactly these keys in order:

```text
run_id: str
role: str                               # latest or best_validation
logical_id: str
epoch: int
validation_loss_hex: str
sha256: str
relative_path: str                      # exactly objects/<sha256>.pt
```

JSON floats are stored as finite hexadecimal strings produced by `float.hex()`.
The catalog is encoded as UTF-8 with sorted keys, compact separators, one final
LF, and no literal corpus content.

### Catalog/payload agreement

After hashing and deserializing the selected already-opened object bytes, load
cross-validates exactly:

- catalog `run_id`, reference `run_id`, payload `run.run_id`, and lineage root;
- reference logical ID and epoch against payload run and progress fields;
- reference validation-loss hex against payload current validation for `latest`
  and against payload best validation fields for `best_validation`;
- reference SHA-256 against the digest of the same bytes supplied to
  deserialization;
- relative path against the sole spelling `objects/<sha256>.pt`;
- `latest` against the catalog's highest completed epoch and current payload;
- `best_validation` against strict-minimum metric state, best logical ID/epoch,
  and the referenced historical payload; and
- catalog role string against the catalog key containing that reference.

A valid hashed object cannot be relabeled for another run, epoch, logical ID,
loss, path, or role. Any mismatch raises `Phase8CheckpointError` with invariant
`phase8.checkpoint.cross_field` before restoration.

## Atomic checkpoint publication

All filesystem work is relative to retained directory descriptors. Starting
from a physically resolved, non-symlinked repository descriptor, the
implementation uses `os.open` with `O_DIRECTORY | O_NOFOLLOW`, `os.mkdir` with
`dir_fd`, `os.open` with `O_CREAT | O_EXCL | O_NOFOLLOW`, `os.link` with
source/destination `dir_fd` and `follow_symlinks=False`, `os.replace` with both
directory descriptors, `os.unlink` with `dir_fd`, and `fstat`/non-following
directory-relative stat. It retains expected device/inode identities for the
repository, `checkpoints`, `phase8`, run, and `objects` directories and verifies
them before and after each irreversible step. The cooperative local-project
trust model does not claim safety against a malicious same-authority process,
but check-then-use path reopening is not permitted.

### First-checkpoint directory bootstrap

The tracked `checkpoints` directory must already be a real contained directory.
The first checkpoint bootstraps children in this exact order:

1. Open and retain the repository and `checkpoints` descriptors; validate
   containment, non-symlink type, and identities.
2. Create `phase8` with directory-relative `mkdir` mode `0o700`, or validate an
   existing exact directory. Open it no-follow. For a new directory, `fsync`
   its descriptor and then `fsync` the `checkpoints` parent descriptor.
3. Create the exact run-ID directory beneath `phase8` the same way. For a new
   run directory, `fsync` it and then `fsync` the `phase8` parent.
4. Create `objects` beneath the run directory the same way. For a new objects
   directory, `fsync` it and then `fsync` the run-directory parent.
5. Revalidate all retained directory identities. No catalog or object write
   begins until every required directory entry is durably committed.

Concurrent `EEXIST` is accepted only after the existing entry passes the exact
no-follow directory and identity checks. A partial safe bootstrap with no valid
catalog contains no authoritative checkpoint and may be completed by a later
call. An unsafe, uncertain, symlinked, or wrong-type entry is preserved and
refused.

### Epoch object and catalog publication

For each completed epoch:

1. Validate complete checkpoint state and build the new catalog value in memory
   before filesystem mutation.
2. Create one unpredictable, contract-owned temporary regular file beneath the
   retained run descriptor with mode `0o600`, `O_CREAT | O_EXCL | O_NOFOLLOW`.
3. Serialize exactly once, flush, and `fsync` that still-open file descriptor.
   Enforce the 67,108,864-byte maximum with `fstat`.
4. Rewind and read the exact bytes from that same open descriptor to compute
   SHA-256; verify byte count and descriptor identity. No checked path is
   reopened.
5. Promote without clobber using an atomic hard-link creation from the retained
   temporary entry into the retained objects descriptor at
   `<sha256>.pt`. `os.link(..., follow_symlinks=False)` must fail with `EEXIST`
   rather than replace. On `EEXIST`, open the existing object no-follow, enforce
   size, read/hash its same opened bytes, and accept only exact byte identity.
6. Verify the promoted object's regular-file type, link/source identity where
   newly linked, exact size, and digest. Remove the known temporary link only
   after successful promotion or verified identical existing object.
7. **`fsync` the retained `objects` directory before any catalog file is
   created.** This durably orders the immutable object directory entry before
   its reference.
8. Create the canonical catalog temporary file under the retained run descriptor
   with mode `0o600` and exclusive/no-follow flags. Write at most 16,384
   canonical bytes, flush, and `fsync` the still-open catalog file.
9. Atomically replace `checkpoint-catalog.json` with directory-relative
   `os.replace` under the retained run descriptor.
10. `fsync` the retained run directory. Only this final successful directory
    sync makes the new catalog durably authoritative.
11. Reopen nothing by ambient path. Revalidate the retained descriptors and
    catalog/object graph before returning the new reference.

The catalog replacement is the single atomic role commit for both `latest` and
`best_validation`. No catalog is published before object-directory durability.
Orphaned immutable objects are safe, retained, and non-authoritative unless the
durable catalog references their exact digest/path.

### Crash/failure interpretation

- **Before object promotion:** the prior durable catalog remains authoritative;
  a temporary file is non-authoritative and may be removed only when its exact
  ownership/name/identity is known in a live clean failure.
- **After object promotion but before `objects` fsync:** the prior catalog
  remains authoritative. After a crash, the new object entry may exist or be
  absent and is never assumed durable or authoritative.
- **After `objects` fsync but before catalog replacement:** the object is a
  durable orphan; the prior catalog remains authoritative.
- **After catalog replacement but before run-directory fsync:** live publication
  outcome is uncertain and raises `Phase8PublicationError`; files are preserved.
  After process restart, recovery opens the actual catalog no-follow and accepts
  it only if its complete catalog/object graph verifies. A valid old or new
  catalog is then the sole observed authority; missing, malformed, or mismatched
  state requires human adjudication.
- **After final run-directory fsync:** the new catalog and its already durable
  referenced object are authoritative. A later return/diagnostic failure cannot
  roll back publication and must report the committed reference.

An ordinary pre-commit failure preserves the previous catalog. An uncertain
commit never triggers retry, replacement, deletion, or a claim that publication
succeeded.

## Checkpoint load and restoration

Continuation loading accepts role `latest` only and performs exactly:

1. Validate public arguments, exact run-ID grammar, dedicated runtime envelope,
   accepted implementation anchor, externally derived pre-registration commit,
   committed planned-record bytes, and complete live repository authority before
   checkpoint path access. The payload `run.code_commit` must equal the planned
   implementation anchor; live `HEAD` must equal the accepted pre-registration
   commit.
2. Snapshot caller global CPU RNG state as a cloned one-dimensional CPU
   `torch.uint8` tensor. From this step until final commit, no operation may
   mutate caller global CPU RNG.
3. Open/retain repository, checkpoints, Phase 8, run, and objects directory
   descriptors using the no-follow identity protocol.
4. Open the catalog once with no-follow relative to the retained run descriptor,
   enforce regular-file type and 16,384-byte limit with `fstat`, read its exact
   recorded length, and parse those bytes as canonical UTF-8 JSON.
5. Validate the complete exact catalog and both references. Select `latest`;
   reject `best_validation` continuation.
6. Open the exact `objects/<sha256>.pt` once relative to the retained objects
   descriptor with no-follow; enforce regular-file type and the positive
   67,108,864-byte limit with `fstat` before allocation.
7. Read that already-opened object's exact bytes once into a bounded byte
   buffer, rejecting short, long, changing, or identity-mismatched reads. Hash
   this buffer and validate the reference digest. Supply the **same buffer**
   through `io.BytesIO` to `torch.load(map_location="cpu", weights_only=True)`;
   never reopen a checked path for deserialization.
8. Validate every exact payload and nested schema field, primitive/container
   type, tensor type/shape/dtype/device, finite value, authority identity,
   configuration field, optimizer relationship, progress equation, metric
   history, RNG state, mode, and lineage with no missing/extra fields.
9. If `best_validation.sha256` differs from `latest.sha256`, open its exact
   object once through the same retained descriptor, size bound, one-read,
   same-buffer hash/deserialization, schema, authority, metric, and cross-field
   procedure. If the digests are equal, reuse the already validated latest
   buffer/payload. No best-validation model or optimizer is constructed.
10. Perform complete catalog/payload cross-validation for both references,
   including run, logical
   ID, epoch/progress, validation loss hex, role, computed SHA-256, exact relative
   path, current/best metric fields, and best semantics.
11. Load the accepted vocabulary binding and construct one fresh accepted
    MiniGPT from exact configuration; validate parameter identities and leave
    caller global RNG equal to its snapshot.
12. Construct one fresh exact AdamW optimizer over the accepted parameter order.
13. Load model state; validate all 90 names, tensors, raw bytes, shapes, dtype,
    device, finiteness, and absence of buffers/extras.
14. Load optimizer state; validate the exact supported group key order including
    `decoupled_weight_decay=True`, the `0..89` parameter mapping, all policy
    flags, and every exact zero-dimensional CPU-float32 finite integral `step`
    tensor equal to `progress.optimizer_updates`, followed by moment shapes,
    dtype, device, and finiteness.
15. Restore and verify the order generator and all eight exact dropout generator
    states on the fresh local objects.
16. Restore and verify complete training mode, all-`None` gradients, progress,
    metrics, configuration, authority, and truthful resume lineage.
17. Re-run every cross-field relationship and runtime/repository check, then
    construct and completely validate the final `Phase8TrainingState` return
    object while caller global RNG still equals its snapshot.
18. Install the checkpoint global CPU RNG state as the **final caller-visible
    commit action**, immediately read it once, and compare its bytes exactly.
19. If installation or that immediate verification fails, restore the caller's
    original snapshot, verify restoration, and raise `Phase8CheckpointError`
    with invariant `phase8.checkpoint.global_rng`. No ordinary fallible work
    follows a successful verification; return the already packaged object.

Any failure before step 18 leaves caller global CPU RNG exactly equal to its
snapshot. Tests force failures at every late step, including final packaging,
and prove no leak. A forced installation/verification failure proves successful
rollback to the exact caller snapshot. A forced rollback failure raises the
same safe invariant with `state_guarantee="unknown"`, chains the triggering
failure, exposes no RNG bytes, returns no object, and requires process disposal;
ordinary supported tensors must make this path unreachable.

Load is fail-closed. It constructs only fresh local objects and exposes none on
failure. It never mutates an existing caller model or optimizer, falls back to
model-weights-only restoration, resets an invalid optimizer, reseeds a missing
local stream, substitutes live configuration for checkpoint configuration, or
continues from `best_validation`.

## Exact-resume equality

On the supported environment, an uninterrupted control state and a restored
state from the same checkpoint are training-semantically equal only when:

- their complete `Phase8RuntimeIdentity` and live process settings are equal;
- all 90 model parameter names, metadata, and raw bytes are equal;
- optimizer groups, hyperparameters, intrinsic
  `decoupled_weight_decay=True`, parameter mapping, all 90 zero-dimensional
  CPU-float32 step tensors and their equality to `progress.optimizer_updates`,
  first moments, second moments, and every tensor/scalar value are equal;
- completed/next epoch, zero cursor, update/example/target counts, and metric
  history are equal;
- order-generator, all eight dropout-generator, and global CPU RNG state bytes
  are equal;
- complete model/child modes and all-`None` gradient states are equal; and
- current/best metric state and next logical batch identity are equal.

Only these branch-provenance or storage-identity fields may differ:
`resume_lineage`, serialized object SHA-256, and the physical
non-authoritative-audit versus authoritative-catalog location. Object-byte/hash
equality is not a semantic-resume requirement; reloaded fields are. Before
serialization, the two payloads may differ only in `resume_lineage`. Runtime,
run ID, configuration, authority, parameters, optimizer, progress, ordering,
RNG, modes, gradients, metrics, best state, logical epoch identity, and every
numerical value remain exactly equal. No wall-clock value is stored in the
semantic payload.

After equality at the restoration boundary, the proof runs the same next epoch
permutation and the complete next epoch on both branches. It requires, after
every sequence and optimizer step:

- identical ordered window identities;
- bitwise-equal per-sequence logits, scalar mean losses, and scaled losses;
- bitwise-equal accumulated and clipped gradients;
- equal pre-clip global norm;
- bitwise-equal post-update model parameters and optimizer state;
- equal progress and all RNG states; and
- equal complete end-epoch training evaluation, validation evaluation,
  current/best metric update, and semantic checkpoint state.

The focused suite proves the complete boundary on small synthetic inputs. The
first authorized real Phase 8 run performs the actual audit from the epoch-1
`latest` checkpoint:

1. retain the uninterrupted in-memory branch;
2. restore a second branch from `latest` and record truthful resume lineage;
3. create the same epoch-2 permutation in each branch;
4. run every epoch-2 logical batch in both branches, comparing every semantic
   state after each update;
5. run complete post-epoch training then validation evaluation in both;
6. update and compare current/best metric state;
7. construct and compare both complete epoch-2 semantic checkpoint payloads,
   ignoring only the explicitly permitted lineage field;
8. serialize both once and run the crash-durable publication protocol to a
   non-authoritative ignored audit root for the control and the canonical
   checkpoint root for the restored branch;
9. verify both catalog/object graphs and compare their reloaded semantic states,
   permitting only lineage-derived object digest and publication-location
   differences; and
10. discard the uninterrupted control only after the complete epoch-2
    evaluation and checkpoint-boundary proof, then continue epochs 3..10 from
    the restored branch.

The audit-root catalog uses the same exact internal latest/best schema but is
never addressable by `load_phase8_checkpoint` and confers no canonical Phase 8
authority because it is outside `checkpoints/phase8`. Its exact path is
`experiments/<run_id>/artifacts/resume-audit-control/`; it remains ignored and
is recorded by SHA-256 in experiment evidence. Before catalog publication, any
older object needed by its best reference is copied from already verified bytes
through the same immutable-object durability protocol, so its local catalog
graph is complete. This extended audit duplicates one complete epoch, not merely
one batch. It is training work and remains unauthorized until the fixed-run
gate.

## Experiment record and evidence

`EXPERIMENT_LOG.md` remains the append-only durable authority. A future run is
assigned one `EXP-YYYYMMDD-NN` ID during a separately authorized
pre-registration gate. The planned record freezes the implementation commit,
configuration, success rules, feasibility evidence, and run authorization.
Later status/result material is appended; the planned text is never rewritten.
Its `Code commit` is exactly
`809834323d53407cb4a54ae539585bb3d78856eb`. No proposed planned-record field
contains the unknown future pre-registration commit.

The planned entry uses exactly these bold labels in order, with no omitted or
extra field:

```text
Experiment ID
Entry kind                         # planned
Recorded at UTC
Status                             # planned
Question
Authorization
Code commit
Phase 7 authority
Phase 8 contract authority
Runtime identity
Dataset authority
Tokenizer authority
Model configuration
Training configuration
Checkpoint configuration
Feasibility evidence
Success predicate
Sealed-test policy
Generation policy
Next gate
```

The final appended result entry uses exactly these bold labels in order:

```text
Experiment ID
Entry kind                         # completed | failed | stopped
Recorded at UTC
Started at UTC
Finished at UTC
Status
Question
Authorization and predecessor
Code commit
Pre-registration commit
Phase 7 authority
Phase 8 contract authority
Runtime identity
Dataset authority
Tokenizer authority
Model configuration
Training configuration
Window counts
Target counts
Update counts
Initialized losses
Epoch training losses
Epoch validation losses
Gradient-norm evidence
Latest checkpoint
Best-validation checkpoint
Checkpoint publication evidence
Resume audit
Resume lineage
Training-loss improvement predicate
Training-loss improvement result
Fitting evidence
Sealed-test access
Generation and samples
Stopping reason
Observations
Conclusions
Limitations
Next gate
```

Simple values use backticked ASCII literals. Authority, runtime, configuration,
count, checkpoint, and audit values use canonical compact JSON with sorted keys.
Every loss and gradient-norm float is represented by an object with exactly
`decimal: str` and `hex: str`; parsing both must yield the same finite Python
float and `hex` must equal `float.hex()`. UTC values use exact second-resolution
`YYYY-MM-DDTHH:MM:SSZ`. Checkpoint records use the exact checkpoint-reference
schema. `Sealed-test access` is exactly `none`; `Generation and samples` is
exactly `none`. A failed/stopped result uses explicit `null` only for checkpoint
or metric fields that were never durably produced and identifies the last valid
catalog reference. No generated prose is treated as a machine authority.

In the result entry, `Code commit` repeats the planned implementation anchor and
`Pre-registration commit` is the exact lowercase 40-hex live launch `HEAD`
established by the pre-registration commit/push/remote-verification gates. This
later result field records the external authority without rewriting the planned
record. Checkpoint identities and resume-audit evidence are reviewed together
with both commits; checkpoint payload `run.code_commit` remains the
implementation anchor.

Machine-generated event details may be stored under ignored
`experiments/<run_id>/artifacts/`, but they are subordinate to the checkpoint
hashes and append-only reviewed `EXPERIMENT_LOG.md` record.

### Phase 8 success predicate

The exact loss-improvement clause is:

```text
min(training_loss at completed epochs 1..10) < initialized training_loss
```

All recorded primary losses must be finite. Phase 8 also requires all four
roadmap criteria plus every DEC-0020 item: deterministic ordering,
supported-environment reproducibility, complete loss history, checkpoint
save/load correctness, retained latest and best checkpoints, complete state
restoration, passing exact-resume audit, and one fully documented run.

Validation improvement is not required for Phase 8 PASS. Best-validation
selection still uses validation throughout. No loss threshold authorizes a
generalization or generation-quality claim.

### Fitting evidence

The record reports, without tuning or early stopping:

- initialized, minimum, final, and per-epoch training loss;
- initialized, minimum, best-epoch, final, and per-epoch validation loss;
- per-epoch validation-minus-training generalization gap;
- whether training continued improving after the best validation epoch;
- whether validation worsened after its best epoch; and
- comparison with `ln(81)` as explanatory context only.

Underfitting or overfitting language must be explicitly described as evidence,
not diagnosis certainty. Falling training loss with flat or rising validation
loss is overfitting evidence. Both losses remaining high or nearly flat is
underfitting or optimization-difficulty evidence. Both falling is evidence of
useful fitting to train with transfer to validation, not sealed-test proof.

## Numerical and failure policy

All inputs, logits, losses, weighted scalars, gradients, gradient norms,
parameters, optimizer tensor state, aggregate metrics, and checkpoint floats
must be finite at their defined boundaries. Existing MiniGPT and explicit-loss
exceptions retain ownership inside their accepted calls.

Phase 8 adds exactly these content-safe exception categories:

- `Phase8TypeError(TypeError)` for wrong public, exact built-in, tensor,
  generator, model, optimizer, record, or child-state types;
- `Phase8ContractError(ValueError)` for invalid typed configuration, shape,
  count, order, progress, mode, gradient, optimizer, or lifecycle state;
- `Phase8GovernanceError(ValueError)` for corpus identity, split, work,
  authority, or sealed-boundary violations;
- `Phase8NumericalError(ArithmeticError)` for nonfinite or unrepresentable
  training/evaluation arithmetic and nonfinite parameter/optimizer state;
- `Phase8CheckpointError(ValueError)` for schema, hash, path, payload,
  catalog, authority, or restoration failures before an uncertain commit; and
- `Phase8PublicationError(RuntimeError)` for an atomic-publication outcome that
  cannot be classified safely.

Every Phase 8 exception constructor accepts exactly one positional
`invariant: str` followed by safe keyword facts, exposes an immutable `details`
mapping, and renders deterministic ASCII JSON with sorted keys. The exact
invariant literals are:

```text
phase8.type.argument
phase8.type.record
phase8.type.tensor
phase8.type.generator
phase8.type.model
phase8.type.optimizer
phase8.contract.configuration
phase8.contract.runtime
phase8.contract.window
phase8.contract.order
phase8.contract.batch
phase8.contract.optimizer
phase8.contract.gradient
phase8.contract.progress
phase8.contract.mode
phase8.contract.metric
phase8.contract.lifecycle
phase8.governance.repository
phase8.governance.dataset
phase8.governance.split
phase8.governance.work
phase8.governance.sealed_test
phase8.numerical.loss
phase8.numerical.gradient
phase8.numerical.global_norm
phase8.numerical.parameter
phase8.numerical.optimizer
phase8.numerical.metric
phase8.checkpoint.role.continuation
phase8.checkpoint.path
phase8.checkpoint.size
phase8.checkpoint.catalog
phase8.checkpoint.reference
phase8.checkpoint.hash
phase8.checkpoint.payload
phase8.checkpoint.schema
phase8.checkpoint.authority
phase8.checkpoint.configuration
phase8.checkpoint.cross_field
phase8.checkpoint.restore
phase8.checkpoint.global_rng
phase8.publication.bootstrap
phase8.publication.object
phase8.publication.objects_fsync
phase8.publication.catalog
phase8.publication.run_fsync
phase8.publication.uncertain
```

No implementation-defined invariant string is public. Field-, argument-,
position-, stage-, role-, and expected/observed facts distinguish failures
within one literal. Type literals belong only to `Phase8TypeError`; contract
literals only to `Phase8ContractError`; governance literals only to
`Phase8GovernanceError`; numerical literals only to `Phase8NumericalError`;
checkpoint literals only to `Phase8CheckpointError`; and publication literals
only to `Phase8PublicationError`.

### Public-operation validation stages

The exact major stages are:

1. `configure_phase8_runtime`: verify dedicated-process eligibility; set
   intra-op, inter-op, deterministic/warn-only, matmul, default dtype/device,
   and MKLDNN in documented order; verify grad/autocast/inference modes; build
   and validate runtime identity.
2. `validate_phase8_runtime`: validate every live setting in runtime-field
   order; derive build/parallel/lock hashes; construct identity; compare the
   recorded identity when supplied by the enclosing operation.
3. `load_phase8_configuration`: validate runtime; construct exact fixed fields
   in record order; validate relationships; return the factory-marked record.
4. `build_phase8_windows`: validate argument types in signature order; split;
   verified vocabulary; manifest authority; exact work count/types/metadata in
   order; only then permitted text/hash/encoding; then window arithmetic,
   identities, coverage, and result packaging.
5. `create_phase8_epoch_order`: validate argument types; positive population;
   epoch range; exact local CPU generator type/device/state ownership; snapshot
   unrelated RNG; one `randperm`; exact permutation; allowed state advancement;
   package result.
6. `iter_phase8_logical_batches`: validate exact containers and epoch; complete
   window records; order length/range/uniqueness; then grouping, target counts,
   final-partial behavior, and iterator creation before yielding.
7. `create_phase8_optimizer`: validate runtime, exact MiniGPT type/structure,
   modes, parameter names/order/types/shapes/device/dtype/finiteness/uniqueness;
   construct AdamW once; validate exact group/state-before-first-step.
8. `run_phase8_logical_batch`: apply the eleven update stages already listed,
   with inherited MiniGPT/loss exceptions retaining ownership inside calls.
9. `evaluate_phase8_split`: validate all arguments and runtime; split/window
   authority; all-None gradients; snapshot semantic state; eval/no-grad
   traversal; aggregate; restore train mode in `finally`; validate complete
   nonmutation; package result.
10. `save_phase8_checkpoint`: validate argument types, runtime, accepted
    implementation anchor, external pre-registration commit, committed planned
    record, and live repository; exact training-state record; configuration/model/optimizer;
    progress/metrics; modes/gradients; RNGs/lineage; complete payload/schema and
    cross-fields; then bootstrap and atomic publication stages.
11. `load_phase8_checkpoint`: use the exact nineteen-stage transactional load
    order in its section; no alternate order is accepted.
12. `run_fixed_phase8_experiment`: validate arguments, runtime, live repository,
    planned implementation-code/config authority, implementation ancestry and
    accepted bytes, external live-HEAD pre-registration authority, and committed
    planned-record bytes before permitted corpus and vocabulary text; build
    windows; initialized evaluation; ten epoch lifecycles; epoch-1 resume audit;
    checkpoint/result completeness; success predicate; package one final result.

Diagnostics may expose invariant names, indices, shapes, counts, hashes, fixed
relative paths, and configuration facts. They never expose text, token IDs,
tensor values, parameters, gradients, logits, RNG bytes, absolute corpus paths,
or arbitrary object reprs. Existing inherited exceptions retain ownership
inside accepted component calls.

A numerical or optimizer-state failure stops immediately, publishes no metric
or checkpoint for the incomplete batch/epoch, and preserves the previous
catalog. No automatic learning-rate change, clipping-threshold change, skipped
update, reset, retry, NaN replacement, checkpoint fallback, or configuration
change is permitted.

## Focused test obligations

Independent tests must derive critical expectations without importing the
production constants or helpers under test.

### Authority and boundaries

1. Exact Phase 8 public signatures, exports, immutable records, error surfaces,
   and fixed configuration are independently verified.
2. Accepted Phase 7 source/tests and earlier accepted contracts remain
   byte-identical; no rank-two model path or architecture change exists.
3. Corpus access constructs exactly six training works and *The Tempest* only,
   with metadata-before-text validation and zero sealed supplier/read path.
4. Static checks reject generation, sampling, test split, MPS/CUDA, mixed
   precision, padding/special tokens, pretrained APIs, schedulers, warmup, and
   random-replacement sampling.

### Windows and ordering

5. Synthetic documents prove starts, length 256, variable tail, exact shift,
   exact-once transition coverage, and no work-boundary crossing.
6. Empty/one-token documents produce zero windows; 2, 257, 258, 513, and
   multi-work boundaries prove exact edge behavior.
7. Canonical ordering and literal safe identities are exact.
8. Same seed/state produces identical epoch permutations; every result is a
   complete permutation; exact order-generator state progression is proved; and
   global/model RNG states do not change. Fixed small populations use literal
   expected PyTorch-2.14.0 permutations for seed 8001 rather than a probabilistic
   "successive permutations differ" assertion.
9. Checkpointed order state reproduces the next permutation and zero cursor.

### Logical batching and optimization

10. Groups of 0, 1, 7, 8, 9, and larger counts prove capacity and final partial
    behavior without omission or duplication.
11. Unequal synthetic sequence lengths prove the accumulated gradient equals
    an independently calculated target-token mean and is not an example mean
    or a second division by eight.
12. Every sequence uses one rank-one forward and one accepted loss call in
    exact batch order; one optimizer step follows the complete batch.
13. Actual supported PyTorch-2.14.0 AdamW construction and `state_dict()` match
    the literal group oracle exactly, including key order,
    `decoupled_weight_decay=True`, 90-parameter order, all DEC-0020 settings,
    uniform decay, and no scheduler.
14. All-None gradient entry, accumulated gradient validation, and the documented
    PyTorch-2.14.0 `1 / (norm + 1e-6)` clipping formula are proved for clearly
    below, epsilon-boundary, exactly-equal, and above-1.0 norms, followed by
    exact call ordering, one step, post-step clearing, state finiteness, and
    counter increments.
15. Failures at every pre-step stage update no parameters or optimizer state;
    post-step failure never publishes a completed boundary or retries.

### Evaluation and metrics

16. Epoch 0 and every completed epoch perform full train then validation
    evaluation in canonical order with target-token weighting.
17. Evaluation uses eval/no-grad, restores train mode, leaves gradients None,
    and preserves parameters, optimizer, progress, order/global RNG, and all
    eight dropout states exactly on success.
18. Forced failures at entry, within each split, aggregation, mode restoration,
    and result packaging publish no metric/checkpoint and preserve the defined
    recoverable state.
19. Strict best comparison, tie behavior, epoch-0 exclusion, initialized
    baseline, loss-improvement predicate, and fitting-evidence arithmetic are
    independently verified.

### Checkpoint and resume

20. Exact schema, primitive-only payload, all 90 model tensors, and complete
    AdamW state are present with no extras. Every optimizer entry has an exact
    zero-dimensional finite CPU-float32 integral `step` equal to
    `progress.optimizer_updates`; nested `configuration.schema_version` is the
    first key and equals exact integer 1; config, progress, counts, metrics,
    mode, lineage, and all ten RNG states are complete.
21. Ordinary model `state_dict` alone is explicitly rejected as incomplete.
22. Object serialization happens once; exact bytes determine SHA-256 and the
    immutable filename; catalog JSON is canonical.
23. Path traversal, symlinked components, wrong file types, hash mismatch,
    malformed catalog/payload, unsupported objects, authority/config mismatch,
    missing/extra states, nonfinite values, and inconsistent progress fail
    before any restored state is returned.
24. Failure injection before/after object promotion and before/during/after
    catalog replacement proves the old catalog remains authoritative or the
    uncertain state is preserved without guessing.
25. Latest and best roles update atomically in one catalog; strict best/tie
    cases and retention of prior objects are verified.
26. Fresh construction and restoration order are observed exactly; restoration
    preserves `decoupled_weight_decay=True` and all 90 exact step tensors/count
    relationships; global RNG is restored last; load failure leaks no partially
    restored caller state.
27. Static and dynamic checks prove no token/window/activation/logit/gradient or
    sealed content enters a checkpoint.
28. Uninterrupted-versus-restored controls prove every semantic equality field,
    including exact AdamW group keys, `decoupled_weight_decay`, step tensor
    metadata/value/progress equality, next permutation, next logical batch,
    per-sequence result, accumulated and clipped gradients, AdamW update,
    parameters, optimizer state, RNG states, progress, and boundary metrics
    bitwise on the supported environment.
29. Resume lineage differs truthfully without weakening training-state
    equality.

### Lifecycle and regression

30. The fixed runner enforces initialized evaluation, ten exact epochs, full
    end-epoch evaluation, checkpoint publication, epoch-1 restoration followed
    by the complete dual-branch epoch-2/evaluation/checkpoint audit, latest/best
    retention, success predicate, stop behavior, and complete result ordering
    without retry or tuning.
31. A bounded synthetic failure matrix proves no partial result can be mistaken
    for a completed run.
32. Existing Phase 3 embedding/vocabulary, Phase 4 loss, Phase 5 attention,
    Phase 6 block, and Phase 7 MiniGPT suites remain passing and byte-identical
    where required.
33. Manual review proves no Phase 9 generation/evaluation or sealed-test access.
34. First-checkpoint bootstrap proves exact directory creation order, child and
    parent fsyncs, safe concurrent `EEXIST`, and refusal of symlink/wrong-type or
    uncertain directory entries.
35. Publication instrumentation proves hard-link no-clobber promotion, object
    verification, `objects` directory fsync before catalog temporary creation,
    catalog-file fsync, atomic replacement, and final run-directory fsync.
36. Crash/failure injection covers before promotion, after promotion before
    objects fsync, after objects fsync before catalog replace, after replace
    before run fsync, and after final fsync, including exact authoritative-state
    and orphan interpretation.
37. Load failures forced after local RNG/model/optimizer/mode/progress restoration
    and during final packaging leave caller global CPU RNG bitwise unchanged;
    final global install/verification failure restores the caller snapshot.
38. Every frozen runtime setting has exact match and one-at-a-time mismatch
    evidence; unsafe/too-late inter-op configuration and boundary grad/autocast/
    inference mismatches are refused before returned state.
39. One literal independent oracle verifies every record field/order/type/
    optionality/repr rule, public signature/annotation/kind, exact owning module,
    every module `__all__`, package re-export/`__all__`, runtime-operation and
    six-exception ownership, invariant, validation stage, checkpoint/catalog/
    reference key, configuration schema-version presence/order/value, rejection
    of missing/unknown versions, planned/result experiment-record labels
    including result-only `Pre-registration commit`, and no production imports.
40. Catalog/payload negatives independently alter run ID, logical ID, epoch,
    progress, validation-loss hex, role, digest, relative path, current/best
    metrics, and best identity while retaining otherwise valid bytes; every case
    fails before restoration.
41. Run/logical/hash/relative-path grammar tests reject traversal, separators,
    alternate equivalent paths, Unicode/case variants, wrong ranges, absolute
    paths, and symlinked roots/components/final files.
42. Descriptor instrumentation proves no-follow directory-relative opens,
    retained device/inode checks, and no ambient-path reopening across write,
    promotion, catalog, load, hash, or deserialize boundaries.
43. Load hashes and deserializes the exact same already-opened bounded byte
    buffer. Catalog and object sizes at limit pass; zero, over-limit, growing,
    shrinking, and short-read cases fail before unbounded allocation.
44. Security tests document that only trusted local project artifacts are
    accepted and that `weights_only=True` is not represented as a hostile-file
    or resource-exhaustion sandbox.
45. Live repository and pre-registration tests prove independently that the
    planned `Code commit` equals accepted implementation commit
    `809834323d53407cb4a54ae539585bb3d78856eb`; that commit is an ancestor of
    run-launch `HEAD`; live `HEAD` equals the externally accepted, remotely
    verified pre-registration commit and refreshed `origin/main`; the planned
    record is committed there and did not require its own future SHA; all latest
    accepted implementation/source/test/spec hashes match; an unapproved
    source/test change after implementation authority fails; wrong planned
    `Code commit`, wrong/missing pre-registration authority, and dirty tracked
    state each fail with the exact corrected invariant/field before corpus,
    model, optimizer, generator, or training-state construction.
46. Continuation succeeds only for canonical `latest`; a valid older
    `best_validation` reference is integrity-checkable but rejected for Phase 8
    training continuation without constructing returned training state.
47. The real-run resume oracle retains both branches through the complete next
    epoch, full train/validation evaluation, metric/best update, serialized
    semantic payload, durable object/catalog boundary, reload, and complete
    semantic comparison; both share one externally validated pre-registration
    authority and only enumerated lineage/hash/location fields differ.
48. A literal gate oracle proves every contract, implementation, feasibility,
    pre-registration, run, result/checkpoint acceptance, exit, and closure
    authorization/action/verification gate is distinct and sequential.

Real-corpus feasibility measurement and training are never unit tests and
remain separately authorized actions.

## Phase 8 / Phase 9 boundary

Phase 8 may train only on the six accepted training works, evaluate only on
*The Tempest*, select the best checkpoint by validation loss, save and restore
complete training state, record fitting evidence, and prove exact resume.

Phase 8 does not generate or sample text, implement generation controls,
produce qualitative samples, access/tokenize/window/score/inspect *Twelfth
Night*, make final sealed-test or generalization claims, or perform Phase 9
evaluation. Sealed-test access remains separately authorized even after Phase
9 begins.

## Feasibility boundary

No runtime feasibility evidence exists for context 256, logical batch capacity
8, CPU float32, four blocks, and ten epochs. After accepted implementation and
only under specific authorization, a bounded feasibility measurement may
observe time and memory without silently changing any setting. If the approved
configuration is impractical, work stops and reports evidence to Master Chat.
A new explicit decision is required before changing context, batch size, model,
optimizer, device, or training policy.

## Proposed Phase 8 gates

1. Phase 7 accepted implementation and exit acceptance. **Complete.**
2. Phase 7 closure commit `33d4510` pushed and remote state synchronized.
   **Complete.**
3. Explicit Phase 8 learning/design authorization. **Complete.**
4. Read-only Phase 8 continuity inspection. **Complete.**
5. Consolidated DEC-0020 conceptual policy acceptance. **Complete.**
6. Documentation-only detailed-contract drafting and stale Phase 7 closure
   reconciliation. **Complete locally; proposed and unaccepted.**
7. Master Chat detailed-contract sanity review. **Complete.**
8. Fresh independent detailed-contract review. **Complete: CORRECT BEFORE
   ACCEPTANCE with three BLOCKER, eight IMPORTANT, and one MINOR finding.**
9. Master Chat finding adjudication. **Complete: all twelve findings accepted
   without changing DEC-0020.**
10. Separate bounded documentation-correction authorization. **Complete.**
11. Apply only the authorized detailed-contract corrections. **Complete locally;
    corrected proposal remains unaccepted.**
12. Focused independent detailed-contract re-review. **Complete: CORRECT AGAIN
    BEFORE ACCEPTANCE with one BLOCKER, two IMPORTANT, and no MINOR findings.**
13. Master Chat second finding adjudication. **Complete: all three accepted
    without changing DEC-0020.**
14. Separate final targeted documentation-correction authorization. **Complete.**
15. Apply only the three authorized final contract corrections. **Complete
    locally; proposal remains unaccepted.**
16. Final focused independent detailed-contract re-review.
17. Master Chat final detailed-contract adjudication.
18. Sebastien's explicit detailed-contract acceptance.
19. Separate accepted-contract commit authorization.
20. Accepted-contract commit creation without amendment or extra files.
21. Separate accepted-contract push authorization.
22. Accepted-contract push.
23. Independent remote verification of the accepted-contract commit.
24. Separate explicit Phase 8 implementation authorization.
25. Phase 8 source and focused synthetic tests only.
26. Master Chat implementation sanity review.
27. Fresh independent implementation review.
28. Master Chat implementation-finding adjudication.
29. Separate bounded implementation-correction authorization if required.
30. Apply only authorized source/test corrections if required.
31. Focused independent implementation re-review if corrections occurred.
32. Master Chat final implementation adjudication.
33. Sebastien's explicit implementation acceptance.
34. Separate accepted-implementation commit authorization.
35. Accepted-implementation commit creation without amendment or extra files.
36. Separate accepted-implementation push authorization.
37. Accepted-implementation push.
38. Independent remote verification of accepted implementation.
39. Separate bounded feasibility-measurement authorization.
40. Execute only the authorized feasibility measurement.
41. Fresh independent feasibility-evidence review.
42. Master Chat feasibility adjudication; stop for a new decision if
    impractical.
43. Sebastien's explicit feasibility result acceptance and authorization to
    proceed toward pre-registration.
44. Stop at the discovered pre-registration self-reference; Master Chat
    adjudication and separate documentation-only correction authorization.
    **Complete.**
45. Apply only the authorized non-circular provenance-contract correction.
    **Complete locally; corrected contract remains unaccepted.**
46. Fresh independent focused review of the provenance correction.
47. Master Chat provenance-correction adjudication.
48. Sebastien's explicit corrected-contract acceptance.
49. Separate corrected-contract documentation commit authorization.
50. Corrected-contract documentation commit creation without amendment or
    extra files.
51. Separate corrected-contract push authorization.
52. Corrected-contract push.
53. Independent remote verification of the corrected-contract commit.
54. Separate provenance-only implementation/test correction authorization.
55. Apply only the accepted runner, checkpoint-provenance, result-schema, and
    focused-test changes required by this correction.
56. Fresh independent focused implementation-correction review.
57. Master Chat implementation-correction adjudication.
58. Sebastien's explicit provenance implementation-correction acceptance.
59. Separate provenance implementation-correction commit authorization.
60. Provenance implementation-correction commit creation without amendment or
    extra files.
61. Separate provenance implementation-correction push authorization.
62. Provenance implementation-correction push.
63. Independent remote verification of the latest accepted implementation and
    exact source/test hashes.
64. Separate experiment pre-registration proposal authorization.
65. Append only the planned `EXPERIMENT_LOG.md` record. Its `Code commit` is
    the accepted implementation anchor; it contains no pre-registration SHA.
66. Fresh independent pre-registration review.
67. Master Chat pre-registration adjudication.
68. Sebastien's explicit pre-registration acceptance.
69. Separate pre-registration commit authorization.
70. Pre-registration commit creation without amendment or extra files. This
    commit becomes external `pre_registration_commit` authority.
71. Separate pre-registration push authorization.
72. Pre-registration push.
73. Independent remote verification of the clean pre-registration/run-launch
    authority commit and exact planned-record bytes.
74. Separate fixed Phase 8 run authorization naming that verified authority.
75. Execute exactly one fixed run at live `HEAD == pre_registration_commit`,
    including checkpointing and the complete dual-branch epoch-2 resume audit;
    no retry or tuning.
76. Separate experiment-result recording authorization.
77. Append only the exact result, external `Pre-registration commit`, and
    checkpoint identities to `EXPERIMENT_LOG.md`.
78. Fresh independent experiment, checkpoint, exact-resume, provenance, and
    Phase 8 exit review.
79. Master Chat result/checkpoint/exit adjudication.
80. Sebastien's explicit experiment, checkpoint, and technical Phase 8 exit
    acceptance.
81. Separate accepted-result documentation commit authorization.
82. Accepted-result commit creation without amendment or extra files.
83. Separate accepted-result push authorization.
84. Accepted-result push.
85. Independent remote verification of the accepted result record.
86. Sebastien's separate Phase 8 closure-state acceptance.
87. Separate documentation-only closure-bookkeeping authorization.
88. Apply only authorized closure bookkeeping.
89. Separate Phase 8 closure-commit authorization.
90. Phase 8 closure-commit creation without amendment or extra files.
91. Separate Phase 8 closure-push authorization.
92. Phase 8 closure push.
93. Independent final remote verification of the Phase 8 closure commit.
94. Only then consider separate Phase 9 learning/design authorization.

No gate silently authorizes a later gate. Review findings authorize no change
until Master Chat accepts them and Sebastien authorizes the bounded correction.

## Current authorization boundary

DEC-0020, the accepted training policy, and the accepted implementation remain
unchanged. The non-circular provenance-contract correction at Gate 45 is
complete locally and remains unaccepted, unstaged, uncommitted, and unpushed
pending Gate 46 fresh focused independent review. No pre-registration record,
production-source/test correction, corpus run, training, evaluation, checkpoint,
exact-resume audit, generation, sampling, sealed-test access, staging, commit,
push, or Phase 9 work is authorized.
