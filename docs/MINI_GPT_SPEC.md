# SebGPT Phase 7 Complete Mini-GPT Specification

## Status and authority

Phase 7 is titled **Complete Mini-GPT**. Its repository goal is exactly:

> Assemble a decoder-only autoregressive Transformer.

Its exact repository exit criteria are:

1. Embeddings, stacked blocks, final normalization, and language-model head are
   integrated.
2. Parameter count and configuration are explicit.
3. Forward pass, loss path, causality, and shape behavior are tested.
4. The architecture is documented and matches recorded decisions.

Sebastien accepted the consolidated conceptual architecture recorded in
DEC-0019 and authorized this documentation-only detailed-contract proposal.
Fresh independent review returned CORRECT BEFORE ACCEPTANCE with five IMPORTANT
findings and no BLOCKER. Master Chat accepted all five without changing
DEC-0019 and authorized the targeted documentation corrections incorporated
here. Focused independent re-review subsequently returned PASS with all five
findings resolved and no BLOCKER, IMPORTANT, or MINOR findings. Codex recommended
ACCEPT CORRECTED CONTRACT, and Master Chat formally accepted this corrected
detailed contract. The mechanics below are authoritative for later separately
authorized Phase 7 implementation.

Acceptance authorizes no source, tests, parameter construction, implementation,
training, experiment, sealed-test access, staging, commit, or push. The next
gate is separate accepted-contract checkpoint commit authorization.

## Accepted conceptual architecture

The complete model processes exactly one nonempty token-ID sequence:

```text
token IDs (T,)
    -> accepted Phase 3 token-plus-position embedding (T, 32)
    -> four distinct accepted Phase 6 Transformer blocks (T, 32)
    -> one final explicit LayerNorm-style normalization (T, 32)
    -> one untied affine language-model head
    -> logits (T, 81)

1 <= T <= 256
```

The four blocks have independent parameter objects and deterministic but
non-identical initial randomized weights. Each retains the accepted width-32,
four-head-by-width-8, pre-norm, `32 -> 128 -> 32` FFN, two-residual, and
two-dropout mathematical structure.

The final normalization uses the same explicit width-32 LayerNorm-style
mathematical form already accepted in Phase 6. It occurs exactly after block 3
and before the language-model head and owns an independent learned `gamma` and
`beta`.

The language-model head is untied. It owns a distinct `(32, 81)` output weight
and `(81,)` bias. It does not alias the accepted token embedding table.

Ordinary forward accepts token IDs and returns logits only. Loss is a separate
operation. Complete-model inspection exposes the embedding representation,
same-call inspection from every block, the final-normalization inspection, and
the logits from that same computation.

## Inherited authority and phase boundary

Phase 7 preserves, rather than redefines:

| Field | Accepted value |
|---|---|
| Tokenizer | `shakespeare-code-point-v1`, schema `1` |
| Vocabulary | size `81`, no special tokens, accepted artifact identity |
| Representation | accepted `TokenPositionEmbedding` |
| Token table | `(81, 32)` |
| Learned absolute-position table | `(256, 32)` |
| Representation seed | `1337` |
| Model width | `32` |
| Attention | four separately visible width-8 causal heads per block |
| Attention visibility | position `i` may use exactly `j <= i` |
| Block topology | accepted sequential pre-norm Phase 6 topology |
| FFN | accepted positionwise `32 -> 128 -> 32` exact-erf GELU path |
| Block dropout | accepted `p = 0.1` at exactly two branch outputs |
| Block parameters | 21 tensors and 12,576 elements per block |
| Semantic device/dtype | CPU `torch.float32`; token/target IDs `torch.long` |

Phase 4's `SimpleNeuralLanguageModel` and experiment remain closed. Phase 7
does not instantiate, wrap, resume, or modify that baseline. It reuses only the
explicit untied affine-head concept and the accepted standalone
`explicit_cross_entropy` operation where this specification says so.

Phase 7 does not introduce production batching, optimization, corpus training,
evaluation scheduling, training/runtime stochastic reproducibility, or
checkpoint/resume behavior. Phase 8 owns those concerns. Phase 7 does not
introduce generation, sampling, qualitative output, final evaluation, or
sealed-test access. Phase 9 owns those concerns.

## Proposed complete-model constants

```text
VOCABULARY_SIZE = 81
MODEL_WIDTH = 32
MAX_SEQUENCE_LENGTH = 256
NUMBER_OF_BLOCKS = 4
NUMBER_OF_HEADS = 4
HEAD_WIDTH = 8
HIDDEN_WIDTH = 128
LAYER_NORM_EPSILON = 1e-5
DROPOUT_PROBABILITY = 0.1
EMBEDDING_INITIALIZATION_SEED = 1337
PHASE5_ATTENTION_CONSTRUCTION_SEED = 5005
PHASE6_PARAMETER_CONSTRUCTION_SEED = 6006
ATTENTION_DROPOUT_SEED = 6007
FEED_FORWARD_DROPOUT_SEED = 6008
BLOCK_PARAMETER_INITIALIZATION_SEEDS = (7001, 7002, 7003, 7004)
OUTPUT_HEAD_INITIALIZATION_SEED = 7005
PARAMETER_DEVICE = torch.device("cpu")
PARAMETER_DTYPE = torch.float32
```

The values `5005`, `6006`, `6007`, and `6008` remain the exact accepted
arguments required to construct each unchanged Phase 6 block. Seeds
`7001..7004` are Phase 7-owned final learned-parameter initialization streams
for blocks 0 through 3. Seed `7005` owns the new output head. The Phase 7
numbers are mnemonic, ordered, distinct, and make ownership inspectable; they
make no claim of statistical superiority.

Final normalization has no random seed because its scale starts at ones and
its bias at zeros. Dropout generator state is runtime stochastic state, not
parameter initialization. Phase 7 does not replace, reseed, synchronize,
advance, or otherwise manage the accepted per-block dropout generators.

## Proposed module and file boundary

Future separately authorized implementation belongs in:

```text
src/sebgpt/model/mini_gpt.py
tests/test_mini_gpt.py
```

The proposed public API is:

```text
@dataclass(frozen=True)
MiniGPTInspection:
    embedding_representation: torch.Tensor = field(repr=False)
    block_inspections: tuple[TransformerBlockInspection, ...] = field(repr=False)
    final_normalization: LayerNormInspection = field(repr=False)
    logits: torch.Tensor = field(repr=False)

LanguageModelHead(
    *,
    model_width: int,
    vocabulary_size: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
)

public parameters:
    output_weight: torch.nn.Parameter  # (32, 81)
    output_bias: torch.nn.Parameter    # (81,)

public method:
    forward(representations: torch.Tensor) -> torch.Tensor

MiniGPT(
    vocabulary: VocabularyBinding,
    *,
    model_width: int,
    max_sequence_length: int,
    number_of_blocks: int,
    number_of_heads: int,
    head_width: int,
    hidden_width: int,
    layer_norm_epsilon: float,
    dropout_probability: float,
    embedding_seed: int,
    block_parameter_seeds: tuple[int, int, int, int],
    output_head_seed: int,
    device: torch.device,
    dtype: torch.dtype,
)

public direct child modules, in registration order:
    representation: TokenPositionEmbedding
    blocks: torch.nn.ModuleList  # exactly four TransformerBlock instances
    final_norm: ExplicitLayerNorm
    head: LanguageModelHead

public methods:
    forward(token_ids: torch.Tensor) -> torch.Tensor
    inspect(token_ids: torch.Tensor) -> MiniGPTInspection

separate accepted loss operation:
    explicit_cross_entropy(logits, targets) -> scalar loss
```

All constructor arguments are required and have no defaults. Exact visible
arguments preserve configuration authority without implying support for other
values. `block_parameter_seeds` must be an exact built-in tuple of four exact
built-in integers and must equal `(7001, 7002, 7003, 7004)`.

`LanguageModelHead` is a small explicit Phase 7 component, not a reuse of
`SimpleNeuralLanguageModel`. It performs only the visible affine map. No
softmax, target access, loss, training, or sampling occurs inside it.

The head constructor first validates exact argument types in signature order,
then exact values `32`, `81`, `7005`, CPU, and float32 in signature order, and
only then creates its local generator or tensors. `bool` is rejected where an
exact integer is required.

## Proposed exact public export oracle

`src/sebgpt/model/mini_gpt.py` defines `__all__` with exactly these new Phase 7
names in this order:

```text
LanguageModelHead
MiniGPT
MiniGPTContractError
MiniGPTInspection
MiniGPTNumericalError
MiniGPTTypeError
```

`src/sebgpt/model/__init__.py` imports and appends exactly the same six new names,
in the same order. Its complete resulting package-level `__all__` is exactly:

```text
AttentionContractError
AttentionInspection
AttentionNumericalError
AttentionTypeError
EmbeddingContractError
EmbeddingTypeError
DropoutInspection
ExplicitDropout
ExplicitLayerNorm
FeedForwardInspection
LayerNormInspection
AggregateMeasurement
Phase4ContractError
Phase4ExperimentConfig
Phase4ExperimentResult
Phase4GovernanceError
Phase4PermittedCorpus
Phase4TypeError
PositionwiseFeedForward
MultiHeadCausalSelfAttention
SimpleNeuralLanguageModel
SingleHeadCausalSelfAttention
TokenPositionEmbedding
TrainingPassResult
TransformerBlock
TransformerBlockContractError
TransformerBlockInspection
TransformerBlockNumericalError
TransformerBlockTransactionError
TransformerBlockTypeError
clear_gradients
explicit_cross_entropy
explicit_gelu
load_phase4_experiment_config
load_phase4_permitted_corpus
manual_sgd_step
measure_example_loss
measure_aggregate_loss
model_parameter_digest
probabilities_from_logits
run_fixed_phase4_experiment
run_training_pass
scale_backward_loss
validate_phase4_experiment_preflight
LanguageModelHead
MiniGPT
MiniGPTContractError
MiniGPTInspection
MiniGPTNumericalError
MiniGPTTypeError
```

No existing export is removed, duplicated, or reordered. The inherited
`explicit_cross_entropy` remains available once and is not redefined or added
as a new Phase 7 export. No Phase 7 constant, private initializer, configuration
helper, or inherited component is newly exported.

## Proposed constructor validation and construction order

`MiniGPT` validates every caller-owned argument before constructing a child,
generator, tensor, or parameter. The order is:

1. exact `VocabularyBinding` type;
2. exact built-in `int` types for model width, maximum sequence length, block
   count, head count, head width, hidden width, and embedding seed;
3. exact built-in `float` types for epsilon and dropout probability;
4. exact built-in tuple type for block seeds, exact length four, then exact
   built-in `int` type for each entry in index order;
5. exact built-in `int` type for output-head seed;
6. exact `torch.device` type and `torch.dtype` instance;
7. accepted factory verification of the vocabulary binding;
8. exact values in public-signature order;
9. pairwise distinct block seeds and distinction from all inherited parameter
   seeds and the output-head seed; and
10. only then child construction and initialization.

After validation, construction order is:

1. accepted representation;
2. block 0, followed immediately by its Phase 7 in-place initialization;
3. block 1 and its initialization;
4. block 2 and its initialization;
5. block 3 and its initialization;
6. final normalization; and
7. language-model head.

The model registers no direct parameters or buffers. Every trainable parameter
belongs to one of those children. Vocabulary identity and configuration values
are immutable non-tensor metadata and are not registered buffers.

## Proposed conservative distinct-block initialization

### Integration problem

The accepted `TransformerBlock` constructor accepts only the exact Phase 5/6
seeds `5005`, `6006`, `6007`, and `6008`. Calling it four times therefore
creates four structurally independent blocks whose initial learned values are
bitwise identical. Generalizing that constructor or changing Phase 5/6 source
would reopen accepted work and is forbidden.

### Resolution

For each block index `b`:

1. Construct one exact accepted `TransformerBlock` using width 32, four heads,
   head width 8, hidden width 128, maximum length 256, epsilon `1e-5`, dropout
   `0.1`, and exact seeds `5005`, `6006`, `6007`, and `6008`.
2. Preserve the block object, all six direct children, all nested head objects,
   every `torch.nn.Parameter` object, both dropout objects, and both dropout
   generator objects and their complete current states.
3. Create one Phase 7 local CPU generator and seed it with
   `BLOCK_PARAMETER_INITIALIZATION_SEEDS[b]`.
4. Under `torch.no_grad()`, fill temporary CPU-float32 tensors from that local
   generator, then copy them into the existing randomized parameter objects in
   the order below. Do not replace or re-register a parameter.
5. Restore or explicitly retain deterministic parameters at their accepted
   initial values without consuming randomness: both normalization `gamma`
   tensors are ones; both normalization `beta` tensors and both FFN biases are
   zeros.
6. Verify structure, parameter identities, shapes, device, dtype, finiteness,
   accepted deterministic values, and unchanged dropout state before exposing
   the block through the `ModuleList`.

The one-stream randomized fill/copy order inside each block is:

1. head 0 query, key, value;
2. head 1 query, key, value;
3. head 2 query, key, value;
4. head 3 query, key, value;
5. attention output weight;
6. FFN first weight; and
7. FFN second weight.

The thirteen attention tensors and FFN first weight use
`Normal(0, 1 / sqrt(32))`. The FFN second weight uses
`Normal(0, 1 / sqrt(128))`. Every fill passes the local generator explicitly.
No global RNG operation is permitted.

This policy changes only the initial numerical values owned by the Phase 7
assembled model. It does not change a block's topology, parameter objects,
forward arithmetic, dropout arithmetic/state, validation, exception ownership,
inspection, causality, or standalone Phase 6 construction contract.

The four final block parameter states must be pairwise non-identical: for every
pair of blocks, at least one randomized weight differs bitwise. This is checked
after construction and independently proved by tests. Equal deterministic
normalization scales and zero biases are intentional and do not violate this
block-level non-identity requirement.

### Determinism and RNG isolation

On the same accepted runtime/platform/build, two complete-model constructions
with the same accepted vocabulary and configuration produce bitwise-identical
initial parameters. Construction preserves global CPU RNG state and is
unaffected by unrelated global random draws.

Phase 7 does not promise bitwise identity across PyTorch versions, builds,
platforms, devices, or different initialization primitives. Phase 8 later owns
runtime stochastic-state capture, training reproducibility, and checkpoint
restoration.

## Proposed final normalization

`final_norm` is one new `ExplicitLayerNorm` with width 32, epsilon `1e-5`, CPU,
and float32. It has its own `(32,)` `gamma` initialized to ones and `(32,)`
`beta` initialized to zeros. Neither aliases a block normalization.

It applies independently at each sequence position across that row's 32
features. It occurs after all four block outputs and before any LM-head
arithmetic. Its ordinary forward returns only its output. Complete-model
inspection calls `final_norm.inspect` once and retains the resulting same-call
inspection object.

## Proposed language-model head

After complete constructor validation, `LanguageModelHead` creates a distinct
local CPU generator seeded with `7005`. It fills `output_weight (32, 81)` with
`Normal(0, 1 / sqrt(32))`, wraps it as a parameter, and creates exact-zero
`output_bias (81,)` without consuming randomness.

For validated `H` with shape `(T, 32)`:

```text
logits = H @ output_weight + output_bias  # (T, 81)
```

The head accepts exactly one CPU-float32 rank-two representation with
`1 <= T <= 256`, width 32, and finite contents. It validates its exact two-
parameter structure, shapes, device, dtype, and finiteness before arithmetic.
It returns finite logits connected through autograd to `H`, the weight, and the
bias. It neither mutates nor aliases its input.

Head forward validation order is input tensor object, rank, dtype, device,
width, sequence length, input finiteness, exact parameter enumeration, both
parameter types, both shapes, both devices, both dtypes, and both finiteness
checks. No matrix multiplication occurs before every check succeeds.

The head weight is a distinct `(32, 81)` allocation. It does not alias or
transpose-alias `representation.token_embeddings (81, 32)`. Reusing the Phase
4 shape, bias, scaled-normal rule, and transparent affine arithmetic does not
reuse the Phase 4 model or seed.

## Proposed parameter ownership and exact count

| Owner | Tensor count | Parameter count |
|---|---:|---:|
| Accepted Phase 3 representation | 2 | 10,784 |
| Block 0 | 21 | 12,576 |
| Block 1 | 21 | 12,576 |
| Block 2 | 21 | 12,576 |
| Block 3 | 21 | 12,576 |
| Final normalization | 2 | 64 |
| Untied LM head weight | 1 | 2,592 |
| LM head bias | 1 | 81 |
| **Complete Mini-GPT** | **90** | **63,825** |

Independent arithmetic is:

```text
embedding = (81 * 32) + (256 * 32)
          = 2,592 + 8,192
          = 10,784

one block = attention + two normalizations + FFN
          = 4,096 + 128 + 8,352
          = 12,576

four blocks = 4 * 12,576
            = 50,304

final normalization = 32 + 32
                    = 64

untied head = (32 * 81) + 81
            = 2,592 + 81
            = 2,673

complete model = 10,784 + 50,304 + 64 + 2,673
               = 63,825
```

All 90 tensors are initially CPU float32 parameters with
`requires_grad=True`. There is no sharing between blocks, between normalization
components, or between the output head and token table. Dropout generators are
runtime state, not parameters or registered buffers. Exact object-identity
checking must establish 90 unique parameter objects.

Standard `named_parameters()` order is exactly:

```text
representation.token_embeddings
representation.position_embeddings
blocks.0.<accepted Phase 6 block parameter order>
blocks.1.<accepted Phase 6 block parameter order>
blocks.2.<accepted Phase 6 block parameter order>
blocks.3.<accepted Phase 6 block parameter order>
final_norm.gamma
final_norm.beta
head.output_weight
head.output_bias
```

For every block index, `<accepted Phase 6 block parameter order>` expands to:

```text
norm1.gamma
norm1.beta
attention.output_weight
attention.heads.0.query_weight
attention.heads.0.key_weight
attention.heads.0.value_weight
attention.heads.1.query_weight
attention.heads.1.key_weight
attention.heads.1.value_weight
attention.heads.2.query_weight
attention.heads.2.key_weight
attention.heads.2.value_weight
attention.heads.3.query_weight
attention.heads.3.key_weight
attention.heads.3.value_weight
norm2.gamma
norm2.beta
feed_forward.first_weight
feed_forward.first_bias
feed_forward.second_weight
feed_forward.second_bias
```

Enumeration order does not redefine the Phase 7 random-fill order, accepted
Phase 5 semantic validation order, block execution order, or inspection order.

## Proposed complete-model input and forward contract

`MiniGPT.forward(token_ids)` accepts when:

- `isinstance(token_ids, torch.Tensor)` is true;
- rank is exactly one `(T,)`;
- dtype is exactly `torch.long`;
- input device is exactly CPU;
- `1 <= T <= 256`;
- every ID is in `0..80`; and
- the input is not mutated or coerced.

Batched, empty, padded, truncated, wrapped-position, offset-position, and
caller-masked input are rejected. Phase 3's broader rank-two and empty support
remains unchanged but is not part of the complete-model boundary.

Before arithmetic, both `forward` and `inspect` validate the input, exact four-
child model structure, exact four-block structure, all 90 parameter objects and
shapes/devices/dtypes/finiteness, no prohibited parameter-object sharing,
head/token untied identity, and parent/child mode consistency. Pairwise
non-identity is an initial-construction guarantee, not a permanent constraint
on parameter values after later authorized optimization.

The exact pre-arithmetic first-failure order for both entry points is:

1. input tensor object;
2. rank;
3. dtype;
4. input device only;
5. sequence length;
6. token-ID range;
7. direct-child names and exact types;
8. exact four-block `ModuleList` length and exact block types;
9. parent, block, and dropout mode-state types and equality;
10. parameter enumeration and object ownership;
11. all parameter types in enumeration order;
12. all shapes in enumeration order;
13. all parameter devices in enumeration order;
14. all dtypes in enumeration order;
15. all parameter finiteness in enumeration order; and
16. all prohibited parameter-object alias checks in semantic owner order.

Only then may representation lookup or any dropout-state advancement occur.

Ordinary forward then performs exactly:

```text
H = representation(token_ids)
for block in blocks, index order 0..3:
    H = block(H)
H = final_norm(H)
logits = head(H)
return logits
```

`forward` returns only finite CPU-float32 logits `(T, 81)`. It performs no
softmax, target access, loss, backward call, update, corpus traversal,
optimization, checkpointing, generation, or sampling.

## Proposed separate loss path and next-token alignment

Phase 7 reuses the accepted standalone Phase 4
`explicit_cross_entropy(logits, targets)` unchanged. It does not route loss
through `SimpleNeuralLanguageModel` and does not duplicate or weaken the
accepted stable arithmetic or representability checks.

A valid Phase 7 next-token example supplies:

```text
input_token_ids = z[s : s + T]       # shape (T,)
target_token_ids = z[s+1 : s+T+1]    # shape (T,)
logits = model(input_token_ids)       # shape (T, 81)
loss = explicit_cross_entropy(logits, target_token_ids)  # scalar
```

Each target is the immediate successor of the input at the same row, within one
document. The model and loss operation do not create the shift. Inputs and
targets have equal positive length `1..256`; targets are CPU `torch.long` IDs
in `0..80`. Cross-document transitions, padding targets, special tokens, target
shifting inside forward, and corpus materialization are excluded.

Synthetic Phase 7 evidence must show that loss is finite, scalar, matches the
accepted explicit arithmetic, and backpropagates finite shape-correct nonzero
gradients to a retained non-leaf embedding representation and every one of the
90 trainable tensors without updating or replacing parameters. Integer token
IDs do not receive gradients. This is gradient evidence only, not training.

## Proposed complete-model inspection

`MiniGPT.inspect(token_ids)` performs one complete model computation. It does
not call ordinary `forward`, reset RNG, clone a model, or recompute any child.

It performs exactly:

1. one `representation(token_ids)` call;
2. `blocks[0].inspect` once and uses that object's `output` as block 1 input;
3. `blocks[1].inspect` once and uses that output as block 2 input;
4. `blocks[2].inspect` once and uses that output as block 3 input;
5. `blocks[3].inspect` once and uses that output as final-norm input;
6. `final_norm.inspect` once; and
7. `head` once on the exact final-normalization output.

The frozen `MiniGPTInspection` stores the identical embedding tensor, a tuple
of exactly four same-call block inspection objects in stack order, the exact
final-normalization inspection object, and the exact logits tensor. Tensor
contents are excluded from generated `repr`. All tensors retain autograd
graphs. Inspection creates no corpus-derived artifact.

In training mode, this one-pass rule prevents inspection from reporting masks
or intermediates from a computation different from the logits. Per-block
dropout transactions remain authoritative. Phase 7 adds no outer stack-wide RNG
rollback: a later-block failure may follow successful state advancement in an
earlier block. Whole-training-operation stochastic rollback and restoration
belong to Phase 8 and must not be inferred here.

### Stack-wide stochastic failure composition

The complete model composes four independently transactional Phase 6 block
calls; it is not one outer dropout transaction. In training mode, a successful
block call commits that block's two accepted local-generator advancements as
soon as the call returns. A later failure does not roll back an already
completed block.

For a failure while block `b` is executing, where `b` is in `0..3`:

- every block with index less than `b` has completed, so both of its dropout
  stream advancements remain committed;
- block `b` applies its inherited complete two-stream Phase 6 transaction and
  restores both of its streams to their states at entry to that block call;
- every block with index greater than `b` is untouched and neither stream
  advances; and
- the global CPU RNG state remains bitwise unchanged.

For a failure after all four blocks return successfully—including a forced
failure in final normalization, LM-head input processing, LM-head arithmetic,
or inspection-result packaging—all eight already-completed block stream
advancements remain committed. No later block exists to restore them, and the
complete model performs no outer rollback. The global CPU RNG remains bitwise
unchanged.

Focused tests must snapshot all eight block-local generator states and global
CPU RNG state, use untouched state-identical controls, and independently force
the following paths for both ordinary `forward` and `inspect` where the stage
exists:

1. failure inside block 0;
2. failure inside a later block after at least one earlier block succeeds;
3. failure inside block 3 after blocks 0 through 2 succeed;
4. failure in final normalization after all blocks succeed;
5. failure in LM-head processing after all blocks succeed; and
6. inspection-result packaging failure after all blocks, final normalization,
   and the head succeed.

For each case, tests prove committed earlier streams equal the corresponding
successful-control states, the failing block equals its entry snapshot,
untouched later blocks equal their original snapshots, and global RNG is
unchanged. Delegating call counters and object-identity captures prove every
attempted block or later component ran at most once and that no recomputation
was used to obtain inspection evidence. Evaluation-mode controls prove that no
dropout stream advances when dropout is disabled. These obligations document
composition only; they do not change Phase 6 rollback behavior or add a
stack-wide transaction.

## Proposed causality, shape, and mode contract

For two valid sequences of the same total length `T`, where `1 <= T <= 256`,
choose `k` with `0 <= k < T`. The token IDs must be exactly equal through
position `k` and may differ only at positions strictly after `k`. Logits through
`k` are exactly equal under controlled deterministic conditions:

- evaluation mode for any otherwise valid pair; and
- training mode only for two independently constructed, parameter-identical
  complete models with equal block-dropout states and equal successful call
  histories.

This evidence imposes no cross-length bitwise-equality obligation.

The proof composes accepted absolute-position lookup, causal attention in every
block, positionwise normalization/FFN/dropout/residual operations, final
positionwise normalization, and the positionwise LM head. A positive control
must show that changing an allowed earlier token can change a later logit. A
separate unmasked reference must show the future perturbation is non-vacuous.

Every representation stage has shape `(T, 32)`; every block inspection contains
attention weights `(4, T, T)`; final logits have shape `(T, 81)`. Each block
receives the exact prior block output object. The stack never changes `T`.

`model.train()` and `model.eval()` propagate normally. All four blocks and
their dropout children must match the complete-model mode before arithmetic.
Only inherited dropout behavior changes between modes; embedding, learned
parameter values, normalization, attention, FFN, head arithmetic, and structure
do not.

## Proposed errors and nonmutation

Phase 7 introduces:

- `MiniGPTTypeError(TypeError)` for wrong public object and exact argument,
  child, parameter, or mode-state types;
- `MiniGPTContractError(ValueError)` for invalid configuration, vocabulary
  binding, rank, shape, length, range, device, dtype, structure, sharing,
  initialization postcondition, or fixed value; and
- `MiniGPTNumericalError(ArithmeticError)` for nonfinite Phase 7-owned input,
  parameter, intermediate, or arithmetic result.

All three follow the accepted project exception surface: construction takes one
literal `invariant` plus safe keyword facts, `str(error)` is deterministic
ASCII JSON with sorted keys, and read-only `details` exposes an immutable
mapping containing that invariant and only the safe facts. Tensor or arbitrary
object representations are never accepted as safe facts.

Diagnostics contain stable invariant names and structural safe facts only.
They never include token IDs, text, tensor values, parameters, logits, targets,
paths supplied from corpus data, or arbitrary object representations.

Rejected calls do not mutate inputs, targets, parameters, gradients, module
mode, global RNG, or Phase 7 configuration. Inherited component exceptions keep
ownership after complete-model prevalidation passes. The separate loss path
retains the accepted Phase 4 exception taxonomy.

### Exact Phase 7 invariant oracle

Independent tests use the literal names below. They must not import production
constants or construct expected names from production helpers.

Mini-GPT constructor type invariants are exactly:

```text
mini_gpt.constructor.vocabulary.exact_type
mini_gpt.constructor.model_width.exact_int
mini_gpt.constructor.max_sequence_length.exact_int
mini_gpt.constructor.number_of_blocks.exact_int
mini_gpt.constructor.number_of_heads.exact_int
mini_gpt.constructor.head_width.exact_int
mini_gpt.constructor.hidden_width.exact_int
mini_gpt.constructor.embedding_seed.exact_int
mini_gpt.constructor.layer_norm_epsilon.exact_float
mini_gpt.constructor.dropout_probability.exact_float
mini_gpt.constructor.block_parameter_seeds.exact_tuple
mini_gpt.constructor.block_parameter_seeds.length
mini_gpt.constructor.block_parameter_seeds.0.exact_int
mini_gpt.constructor.block_parameter_seeds.1.exact_int
mini_gpt.constructor.block_parameter_seeds.2.exact_int
mini_gpt.constructor.block_parameter_seeds.3.exact_int
mini_gpt.constructor.output_head_seed.exact_int
mini_gpt.constructor.device.exact_torch_device
mini_gpt.constructor.dtype.torch_dtype
```

`block_parameter_seeds.length` is a contract invariant because exact tuple type
has already passed. Every other `exact_*` or `torch_dtype` invariant above is a
type invariant. Constructor value and relationship invariants are exactly:

```text
mini_gpt.constructor.vocabulary.verified
mini_gpt.constructor.model_width.value
mini_gpt.constructor.max_sequence_length.value
mini_gpt.constructor.number_of_blocks.value
mini_gpt.constructor.number_of_heads.value
mini_gpt.constructor.head_width.value
mini_gpt.constructor.hidden_width.value
mini_gpt.constructor.layer_norm_epsilon.value
mini_gpt.constructor.dropout_probability.value
mini_gpt.constructor.embedding_seed.value
mini_gpt.constructor.block_parameter_seeds.value
mini_gpt.constructor.output_head_seed.value
mini_gpt.constructor.device.value
mini_gpt.constructor.dtype.value
mini_gpt.constructor.block_parameter_seeds.distinct
mini_gpt.constructor.block_parameter_seeds.disjoint
```

Standalone LM-head constructor invariants are exactly:

```text
lm_head.constructor.model_width.exact_int
lm_head.constructor.vocabulary_size.exact_int
lm_head.constructor.seed.exact_int
lm_head.constructor.device.exact_torch_device
lm_head.constructor.dtype.torch_dtype
lm_head.constructor.model_width.value
lm_head.constructor.vocabulary_size.value
lm_head.constructor.seed.value
lm_head.constructor.device.value
lm_head.constructor.dtype.value
```

Complete-model input invariants are exactly:

```text
mini_gpt.input.tensor
mini_gpt.input.rank
mini_gpt.input.dtype
mini_gpt.input.device
mini_gpt.input.sequence_length
mini_gpt.input.token_id_range
```

Standalone-head input and result invariants are exactly:

```text
lm_head.input.tensor
lm_head.input.rank
lm_head.input.dtype
lm_head.input.device
lm_head.input.model_width
lm_head.input.sequence_length
lm_head.input.finite
lm_head.output.finite
```

Complete-model structure, mode, parameter, sharing, inspection, and numerical
invariants are exactly:

```text
mini_gpt.submodules.structure
mini_gpt.blocks.structure
mini_gpt.modes.type
mini_gpt.modes.consistent
mini_gpt.parameters.enumeration
mini_gpt.parameters.unique
mini_gpt.head.untied
mini_gpt.initialization.block.0.structure
mini_gpt.initialization.block.1.structure
mini_gpt.initialization.block.2.structure
mini_gpt.initialization.block.3.structure
mini_gpt.initialization.block.0.parameter_identity
mini_gpt.initialization.block.1.parameter_identity
mini_gpt.initialization.block.2.parameter_identity
mini_gpt.initialization.block.3.parameter_identity
mini_gpt.initialization.block.0.deterministic_values
mini_gpt.initialization.block.1.deterministic_values
mini_gpt.initialization.block.2.deterministic_values
mini_gpt.initialization.block.3.deterministic_values
mini_gpt.initialization.block.0.dropout_state
mini_gpt.initialization.block.1.dropout_state
mini_gpt.initialization.block.2.dropout_state
mini_gpt.initialization.block.3.dropout_state
mini_gpt.initialization.blocks.nonidentical
mini_gpt.initialization.global_rng
mini_gpt.embedding_output.finite
mini_gpt.block_output.0.finite
mini_gpt.block_output.1.finite
mini_gpt.block_output.2.finite
mini_gpt.block_output.3.finite
mini_gpt.final_normalization_output.finite
mini_gpt.output.finite
mini_gpt.inspection.block_count
mini_gpt.inspection.handoff_identity
mini_gpt.inspection.final_normalization_identity
mini_gpt.inspection.logits_identity
```

For `qualified_name` in the exact 90-name expansion under **Proposed parameter
ownership and exact count**, parameter invariants are exactly:

```text
mini_gpt.parameter.<qualified_name>.ownership
mini_gpt.parameter.<qualified_name>.type
mini_gpt.parameter.<qualified_name>.shape
mini_gpt.parameter.<qualified_name>.device
mini_gpt.parameter.<qualified_name>.dtype
mini_gpt.parameter.<qualified_name>.finite
```

Standalone-head parameter invariants are exactly:

```text
lm_head.parameters.structure
lm_head.parameter.output_weight.type
lm_head.parameter.output_weight.shape
lm_head.parameter.output_weight.device
lm_head.parameter.output_weight.dtype
lm_head.parameter.output_weight.finite
lm_head.parameter.output_bias.type
lm_head.parameter.output_bias.shape
lm_head.parameter.output_bias.device
lm_head.parameter.output_bias.dtype
lm_head.parameter.output_bias.finite
```

### Validation-stage-to-exception mapping

| Validation stage | Exact exception class |
|---|---|
| Mini-GPT or LM-head constructor `exact_*`, `torch_dtype`, public input `tensor`, child type, parameter `type`, or mode `type` | `MiniGPTTypeError` |
| Verified binding, constructor `.value`, seed tuple length/relationships, rank, dtype, device, length, range, module/block/parameter structure, ownership, shape, sharing, untied identity, initialization postcondition, or inspection identity/count | `MiniGPTContractError` |
| Nonfinite standalone-head input | `MiniGPTNumericalError` with `lm_head.input.finite` |
| Nonfinite standalone-head arithmetic result | `MiniGPTNumericalError` with `lm_head.output.finite` |
| Nonfinite Phase 7 parameter or complete-model boundary/intermediate | `MiniGPTNumericalError` with its exact `.finite` invariant |
| Accepted representation, block, final-normalization, or explicit-cross-entropy failure after Phase 7 prevalidation | That inherited component's accepted exception class and invariant |

The standalone-head input and output failures are intentionally distinct:
`lm_head.input.finite` proves caller-supplied nonfinite data was rejected before
arithmetic, while `lm_head.output.finite` proves finite accepted inputs and
parameters produced a nonfinite affine result. Neither is a generic contract
failure.

## Proposed focused test and evidence obligations

Future tests must be synthetic, independent, and cover at least:

1. exact public signatures, required arguments, exports, and frozen inspection;
2. constructor type/value order before any allocation;
3. accepted vocabulary binding and exact complete configuration;
4. exact child order, four distinct block objects, no sharing, and mode
   propagation;
5. exact 90 tensors, names, shapes, device, dtype, trainability, and 63,825
   independently derived elements;
6. accepted Phase 3 embedding parameters and initial bytes remain reproduced;
7. Phase 7 per-block streams independently reproduce every randomized weight;
8. every block preserves parameter object identity and accepted Phase 6
   structure while final randomized states differ pairwise;
9. deterministic block parameters keep accepted ones/zeros and no block
   dropout state changes during Phase 7 parameter initialization;
10. head seed 7005, scaled-normal weight, zero bias, global RNG isolation, and
    complete-model repeated-construction determinism;
11. head/token weight non-aliasing and absence of every prohibited sharing;
12. input type/rank/dtype/device/length/range order, including lengths 1 and
    256 and rejection of 0, 257, and batched input;
13. exact embedding-to-four-block-to-final-norm-to-head topology and object
    handoffs;
14. forward returns only logits `(T,81)` and matches independent affine-head
    arithmetic;
15. inspection invokes every stochastic block exactly once and preserves all
    same-call identities and autograd graphs;
16. forward and inspection failures at block 0, later blocks, block 3, final
    normalization, head processing, and inspection packaging prove committed
    earlier streams, inherited failing-block rollback, untouched later streams,
    unchanged global RNG, and absence of recomputation;
17. exact same-length protected-prefix causality in evaluation and controlled paired
    training mode, plus permitted-history and non-vacuous controls;
18. final normalization positionwise arithmetic and independent parameters;
19. separate next-token alignment and exact reuse of accepted explicit
    cross-entropy without invoking the Phase 4 model;
20. finite scalar loss and finite, shape-correct, nonzero gradients reaching a
    retained embedding representation and all 90 parameters without
    value/object mutation; token IDs remain non-differentiable;
21. finite guards, deterministic content-safe errors, rejected-call
    nonmutation, and exact first-failure behavior;
22. accepted Phase 3, Phase 4 loss, Phase 5, and Phase 6 regression suites
    remain passing and their production files remain unchanged; and
23. source review finds no `SimpleNeuralLanguageModel`, weight tying, batching,
    optimizer, training loop, checkpoint, generation, corpus access, sealed
    test, high-level Transformer, or pretrained/hosted model path.

Tests must not derive critical expected values solely from production constants
or helpers. They must not persist parameters, gradients, activations, logits,
losses, attention weights, masks, token sequences, or checkpoints.

## Educational demonstrations required before exit acceptance

Before Phase 7 exit acceptance, Sebastien should inspect small synthetic
evidence showing:

1. token IDs becoming token-plus-position representations;
2. four distinct block parameter sets and the representation flowing through
   all four blocks;
3. final normalization statistics at a selected position;
4. one final width-32 row mapping to 81 logits through the untied head;
5. exact next-token row alignment and separate cross-entropy;
6. protected-prefix causality through the complete model and permitted earlier
   influence;
7. same-call block inspection and attention weights without treating weights
   as complete explanations; and
8. gradient flow through all 90 tensors and the exact 63,825-parameter count.

This is synthetic inspection only. It is not a training run, checkpoint,
generation event, validation measurement, or sealed-test evaluation.

## Explicit Phase 7 exclusions

Phase 7 does not modify accepted Phase 3, Phase 5, or Phase 6 source, tests,
contracts, mathematical behavior, or standalone initialization semantics. It
does not use `SimpleNeuralLanguageModel` as its architecture and does not tie
the LM head to token embeddings.

It does not introduce empty complete-model input, batching, padding,
caller-configurable masks, cross-attention, new token types, new vocabulary,
position offsets or wrapping, optimizer infrastructure, parameter updates,
training/evaluation loops, data sampling, corpus access, experiment execution,
checkpoint/resume, generation, final evaluation, sealed-test access, MPS/mixed
precision, pretrained models, hosted APIs, or high-level/fused Transformer
implementations.

## Proposed Phase 7 gates

1. Phase 6 accepted implementation push and exit review. **Complete.**
2. Phase 6 formal remote closure at `e84b364`. **Complete.**
3. Explicit Phase 7 learning/design authorization. **Complete.**
4. Read-only Phase 7 continuity inspection. **Complete.**
5. Consolidated conceptual architecture discussion and DEC-0019 acceptance.
   **Complete.**
6. Documentation-only detailed-contract drafting and continuity reconciliation.
   **Complete locally; proposed and unaccepted.**
7. Master Chat detailed-contract sanity review and independent-review dispatch.
   **Complete.**
8. Fresh independent detailed-contract review. **Complete: CORRECT BEFORE
   ACCEPTANCE, with five IMPORTANT findings and no BLOCKER.**
9. Master Chat finding adjudication and bounded documentation-correction
   authorization. **Complete: all five findings accepted.**
10. Apply only the five authorized documentation corrections. **Complete
    locally; corrected proposal remains unaccepted.**
11. Focused independent detailed-contract re-review. **Complete: PASS; all five
    prior IMPORTANT findings resolved with no remaining finding.**
12. Master Chat final detailed-contract adjudication. **Complete: ACCEPT
    CORRECTED CONTRACT.**
13. Sebastien's explicit corrected detailed-contract acceptance. **Complete
    through the formal acceptance instruction.**
14. Separate accepted-contract commit authorization.
15. Accepted-contract commit creation without amendment or extra files.
16. Separate accepted-contract push authorization.
17. Accepted-contract push.
18. Independent remote verification of the accepted-contract commit.
19. Separate explicit Phase 7 implementation authorization.
20. Phase 7 implementation and focused synthetic testing only.
21. Master Chat implementation sanity review.
22. Fresh independent implementation review.
23. Master Chat adjudication of any implementation findings.
24. Separate focused-correction authorization if required.
25. Apply only the authorized implementation corrections if required.
26. Focused independent implementation re-review if corrections occurred.
27. Master Chat final implementation adjudication.
28. Sebastien's explicit implementation acceptance.
29. Separate accepted-implementation commit authorization.
30. Accepted-implementation commit creation without amendment or extra files.
31. Separate accepted-implementation push authorization.
32. Accepted-implementation push.
33. Independent remote verification of the accepted-implementation commit.
34. Separate educational-inspection and Phase 7 exit-review authorization.
35. Educational inspection and independent Phase 7 exit review.
36. Master Chat exit adjudication.
37. Sebastien's explicit Phase 7 exit and closure-state acceptance.
38. Separate documentation-only closure-bookkeeping authorization.
39. Apply the authorized closure bookkeeping only.
40. Separate Phase 7 closure-commit authorization.
41. Phase 7 closure-commit creation without amendment or extra files.
42. Separate Phase 7 closure-push authorization.
43. Phase 7 closure push.
44. Independent final remote verification of the Phase 7 closure commit.
45. Only then consider separate Phase 8 learning/design authorization.

Every irreversible action remains separately authorized. A failed review never
authorizes correction unless Master Chat accepts the finding and Sebastien
authorizes the bounded work.

## Current authorization boundary

The corrected detailed contract is formally accepted. Gate 14, separate
accepted-contract checkpoint commit authorization, is next and remains
unauthorized. No Phase 7 source, tests, parameter construction, implementation,
training, experiment, sealed-test access, staging, commit, push, Phase 8, or
Phase 9 work is authorized.
