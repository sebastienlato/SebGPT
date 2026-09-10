# SebGPT Phase 6 Transformer Block Specification

## Status and authority

Phase 6 is titled **Transformer Block**. Its repository goal is exactly:

> Combine attention and feed-forward computation into a stable reusable block.

Its exact repository exit criteria are:

1. Normalization, residual connections, feed-forward layers, and dropout are
   understood.
2. A Transformer block is implemented from explicit components.
3. Shape, causality, gradient, and residual-path behavior are tested.

Sebastien completed the Phase 6 learning/design discussion and explicitly
accepted the conceptual architecture recorded in DEC-0018. The authorized
documentation-only detailed-contract proposal was drafted, Master Chat sanity
review returned PASS, and the fresh Codex independent detailed-contract review
returned FAIL with four MUST-FIX findings and one SHOULD-FIX finding. Master
Chat accepted all five findings and confirmed that none changes or reopens
DEC-0018. The authorized documentation-only corrections are incorporated in
this revision. Gate 10 focused independent re-review returned PASS with all four
MUST-FIX and the SHOULD-FIX finding resolved. Master Chat adjudicated PASS, and
Sebastien explicitly accepted the corrected detailed contract without changing
DEC-0018. The detailed mechanics below are authoritative for later separately
authorized Phase 6 implementation.

The dedicated accepted-contract checkpoint commit is authorized. This does not
authorize push, Phase 6 source, tests, parameter construction, implementation,
training, experiment, sealed-test access, or Phase 7 work. Implementation
requires this accepted contract to be pushed, independently remote-verified,
and then separately authorized.

## Accepted conceptual architecture

The accepted architecture is one representation-to-representation block:

```text
X (T, 32) -> TransformerBlock -> Y (T, 32)
1 <= T <= 256
```

Exactly one sequence is processed per call. The block does not own token IDs,
vocabulary loading, embeddings, targets, logits, loss, corpus access, or a
language-model output head.

The topology is exactly pre-norm with two sequential residual paths:

```text
N1 = norm1(X)
A  = attention(N1)
AD = attention_dropout(A)
R1 = X + AD

N2 = norm2(R1)
F  = feed_forward(N2)
FD = feed_forward_dropout(F)
Y  = R1 + FD
```

The attention component is the accepted Phase 5
`MultiHeadCausalSelfAttention`: four visible width-8 causal heads followed by
its accepted bias-free width-32 output projection. Phase 6 does not alter its
parameters, seed-5005 initialization, causal relation, numerical contract, or
inspection-weight semantics. It adds no dropout inside attention.

`norm1` and `norm2` are separate explicit LayerNorm-style components. Each
normalizes every position independently across its 32 features and owns an
independent learned scale `gamma` and bias `beta`, each shaped `(32,)`. Their
parameters are not shared. The mechanism does not use `torch.nn.LayerNorm`.

The feed-forward component operates independently at each position:

```text
(T, 32) -> affine (32, 128) -> GELU -> affine (128, 32) -> (T, 32)
```

It owns weight shapes `(32, 128)` and `(128, 32)` and bias shapes `(128,)` and
`(32,)`. It never mixes sequence positions.

Actual dropout with probability `0.1` occurs at exactly two locations:
after the attention branch output and after the feed-forward branch output,
each immediately before its residual addition. Training mode applies stochastic
feature/activation masking; evaluation mode is deterministic identity. No other
dropout exists.

Ordinary block forward returns only `Y`. A separate educational inspection path
exposes the accepted same-call intermediate stages, including inherited
attention weights and both pre- and post-dropout branch values. Attention
weights remain inspectable routing quantities, not complete explanations of
block behavior.

Future distinct initialization across multiple stacked blocks is explicitly
deferred to Phase 7. One Phase 6 block owns one unchanged accepted Phase 5
attention module with seed 5005.

## Inherited authority and phase boundary

Phase 6 preserves these accepted identities:

| Field | Accepted value |
|---|---|
| Tokenizer | `shakespeare-code-point-v1`, schema `1`, vocabulary size `81`, no special tokens |
| Representation width | `32` |
| Positional capacity | `256` |
| Representation | accepted token lookup plus learned absolute-position lookup, then addition |
| Attention | accepted `MultiHeadCausalSelfAttention` |
| Attention input/output | one finite CPU-float32 tensor `(T, 32)`, `1 <= T <= 256` |
| Attention heads | four separate width-8 heads |
| Attention causality | position `i` may use exactly positions `j <= i` |
| Attention parameters | thirteen bias-free tensors, `4,096` elements |
| Attention seed | exact `5005` under its accepted initialization contract |
| Semantic device/dtype | CPU `torch.float32` |

The Phase 4 `SimpleNeuralLanguageModel` and `EXP-20260909-01` remain closed and
unchanged. Phase 6 does not wrap, resume, or train that model. Phase 4's
64-position experiment window does not reduce the accepted Phase 6 block range
of `1..256`.

Phase 6 owns only one block's parameter-initialization boundary, its two runtime
dropout streams, and synthetic educational evidence. Phase 7 owns embedding
integration, multiple stacked blocks, final model normalization, the
language-model head, and parameter-initialization seed allocation for distinct
stacked blocks and the assembled mini-GPT model. Phase 8 owns production
batching and optimization plus training/runtime stochastic state and
reproducibility: data ordering, training stochasticity, evaluation scheduling
and state, and checkpoint/resume restoration. Phase 9 owns generation and final
evaluation, including any separately authorized sealed-test event.

## Proposed constants

```text
MODEL_WIDTH = 32
NUMBER_OF_HEADS = 4
HEAD_WIDTH = 8
HIDDEN_WIDTH = 128
MAX_SEQUENCE_LENGTH = 256
LAYER_NORM_EPSILON = 1e-5
DROPOUT_PROBABILITY = 0.1
ATTENTION_INITIALIZATION_SEED = 5005
PHASE6_PARAMETER_INITIALIZATION_SEED = 6006
ATTENTION_DROPOUT_SEED = 6007
FEED_FORWARD_DROPOUT_SEED = 6008
PARAMETER_DEVICE = torch.device("cpu")
PARAMETER_DTYPE = torch.float32
```

The Phase 6 seed numbers make independent responsibilities visible; they make no
claim of statistical superiority. Seed 5005 remains owned by Phase 5. Seed 6006
initializes only new random Phase 6 parameters. Seeds 6007 and 6008 drive the
two independent runtime dropout streams.

## Proposed module and file boundary

Future separately authorized implementation belongs in:

```text
src/sebgpt/model/transformer_block.py
tests/test_transformer_block.py
```

The proposed public API is:

```text
@dataclass(frozen=True)
LayerNormInspection:
    output: torch.Tensor = field(repr=False)
    mean: torch.Tensor = field(repr=False)
    variance: torch.Tensor = field(repr=False)
    normalized: torch.Tensor = field(repr=False)

@dataclass(frozen=True)
DropoutInspection:
    output: torch.Tensor = field(repr=False)
    keep_mask: torch.Tensor = field(repr=False)

@dataclass(frozen=True)
FeedForwardInspection:
    pre_activation: torch.Tensor = field(repr=False)
    hidden_activation: torch.Tensor = field(repr=False)
    output: torch.Tensor = field(repr=False)

@dataclass(frozen=True)
TransformerBlockInspection:
    normalized_attention_input: torch.Tensor = field(repr=False)
    attention_branch_output_before_dropout: torch.Tensor = field(repr=False)
    attention_weights: torch.Tensor = field(repr=False)
    attention_dropout_keep_mask: torch.Tensor = field(repr=False)
    attention_branch_output_after_dropout: torch.Tensor = field(repr=False)
    first_residual: torch.Tensor = field(repr=False)
    normalized_feed_forward_input: torch.Tensor = field(repr=False)
    feed_forward_pre_activation: torch.Tensor = field(repr=False)
    feed_forward_hidden_activation: torch.Tensor = field(repr=False)
    feed_forward_branch_output_before_dropout: torch.Tensor = field(repr=False)
    feed_forward_dropout_keep_mask: torch.Tensor = field(repr=False)
    feed_forward_branch_output_after_dropout: torch.Tensor = field(repr=False)
    output: torch.Tensor = field(repr=False)

ExplicitLayerNorm(
    *, model_width: int, epsilon: float,
    device: torch.device, dtype: torch.dtype,
)
public parameters: gamma (32,), beta (32,)
public methods:
    forward(representations: torch.Tensor) -> torch.Tensor
    inspect(representations: torch.Tensor) -> LayerNormInspection

explicit_gelu(values: torch.Tensor) -> torch.Tensor

PositionwiseFeedForward(
    *, model_width: int, hidden_width: int, seed: int,
    device: torch.device, dtype: torch.dtype,
)
public parameters:
    first_weight (32, 128), first_bias (128,),
    second_weight (128, 32), second_bias (32,)
public methods:
    forward(representations: torch.Tensor) -> torch.Tensor
    inspect(representations: torch.Tensor) -> FeedForwardInspection

ExplicitDropout(
    *, probability: float, seed: int,
    device: torch.device, dtype: torch.dtype,
)
public methods:
    forward(values: torch.Tensor) -> torch.Tensor
    inspect(values: torch.Tensor) -> DropoutInspection

TransformerBlock(
    *,
    model_width: int,
    number_of_heads: int,
    head_width: int,
    hidden_width: int,
    max_sequence_length: int,
    layer_norm_epsilon: float,
    dropout_probability: float,
    attention_seed: int,
    parameter_seed: int,
    attention_dropout_seed: int,
    feed_forward_dropout_seed: int,
    device: torch.device,
    dtype: torch.dtype,
)
public direct child modules, in registration order:
    norm1: ExplicitLayerNorm
    attention: MultiHeadCausalSelfAttention
    attention_dropout: ExplicitDropout
    norm2: ExplicitLayerNorm
    feed_forward: PositionwiseFeedForward
    feed_forward_dropout: ExplicitDropout
public methods:
    forward(representations: torch.Tensor) -> torch.Tensor
    inspect(representations: torch.Tensor) -> TransformerBlockInspection
```

"Six direct child modules" refers only to children registered immediately on
the block. The inherited attention child retains its own accepted `heads`
container and four head children; those nested Phase 5 modules do not increase
the block's direct-child count.

All constructor arguments are required and have no defaults. Public component
constructors accept only the exact applicable constants above. An
`ExplicitDropout` seed is exactly 6007 or 6008; the complete block additionally
requires 6007 at the attention location and 6008 at the feed-forward location.
The block constructor validates all argument types and values in signature
order before constructing any submodule, generator, tensor, or parameter.
Integer arguments require exact built-in `int` and reject `bool`; epsilon and
probability require exact built-in `float`; device requires exact
`torch.device("cpu")`; and dtype is exactly `torch.float32`.

`explicit_gelu` is a small visible arithmetic operation, not a configurable
activation framework. `torch.nn.LayerNorm`, `torch.nn.Dropout`,
`torch.nn.functional.gelu`, `torch.nn.GELU`, `torch.nn.Transformer`, and fused
Transformer-block implementations are not permitted in the Phase 6 educational
core.

## Proposed parameter ownership and count

### New Phase 6 parameters

| Owner and parameter | Shape | Count |
|---|---:|---:|
| `norm1.gamma` | `(32,)` | 32 |
| `norm1.beta` | `(32,)` | 32 |
| `norm2.gamma` | `(32,)` | 32 |
| `norm2.beta` | `(32,)` | 32 |
| `feed_forward.first_weight` | `(32, 128)` | 4,096 |
| `feed_forward.first_bias` | `(128,)` | 128 |
| `feed_forward.second_weight` | `(128, 32)` | 4,096 |
| `feed_forward.second_bias` | `(32,)` | 32 |
| **New Phase 6 total** | **8 tensors** | **8,480** |

### Complete block

| Group | Tensor count | Parameter count |
|---|---:|---:|
| Accepted Phase 5 multi-head attention | 13 | 4,096 |
| Two explicit normalizations | 4 | 128 |
| Positionwise feed-forward | 4 | 8,352 |
| Dropout components | 0 | 0 |
| **Complete Transformer block** | **21** | **12,576** |

There are no other parameters or registered buffers. Dropout RNG state is
runtime state, not a trainable parameter or a registered tensor buffer. Every
parameter is initially CPU `torch.float32` with `requires_grad=True`.

With the proposed submodule registration order, standard block
`named_parameters()` order is exactly:

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

This enumeration order does not alter Phase 5's internal semantic validation or
initialization order.

## Proposed initialization contract

Construction first validates the entire block configuration. It then constructs
submodules in the documented registration order.

The accepted attention module is constructed with its exact accepted arguments
and independently performs its seed-5005 initialization. Phase 6 neither draws
from nor resets that generator.

Each normalization is initialized without randomness:

```text
gamma = ones((32,), CPU float32)
beta  = zeros((32,), CPU float32)
```

The two normalization tensors are distinct allocations for each component.

The feed-forward component creates one local CPU generator, seeds it with exact
integer 6006, and uses one uninterrupted stream:

1. allocate and fill `first_weight (32, 128)` with
   `Normal(0, 1 / sqrt(32))`;
2. allocate exact-zero `first_bias (128,)` without consuming randomness;
3. allocate and fill `second_weight (128, 32)` with
   `Normal(0, 1 / sqrt(128))`; and
4. allocate exact-zero `second_bias (32,)` without consuming randomness.

Every randomized fill passes the local generator explicitly. PyTorch default
initializers and global RNG calls are forbidden.

The two dropout components each create their own local CPU generator after
complete constructor validation. The attention-branch generator is seeded 6007
and the feed-forward-branch generator 6008. Creating or using the block leaves
global CPU RNG state unchanged.

On the accepted runtime/platform/build, independently constructed blocks have
bitwise-identical initial parameters. Two blocks with identical parameter state,
mode, inputs, and successful call history produce the same dropout masks and
outputs. No cross-version or cross-platform bitwise guarantee is made.

Distinct initialization for multiple future stacked block instances is not
defined here. Phase 7 must decide complete-model seed allocation without
silently changing this single-block Phase 6 contract.

## Proposed common tensor contract

Every representation-valued Phase 6 public operation accepts an object only if
`isinstance(value, torch.Tensor)` is true. Lists, tuples, token IDs, NumPy
objects, and other values are not coerced. Valid tensor subclasses and
non-contiguous tensors are accepted.

Unless a narrower internal shape is explicitly documented, the tensor must be:

- rank two;
- shape `(T, 32)` with `1 <= T <= 256`;
- exact dtype `torch.float32`;
- on exact CPU, equal to all applicable parameter devices; and
- finite everywhere.

Input tensors are never mutated. Empty sequences, batches `(B, T, 32)`, widths
other than 32, token IDs, caller masks, padding, and cross-attention inputs are
rejected or absent.

Construction verifies initial parameter ownership, registration, object
identity, and `requires_grad=True`. Public computations validate structure,
parameter type, shape, device, dtype, and finiteness, but do not require original
construction identity or `requires_grad=True`. This preserves ordinary freezing
and compatible state restoration.

## Proposed explicit LayerNorm-style arithmetic

For each input row `x_i` independently across its 32 features:

```text
mean_i       = mean(x_i)
centered_i   = x_i - mean_i
variance_i   = mean(centered_i * centered_i)
inverse_std  = rsqrt(variance_i + 1e-5)
normalized_i = centered_i * inverse_std
output_i     = normalized_i * gamma + beta
```

Variance is the population variance with divisor 32, not an unbiased sample
variance. `gamma` and `beta` broadcast only across positions. No statistic is
computed across `T`, so changing one position cannot change normalization at a
different position.

`forward` invokes one private `_compute_normalization` boundary once and returns
its output. `inspect` invokes it once and returns the identical output, mean,
variance, and normalized tensor objects produced by that call. It does not
clone, detach, or recompute them. For input `(T, 32)`, mean and variance each
have shape `(T, 1)`, while normalized and output each have shape `(T, 32)`.

Required numerical checks occur immediately after mean, centering, squaring,
variance, epsilon addition, reciprocal square root, normalization, gamma
multiplication, and beta addition. Nonfinite results or a nonpositive
`variance + epsilon` raise the Phase 6 numerical exception rather than being
clamped or recomputed in another dtype.

## Proposed exact GELU and feed-forward arithmetic

GELU uses the exact error-function form, not the tanh approximation:

```text
explicit_gelu(z) = 0.5 * z * (1 + erf(z / sqrt(2)))
```

The function accepts any finite CPU-float32 tensor and preserves its shape. It
validates type, dtype, device, and finiteness, then checks every intermediate and
the result for finiteness. It does not call a high-level GELU primitive.

For validated `(T, 32)` input:

```text
pre_activation    = X @ first_weight + first_bias       # (T, 128)
hidden_activation = explicit_gelu(pre_activation)       # (T, 128)
output            = hidden_activation @ second_weight
                    + second_bias                        # (T, 32)
```

Each matrix multiplication acts only on the last feature dimension. No
operation reduces, attends, convolves, or otherwise mixes along `T`.

`forward` invokes one private `_compute_feed_forward` boundary exactly once and
returns its output. `inspect` invokes the same boundary once and exposes the
identical pre-activation, hidden-activation, and output objects from that call.

## Proposed explicit dropout arithmetic and RNG semantics

For training-mode input `V`, each dropout component performs inverted dropout:

```text
uniform   = rand(V.shape, CPU float32, local_generator)  # values in [0, 1)
keep_mask = uniform >= 0.1                               # Boolean
output    = V * keep_mask.to(float32) / 0.9
```

One uniform value is drawn for every tensor element. Masks therefore act on
features/activations without mixing positions. No mask is reused between the
attention and feed-forward branches, and no global RNG is consumed.

In evaluation mode:

```text
keep_mask = ones(V.shape, dtype=bool, device=CPU)
output = V
```

Evaluation makes no random draw and does not advance either local generator.
Returning the input object as the identity output is permitted and documented;
inspection therefore may report the same tensor object before and after dropout
in evaluation mode.

Each successful training-mode dropout call advances only its own generator once
for the full shape. Successive calls use successive masks.

`forward` calls the private dropout computation once and returns its output.
`inspect` calls it once and returns the identical output and keep-mask objects.
Neither path reruns masking for reporting.

Each dropout component owns no parameter or buffer and retains its exact
probability, seed, private local CPU `_generator`, and module `training` flag.
Its public calls require the mode flag to be an exact Boolean. Complete block
validation also requires both dropout child modes to equal the parent block mode; a manually
created mixed-mode block fails before arithmetic or RNG consumption. Ordinary
`block.train()` and `block.eval()` propagate the mode normally to both children.

### Authoritative block dropout transaction

`TransformerBlock.forward` and `TransformerBlock.inspect` use exactly the same
transaction boundary and rollback procedure.

Before reading either generator state, the block completes ordinary input,
direct-child, mode, and new-parameter validation and validates both dropout
components in attention-then-feed-forward order. For each dropout component it
verifies the exact component type, configured probability, configured seed,
CPU device, float32 dtype, absence of parameters and buffers, exact Boolean mode,
and a local generator that is an exact `torch.Generator` on CPU. It also verifies
that the two generator objects are distinct and that the child modes equal the
parent block mode. Any failure here occurs before generator-state snapshotting
and before either stream draws.

After both components and both generator objects are valid, the block snapshots
the complete current state of both streams in this fixed order:

```text
attention_state = attention_dropout._generator.get_state().clone()
feed_forward_state = feed_forward_dropout._generator.get_state().clone()
```

Each snapshot is the full independent CPU byte tensor returned for that exact
validated generator, not merely the initial seed, draw count, or a reference to
mutable generator state. Snapshotting does not use or mutate global CPU RNG. A
failure while obtaining either snapshot occurs before any dropout draw and raises
`TransformerBlockTransactionError` with invariant
`block.dropout_transaction.snapshot.attention` or
`block.dropout_transaction.snapshot.feed_forward`; the transaction has not
started, so no restoration is required.

The transaction begins only after both snapshots succeed. It includes all block
arithmetic, both possible dropout draws, numerical checks, construction of every
inspection dataclass, construction of the final public return value, and the
last operation before that value is returned. Any catchable `BaseException`
escaping that scope triggers rollback. This one rule includes Phase 6 contract,
type, numerical, transaction, Python, and PyTorch failures; inherited Phase 5
attention failures; failures before either stream draws; failures after
attention dropout draws but before feed-forward dropout draws; failures after
both streams draw; and failures while packaging an inspection result.

Rollback always attempts to restore both originally validated generator objects
to their complete snapshots in attention-then-feed-forward order, even when no
draw occurred or only the first stream advanced. If the first restoration fails,
the second restoration is still attempted. If both restorations succeed, the
original exception object is re-raised with its original traceback and ownership.
If either restoration fails, the block raises
`TransformerBlockTransactionError(RuntimeError)` with invariant
`block.dropout_transaction.rollback`, identifies only the failed stream name or
names as safe facts, and chains the original escaping exception as its cause. It
never exposes generator-state bytes. This transaction error owns the fact that
future dropout reproducibility can no longer be guaranteed.

A successful call commits both current states and performs no restoration. In
training mode both streams advance exactly once. In evaluation mode neither
stream draws, so the successfully committed states remain byte-identical to the
snapshots. Global CPU RNG state is unchanged on success, ordinary failure,
snapshot failure, and successful rollback.

The public `ExplicitDropout.forward` and `ExplicitDropout.inspect` operations
apply the analogous one-generator transaction to their own computation and, for
inspection, result packaging. The enclosing two-generator block transaction is
the sole authority for atomicity across both branch locations.

Dropout generator state is intentionally not assigned checkpoint semantics in
Phase 6. Phase 8 must explicitly decide how complete training-state checkpoints
capture and restore all runtime RNG state.

## Proposed block forward contract

The block validates the input, exact child-module structure, and every new
Phase 6 parameter before arithmetic. The inherited attention module retains its
own accepted public validation and exception ownership.

After validation, normal forward performs exactly:

```text
normalized_attention_input = norm1(X)
attention_output = attention(normalized_attention_input)
attention_after_dropout = attention_dropout(attention_output)
first_residual = X + attention_after_dropout

normalized_feed_forward_input = norm2(first_residual)
feed_forward_output = feed_forward(normalized_feed_forward_input)
feed_forward_after_dropout = feed_forward_dropout(feed_forward_output)
output = first_residual + feed_forward_after_dropout
```

Every named tensor except attention weights has shape `(T, 32)`. The hidden
pre-activation and activation have shape `(T, 128)`. Attention inspection
weights have accepted shape `(4, T, T)`.

Residual additions are exact elementwise additions with no scale, gate,
projection, or learned coefficient. Each result is checked for finiteness
immediately. Output is a finite CPU-float32 tensor connected through autograd to
the input and every unfrozen block parameter.

`forward` returns the final output only. It performs no target access, logits,
loss, backward call, update, training orchestration, state reset, checkpoint,
or generation.

## Proposed block inspection contract

`TransformerBlock.inspect(X)` performs one block computation. It must not call
`forward`, recompute a sublayer, restore RNG merely to obtain another view, or
run attention or dropout twice.

During that one computation it uses:

- `norm1.inspect` once;
- inherited `attention.inspect` once;
- `attention_dropout.inspect` once;
- `norm2.inspect` once;
- `feed_forward.inspect` once; and
- `feed_forward_dropout.inspect` once.

The returned `TransformerBlockInspection` contains the identical tensor objects
produced at those same-call boundaries. Both branch fields explicitly distinguish
the learned branch output before dropout from the value actually added after
dropout. Keep masks are visible. `first_residual` is the exact input-plus-
attention-dropout result. `output` is the exact first-residual-plus-FFN-dropout
result and is the same object the private block computation designates as final.

All inspection tensors retain their autograd graphs. The frozen dataclasses omit
tensor fields from generated representations so `repr` cannot dump their
contents. Inspection is restricted to caller-provided synthetic tensors; no
corpus-derived activation artifact is created in Phase 6.

Inherited attention weights are those used to create the pre-dropout attention
branch output in that same call. Later dropout, residual, normalization, GELU,
and feed-forward operations mean they are not complete explanations of final
block behavior.

## Proposed causality contract

For two accepted inputs with an exactly equal prefix through position `k`, block
outputs through `k` must be exactly equal under controlled deterministic
conditions:

- in evaluation mode for any two otherwise valid inputs; and
- in training mode for two independently constructed, parameter-identical
  blocks with identical dropout state and successful call history.

The proof composes four facts:

1. accepted attention at destination `i` reads only `j <= i`;
2. each normalization reads only the 32 features at its own position;
3. GELU, affine feed-forward computation, dropout, and residual addition are
   positionwise; and
4. paired training calls draw equal masks by position and feature from equal
   local-generator states.

A separate positive control changes an allowed earlier input and demonstrates a
later output change. Causality tests must also independently show that the future
perturbation would affect an unmasked reference, avoiding a vacuous equality.

## Proposed residual and gradient evidence

Focused synthetic tests must expose both residual identities.

For the first path, set the accepted attention `output_weight` to exact finite
zeros while leaving its structure intact; this makes the attention branch
output exact zero. Inspection must show:

```text
attention_branch_output_before_dropout == 0
attention_branch_output_after_dropout == 0
first_residual == input
```

A scalar sum of `first_residual` then gives the input an exact all-ones gradient,
demonstrating the identity gradient route when the attention branch derivative
is zero.

For the second path, set FFN `second_weight` and `second_bias` to exact finite
zeros while leaving its structure intact; this makes the feed-forward output
exact zero. Inspection must show:

```text
feed_forward_branch_output_before_dropout == 0
feed_forward_branch_output_after_dropout == 0
output == first_residual
```

After retaining the intermediate gradient, a scalar sum of final output gives
`first_residual` an exact all-ones gradient. A combined controlled example makes
both learned branches zero, proves final output equals original input exactly,
and proves the end-to-end input gradient is exact ones.

A separate nondegenerate evaluation-mode scalar objective must produce existing,
shape-correct, finite gradients for the input and all 21 block parameters, with
at least one nonzero element in every gradient. Backward must not replace or
mutate parameter objects, shapes, devices, or dtypes. This is differentiability
evidence only, not an update or training run.

## Proposed error taxonomy and safe diagnostics

Phase 6 owns:

- `TransformerBlockTypeError(TypeError)` for wrong public argument, tensor, child
  module, parameter, generator, or mode-state object types;
- `TransformerBlockContractError(ValueError)` for wrong configuration, shape,
  rank, length, dtype, device, structure, or fixed value;
- `TransformerBlockNumericalError(ArithmeticError)` for nonfinite inputs,
  parameters, intermediates, results, invalid normalization denominators, or
  invalid random draws; and
- `TransformerBlockTransactionError(RuntimeError)` for a generator-state
  snapshot failure or a rollback restoration failure.

Accepted Phase 5 exception classes propagate unchanged for failures owned by the
attention component. Phase 6 must not catch and relabel them except to restore
dropout RNG state before re-raising the same exception object.

Every Phase 6 exception carries one stable `invariant` identifier and only safe
structural facts. Its public `details` property is an immutable mapping, and its
message is deterministic ASCII JSON with sorted keys. Diagnostics never render
tensor values, activations, masks, attention weights, token IDs, text, paths, or
corpus content.

Stable invariant namespaces are:

```text
normalization.constructor.<argument>.<stage>
    argument = model_width | epsilon | device | dtype
    stage = type | value
normalization.input.<stage>
    stage = tensor | rank | dtype | device | model_width |
            sequence_length | finite
normalization.parameters.structure
normalization.parameter.<name>.<stage>
    name = gamma | beta
    stage = type | shape | device | dtype | finite
normalization.<stage>.finite
    stage = mean | centered | squared | variance |
            variance_plus_epsilon | inverse_std | normalized |
            gamma_product | output
normalization.variance_plus_epsilon.positive

gelu.input.<stage>
    stage = tensor | dtype | device | finite
gelu.<stage>.finite
    stage = scaled | erf | one_plus_erf | half_input | output

ffn.constructor.<argument>.<stage>
    argument = model_width | hidden_width | seed | device | dtype
    stage = type | value
ffn.input.<stage>
    stage = tensor | rank | dtype | device | model_width |
            sequence_length | finite
ffn.parameters.structure
ffn.parameter.<name>.<stage>
    name = first_weight | first_bias | second_weight | second_bias
    stage = type | shape | device | dtype | finite
ffn.<stage>.finite
    stage = pre_activation | hidden_activation | output

dropout.constructor.<argument>.<stage>
    argument = probability | seed | device | dtype
    stage = type | value
dropout.input.<stage>
    stage = tensor | rank | dtype | device | model_width |
            sequence_length | finite
dropout.structure
dropout.mode.type
dropout.transaction.snapshot
dropout.transaction.rollback
dropout.uniform.finite
dropout.uniform.range
dropout.keep_mask.structure
dropout.output.finite

block.constructor.<argument>.<stage>
    argument = model_width | number_of_heads | head_width | hidden_width |
               max_sequence_length | layer_norm_epsilon |
               dropout_probability | attention_seed | parameter_seed |
               attention_dropout_seed | feed_forward_dropout_seed |
               device | dtype
    stage = type | value
block.input.<stage>
    stage = tensor | rank | dtype | device | model_width |
            sequence_length | finite
block.submodules.structure
block.dropout_modes.consistent
block.dropout_transaction.snapshot.attention
block.dropout_transaction.snapshot.feed_forward
block.dropout_transaction.rollback
block.parameters.structure
block.parameter.<qualified_name>.<stage>
    qualified_name = norm1.gamma | norm1.beta | norm2.gamma | norm2.beta |
                     feed_forward.first_weight |
                     feed_forward.first_bias |
                     feed_forward.second_weight |
                     feed_forward.second_bias
    stage = type | shape | device | dtype | finite
block.attention_dropout_output.finite
block.first_residual.finite
block.feed_forward_dropout_output.finite
block.output.finite
```

The pipe notation expands to a finite literal set. For example, a wrong
`norm2.beta` shape is exactly `block.parameter.norm2.beta.shape`. Implementation
may not invent content-bearing or dynamically unbounded identifiers.

## Proposed deterministic first-failure order

Each constructor validates argument types in signature order, then exact values
in signature order, before allocation or generator creation.

Representation-valued component calls validate:

1. tensor object type;
2. rank where applicable;
3. dtype;
4. device;
5. final width where applicable;
6. sequence length where applicable;
7. input finiteness;
8. exact owned structure;
9. every owned parameter type in semantic computation order;
10. every owned parameter shape in that order;
11. every owned parameter device;
12. every owned parameter dtype; and
13. every owned parameter's finiteness.

Both block entry points apply this pre-transaction order:

1. common block input validation;
2. exact six-direct-child structure and child-type validation;
3. parent and dropout-child mode validation;
4. complete attention-dropout component and local-generator validation;
5. complete feed-forward-dropout component and local-generator validation;
6. proof that the two validated generator objects are distinct;
7. all new Phase 6 parameter validation; and
8. attention then feed-forward complete generator-state snapshotting.

New parameter semantic order is `norm1` gamma/beta, `norm2` gamma/beta, then
feed-forward first weight/bias and second weight/bias. The accepted attention
validates itself at its public call boundary using its accepted Phase 5 order.
The transaction begins only after step 8 succeeds.

Arithmetic numerical checks occur immediately after their producing operation.
The authoritative transaction section defines rollback for every escaping
failure after snapshotting. No failed public call may mutate caller tensors,
parameter values, parameter identity, module mode, global CPU RNG state, or the
next successful dropout masks.

## Proposed focused test and evidence matrix

Critical expectations must be test-local literals or independently derived
references rather than imports of the production constants or helpers under
test.

### Authority, construction, and parameters

1. Exact constructor types and values fail in deterministic order before any
   allocation or generator creation.
2. The block exposes exactly the six approved child modules in the approved
   order and one accepted Phase 5 attention instance.
3. Exact parameter names, shapes, tensor count 21, and total count 12,576 are
   verified; new Phase 6 count is independently verified as 8,480.
4. There are exactly two independent normalization gamma/beta pairs, exactly
   four FFN tensors, no dropout parameters/buffers, and no parameter sharing.
5. Static checks prove the absence of high-level LayerNorm, GELU, Dropout, or
   Transformer implementations and absence of any attention redesign.

### Initialization and RNG isolation

6. Normalization gamma values are exact ones and beta values exact zeros.
7. A test-local generator seeded literal 6006 reproduces both FFN weights
   bitwise in exact draw order with the two literal fan-in scales; biases are
   exact zero and consume no random draw.
8. The thirteen inherited attention parameters remain bitwise equal to a
   separately constructed accepted Phase 5 module.
9. Independent complete block constructions are bitwise equal and construction
   leaves global CPU RNG state unchanged.
10. Test-local generators seeded 6007 and 6008 independently reproduce exact
    first and later masks at both dropout locations without consulting
    production helpers.

### Normalization

11. Hand-derived rows verify exact population mean, variance, epsilon use,
    normalized values, affine gamma/beta behavior, and `(T, 32)` output.
12. A constant row produces zero normalized values and therefore the learned
    beta after affine transformation.
13. Changing one position leaves all other normalized rows exactly unchanged.
14. Forward/inspection same-call identity, autograd, parameter gradients,
    non-contiguous input, and numerical guards are covered.

### GELU and feed-forward

15. Literal negative, zero, and positive values match an independent exact-erf
    GELU reference and distinguish it from the tanh approximation where chosen
    test values differ.
16. Independent matrix arithmetic verifies `(T,32)->(T,128)->(T,32)`, both
    biases, exact GELU placement, and same-call inspection identities.
17. Changing one input position leaves every other FFN output row exactly
    unchanged.
18. Input and all four FFN parameter gradients are finite, shape-correct, and
    nonzero for a controlled nondegenerate example.

### Dropout

19. Evaluation mode returns identity, exposes an all-true Boolean mask, consumes
    no RNG, and is repeatable.
20. Training mode uses exact keep predicate `uniform >= 0.1`, zeroes dropped
    elements, scales kept elements by exact division by `0.9`, and exposes the
    same mask used for its output.
21. The two sites use distinct deterministic streams; repeated successful calls
    advance them; global RNG is unchanged.
22. Identical blocks with identical successful call histories reproduce both
    masks exactly.
23. The `forward` transaction is independently forced to fail (a) after both
    snapshots but before either draw, (b) after attention dropout advances but
    before feed-forward dropout draws, and (c) after both streams advance. For
    every case, the next successful attention and feed-forward masks from the
    failed block exactly match those from untouched, state-identical control
    blocks, proving restoration of both streams rather than only the stream that
    visibly advanced.
24. The complete three-stage failure matrix in item 23 is repeated through
    `inspect`. A separate injected failure while packaging
    `TransformerBlockInspection`, after both draws, proves that inspection-result
    construction remains inside the transaction and restores both streams.
25. Snapshot failures prove no draw and the exact transaction-error invariant.
    Rollback tests force each restoration to fail independently and together,
    prove both restorations are always attempted in attention-then-feed-forward
    order, verify safe transaction-error ownership and chaining, and never expose
    generator-state bytes. Global CPU RNG remains bitwise unchanged for every
    successful and failing transaction case.
26. `train()` and `eval()` propagate to both dropout components without changing
    parameters or resetting streams; a corrupted mixed-mode block is rejected
    before snapshotting, arithmetic, or RNG consumption.
27. Mode-isolation evidence uses identical controlled inputs and unchanged
    parameters. It proves `norm1`, accepted Phase 5 attention output and weights,
    `norm2`, both FFN projections, GELU, and the pre-dropout FFN output are
    bitwise identical with `torch.equal` across modes when the first learned
    branch is controlled to zero so both modes feed the same first residual
    forward. Complete parameter snapshots remain bitwise unchanged. Separate
    nonzero component tests prove only the two explicit dropout masks and
    post-dropout values are mode-dependent; neither test modifies Phase 5.

### Topology, inspection, and residuals

28. Independent arithmetic verifies exact pre-norm sequencing and exactly two
    branch-output dropout locations before exactly two residual additions.
29. Forward returns only final `(T,32)` output. Inspection invokes every
    prescribed child boundary exactly once and exposes identical same-call
    tensor objects without clone, detach, or recomputation.
30. Inspection distinguishes pre- and post-dropout values and exposes inherited
    attention weights with shape `(4,T,T)` without presenting them as a complete
    explanation.
31. Separate controlled tests prove exact identity values and exact identity
    gradients through the first residual by zeroing the accepted attention
    `output_weight`, through the second residual by zeroing FFN `second_weight`
    and `second_bias`, and through both residuals together. All controlled values
    remain finite and no architecture or parameter structure is changed.

### Shape, causality, gradients, and safety

32. Lengths 1 and 256 succeed with exact output shapes; empty, 257, batched,
    wrong-width, wrong-dtype/device, nonfinite, and non-tensor inputs fail before
    arithmetic or RNG consumption.
33. Evaluation-mode and paired-training-mode future-only perturbations preserve
    protected prefix outputs exactly, with an unmasked-effect control and a
    permitted-history positive control.
34. A nondegenerate synthetic scalar reaches the input and all 21 parameters
    with finite, shape-correct, nonzero gradients and no parameter mutation.
35. Reachable normalization, GELU, projection, dropout-scaling, residual-addition,
    and inherited attention numerical failures raise the exact owning exception
    and invariant without state mutation; defensive-only guards receive direct
    sentinel and static call-order evidence.
36. Simultaneous-defect tests prove the documented first-failure order and safe,
    deterministic diagnostics.

### Regression and phase boundary

37. All accepted Phase 3 embedding, Phase 4 model/loss/runner, and Phase 5
    attention tests remain unchanged and passing.
38. Static tree/import/diff checks prove there is no batching, padding, caller
    mask, cross-attention, stacked block, complete mini-GPT, final normalization,
    LM-head integration, optimizer, training loop, corpus access, checkpoint,
    generation, evaluation, sealed-test access, pretrained model, hosted API,
    or Phase 7 work.

Every Phase 6 test uses small synthetic tensors only. Tests must not read raw or
processed corpus text, instantiate production examples, access *Twelfth Night*,
or persist parameters, gradients, activations, masks, or attention weights.

## Educational demonstrations required before exit acceptance

After implementation is separately authorized, reviewed, and accepted,
Sebastien should inspect small synthetic evidence showing:

1. normalization statistics are per position and across width 32;
2. the two pre-norm equations and exact shape trace;
3. FFN expansion `32 -> 128`, exact GELU, and contraction `128 -> 32`;
4. training dropout masks and `1 / 0.9` scaling versus evaluation identity;
5. both residual identity values and identity gradient routes under zero learned
   branches;
6. exact protected-prefix causality for the complete block; and
7. inherited attention weights alongside the later computations that prevent
   treating them as a complete explanation.

These demonstrations construct only ephemeral synthetic modules and values.
They perform no update, training, experiment, corpus evaluation, checkpoint, or
sealed-test access.

## Explicit Phase 6 exclusions

Phase 6 does not introduce batched block input, padding, caller-configurable
masks, cross-attention, multiple stacked blocks, complete mini-GPT assembly,
final model normalization, language-model head integration, targets, logits,
loss, production sampling/batching, optimizer or training orchestration, corpus
training, checkpoints/resume, generation, final evaluation, sealed-test use,
MPS or mixed-precision optimization, pretrained models, hosted APIs, or Phase 7
work.

It does not modify accepted Phase 3 representation, Phase 4 baseline/experiment,
or Phase 5 attention source, tests, contract, initialization, causality, or
inspection semantics.

## Proposed Phase 6 gates

No gate silently authorizes a later gate. Any correction after independent
review requires focused re-review before acceptance.

1. **Explicit Phase 6 learning/design authorization.** Complete after remotely
   verified Phase 5 closure.
2. **Read-only continuity inspection.** Complete and accepted.
3. **First-principles learning/design discussion.** Complete.
4. **Consolidated conceptual architecture approval.** Complete through DEC-0018.
5. **Explicit documentation-only detailed-contract authorization.** Complete.
6. **Detailed-contract drafting.** Complete in this proposal and accompanying
   continuity updates.
7. **Master Chat detailed-contract sanity review.** Complete with PASS; no later
   gate was authorized by that result.
8. **Fresh independent detailed-contract review.** Complete with FAIL: four
   MUST-FIX and one SHOULD-FIX findings were accepted without changing DEC-0018.
9. **Authorized documentation-only contract correction.** Complete locally in
   this revision for all five findings.
10. **Focused independent detailed-contract re-review.** Complete with PASS: all
    four prior MUST-FIX and the prior SHOULD-FIX finding are resolved, and
    DEC-0018 remains unchanged.
11. **Explicit detailed-contract acceptance.** Complete: Master Chat adjudicated
    PASS and Sebastien explicitly accepted the corrected contract.
12. **Accepted-contract checkpoint authorization.** Complete for one dedicated
    six-file documentation commit titled `Define Phase 6 transformer block
    contract`.
13. **Accepted-contract checkpoint commit.** This authorized dedicated commit
    completes Gate 13. Gate 12 does not itself authorize push.
14. **Contract push and independent remote verification.** Separately authorized
    after the accepted-contract commit.
15. **Explicit Phase 6 implementation authorization.** Required only after Gate
    14 succeeds; no earlier gate authorizes implementation.
16. **Phase 6 implementation and focused synthetic tests.** No training or
    experiment.
17. **Master Chat implementation sanity review.** Review only.
18. **Fresh independent implementation review.** Review only.
19. **Implementation correction, if required.** Only accepted findings may be
    changed.
20. **Focused independent implementation re-review, if correction occurred.**
21. **Explicit implementation acceptance.** Requires a passing independent
    result and separate Sebastien approval.
22. **Accepted-implementation commit authorization.** Separate explicit
    authorization is required after acceptance.
23. **Accepted-implementation commit.** Only the authorized accepted files may
    be staged and committed.
24. **Implementation push and independent remote verification.** Separately
    authorized after Gate 23.
25. **Educational inspection and exact Phase 6 exit review.** Evaluate only the
    three repository exit criteria using accepted synthetic evidence.
26. **Explicit Phase 6 technical acceptance.** Requires a passing exit review
    and separate Sebastien approval.
27. **Phase 6 closure documentation.** Documentation-only status work after
    technical acceptance.
28. **Phase 6 closure commit authorization.** Separate explicit authorization is
    required after closure documentation is reviewed.
29. **Phase 6 closure commit.** Only the authorized closure documentation may be
    staged and committed.
30. **Closure push and independent remote verification.** Separately authorized
    after Gate 29; this is distinct from the closure commit.
31. **Explicit Phase 7 authorization.** Phase 7 cannot begin implicitly and is
    not authorized by successful Phase 6 closure.

## Current authorization boundary

Gates 1–12 are complete. Gate 10 focused independent re-review returned PASS
with all five prior findings resolved; Master Chat adjudicated PASS; Sebastien
explicitly accepted the corrected contract; and DEC-0018 remains unchanged.
The accepted detailed mechanics are authoritative for later Phase 6
implementation. This dedicated six-file accepted-contract commit completes Gate
13. Gate 14 push and independent remote verification require separate
authorization. Phase 6 source, tests, parameters, implementation, experiment,
sealed-test access, push, and Phase 7 work remain unauthorized.
