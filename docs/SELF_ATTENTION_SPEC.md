# SebGPT Phase 5 Self-Attention Specification

## Status and authority

Phase 5 is titled **Self-Attention**. Its repository goal is exactly:

> Derive and implement causal self-attention before using it inside a
> Transformer.

Its exact repository exit criteria remain:

1. Queries, keys, values, scaling, masking, and attention weights are
   explainable.
2. Single-head causal self-attention is implemented transparently and tested.
3. Multi-head attention is built from understood components and tested.
4. Attention shapes and selected weights can be inspected.

Sebastien completed the Phase 5 learning/design discussion, accepted the
conceptual architecture recorded in DEC-0017, and authorized this
documentation-only detailed-contract proposal. Gate 5 drafting completed,
Gate 6 Master Chat sanity review returned PASS, and the fresh Gate 7 Codex
independent review returned FAIL with five MUST-FIX and three SHOULD-FIX
findings. Master Chat accepted all eight findings without changing DEC-0017.
The Gate 8 documentation-only corrections are complete in this revision and
focused independent Gate 9 re-review returned PASS. Master Chat adjudicated the
result as PASS, and Sebastien explicitly accepted the corrected detailed
contract. All five Gate 7 MUST-FIX findings and all three SHOULD-FIX findings
are resolved. DEC-0017's conceptual architecture remains unchanged. The
mechanics below are now authoritative for later separately authorized Phase 5
implementation.

The accepted-contract checkpoint commit is authorized. This does not authorize
push, Phase 5 source, tests, parameter construction, training, experiment,
sealed-test access, or Phase 6 work. Implementation requires the accepted
contract to be pushed, independently remote-verified, and then separately
authorized.

## Approved conceptual architecture

The approved architecture is:

- standalone causal self-attention over the accepted token-plus-position
  representation `X` with shape `(T, 32)`;
- exactly one sequence per call, with `1 <= T <= 256` and no batched attention;
- one full-width head with bias-free query, key, and value weights, each shape
  `(32, 32)`, scaled by `sqrt(32)`, and no output projection;
- causal visibility `j <= i`, so position `i` may use itself and earlier
  positions but never a future position;
- four-head attention with four independently learned, visible heads, each
  receiving the full `(T, 32)` input and owning bias-free `(32, 8)` query, key,
  and value weights;
- per-head scaling by `sqrt(8)`, per-head output `(T, 8)`, concatenation in
  head order to `(T, 32)`, and one bias-free output weight `(32, 32)`;
- ordinary forward calls return only the contextual representation, while an
  explicit inspection path exposes the exact post-softmax weights used for the
  output: `(T, T)` for one head and `(4, T, T)` for four heads; and
- no residual path, normalization, feed-forward layer, dropout, Transformer
  block, stacking, training experiment, checkpoint, generation, or final
  evaluation.

Attention weights are inspectable mathematical quantities. Neither the API nor
project documentation may present them as complete explanations of model
behavior.

## Inherited authority and composition boundary

Phase 5 preserves, rather than redefines, the accepted Phase 2–4 identities:

| Field | Accepted value |
|---|---|
| Tokenizer | `shakespeare-code-point-v1`, schema `1` |
| Vocabulary size | `81`, no special tokens |
| Vocabulary artifact SHA-256 | `9f4235f9dab3e0361221a90c5ae7ca7fee540760fcc1708964368e6f17d6d70e` |
| Model width | `32` |
| Token table | `(81, 32)` |
| Learned absolute-position table | `(256, 32)` |
| Representation | token lookup plus position lookup, then addition |
| Representation implementation | accepted `TokenPositionEmbedding` |
| Semantic device and dtype | CPU `torch.float32` |

The Phase 5 core consumes an already constructed representation tensor. It does
not own, copy, replace, reload, or semantically revalidate the vocabulary,
token table, or position table. It does not accept token IDs, targets, text,
work objects, corpus manifests, or paths. This standalone boundary makes
cross-position information flow inspectable without changing the accepted
Phase 3 representation or Phase 4 baseline.

The accepted Phase 4 `SimpleNeuralLanguageModel`, its four parameters, and
`EXP-20260909-01` remain unchanged and closed. No persisted trained Phase 4
checkpoint exists, so Phase 5 must not load, resume, or claim to extend one.

A later synthetic demonstration may show this shape-only composition:

```text
token IDs -> accepted representation -> X (T, 32)
X -> Phase 5 attention -> contextual representation (T, 32)
contextual representation @ synthetic output weight (32, 81)
    + synthetic output bias (81,) -> logits (T, 81)
```

Such a demonstration uses caller-owned synthetic values only. It is not a new
language-model architecture, a training run, a continuation of Phase 4, or
evidence that attention improves loss or generalization. It is not required to
satisfy a Phase 5 exit criterion.

## Terminology and proposed constants

```text
MODEL_WIDTH = 32
MAX_SEQUENCE_LENGTH = 256
SINGLE_HEAD_WIDTH = 32
NUMBER_OF_HEADS = 4
MULTI_HEAD_WIDTH = 8
ATTENTION_INITIALIZATION_SEED = 5005
PARAMETER_DEVICE = torch.device("cpu")
PARAMETER_DTYPE = torch.float32
```

For a sequence representation `X`:

- row `i` is the representation at destination/query position `i`;
- row `j` is a candidate source/key-value position `j`;
- `Q = X @ W_Q`, `K = X @ W_K`, and `V = X @ W_V`;
- score entry `(i, j)` compares query row `i` with key row `j`;
- visibility entry `(i, j)` is permitted exactly when `j <= i`;
- an attention row is the normalized distribution over source positions used
  to construct one destination output row; and
- a contextual output row is the weighted sum of value rows visible to that
  destination.

The seed `5005` is a proposed Phase-5-specific reproducibility constant. It was
chosen to be explicit and distinct from accepted embedding seed `1337` and
Phase 4 output-head seed `4004`; the number has no learned or mathematical
meaning.

## Proposed module and file boundary

Future authorized implementation belongs in:

```text
src/sebgpt/model/self_attention.py
tests/test_self_attention.py
```

The proposed public API is:

```text
@dataclass(frozen=True)
AttentionInspection:
    output: torch.Tensor = field(repr=False)
    attention_weights: torch.Tensor = field(repr=False)

SingleHeadCausalSelfAttention(
    *,
    model_width: int,
    max_sequence_length: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
)

public parameters:
    query_weight: torch.nn.Parameter   # (32, 32)
    key_weight: torch.nn.Parameter     # (32, 32)
    value_weight: torch.nn.Parameter   # (32, 32)

public methods:
    forward(representations: torch.Tensor) -> torch.Tensor
    inspect(representations: torch.Tensor) -> AttentionInspection

MultiHeadCausalSelfAttention(
    *,
    model_width: int,
    number_of_heads: int,
    head_width: int,
    max_sequence_length: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
)

public submodules:
    heads: torch.nn.ModuleList        # exactly four visible head modules

each head exposes:
    query_weight: torch.nn.Parameter  # (32, 8)
    key_weight: torch.nn.Parameter    # (32, 8)
    value_weight: torch.nn.Parameter  # (32, 8)

public parameter:
    output_weight: torch.nn.Parameter # (32, 32)

public methods:
    forward(representations: torch.Tensor) -> torch.Tensor
    inspect(representations: torch.Tensor) -> AttentionInspection
```

The internal per-head holder is not a separately configurable public
architecture. Its purpose is to keep the four heads and their parameters
individually visible. Multi-head implementation must not replace these visible
heads with one packed `(32, 24)` QKV matrix or another opaque fused operation.

All constructor arguments are required and have no defaults. Exact accepted
values stay visible at every call site. Constructor validation completes before
creating a local generator, allocating a tensor, or registering a parameter.

The single-head constructor accepts only exact values `32`, `256`, `5005`,
`torch.device("cpu")`, and `torch.float32` for its five arguments. The
multi-head constructor accepts only exact values `32`, `4`, `8`, `256`, `5005`,
`torch.device("cpu")`, and `torch.float32` in signature order.

## Proposed parameter ownership and counts

### Single head

| Parameter | Shape | Count |
|---|---:|---:|
| `query_weight` | `(32, 32)` | 1,024 |
| `key_weight` | `(32, 32)` | 1,024 |
| `value_weight` | `(32, 32)` | 1,024 |
| **Total** | | **3,072** |

There are no biases, output weight, buffers, embedding parameters, output-head
parameters, normalization parameters, or other trainable tensors.

### Four heads

| Parameter group | Shape per tensor | Tensor count | Element count |
|---|---:|---:|---:|
| Four query weights | `(32, 8)` | 4 | 1,024 |
| Four key weights | `(32, 8)` | 4 | 1,024 |
| Four value weights | `(32, 8)` | 4 | 1,024 |
| `output_weight` | `(32, 32)` | 1 | 1,024 |
| **Total** | | **13 tensors** | **4,096** |

There are exactly four head submodules. Head order `0, 1, 2, 3` governs
inspection axis zero, concatenation, RNG allocation/fill, and the proposed
head-first semantic validation order. These orders are distinct from standard
PyTorch public parameter enumeration.

Because `output_weight` is a direct parameter of the parent module and `heads`
contains child modules, standard `named_parameters()` enumeration is exactly:

```text
output_weight
heads.0.query_weight
heads.0.key_weight
heads.0.value_weight
heads.1.query_weight
heads.1.key_weight
heads.1.value_weight
heads.2.query_weight
heads.2.key_weight
heads.2.value_weight
heads.3.query_weight
heads.3.key_weight
heads.3.value_weight
```

This output-first enumeration does not select RNG draw order, concatenation
order, inspection order, or semantic validation order. Tests must verify both
orders independently and must not infer one from the other.

## Proposed deterministic initialization

Both public modules initialize independently. Each construction:

1. validates the complete constructor contract;
2. creates one local `torch.Generator(device="cpu")`;
3. seeds it with exact integer `5005`;
4. allocates each required CPU `torch.float32` tensor in the fixed order below;
5. completely fills the tensor with `torch.nn.init.normal_`, using mean `0.0`,
   standard deviation `1 / math.sqrt(32)`, and the local generator explicitly;
6. only then wraps and registers it as `torch.nn.Parameter`.

The single-head RNG allocation/fill order is:

1. `query_weight`
2. `key_weight`
3. `value_weight`

The multi-head RNG allocation/fill order is head-major and Q/K/V-minor:

1. head 0 query, key, value;
2. head 1 query, key, value;
3. head 2 query, key, value;
4. head 3 query, key, value; and
5. `output_weight`.

All thirteen multi-head tensors continue one generator stream; a new generator
must not be seeded per head, because doing so would initialize equal-shaped
heads identically. The output weight uses the same scaled-normal rule because
its input width is 32. No PyTorch default parameter initialization or global
RNG call is permitted. Parameter registration may occur as required to build
the parent and child modules, but it must not consume randomness or redefine
the allocation/fill sequence above.

On the accepted runtime/platform/build, two independent constructions of the
same module produce bitwise-identical parameters, leave the global CPU RNG
state unchanged, and are unaffected by unrelated global random draws. The
guarantee does not promise identical bytes across arbitrary PyTorch versions,
platforms, or builds.

## Proposed representation-input contract

`forward` and `inspect` accept an object only when
`isinstance(representations, torch.Tensor)` is true. Lists, tuples, NumPy
objects, token IDs, and other values are not coerced. A valid tensor subclass is
accepted. Contiguous and non-contiguous tensors are both permitted without a
validation-only contiguous copy.

The tensor must satisfy:

- rank exactly two;
- shape exactly `(T, 32)` with `1 <= T <= 256`;
- dtype exactly `torch.float32`;
- device exactly CPU and equal to every module parameter device;
- every value finite; and
- no caller-visible mutation during validation or computation.

Empty sequences are rejected. The causal softmax contract relies on each row
having at least the diagonal entry, and the approved architecture explicitly
sets `T >= 1`. Batched `(B, T, 32)` input, rank-one input, widths other than 32,
caller-supplied positions, padding masks, arbitrary attention masks, and
cross-attention inputs are rejected or absent from the API.

Validation finishes before Q/K/V projection. The exact forward/inspection
first-failure order is:

1. tensor object type;
2. rank exactly two;
3. dtype exactly `torch.float32`;
4. input device exactly CPU;
5. final dimension exactly 32;
6. sequence length in `1..256`;
7. every input value finite;
8. exact parameter structure and approved names;
9. parameter object type in semantic parameter order;
10. parameter shape in semantic parameter order;
11. parameter CPU device in semantic parameter order;
12. parameter `torch.float32` dtype in semantic parameter order; and
13. parameter finiteness in semantic parameter order.

Single-head semantic parameter order is query, key, then value. Multi-head
semantic parameter order is head 0 Q/K/V through head 3 Q/K/V, then output
weight. This is explicitly a contract validation order, not standard PyTorch
registration or `named_parameters()` enumeration order. Within each validation
category every semantic parameter is checked before moving to the next
category. A failure stops the call before later validation or arithmetic.

Construction and focused tests establish initial parameter ownership,
registration, `requires_grad=True`, and object identity. Ordinary `forward` and
`inspect` do not require original-construction object identity or
`requires_grad=True`. This permits ordinary parameter freezing and compatible
PyTorch state restoration while retaining structure, shape, dtype, device, and
finiteness checks required for correct arithmetic.

## Proposed exceptions and safe diagnostics

Phase 5 owns three public exception classes:

- `AttentionTypeError(TypeError)` for non-tensor inputs and wrong constructor
  argument or parameter object types;
- `AttentionContractError(ValueError)` for invalid configuration, rank, shape,
  length, dtype, device, or parameter structure; and
- `AttentionNumericalError(ArithmeticError)` for nonfinite input, parameter, or
  intermediate/result values and an invalid normalization denominator.

Errors are deterministic and expose only an invariant name and safe structural
facts such as expected/observed rank, shape, dtype, device, parameter name, or
bounds. They must never render tensor values, attention scores or weights,
token sequences, corpus prose, text excerpts, paths, or sealed-test content.

Constructor validation order is argument type in signature order, exact value
in signature order, and only then initialization. Exact built-in integers are
required; `bool` is rejected. Device must be an exact `torch.device` equal to
CPU, and dtype must be exactly `torch.float32`.

### Stable invariant identifiers

Every exception carries one stable, content-safe `invariant` identifier. Tests
assert these literal strings and must not import identifiers from production.
The identifier namespaces and fixed substitutions are:

- constructor arguments:
  `single.constructor.<argument>.type|value` and
  `multi.constructor.<argument>.type|value`, where single-head arguments are
  `model_width`, `max_sequence_length`, `seed`, `device`, and `dtype`, and
  multi-head additionally has `number_of_heads` and `head_width` in signature
  order;
- common input stages: `attention.input.tensor`, `attention.input.rank`,
  `attention.input.dtype`, `attention.input.device`,
  `attention.input.model_width`, `attention.input.sequence_length`, and
  `attention.input.finite`;
- structure: `single.parameters.structure` and `multi.parameters.structure`;
- single-head parameter stages:
  `single.parameter.<name>.type|shape|device|dtype|finite`, where `<name>` is
  exactly `query_weight`, `key_weight`, or `value_weight`;
- multi-head parameter stages:
  `multi.parameter.heads.<h>.<name>.type|shape|device|dtype|finite` for literal
  head indices `<h>` in `0`, `1`, `2`, `3` and exact names `query_weight`,
  `key_weight`, `value_weight`, followed by
  `multi.parameter.output_weight.type|shape|device|dtype|finite`;
- single-head numerical stages: `single.projection.query.finite`,
  `single.projection.key.finite`, `single.projection.value.finite`,
  `single.scores.finite`, `single.scaled_scores.finite`,
  `single.masked_scores.permitted_finite`, `single.row_max.finite`,
  `single.shifted_scores.permitted_finite`,
  `single.exponentials.finite`, `single.denominator.finite`,
  `single.denominator.positive`, `single.weights.finite`, and
  `single.output.finite`;
- per-head multi-head numerical stages:
  `multi.head.<h>.projection.query.finite`, `.projection.key.finite`,
  `.projection.value.finite`, `.scores.finite`, `.scaled_scores.finite`,
  `.masked_scores.permitted_finite`, `.row_max.finite`,
  `.shifted_scores.permitted_finite`, `.exponentials.finite`,
  `.denominator.finite`, `.denominator.positive`, `.weights.finite`, and
  `.output.finite`, with literal `<h>` in `0..3`; and
- final multi-head stages: `multi.concatenated.finite` and
  `multi.output_projection.finite`.

The pipe notation above describes a finite set of literal identifiers; it is
not license to expose arbitrary dynamic text. For example, a wrong head-2 key
shape is exactly `multi.parameter.heads.2.key_weight.shape`. Type identifiers
produce `AttentionTypeError`, structural/value identifiers produce
`AttentionContractError`, and every `finite` or `positive` numerical identifier
produces `AttentionNumericalError`.

## Proposed causal-mask construction

For sequence length `T`, construct fresh position indices on the input device:

```text
row_positions = torch.arange(T, dtype=torch.long, device=X.device)[:, None]
column_positions = torch.arange(T, dtype=torch.long, device=X.device)[None, :]
allowed = column_positions <= row_positions
```

`allowed` has shape `(T, T)` and Boolean dtype. Entry `(i, j)` is true exactly
when `j <= i`. The diagonal is therefore visible. No mask is accepted from a
caller, no padding position exists, and no mask is stored as a learnable
parameter. An implementation may use one arange tensor to derive the two views,
but it may not change the resulting relation.

After scaling, forbidden score entries are replaced with exact negative
infinity:

```text
masked_scores = scaled_scores.masked_fill(~allowed, -torch.inf)
```

The unmasked scaled scores must be finite before masking. Negative infinity is
permitted only where `allowed` is false.

## Proposed transparent stable softmax

The implementation exposes the stable row-wise normalization rather than
delegating the educational mechanism to a high-level attention primitive:

```text
row_max = masked_scores.max(dim=-1, keepdim=True).values
shifted = masked_scores - row_max
unnormalized = torch.exp(shifted)
denominator = unnormalized.sum(dim=-1, keepdim=True)
attention_weights = unnormalized / denominator
```

Every row has at least one permitted finite score because self-visibility is
allowed and `T >= 1`; `row_max` must therefore be finite. Forbidden shifted
entries remain negative infinity, their exponentials are exact zero, and their
post-softmax weights are exact zero. Every denominator must be finite and
strictly positive. Every weight must be finite and nonnegative. Rows normalize
to one subject to float32 rounding.

`torch.nn.MultiheadAttention`, `torch.nn.functional.scaled_dot_product_attention`,
or another fused/high-level attention implementation is not permitted for the
Phase 5 educational core. A test may compare results with an independently
constructed mathematical reference, but that reference is not production
authority.

## Proposed single-head operation

For validated `X` with shape `(T, 32)`:

```text
Q = X @ query_weight                # (T, 32)
K = X @ key_weight                  # (T, 32)
V = X @ value_weight                # (T, 32)
scores = Q @ K.transpose(0, 1)      # (T, T)
scaled_scores = scores / sqrt(32)   # (T, T)
allowed = causal_visibility(T)      # (T, T)
weights = stable_masked_softmax(scaled_scores, allowed)  # (T, T)
output = weights @ V                # (T, 32)
```

The output must be finite CPU `torch.float32`, must not alias or mutate `X` or
a parameter, and must remain connected through autograd to `X` and all three
weights. There is no bias and no operation after the weighted value sum.

For `T = 1`, the only attention weight is exactly one and output row zero equals
value row zero. For all `T`, output row `i` is independent of input rows
`i + 1 .. T - 1`.

## Proposed multi-head operation

For each head `h` in exact order `0..3`:

```text
Q_h = X @ heads[h].query_weight             # (T, 8)
K_h = X @ heads[h].key_weight               # (T, 8)
V_h = X @ heads[h].value_weight             # (T, 8)
scores_h = Q_h @ K_h.transpose(0, 1)         # (T, T)
scaled_scores_h = scores_h / sqrt(8)         # (T, T)
weights_h = stable_masked_softmax(scaled_scores_h, allowed) # (T, T)
head_output_h = weights_h @ V_h              # (T, 8)
```

The same exact causal visibility relation is used for every head. The four
head outputs concatenate in head order along the final dimension:

```text
concatenated = torch.cat(
    (head_output_0, head_output_1, head_output_2, head_output_3),
    dim=-1,
)                                             # (T, 32)
output = concatenated @ output_weight         # (T, 32)
inspection_weights = torch.stack(
    (weights_0, weights_1, weights_2, weights_3),
    dim=0,
)                                             # (4, T, T)
```

The output weight is bias-free. Output projection mixes the concatenated head
features at each position but never mixes sequence positions, so it cannot
weaken causality. The final output remains connected through autograd to `X`,
all twelve head weights, and `output_weight`.

## Proposed numerical checks

Each public computation fails immediately with `AttentionNumericalError` if
any of these required values is nonfinite or invalid:

1. validated input or parameter values;
2. Q, K, or V projections, checked in operation order;
3. raw scores or scaled scores before masking;
4. permitted masked-score entries or each row maximum;
5. shifted permitted scores or their exponentials;
6. normalization denominator, including nonpositive values;
7. attention weights; or
8. per-head, concatenated, projected, or final output as applicable.

Mask-created negative infinity and shifted forbidden entries are expected and
excluded from the relevant finiteness reductions. The implementation must not
silently convert a nonfinite result to a finite value, clamp scores, change
dtype, or retry in higher precision.

### Numerical-failure evidence and reachability

Focused public-boundary tests must cover at least these reachable failures with
finite caller inputs and finite parameters before the named operation:

1. a controlled single-head Q projection overflow raises
   `AttentionNumericalError` with literal invariant
   `single.projection.query.finite`;
2. controlled finite single-head Q and K projections whose dot product
   overflows raise at `single.scores.finite`; and
3. controlled finite multi-head outputs and finite `output_weight` whose final
   matrix product overflows raise at `multi.output_projection.finite`.

Equivalent K/V projection and per-head multi-head paths are covered through a
table-driven test that forces each projection stage in operation order and
asserts its literal invariant. Every public failure test snapshots the caller
input and every parameter beforehand, then proves exact caller-visible value,
shape, dtype, device, and object-identity preservation afterward. It also
asserts the exact exception class and verifies that diagnostics contain only
the literal invariant and safe structural facts, never tensor values.

A successful `T >= 2` instrumented computation must prove that masking creates
negative infinity exactly at forbidden entries, that the permitted entries
remain finite, that the call does not raise a finiteness failure for those
intentional forbidden values, and that their resulting weights are exact zero.

Some later guards are defensive rather than naturally reachable after earlier
invariants:

- division of already finite raw scores by positive `sqrt(32)` or `sqrt(8)`
  cannot newly overflow;
- masked permitted scores are unchanged finite scaled scores;
- each row maximum is finite because the visible diagonal is finite;
- if permitted shifted scores remain finite, their values are nonpositive and
  their exponentials are finite in `[0, 1]`;
- at least one shifted entry per row is zero, so its exponential is one and a
  denominator over at most 256 positions is finite and positive;
- dividing finite exponentials by that denominator yields finite weights; and
- concatenation cannot create a nonfinite value from finite head outputs.

Subtraction at `shifted_scores` can itself produce negative infinity from two
opposite extreme finite scores and therefore has a reachable focused helper
test. Head/output finiteness checks remain mandatory defensive checks against
backend arithmetic results.

For every mandatory stage not naturally reachable through a valid public call,
focused tests use prescribed module-private instrumentation: a spy delegates to
the real finite/positive checker and records the literal invariant sequence in
a successful call; direct checker tests inject a nonfinite or nonpositive
sentinel at each defensive invariant and assert the exact exception; and static
call-order review verifies that the checker is invoked immediately after the
documented producing operation. This evidence is explicitly instrumentation of
defensive branches, not a claim that ordinary black-box inputs naturally reach
them.

## Proposed inspection contract

Each module has one prescribed private boundary:

```text
_compute_attention(representations) -> tuple[output, attention_weights]
```

`forward(X)` invokes that boundary exactly once and returns its `output` object.
`inspect(X)` invokes it exactly once and constructs one frozen
`AttentionInspection` directly from the two returned tensor objects. It does
not clone, detach, replace, reconstruct, or recompute either tensor. It must not
compute an output with one set of weights and rerun attention to obtain a
second set for reporting. The inspection output follows the same validation,
numerical, shape, causality, device, dtype, and autograd contract as ordinary
forward use.

- Single-head inspection weights have exact shape `(T, T)`.
- Multi-head inspection weights have exact shape `(4, T, T)`.
- Axis zero of multi-head inspection is head order `0, 1, 2, 3` and is never
  averaged or collapsed.
- Future entries are exact zero.
- Each row sums to one within the approved float32 tolerance.
- The output is reconstructable from the reported weights, the same-call value
  projections, concatenation, and output projection.

The result dataclass excludes both tensors from its generated representation so
`repr(result)` cannot print tensor contents. Callers may explicitly inspect
selected synthetic weight values. Production or corpus-derived weight artifacts
are outside Phase 5.

Focused same-call evidence independently instruments `_compute_attention` with
a delegating wrapper. For each public module it proves one call exactly,
captures the output and weight object identities returned at that boundary, and
asserts `inspection.output is captured_output` and
`inspection.attention_weights is captured_weights`. A second-computation trap
fails if the boundary is invoked again. Separate numerical reconstruction tests
remain useful for mathematical correctness but are not accepted as proof of
same-call identity.

## Proposed gradient contract

No optimizer, training loop, corpus loss, or update operation belongs to Phase
5. Synthetic tests may create a small valid representation with
`requires_grad=True`, compute a finite scalar from the attention output, and
call `backward()` once.

For a controlled non-degenerate example:

- single-head input gradient and all three parameter gradients exist, have the
  expected shapes, are finite, and have at least one nonzero element;
- multi-head input gradient, all twelve per-head parameter gradients, and the
  output-weight gradient exist, have expected shapes, are finite, and have at
  least one nonzero element; and
- backward does not replace parameter objects or mutate their shapes, devices,
  or dtypes.

This demonstrates differentiability only. It is not a parameter update,
training run, optimization result, or claim about language-model quality.

Inspection weights retain their autograd graph. Separate controlled tests form
a nonconstant scalar objective directly from
`inspection.attention_weights`—for example, an elementwise product with a
test-local nonuniform coefficient matrix followed by a sum—and call
`backward()`. For single-head inspection, the input representation and Q/K
parameters must receive finite gradients with at least one nonzero element. For
multi-head inspection, test-local distinct coefficients on all four weight
matrices must produce finite, nonzero gradients for the input and every head's
Q and K parameters. A weights-only objective has no mathematical dependency on
V or the multi-head output projection, so those gradients may remain `None`
and are not required by this inspection-specific proof.

## Proposed test and evidence contract

Future implementation requires focused deterministic tests whose critical
expected values are literals or independently derived references rather than
imports of the same production constants or helpers under test.

### Constructor, identity, and initialization

1. Exact constructor types and values reject in deterministic order before any
   allocation.
2. Single-head owns exactly three approved bias-free parameters and 3,072
   elements; construction registers each with `requires_grad=True`.
3. Multi-head exposes exactly four ordered heads plus one output weight, owns
   exactly thirteen approved bias-free parameters and 4,096 elements, and has
   no packed QKV tensor; construction registers each with
   `requires_grad=True`.
4. Single-head `named_parameters()` is exactly query, key, value. Multi-head
   `named_parameters()` is the literal output-first thirteen-name sequence
   documented above, distinct from head-major RNG allocation and semantic
   validation order.
5. A test-local independent CPU generator is seeded with literal `5005`. It
   allocates literal shapes `(32, 32)`, `(32, 32)`, `(32, 32)` in single-head
   Q/K/V order and fills each using literal `mean=0.0` and
   `std=1 / math.sqrt(32)`. Every actual single-head parameter is bitwise equal
   to its independently generated expected tensor.
6. A fresh test-local independent CPU generator is seeded with literal `5005`.
   It allocates twelve literal `(32, 8)` tensors in head-0 Q/K/V through head-3
   Q/K/V order, then one literal `(32, 32)` output tensor, filling every tensor
   with literal `Normal(0, 1 / math.sqrt(32))`. Each expected tensor is mapped
   to its semantic attribute—not `named_parameters()` position—and compared
   bitwise with the actual parameter. The test imports no production constant
   or initialization helper and never resets the reference generator, proving
   exact shapes, seed, distribution, draw order, and one uninterrupted stream.
7. Two independent constructions are bitwise identical on the supported
   runtime, unrelated global RNG draws do not affect them, and construction
   preserves global CPU RNG state.
8. Distinct equal-shaped multi-head parameters are not all initialized
   identically, proving one continuing generator stream rather than reseeding
   each head.

### Input and safety boundary

9. Non-tensor, wrong rank, dtype, device, width, empty length, excessive length,
   and nonfinite inputs fail in exact order without projection.
10. Valid tensor subclasses and non-contiguous `(T, 32)` views are accepted.
11. Input and parameters are not mutated by `forward`, `inspect`, or any
    rejected call.
12. Corrupted parameter structure, type, shape, device, dtype, and finiteness
    fail in the fixed semantic validation order before attention arithmetic.
    Per-call tests do not require original-construction identity or
    `requires_grad=True`.
13. Frozen parameters and parameters restored compatibly through normal
    PyTorch state loading retain valid forward/inspection behavior.
14. Every failure asserts a literal stable invariant identifier and exact
    exception class without importing the identifier from production;
    diagnostics contain safe structural facts and no tensor values or corpus
    content.

### Numerical-failure guards

15. Finite inputs and parameters causing Q, K, or V projection overflow fail at
    the exact first projection invariant for single-head and every multi-head
    position, with no caller-visible mutation and content-safe diagnostics.
16. Finite Q/K projections whose dot product overflows fail at the literal raw
    score invariant before scaling or masking.
17. Finite multi-head outputs and finite output weight whose product overflows
    fail at `multi.output_projection.finite`.
18. Instrumented valid `T >= 2` attention accepts mask-created forbidden
    negative infinity, excludes it from inappropriate finiteness failures, and
    produces exact zero future weights.
19. A delegating checker spy, direct defensive-branch injection, and static
    call-order review cover every remaining mandatory numerical guard according
    to the reachability analysis above, including shifted-score overflow and
    all explicitly identified defensive-only stages.

### Single-head mathematics and causality

20. Q, K, and V have shape `(T, 32)` and match independently derived matrix
    references.
21. Raw and scaled score shapes are `(T, T)` and scaling is exactly by
    `sqrt(32)`.
22. The causal visibility matrix is exactly `j <= i`, including the diagonal.
23. Every future post-softmax weight is exactly zero.
24. All permitted weights are finite and nonnegative, and each attention row
    sums to one with `rtol=0` and `atol=1e-6`.
25. The reported output equals an independent weighted-value reference within
    `rtol=1e-5`, `atol=1e-6` and has shape `(T, 32)`.
26. `T=1` produces weight `[[1.0]]` exactly and output equal to `V`.
27. With controlled single-head projections, two inputs share an exact prefix
    and differ by a deliberate finite future-only perturbation large enough to
    change an independently computed unmasked result. Protected prefix outputs
    are exactly equal with `torch.equal`, proving structural causal invariance.
28. The equivalent controlled four-head test, with a controlled finite output
    projection, proves exact protected-prefix equality and independently shows
    the perturbation would be visible without masking.
29. A separate controlled positive test changes an allowed earlier input and
    produces a clear later-output change for both single-head and multi-head
    attention, proving permitted information flow rather than only future
    blocking.

### Multi-head mathematics and inspection

30. Every head has Q/K/V shape `(T, 8)`, score and weight shape `(T, T)`, scale
    `sqrt(8)`, and output shape `(T, 8)`.
31. Independently derived head results match implementation results without
    using the production attention helper.
32. Concatenation in head order has exact shape `(T, 32)` and preserves
    independently identifiable controlled head slices.
33. `output_weight` maps `(T, 32)` to `(T, 32)` with no bias and no
    cross-position mixing.
34. Multi-head future weights are exact zero and every permitted row of every
    head normalizes within `rtol=0`, `atol=1e-6`.
35. Inspection exposes exact shape `(4, T, T)` with four separate, correctly
    ordered heads; it never averages them.
36. Single-head inspection exposes exact shape `(T, T)`.
37. Each inspection output is reconstructed from the weights returned by that
    same inspection call within `rtol=1e-5`, `atol=1e-6`.
38. Independent boundary instrumentation proves `inspect()` calls
    `_compute_attention` once exactly and returns the identical output and
    weight objects produced by that call; a second-computation trap remains
    untriggered.
39. A controlled nonconstant scalar formed directly from inspection weights
    produces finite, nonzero input and Q/K gradients for single-head and every
    multi-head head. No V or output-weight gradient is required from this
    weights-only objective.

### Gradients, inherited regressions, and phase boundary

40. Controlled single-head and multi-head output scalar examples produce finite,
    shape-correct, nonzero input and parameter gradients for every approved
    attention parameter.
41. Existing Phase 3 vocabulary/embedding and Phase 4 example/model/loss/runner
    tests remain unchanged and passing.
42. Import and static-diff review prove there is no batched attention, caller
    mask, padding, cross-attention, residual path, normalization, feed-forward
    layer, dropout, Transformer block, stacking, optimizer, training loop,
    checkpoint, generation, final evaluation, sealed-test access, high-level
    attention primitive, pretrained model, or hosted model API.

Tests use synthetic representation tensors only. They must not read raw or
processed corpus text, load `EXP-20260909-01`, access *Twelfth Night*, persist
attention values, or construct a training experiment.

## Educational demonstrations required before exit acceptance

After implementation acceptance, but before declaring Phase 5 complete,
Sebastien should inspect small synthetic examples that make these facts visible:

1. one token produces the single weight `1`;
2. a three-position causal matrix has visible pattern
   `[[1,0,0], [*,*,0], [*,*,*]]` after softmax, with each row normalized;
3. changing only a future input leaves an earlier output unchanged;
4. changing an allowed earlier input can change a later output;
5. Q/K/V, score, weight, value-sum, per-head, concatenated, and projected shapes
   are traced explicitly; and
6. four head matrices are inspected separately, alongside the reminder that
   weights are not complete explanations.

These demonstrations may instantiate only synthetic Phase 5 modules after
implementation has been separately authorized and accepted. They perform no
parameter update and create no durable model or attention artifact.

## Phase boundary

Phase 5 introduces only standalone single-head and four-head causal
self-attention plus synthetic inspection and gradient evidence. It does not
modify the accepted Phase 1–4 architecture, reopen the Phase 4 experiment, or
claim improved loss, generalization, or test performance.

Phase 5 explicitly excludes residual connections, normalization, feed-forward
layers, dropout, Transformer-block composition, stacked Transformer blocks,
production batching, optimization/training infrastructure, corpus training,
validation or sealed-test evaluation, checkpoints and resume behavior,
generation, final evaluation, MPS/mixed-precision optimization, pretrained
models, hosted APIs, padding, cross-attention, and caller-configurable masks.

Phase 6 owns Transformer-block composition. Phase 7 owns complete mini-GPT
assembly. Phase 8 owns production training/batching and checkpointing. Phase 9
owns generation and final evaluation, including any separately authorized
sealed-test event.

## Proposed Phase 5 gates

No gate silently authorizes a later gate. Any correction after independent
review requires focused re-review before acceptance.

1. **Explicit Phase 5 learning/design authorization.** Complete through the
   Gate 30 authorization after remotely verified Phase 4 closure.
2. **First-principles learning and repository orientation.** Complete.
3. **Conceptual architecture approval.** Complete: Sebastien accepted the
   architecture recorded in DEC-0017.
4. **Explicit detailed-contract authorization.** Complete for documentation
   only.
5. **Detailed contract drafting.** Complete in the initial proposal and its
   continuity updates.
6. **Master Chat sanity review.** Complete with PASS.
7. **Fresh independent detailed-contract review.** Complete with FAIL: five
   MUST-FIX and three SHOULD-FIX findings were accepted by Master Chat without
   changing DEC-0017.
8. **Documentation correction, if required.** Complete locally in this
   revision for all eight accepted findings, without implementation.
9. **Focused independent re-review after corrections and explicit contract
   acceptance.** Complete: focused review returned PASS, Master Chat
   adjudicated PASS, and Sebastien explicitly accepted the corrected contract
   with all eight Gate 7 findings resolved and DEC-0017 unchanged.
10. **Accepted-contract checkpoint commit.** Separately authorized only after
    Gate 7 or Gate 9 passes and Sebastien accepts the contract. Authorized for
    this dedicated six-file documentation checkpoint.
11. **Contract push and independent remote verification.** Separately
    authorized after Gate 10.
12. **Explicit Phase 5 implementation authorization.** Required after the
    accepted contract is remotely verified.
13. **Single-head and multi-head implementation with focused synthetic tests.**
    No training or experiment.
14. **Master Chat implementation sanity check and fresh independent review.**
15. **Implementation corrections, focused re-review when corrections occur,
    and explicit implementation acceptance.**
16. **Accepted-implementation commit.** Separately authorized after acceptance.
17. **Implementation push and independent remote verification.** Separately
    authorized after Gate 16.
18. **Educational inspection and exact Phase 5 exit review.** Evaluate only the
    four repository exit criteria using accepted synthetic evidence.
19. **Explicit Phase 5 acceptance and closure documentation.** Only after a
    passing exit review.
20. **Phase 5 closure commit and later push/remote verification.** Each action
    separately authorized.
21. **Explicit Phase 6 authorization.** Phase 6 cannot begin implicitly.

## Current authorization boundary

Gates 1–9 are complete. The corrected detailed mechanics are accepted and
authoritative for later implementation. Gate 10's dedicated accepted-contract
checkpoint commit is authorized. Push, source, tests, parameters,
implementation execution, training, experiment, sealed-test access, and Phase
6 work remain unauthorized.
