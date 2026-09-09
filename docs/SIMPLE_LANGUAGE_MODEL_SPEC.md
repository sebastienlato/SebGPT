# SebGPT Phase 4 Simple Neural Language Model Specification

## Status and authority

Phase 4 is titled **Simple Neural Language Model**. Its repository goal is
exactly:

> Build and train a small non-attention baseline to understand next-token
> prediction.

Its exact exit criteria remain:

1. Inputs, targets, logits, probabilities, and cross-entropy loss are
   understood.
2. A simple neural language model is implemented and tested.
3. Backpropagation and parameter updates are inspected on a small example.
4. A reproducible run shows loss improving over a baseline.

Sebastien accepted the conceptual architecture recorded in DEC-0016 and
authorized its documentation-only detailed contract. Independent Gate 6 review
returned FAIL with seven must-fix findings. Gate 7 corrected all seven without
changing the conceptual design. Focused independent Gate 8 re-review returned
PASS, master-chat adjudication returned PASS, and Sebastien explicitly accepted
this corrected detailed contract. Every API, exception, initialization,
training, experiment, success-rule, test, and gate mechanic below is accepted
contract authority, but acceptance does not authorize implementation or an
experiment.

The Gate 8 PASS confirmed that all seven Gate 6 findings are resolved, the
twelve conceptual decisions remain unchanged, and no accepted Phase 1–3
authority is contradicted.

This checkpoint authorizes no source, test, model-parameter, example-artifact,
experiment, commit, or push work. In particular, it does not authorize access
to the sealed test work.

## Accepted conceptual architecture

The accepted Phase 4 design is:

- a linear/log-bilinear non-attention baseline over the accepted Phase 3
  token-plus-position representation;
- no communication between sequence positions, so a prediction's effective
  token context is exactly its token at the same position;
- equal-length inputs and targets shifted by one token;
- maximum example input length 64 and stride 64;
- a final variable-length example whenever at least one adjacent transition
  remains, with no padding or padding token;
- deterministic on-demand examples that are never persisted;
- one runtime example at a time, without production batching or random
  sampling;
- an untied explicit output weight of shape `(32, 81)` and bias of shape
  `(81,)`;
- transparent, numerically stable cross-entropy from logits;
- explicit manual SGD for the inspected update and bounded training run;
- parameter updates from the six training works only, evaluation-only use of
  *The Tempest*, and no Phase 4 access to *Twelfth Night*; and
- `ln(81)` as the fixed conceptual uniform baseline, with initial and final
  aggregate training and validation losses reported as observations.

No recurrence, convolution, attention, or other cross-position transformation
is permitted. A length-64 dataset example groups 64 adjacent predictions; it
does not give this baseline a 64-token effective receptive field.

## Inherited accepted authority

Phase 4 preserves, rather than redefines:

| Field | Accepted value |
|---|---|
| Tokenizer ID | `shakespeare-code-point-v1` |
| Vocabulary artifact schema | `1` |
| Vocabulary size | `81` |
| Vocabulary artifact path | `artifacts/tokenizers/shakespeare-code-point-v1/vocabulary.json` |
| Vocabulary artifact SHA-256 | `9f4235f9dab3e0361221a90c5ae7ca7fee540760fcc1708964368e6f17d6d70e` |
| Special tokens | none |
| Embedding dimension | `32` |
| Token table | `(81, 32)` |
| Position table | `(256, 32)` |
| Representation | token lookup plus learned absolute-position lookup |
| Representation parameters | `10,784` |
| Representation initialization seed | `1337` |
| Parameter device and dtype | CPU `torch.float32` |
| Token and target dtype | `torch.long` |

The accepted `VocabularyBinding` loader remains the only factory for runtime
vocabulary identity. The accepted `TokenPositionEmbedding` remains the
representation implementation. Phase 4 must compose it without copying,
regenerating, modifying, or semantically revalidating its vocabulary or tables.

## Accepted terminology and constants

```text
VOCABULARY_SIZE = 81
EMBEDDING_DIM = 32
MAX_POSITIONS = 256
CONTEXT_LENGTH = 64
STRIDE = 64
EMBEDDING_SEED = 1337
OUTPUT_HEAD_SEED = 4004
LEARNING_RATE = 0.05
TRAINING_PASSES = 2
```

A **source token index** identifies a token inside exactly one independently
processed document. Transition index `i` is the prediction relation from source
token `z_i` to source token `z_(i+1)`. An **example** contains one immutable
input tuple, its shifted target tuple, and the source start index. A
**collection** is a deterministic iteration of examples. It is not a runtime
batch. Phase 4 converts only the current example to rank-one tensors.

## Accepted corpus-neutral shifted-example contract

### Public data model and API

The low-level layer is corpus-neutral and contains no Shakespeare work, split,
path, manifest, or sealed-test knowledge.

```text
ShiftedTokenExample(
    start_index: int,
    input_ids: tuple[int, ...],
    target_ids: tuple[int, ...],
)

iter_shifted_examples(
    token_ids: tuple[int, ...],
    *,
    context_length: int,
    stride: int,
) -> Iterator[ShiftedTokenExample]
```

`ShiftedTokenExample` is a frozen dataclass. Its two ID fields are excluded
from `repr`; a representation may expose only `start_index` and lengths. The
tuples are newly sliced immutable values and never alias a mutable caller
container.

The function accepts only an exact built-in tuple for `token_ids`. Every
element must be an exact built-in integer, so `bool` is rejected, and satisfy
`0 <= token_id < 81`. `context_length` and `stride` must be exact built-in
integers equal to 64. The strict fixed arguments keep the Phase 4 configuration
visible without silently generalizing it.

Validation is eager: every input invariant is checked before the iterator is
returned. The public function is a non-generator outer function: it performs
all validation and then returns a separate private inner iterator. Validation
order and exception ownership are:

1. `token_ids` exact tuple type, else `Phase4TypeError`;
2. `context_length` exact built-in integer type, else `Phase4TypeError`;
3. `context_length == 64`, else `Phase4ContractError`;
4. `stride` exact built-in integer type, else `Phase4TypeError`;
5. `stride == 64`, else `Phase4ContractError`;
6. token elements from left to right: exact built-in integer type, else
   `Phase4TypeError`; then range `0..80`, else `Phase4ContractError`.

No partial iterator is returned after failure. Errors expose an invariant name,
the first invalid position when applicable, and safe expected bounds only;
they never expose an input ID sequence or value.

### Start indices and example construction

For `N = len(token_ids)`:

- if `N <= 1`, the iterator is empty;
- otherwise, start indices are exactly `range(0, N - 1, 64)`;
- for each start `s`, define `L = min(64, N - 1 - s)`;
- `input_ids = token_ids[s : s + L]`;
- `target_ids = token_ids[s + 1 : s + L + 1]`.

Every yielded example therefore satisfies:

- `0 <= start_index <= N - 2`;
- `1 <= L <= 64`;
- `len(input_ids) == len(target_ids) == L`; and
- `target_ids[j] == token_ids[start_index + j + 1]`.

Iteration order is strictly increasing source start index. The helper consumes
one already independent document tuple per call. It cannot accept multiple
documents, concatenate inputs, infer a split, tokenize text, or open any file.

### Exact boundary behavior

| Source length `N` | Start indices | Example lengths | Covered transitions |
|---:|---|---|---|
| 0 | none | none | none |
| 1 | none | none | none |
| 2 | `0` | `1` | `0` |
| 63 | `0` | `62` | `0..61` |
| 64 | `0` | `63` | `0..62` |
| 65 | `0` | `64` | `0..63` |
| 66 | `0, 64` | `64, 1` | `0..64` |
| 128 | `0, 64` | `64, 63` | `0..126` |
| 129 | `0, 64` | `64, 64` | `0..127` |
| 130 | `0, 64, 128` | `64, 64, 1` | `0..128` |

More generally, for `N >= 1`, write `N - 1 = 64q + r` with
`0 <= r < 64`. Construction yields `q` full length-64 examples and, exactly
when `r > 0`, one length-`r` tail. There is no padding and no empty example.

### Transition-coverage invariant and proof

The transition indices for one document are the integers `0..N-2`. Example
`k`, starting at `s_k = 64k`, covers:

```text
[64k, min(64k + 63, N - 2)]
```

Two consecutive ranges end at `64k + 63` and begin at `64(k + 1)`, so they are
disjoint and adjacent. The first begins at zero. The final range ends at
`N - 2` because its length is `min(64, N - 1 - s_k)`. The ranges therefore
partition `0..N-2`: every permitted adjacent transition appears exactly once,
none is omitted, and none is duplicated.

The proof applies to one function call and hence one document. Because no call
accepts more than one token tuple, the low-level layer cannot create a
cross-document transition.

### On-demand and non-persistence behavior

The validated source tuple may remain referenced by the iterator. Only the
current input and target slices, each at most 64 IDs, are materialized when
requested. The implementation creates no list or tuple of all examples, no
dataset object containing every tensor, and no filesystem output. Callers may
collect synthetic results in tests, but production orchestration and the future
experiment must consume examples incrementally and discard each example after
its update or measurement.

## Accepted Shakespeare governance contract

### Separation from the neutral helper

The Shakespeare-specific layer owns dataset identity, work identity, split,
manifest order, processed provenance, text access, encoding, and sealed-test
enforcement. The neutral helper owns only the arithmetic above.

Conceptually, production orchestration provides:

```text
iter_shakespeare_phase_4_examples(
    works: tuple[ExtractedWork, ...],
    vocabulary: VocabularyBinding,
    phase_1_manifest: Mapping[str, object],
    *,
    split: str,
) -> Iterator[DocumentShiftedExample]
```

`split` accepts exactly `train` or `validation`; `test` and every other value
fail before work content access. A frozen `DocumentShiftedExample` wraps safe
training/validation metadata (`work_id`, `manifest_order`, `split`) and one
`ShiftedTokenExample`, with ID tuples excluded from `repr`.

The implementation may refine module-private helper structure during review,
but it may not weaken any observable behavior in this section.

The orchestration public operation validates in this exact first-failure order:

1. `works` is an exact tuple, else `Phase4TypeError`;
2. `phase_1_manifest` is a `Mapping`, else `Phase4TypeError`;
3. `split` is an exact built-in string, else `Phase4TypeError`;
4. `vocabulary` is a `VocabularyBinding`, else `Phase4TypeError`;
5. the binding is factory-verified, else `Phase4ContractError`;
6. `split` is exactly `train` or `validation`, else
   `Phase4GovernanceError`;
7. manifest schema and dataset identity, else `Phase4GovernanceError`;
8. expected count for the requested split, else `Phase4GovernanceError`;
9. each work object has the accepted `ExtractedWork` boundary, in tuple order,
   else `Phase4TypeError` at the first invalid position;
10. every supplied work's safe identity, order, split, normalization, hash,
    byte count, and code-point count, in tuple order, else
    `Phase4GovernanceError` at the first mismatch; and
11. only after all prior checks succeed, permitted text access and recomputed
    text provenance in tuple order, with the first mismatch producing
    `Phase4GovernanceError`.

A simultaneous wrong split and bad work provenance therefore fails at split
validation; simultaneous metadata defects fail at the first work and first
field in the fixed field order above. No later content or invariant is touched
after the first failure.

### Complete metadata validation before text access

Before accessing `processed_text` on any supplied work, orchestration validates
all supplied work metadata against the accepted Phase 1 manifest:

- dataset and manifest schema identity;
- exact requested split;
- exact work count;
- exact identities and manifest order;
- no duplicate, missing, unexpected, reordered, or test work;
- exact normalization identity;
- processed SHA-256, byte count, and code-point count; and
- a factory-produced accepted vocabulary binding with the inherited identity.

Training accepts exactly the six works in manifest order 1 through 6:

1. `hamlet`
2. `romeo-and-juliet`
3. `macbeth`
4. `a-midsummer-nights-dream`
5. `much-ado-about-nothing`
6. `henry-v`

Validation accepts exactly `the-tempest`, manifest order 7. No Phase 4
orchestration accepts or extracts a test work. Its upstream producer must
filter safe work specifications before extracting or loading any text; calling
an all-eight-work content loader and discarding the test afterward is
forbidden.

After every safe metadata check succeeds, orchestration accesses permitted
texts one document at a time, strictly UTF-8 re-encodes them, recomputes their
byte count, SHA-256, and code-point count, and compares all three to both the
work provenance and manifest. It then creates a `CodePointTokenizer` from the
accepted binding's immutable code points, encodes the permitted document in
memory, and passes that one immutable ID tuple to the neutral iterator. It
never joins texts or ID tuples.

The actual test file is never opened, decoded, encoded, windowed, scored, or
measured. Tests must use a sentinel object whose content property raises and
must prove rejection before that property is read.

## Accepted tensor boundary

Example construction uses plain tuples because document/window
semantics do not require PyTorch and immutable tuples make exact examples easy
to inspect. Immediately before model use, only the current example is converted
to:

- one rank-one CPU `torch.long` input tensor of shape `(L,)`; and
- one rank-one CPU `torch.long` target tensor of shape `(L,)`.

Both tensors are ephemeral. Rank-two model input is intentionally excluded
from the Phase 4 model and training API. Existing accepted Phase 3 rank-two
support and synthetic tests remain unchanged; Phase 4 simply does not use that
optional representation boundary.

## Accepted simple neural language model contract

### API and composition

```text
SimpleNeuralLanguageModel(
    vocabulary: VocabularyBinding,
    *,
    embedding_dim: int,
    max_positions: int,
    embedding_seed: int,
    output_head_seed: int,
    device: torch.device,
    dtype: torch.dtype,
)

public submodule:
    representation: TokenPositionEmbedding

public parameters:
    output_weight: torch.nn.Parameter
    output_bias: torch.nn.Parameter

public method:
    forward(token_ids: torch.Tensor) -> torch.Tensor
```

All keyword arguments are required and accept only exact inherited or accepted
values: 32, 256, 1337, 4004, `torch.device("cpu")`, and `torch.float32`.
Configuration and head-seed validation occurs before parameter allocation. The
model constructs exactly one accepted `TokenPositionEmbedding` as
`representation`; it does not copy or replace its parameters.

Model construction uses this exact first-failure validation order:

1. `vocabulary` is a `VocabularyBinding`, else `Phase4TypeError`;
2. `embedding_dim`, `max_positions`, `embedding_seed`, and `output_head_seed`,
   in that order, are exact built-in integers, else `Phase4TypeError`;
3. `device` is an exact `torch.device` and `dtype` is a `torch.dtype`, in that
   order, else `Phase4TypeError`;
4. the vocabulary binding is factory-verified, else `Phase4ContractError`;
5. configuration values, in argument order, are exactly 32, 256, 1337, 4004,
   CPU, and `torch.float32`, else `Phase4ContractError` at the first mismatch;
6. only then construct the accepted representation and initialize the output
   head.

All Phase 4 configuration checks therefore finish before any parameter is
allocated. Accepted tokenizer or embedding exceptions retain ownership only
after this Phase 4 constructor boundary passes.

`forward` accepts `isinstance(token_ids, torch.Tensor)` inputs, including valid
tensor subclasses, but requires rank one. It then preserves the accepted
embedding invariants: CPU, `torch.long`, `0 <= L <= 64`, IDs in `0..80`, and no
input mutation. Although the representation can accept up to 256 positions,
the Phase 4 model rejects `L > 64` before representation lookup.

`forward` validates in this exact order:

1. tensor object type, else `Phase4TypeError`;
2. rank exactly one, else `Phase4ContractError`;
3. dtype exactly `torch.long`, else `Phase4ContractError`;
4. input and all four parameter devices exactly CPU, else
   `Phase4ContractError`;
5. sequence length `0 <= L <= 64`, else `Phase4ContractError`;
6. every ID in `0..80`, with empty input skipping reduction, else
   `Phase4ContractError`; and
7. only then representation lookup and output-head arithmetic.

After validation:

```text
representations = representation(token_ids)        # (L, 32)
logits = representations @ output_weight + output_bias  # (L, 81)
```

`forward` returns logits only. It performs no softmax, target access, loss,
backward call, update, state mutation, recurrence, or cross-position operation.
Each logits row depends only on the representation row at the same position.

Empty rank-one input remains composable and returns logits of shape `(0, 81)`,
but it is never a shifted training example and is rejected by the loss contract.

### Parameter ownership and count

The model owns exactly four learnable parameters:

| Parameter | Shape | Count |
|---|---:|---:|
| `representation.token_embeddings` | `(81, 32)` | 2,592 |
| `representation.position_embeddings` | `(256, 32)` | 8,192 |
| `output_weight` | `(32, 81)` | 2,592 |
| `output_bias` | `(81,)` | 81 |
| **Total** | | **13,457** |

There are no other parameters or registered buffers. All four parameters are
CPU `torch.float32` with `requires_grad=True`, and all four are updated during
Phase 4 training. The Phase 3 tables are not frozen. Weight tying is forbidden:
`output_weight` is a distinct allocation and does not alias or transpose-alias
`token_embeddings`.

Because PyTorch enumerates parameters registered directly on the root module
before parameters in child modules, `tuple(model.named_parameters())` is
required to contain exactly, in order:

1. `output_weight`;
2. `output_bias`;
3. `representation.token_embeddings`;
4. `representation.position_embeddings`.

## Accepted output-head initialization

`OUTPUT_HEAD_SEED = 4004` is a Phase-4-specific seed. It is intentionally
different from 1337, is mnemonic for the phase and component, and prevents the
head from restarting the exact Phase 3 random stream. Seed choice has no claim
of statistical superiority.

After the complete model configuration and vocabulary binding pass validation:

1. Construct the accepted representation, which privately uses its accepted
   local CPU generator seeded with 1337.
2. Create a distinct local `torch.Generator(device="cpu")` for the head.
3. Seed that generator with exact built-in integer 4004.
4. Allocate an empty CPU `torch.float32` tensor of shape `(32, 81)`.
5. Fill every element with `torch.nn.init.normal_`, mean `0.0`, standard
   deviation `1 / sqrt(32)`, and the head generator explicitly.
6. Wrap it as `output_weight`.
7. Allocate an exact-zero CPU `torch.float32` tensor of shape `(81,)` without
   consuming any RNG value and wrap it as `output_bias`.

Head initialization consumes no global RNG state. Together with the accepted
representation initializer, complete model construction preserves the global
CPU RNG state. Independent constructions are bitwise equal only under the same
supported runtime, platform, PyTorch build, implementation, vocabulary
identity, device, dtype, seeds, draw order, and initialization primitives. No
cross-version, MPS, or future-training bitwise promise is made.

## Accepted logits and probability contract

For input length `L`:

- representations have shape `(L, 32)`;
- logits have shape `(L, 81)`; and
- derived probabilities have shape `(L, 81)`.

Row `j` contains the scores or probabilities for target token `y[j]`.
Vocabulary class dimension is exactly final dimension 1 for rank-two logits.

Probabilities are derived only for educational inspection by a separate
public `probabilities_from_logits(logits)` operation. Its input contract is
independent from the positive-length loss contract. It accepts exact shape
`(L, 81)` for every integer `L >= 0` and validates in this exact order:

1. `logits` is a tensor object, else `Phase4TypeError`;
2. rank is exactly two, else `Phase4ContractError`;
3. dtype is exactly `torch.float32`, else `Phase4ContractError`;
4. device is exactly CPU, else `Phase4ContractError`;
5. final class dimension is exactly 81, else `Phase4ContractError`;
6. sequence length satisfies `L >= 0`, explicitly including zero, else
   `Phase4ContractError`; and
7. every element is finite, else `Phase4ContractError`.

For `L = 0`, it returns a new empty CPU `torch.float32` tensor of exact shape
`(0, 81)` without calling a row reduction. It does not route through or reuse
the loss input validator. For `L > 0`, it computes along `dim=1`:

```text
row_max = logits.max(dim=1, keepdim=True).values
shifted = logits - row_max
unnormalized = exp(shifted)
probabilities = unnormalized / unnormalized.sum(dim=1, keepdim=True)
```

It returns a new CPU `torch.float32` tensor connected to logits through
autograd, although Phase 4 loss must not consume these probabilities. Each
nonempty accepted row is finite, nonnegative, and sums to one within the
accepted float32 comparison tolerance.

## Accepted explicit cross-entropy contract

```text
explicit_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor
```

Accepted logits are rank-two CPU `torch.float32`, shape `(L, 81)`, with every
element finite. Accepted targets are rank-one CPU `torch.long`, shape `(L,)`,
with every value in `0..80`. `L` must be positive. Neither input is mutated or
coerced.

Validation order and exception ownership are:

1. logits tensor object type, else `Phase4TypeError`;
2. targets tensor object type, else `Phase4TypeError`;
3. logits rank exactly two, else `Phase4ContractError`;
4. target rank exactly one, else `Phase4ContractError`;
5. logits dtype exactly `torch.float32`, else `Phase4ContractError`;
6. target dtype exactly `torch.long`, else `Phase4ContractError`;
7. both devices exactly CPU and equal, else `Phase4ContractError`;
8. logits class dimension exactly 81, else `Phase4ContractError`;
9. leading lengths equal, else `Phase4ContractError`;
10. length positive, else `Phase4ContractError`;
11. all logits finite, else `Phase4ContractError`;
12. every target in `0..80`, else `Phase4ContractError`.

After complete validation, calculate:

```text
row_max = logits.max(dim=1, keepdim=True).values
shifted = logits - row_max
shifted_exp = exp(shifted)
shifted_sum = shifted_exp.sum(dim=1)
log_shifted_sum = log(shifted_sum)
target_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
target_margin = row_max.squeeze(1) - target_logits
negative_log_likelihood = log_shifted_sum + target_margin
scaled_rows = negative_log_likelihood / L
loss = scaled_rows.sum()
```

Subtracting the target logit from the row maximum before adding the small
`log_shifted_sum` term avoids reconstructing an absolute log normalizer that
could overflow and then attempting to cancel it. Dividing rows before the
reduction likewise avoids a representable mean failing only because an
unnecessary unscaled sum overflowed.

Finite input logits do not guarantee that the mathematically required loss is
representable in float32. After input validation, representability checks occur
in this exact order:

1. `shifted` contains only finite values or negative infinity created by an
   out-of-range negative finite subtraction; positive infinity or NaN produces
   `Phase4ContractError("loss.unrepresentable.shifted")`;
2. every `shifted_exp` value is finite and nonnegative, else
   `Phase4ContractError("loss.unrepresentable.shifted_exp")`;
3. `shifted_sum` is finite and strictly positive, else
   `Phase4ContractError("loss.unrepresentable.shifted_sum")`;
4. `log_shifted_sum` is finite, else
   `Phase4ContractError("loss.unrepresentable.log_shifted_sum")`;
5. `target_margin` is finite, else
   `Phase4ContractError("loss.unrepresentable.target_margin")`;
6. every per-row `negative_log_likelihood` is finite, else
   `Phase4ContractError("loss.unrepresentable.row")`;
7. every `scaled_rows` value is finite, else
   `Phase4ContractError("loss.unrepresentable.scaled_row")`; and
8. reduced `loss` is finite, else
   `Phase4ContractError("loss.unrepresentable.reduction")`.

These failures classify a mathematically required value that cannot be
represented by the accepted float32 procedure; they do not relabel the finite
input logits as malformed. Negative infinity in `shifted` is deliberately
permitted because `exp(-inf) == 0` can preserve a representable result when an
irrelevant competitor is enormously below the row maximum.

For a representable case, the result is one scalar, shape `()`, CPU
`torch.float32`, and autograd-connected to logits and model parameters. This is
the arithmetic mean over target-token negative log-probabilities. The
implementation may use low-level tensor primitives but must not call
`torch.nn.functional.cross_entropy`, `torch.nn.CrossEntropyLoss`, or another
combined reference loss.

Independent tests compare it to `torch.nn.functional.cross_entropy` using
`rtol=1e-6` and `atol=1e-6` on ordinary and representable test-local logits.
The reference function is test evidence only. Dedicated cases must cover
representable logits near `-10000` and `10000`, a target equal to
`torch.finfo(torch.float32).max` with an irrelevant competitor equal to the
most-negative finite value `torch.finfo(torch.float32).min` whose negatively
infinite shift must not spoil the representable result, and the reverse target
assignment whose required margin is unrepresentable and must fail
deterministically rather than return accidental infinity.

## Accepted backpropagation inspection

The educational inspection uses the accepted vocabulary binding and a freshly
initialized model but only synthetic IDs:

```text
input  = [3, 1, 3]
target = [1, 3, 2]
```

It performs one forward call, one explicit scalar mean loss, and one
`backward()`, with no update until all observations are captured. It records
only structural and numeric facts:

- input, representation, logits, and target shapes;
- scalar loss value;
- a pre-backward snapshot proving `backward()` alone changes no parameter;
- non-`None`, finite, shape-equal gradients on all four parameters;
- nonzero output-weight and output-bias gradients;
- nonzero gradients for used token rows 1 and 3 and zero gradient for a
  test-local unused token row;
- nonzero gradients for used position rows 0, 1, and 2 and zero gradient for
  position row 3; and
- exact manual-update arithmetic for every parameter after inspection.

The inspection persists no model state, gradient, tensor, token stream, or
example artifact. It is a synthetic mechanistic check, not the corpus training
experiment.

## Accepted manual-SGD contract

The only Phase 4 update is:

```text
parameter -= learning_rate * parameter.grad
```

inside `torch.no_grad()`. No `torch.optim` object or optimizer state exists.

The exact trainable parameter order is:

1. `output_weight`;
2. `output_bias`;
3. `representation.token_embeddings`;
4. `representation.position_embeddings`.

The public manual-update operation validates in this exact order:

1. learning-rate exact built-in float type, else `Phase4TypeError`;
2. learning rate finite and greater than zero, else `Phase4ContractError`;
3. model exact `SimpleNeuralLanguageModel` boundary, else `Phase4TypeError`;
4. exact four-name parameter enumeration, else `Phase4ContractError`;
5. for every parameter in fixed order, complete parameter invariants:
   `torch.nn.Parameter` object, expected identity with the corresponding model
   attribute, shape, CPU device,
   `torch.float32`, `requires_grad=True`, and finite value; a wrong object type
   produces `Phase4TypeError` and another mismatch produces
   `Phase4ContractError`;
6. gradient presence for every parameter, else `Phase4ContractError`;
7. gradient shape for every parameter, else `Phase4ContractError`;
8. gradient dtype for every parameter, else `Phase4ContractError`;
9. gradient device for every parameter, else `Phase4ContractError`; and
10. gradient finiteness for every parameter, else `Phase4ContractError`.

Each numbered stage checks all four parameters in the fixed order before the
next stage. Thus, for example, a missing earlier gradient deterministically
precedes a wrong-shaped later gradient. Every stage completes for all
parameters before any mutation occurs. The fixed experiment supplies learning
rate `0.05`.

If any preflight check fails, no parameter is updated and gradients remain
available for diagnosis. After successful preflight, all parameters are
updated in the fixed order under one `torch.no_grad()` context. Exact arithmetic
is tested against pre-update clones.

The public `clear_gradients(model)` boundary validates in exact order: model
exact type, else `Phase4TypeError`; exact four-name enumeration, else
`Phase4ContractError`; every parameter object type in fixed order, else
`Phase4TypeError`; then identity, shape, device, dtype, learnability, and
finiteness by category across all parameters, else `Phase4ContractError`. For
every non-`None` gradient it next validates tensor object type, else
`Phase4TypeError`, followed by shape, dtype, device, and finiteness by category
across all parameters, else `Phase4ContractError`. Only after complete
preflight does it assign `parameter.grad = None` in fixed order. It accepts an
all-`None` gradient state. A failure clears nothing.

Gradient clearing occurs before the first training example and immediately
after every successful update. A failed update preflight does not clear
gradients. Validation performs no backward call and no update, so it cannot
create or change gradients. Representative simultaneous-defect tests verify
the deterministic first failure for both public operations.

## Accepted deterministic training and validation orchestration

### Fixed traversal

Training order is fixed and never shuffled:

1. the six works in accepted manifest order 1 through 6;
2. within each work, example starts `0, 64, 128, ...`; and
3. repeat that exact complete order for each training pass.

For each example, tensorize only its current tuples, clear gradients, compute
logits and mean cross-entropy, scale the backward scalar by `L / 64`, call
`backward()`, preflight and perform one manual-SGD update, clear gradients, and
discard the tensors. The scale makes the backward scalar equal to the sum of
that example's token losses divided by 64, so a short tail does not amplify
each of its transitions by `64 / L` relative to a full example. Every source
transition still appears in exactly one update per pass.

No random number is consumed during traversal, tensorization, loss, backward,
or update. No shuffling, early stopping, retry, learning-rate scheduling, or
hyperparameter search is permitted.

Validation traverses *The Tempest* in ascending start order under
`torch.no_grad()`. It performs no backward call, parameter update, or gradient
clear that could hide an accidental gradient. Parameter identities and the
canonical raw-byte digest defined in the experiment lifecycle are required to
match before and after validation.

Initial and final aggregate training measurements likewise run under
`torch.no_grad()` with no backward call, update, or gradient clearing.
Parameter identities and canonical raw-byte digests must match before and
after every aggregate measurement.

### Aggregate loss

Reported aggregate loss is weighted by predicted target tokens, never by
examples. For each example `e`, let `m_e` be its detached mean loss and `L_e`
its target count. In fixed traversal order:

```text
total_negative_log_likelihood = math.fsum(float(m_e.item()) * L_e for e)
total_target_count = sum(L_e for e)
aggregate_loss = total_negative_log_likelihood / total_target_count
```

The accepted corpus counts imply these independently reverified Phase 4
expectations:

| Split | Source tokens | Documents | Target transitions | Examples |
|---|---:|---:|---:|---:|
| training | 792,705 | 6 | 792,699 | 12,389 |
| validation | 98,296 | 1 | 98,295 | 1,536 |

Counts are derived from permitted encoded documents and checked against
accepted provenance; they are not reasons to open the sealed test. Aggregate
measurement rejects a zero total target count. Using `math.fsum` and fixed
order makes host-side accumulation explicit and avoids the incorrect
mean-of-example-means calculation.

## Accepted pre-registered bounded experiment

Before any experimental result exists, freeze the following single run:

`Phase4ExperimentConfig` is a frozen dataclass containing the exact identity,
architecture, traversal, update, measurement, baseline, and success fields in
the table below. It accepts no unlisted extension mapping and stores no model,
parameter, gradient, text, token sequence, example, or callback. The future
experiment entry point accepts exactly one such config, one accepted vocabulary
binding, and one already metadata-validated seven-work permitted-corpus
handoff. It has no override keyword arguments; changing a setting requires a
new reviewed config identity rather than an invocation-time override.

| Setting | Accepted fixed value |
|---|---|
| Code | future accepted, clean, committed Phase 4 implementation checkpoint |
| Tokenizer implementation | `de7a7f096fbbd8607c944412eaef30be9b686b56` |
| Vocabulary | accepted `shakespeare-code-point-v1` binding, schema 1, canonical path, and pinned SHA-256 |
| Dataset | accepted Phase 1 manifest SHA-256 `157e324c41c6aee756b9c554ae465388a892ea9a0f8fb8296ef0d986a0c9f6fb` and six train/one validation provenance |
| Embedding contract | DEC-0015; checkpoint `0e458cc8b8bce9d8e89b23abbb8ce6385669b7d5` |
| Embedding implementation | `68b47bb404d55357cadce35b97036c72c5876d62` |
| Device/dtype | CPU `torch.float32`; IDs `torch.long` |
| Embedding seed | 1337, inherited |
| Output-head seed | 4004 |
| Context/stride | 64 / 64 |
| Ordering | fixed manifest order, then ascending start; no shuffle |
| Trainable parameters | all four model parameters |
| Update | manual SGD, one update per example |
| Learning rate | 0.05 |
| Training passes | 2 |
| Updates | exactly `12,389 * 2 = 24,778` |
| Gradient scalar | example mean loss multiplied by `L / 64` |
| Initial measurement | aggregate training, then aggregate validation, before any update |
| Final measurement | aggregate training, then aggregate validation, after all 24,778 updates |
| Intermediate evaluation | none |
| Baseline | mathematical `ln(81)` |
| Test access | none |

The immutable experiment configuration and its supplied authority objects are
validated before model construction in this exact first-failure order:

1. experiment-configuration exact object type, else `Phase4TypeError`;
2. vocabulary-binding object type and permitted-corpus container/work object
   types, in that order, else `Phase4TypeError`;
3. future Phase 4 implementation commit grammar, clean-checkpoint status, and
   exact checked-out identity, else `Phase4ContractError`;
4. accepted tokenizer implementation commit identity, else
   `Phase4ContractError`;
5. DEC-0015 and accepted embedding contract/implementation commit identities,
   else `Phase4ContractError`;
6. vocabulary ID, schema, size, path, SHA-256, and factory verification, in
   that order, else `Phase4ContractError`;
7. Phase 1 dataset/manifest identity and seven permitted work provenance
   records, else `Phase4GovernanceError`;
8. model architecture, parameter names/shapes/count, device, and dtype, else
   `Phase4ContractError`;
9. embedding seed then head seed, else `Phase4ContractError`;
10. context length, stride, and variable-tail/no-padding policy, else
    `Phase4ContractError`;
11. learning-rate exact type, then value, using the manual-SGD ownership rule,
    else `Phase4TypeError` or `Phase4ContractError` respectively;
12. pass count, fixed traversal, no-shuffle rule, and tail gradient scale, else
    `Phase4ContractError`;
13. training/validation token, target, example, and update counts, else
    `Phase4GovernanceError` for data disagreements or `Phase4ContractError` for
    configuration arithmetic disagreement;
14. exact baseline and success-predicate identity, else
    `Phase4ContractError`; and
15. no-test supplier/access boundary, else `Phase4GovernanceError`.

No model or parameter is constructed before all fifteen stages pass.
Representative multi-fault tests require the first stage above to win
deterministically and prove later inputs remain untouched.

### One uninterrupted model lifecycle

The experiment has exactly one model lifecycle:

1. Construct exactly one fresh accepted Phase 4 model from the fully validated
   frozen configuration.
2. Verify all four trainable parameters have `parameter.grad is None`.
3. Measure initial aggregate training loss with that model.
4. Measure initial aggregate validation loss with that same model.
5. Do not reconstruct, restore, reseed, reload, copy-replace, reset, or
   otherwise substitute the model or any parameter object.
6. Perform both complete training passes on those same four parameter objects.
7. Measure final aggregate training loss on those same objects.
8. Measure final aggregate validation loss on those same objects.
9. Keep the model only in memory for the bounded run and discard it without a
   checkpoint after the record is complete.

The orchestration captures the Python identity of the model and each named
parameter immediately after construction and requires those identities to be
unchanged before and after every measurement, pass, and final record step. A
constructor/factory call count must remain exactly one. There is no accepted
state-loading, reset, reseed, or parameter-replacement path.

Every measurement begins and ends with all four gradients `None`, runs under
`torch.no_grad()`, and performs no mutation. For exact no-update evidence,
measurement computes a SHA-256 digest before and after over canonical parameter
bytes in exact named-parameter order. Canonical bytes include length-prefixed
UTF-8 parameter name, shape dimensions, dtype/device identity, and raw element
bytes obtained from the detached contiguous CPU tensor viewed and flattened as
`torch.uint8`. Digest equality, unchanged object identities, and all gradients
remaining `None` are jointly required. This raw-byte digest distinguishes
`+0.0` from `-0.0`; `torch.equal()` alone is insufficient evidence.

The future run must first verify exact code, vocabulary, dataset, configuration,
counts, and a clean enough repository to identify its outputs. It performs no
tuning and persists no weights or checkpoint. If a precondition or invariant
fails, record a failed/stopped run without changing settings and stop for
review. If the success rule fails, record the failure; do not adjust and rerun
under the same experiment ID.

If the Gate 24 run fails its pre-registered predicate, its complete failed
record is appended permanently to `EXPERIMENT_LOG.md` and independently
analyzed read-only. The entry may never be overwritten, removed, or reclassified
by a later result. Any proposed change requires an explicit amended Phase 4
experiment pre-registration/decision, independent review, and Sebastien's
explicit acceptance and run authorization. A later run uses a distinct durable
experiment identity and receives a separate record.

Repeated unchanged runs under new identities merely to seek a stochastic or
environmental pass are forbidden. Seed, learning rate, pass count, traversal,
architecture, gradient scaling, success criterion, and every other frozen
setting may not change silently. An unchanged reproduction/debug rerun requires
explicit purpose-specific authorization, remains linked to the original
failed record, and cannot replace or reverse the original PASS/FAIL
adjudication.

The learning rate and two-pass duration are deliberately modest accepted
pre-registered choices, not observed winners. Per-example deterministic SGD
provides many small inspected updates while remaining bounded. Gate 8 accepted
their reasonableness; the values remain frozen before any separately authorized
run.

## Accepted exact success criterion

Define:

```text
uniform_baseline = math.log(81)  # approximately 4.394449154672439
```

The single primary PASS rule is the conjunction:

```text
final_training_aggregate_loss < uniform_baseline
and
final_training_aggregate_loss < initial_training_aggregate_loss
```

Both comparisons are strict and use the recorded finite scalar values without
rounding for adjudication. This is the smallest rule that demonstrates that
the bounded training run improved training loss from its initialized model and
surpassed fixed uniform guessing.

Initial and final validation aggregate losses are mandatory observations. The
record must state whether validation improved, stayed equal, or worsened, but
validation change is not a Phase 4 PASS condition. Lower training loss does not
establish generalization, and Phase 4 must not silently claim that it does. No
test metric exists.

## Accepted error model

Phase 4 introduces exactly three public exception classes:

- `Phase4TypeError(TypeError)` for invalid non-tensor/container or exact-type
  boundaries;
- `Phase4ContractError(ValueError)` for configuration, token, shape, dtype,
  device, finiteness, loss, probability, gradient, and update invariants; and
- `Phase4GovernanceError(ValueError)` for dataset, document, split, order,
  provenance, text-access, and sealed-test boundaries.

Each carries an immutable mapping of safe facts and formats deterministic JSON
for `str` and `repr`. Permitted facts are invariant names, first failing
positions, expected numeric bounds, shapes, counts, dtypes, devices, accepted
split labels, and booleans. Errors must not contain input or target values,
token sequences, corpus prose, text excerpts, absolute paths, parameter values,
or sealed-test-specific observed identity/content.

Tokenizer-owned `VocabularyArtifactError` and accepted embedding-owned errors
retain their ownership and may propagate unchanged. Phase 4 must not catch and
reinterpret them in a way that loses their accepted boundary.

Validation occurs completely before a mutating operation. Orchestration
validates safe metadata for every supplied work before accessing any permitted
text. Manual SGD validates every parameter and gradient before updating the
first one.

### Authoritative public-operation ordering index

The exact validation sequences are part of the public contract, not examples:

- `iter_shifted_examples`: six stages under **Accepted corpus-neutral
  shifted-example contract**;
- `iter_shakespeare_phase_4_examples`: eleven stages under **Accepted
  Shakespeare governance contract**;
- `SimpleNeuralLanguageModel` construction: six stages under **API and
  composition**;
- model `forward`: seven stages under **API and composition**;
- `probabilities_from_logits`: seven stages under **Accepted logits and
  probability contract**;
- `explicit_cross_entropy`: twelve input-validation stages followed by eight
  representability stages under **Accepted explicit cross-entropy contract**;
- manual SGD: learning-rate type/value, model type, enumeration, all parameter
  categories, then all gradient categories under **Accepted manual-SGD
  contract**;
- `clear_gradients`: model type, enumeration, all parameter categories, then
  every applicable gradient category before clearing under **Accepted
  manual-SGD contract**; and
- experiment configuration: fifteen stages under **Accepted pre-registered
  bounded experiment**.

When an input violates multiple stages, the lowest numbered applicable stage
wins. Within a stage covering multiple parameters or works, fixed enumeration
order wins. No losing-stage access, computation, allocation, or mutation is
permitted.

## Accepted independent test contract

Critical expectations are test-local accepted literals or independently
derived from inputs. Tests may import the implementation under test but may not
derive their expected values by importing the same production constants,
boundary helper, parameter counter, formula helper, or manifest-selection code
being tested.

### Shifted examples and coverage

- Exact examples for synthetic lengths 0, 1, 2, 63, 64, 65, 66, 128, 129,
  and 130.
- General independently computed coverage for multiple longer lengths: every
  transition index occurs once, no other index occurs, starts are ordered, and
  input/target shifts are exact.
- Exact tuple/type/range rejection, first-failure ordering, eager validation,
  immutable records, content-safe representation, and no input mutation.
- Separate-document sentinels prove no concatenation or cross-boundary pair.
- Instrumentation proves incremental consumption rather than production
  materialization or persistence.

### Shakespeare governance

- Independent manifest-local expectations for the six training works and one
  validation work.
- Missing, duplicate, reordered, wrong-split, wrong-provenance, and test work
  are rejected before any `processed_text` access.
- A sealed-test access guard raises if touched and records exactly zero content
  accesses.
- Training and validation texts are encoded separately; no joined text or ID
  tuple reaches the neutral helper.
- Traversal is manifest order then ascending start, repeatably.

### Model and initialization

- Exact composition, four parameter names/identities/enumeration order, shapes, count 13,457,
  CPU float32 dtype, learnability, no buffers, and no hidden module or weight
  aliasing.
- Constructor validation precedes parameter allocation where observable.
- A test-local generator seeded 4004 independently reproduces the output
  weight; bias is exactly zero.
- Repeated complete constructions are bitwise equal on the supported runtime,
  global RNG consumption does not change construction, and construction leaves
  global CPU RNG state unchanged.
- Rank-one lengths 0, 1, 64 succeed with exact logits shapes; length 65,
  rank-two, wrong dtype/device, and invalid IDs fail before lookup/head work.
- Logits equal an independently computed representation matrix multiplication
  plus bias; positions do not communicate.

### Probabilities and loss

- Probability validation independently covers exact object type, rank, dtype,
  device, class width, nonnegative length, and finiteness in the specified
  order; it does not call the loss validator.
- `(0, 81)` is accepted and returns a distinct empty CPU float32 `(0, 81)`
  tensor; wrong class widths, rank, dtype, device, and nonfinite values are
  rejected with exact exception ownership.
- Nonempty probabilities use dimension 1, are nonnegative and finite, and rows
  sum to one within the specified tolerance.
- Cross-entropy validation order covers both object types, ranks, dtypes,
  devices, class count, paired length, positive length, input finiteness, and
  target range.
- Hand-calculated tiny logits demonstrate target log-probability and mean loss.
- Ordinary and finite representable `+/-10000` logits agree with PyTorch's
  reference within `rtol=1e-6`, `atol=1e-6`.
- A `torch.finfo(torch.float32).max` correct target plus a
  `torch.finfo(torch.float32).min` irrelevant competitor proves a negatively
  infinite shifted competitor does not cause premature refusal or overflow
  when the result is representable.
- The reverse target assignment proves an
  unrepresentable required float32 loss raises the exact deterministic
  `Phase4ContractError` rather than returning infinity.
- Independent arithmetic proves the implementation uses target-margin-first
  cancellation and divide-before-reduction rather than reconstructing an
  overflowing absolute log normalizer or unscaled sum.
- Loss is a new scalar connected through logits to all expected model
  parameters and does not mutate logits or targets.

### Exception ownership and first failure

- Each major public API is tested against its complete documented validation
  order: shifted examples, Shakespeare orchestration, model construction,
  model forward, probability inspection, cross-entropy, manual SGD, gradient
  clearing, and experiment-configuration validation.
- Wrong object/exact Python type produces `Phase4TypeError`; correct type with
  invalid value/shape/dtype/device/range/configuration produces
  `Phase4ContractError`; split/provenance/sealed-policy violations produce
  `Phase4GovernanceError`.
- Representative simultaneous defects for every API prove one deterministic
  first exception and prove later values, content guards, parameters, and
  gradients remain untouched.
- Accepted tokenizer and embedding exceptions retain their existing ownership
  after Phase 4 boundary validation succeeds.

### Gradients, updates, and aggregation

- The exact synthetic inspection reaches output weight, bias, used token rows,
  and used position rows while unused test-local rows remain zero.
- `backward()` alone leaves parameters bitwise unchanged.
- Manual SGD matches independent clone arithmetic for all four parameters.
- Missing, wrong-shaped, wrong-device/dtype, or nonfinite gradient fails before
  any parameter mutation.
- Gradient clearing sets every gradient to `None`; failure preserves gradients.
- Validation and aggregate training measurements run under no-grad, leave all
  gradients `None`, preserve parameter Python identities, and produce identical
  pre/post SHA-256 digests over the specified canonical raw parameter bytes.
  A dedicated `+0.0`/`-0.0` test proves this digest is stricter than
  `torch.equal()`.
- Short-tail training scales its backward scalar by `L / 64`.
- Aggregate loss is independently reconstructed from all target-token losses,
  differs from a deliberately chosen mean-of-example-means case, and has the
  exact target counts.
- Two fresh synthetic traversals yield identical order, update count, parameter
  state, and reported losses on the supported runtime.

### Experiment lifecycle and failed-run governance

- Instrumentation proves exactly one model constructor call, initial gradients
  all `None`, and the same model and four parameter identities across initial
  training measurement, initial validation measurement, both passes, final
  training measurement, and final validation measurement.
- No reconstruction, reseed, state load, restore, reset, copy-replacement, or
  parameter-replacement call path is available to the experiment.
- Measurement tests prove no update, persistent gradient, or identity change.
- Static record tests prove an original failed experiment remains append-only;
  changed settings require an amended pre-registration, independent review,
  and explicit Sebastien acceptance/authorization before a distinct run ID.
- Static review proves an unchanged repeated run cannot silently become a new
  acceptance attempt; any explicitly authorized reproduction remains linked to
  and cannot replace the original adjudication.

### Phase boundary review

A manual import/diff review confirms no test or implementation introduces
attention, Q/K/V, causal attention masks, recurrence, convolution,
Transformer blocks, residual paths, normalization, dropout, padding, special
tokens, cross-document examples, sealed-test access, runtime batching,
sampling, `torch.optim`, checkpoints, generation, MPS/mixed precision, or
hosted/pretrained models.

## Accepted experiment record

The future run appends one `EXPERIMENT_LOG.md` entry. It is committed as reviewed
documentation after the experiment checkpoint; no binary model, checkpoint,
token stream, window, tensor, or generated sample is retained.

The entry records at least:

- experiment ID, UTC date, status, question, and pre-registered success rule;
- exact clean implementation Git commit;
- Python, PyTorch, operating system, architecture, device, and dtype;
- accepted tokenizer implementation commit
  `de7a7f096fbbd8607c944412eaef30be9b686b56`;
- tokenizer ID, schema, vocabulary size, artifact path, and SHA-256;
- Phase 1 dataset ID, manifest SHA-256
  `157e324c41c6aee756b9c554ae465388a892ea9a0f8fb8296ef0d986a0c9f6fb`,
  and seven permitted work hashes;
- accepted embedding decision DEC-0015 and contract checkpoint
  `0e458cc8b8bce9d8e89b23abbb8ce6385669b7d5`;
- accepted embedding implementation commit
  `68b47bb404d55357cadce35b97036c72c5876d62`;
- model name, complete parameter shapes/count, trainable set, and absence of
  weight tying;
- embedding and head seeds, initialization distribution/order, and RNG scope;
- context length, stride, tail policy, work/example order, and no-shuffle rule;
- exact training/validation source token, target, and example counts;
- manual-SGD rule, learning rate, gradient scaling/clearing, passes, and update
  count;
- exact initial/final aggregate training and validation losses;
- exact `ln(81)` baseline used for adjudication;
- each primary predicate and overall PASS/FAIL;
- one-constructor lifecycle verification, unchanged parameter-object identities,
  all measurement-boundary gradients `None`, and matching pre/post canonical
  parameter-byte digests for each measurement;
- validation direction described as observation only;
- confirmation of zero sealed-test access and zero retained model/data
  artifacts;
- any failed-run predecessor or amended pre-registration identity and its
  independent-review/authorization lineage;
- observations, conclusions limited to the evidence, and follow-up.

No planned entry is created in this acceptance checkpoint because Phase 4
implementation and the experiment remain unauthorized and no experiment has
run.

## Phase 4 boundary

Phase 4 does not introduce attention, Q/K/V projections, causal attention
masks, multi-head machinery, recurrence, convolution, Transformer blocks,
residual connections, normalization, dropout, full mini-GPT stacking, padding,
new special tokens, cross-document or cross-split examples, sealed-test use,
production random sampling/batching, optimizer infrastructure, checkpoints,
resume behavior, generation, final evaluation, MPS optimization, mixed
precision, pretrained models, or hosted model APIs.

Phase 5 owns self-attention. Phase 6 owns Transformer-block composition. Phase
7 owns the complete mini-GPT. Phase 8 owns production training/batching and
checkpointing. Phase 9 owns generation and final evaluation, including any
separately authorized sealed-test event.

## Accepted Phase 4 gates

No gate may silently authorize a later gate. Corrections after an independent
review require focused re-review before acceptance.

1. **Explicit Phase 4 learning/design authorization.** Complete through Gate
   18 authorization after the remotely verified Phase 3 closure.
2. **First-principles learning and repository orientation.** Complete.
3. **Conceptual architecture approval.** Complete: Sebastien accepted all
   twelve decisions recorded in DEC-0016.
4. **Explicit detailed-contract authorization.** Complete for documentation
   only.
5. **Detailed contract drafting.** Complete in this
   specification and related continuity documentation.
6. **Independent detailed-contract review.** Complete with FAIL: seven
   must-fix findings were accepted without reopening the conceptual design.
7. **Contract correction, if required.** Complete locally for all seven
   findings as documentation only; no implementation or experiment occurred.
8. **Focused independent re-review after corrections and explicit contract
   acceptance.** Complete: focused review returned PASS for all seven findings,
   master-chat adjudication returned PASS, and Sebastien explicitly accepted
   the corrected detailed contract. Implementation and experiment remain
   unauthorized.
9. **Accepted-contract checkpoint commit.** Separately authorized after Gate 8.
10. **Contract push and independent remote verification.** Separately
    authorized after Gate 9.
11. **Explicit shifted-example implementation authorization.** Separate from
    the model implementation.
12. **Shifted-example and Shakespeare-governance implementation with focused
    tests.** No model, loss, or training work.
13. **Independent example-implementation review.** PASS/FAIL against the
    accepted contract.
14. **Example corrections, focused re-review when corrections occur, and
    explicit acceptance.** No checkpoint acceptance before required re-review.
15. **Accepted example-implementation commit.** Separately authorized.
16. **Example checkpoint push and independent remote verification.**
    Separately authorized.
17. **Explicit model/loss/update implementation authorization.** Separate from
    example construction and experiment authorization.
18. **Model, explicit loss, gradient inspection, manual update, aggregation,
    and deterministic tests.** No corpus training experiment.
19. **Independent model/loss/update implementation review.** PASS/FAIL against
    the accepted contract.
20. **Implementation corrections, focused re-review when corrections occur,
    and explicit acceptance.** No experiment is authorized by acceptance.
21. **Accepted model/loss/update implementation commit.** Separately
    authorized.
22. **Implementation checkpoint push and independent remote verification.**
    Separately authorized.
23. **Explicit fixed-experiment authorization.** Only after Gate 22 and a
    fresh precondition check.
24. **Run exactly the pre-registered bounded experiment and append its record.**
    A failed criterion is recorded, not tuned around.
25. **Independent experiment and Phase 4 exit review.** Evaluate the record,
    reproducibility, boundary, and exactly the four roadmap exit criteria.
26. **Controlled failed-run analysis and re-registration, if required.** Keep
    the failed Gate 24 record permanently, perform independent read-only cause
    analysis, and make any proposed setting or technical change through an
    explicit amended pre-registration/decision. Independently review that
    amendment and obtain Sebastien's explicit acceptance and run authorization
    before a distinct durable experiment identity may run. Never overwrite,
    remove, relabel, tune around, or repeatedly rerun the failed result.
    Unchanged reproduction/debug execution also requires explicit
    purpose-specific authorization, remains linked to the original record, and
    cannot replace its adjudication.
27. **Explicit Phase 4 acceptance and closure documentation.** Only after a
    passing independent exit review.
28. **Phase 4 closure commit.** Separately authorized after Gate 27.
29. **Closure push and independent remote verification.** Separately authorized
    after Gate 28.
30. **Explicit Phase 5 authorization.** Phase 5 cannot begin implicitly.
