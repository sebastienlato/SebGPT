# SebGPT Phase 3 Embedding Specification

## Status and authority

This document separates two kinds of Phase 3 statements:

- **Accepted conceptual architecture:** explicitly approved by Sebastien on
  2026-09-08 and recorded in DEC-0015.
- **Accepted detailed contract:** the API, validation, error, initialization,
  authority-loading, test, and gate mechanics below. The first independent
  review returned six required corrections; focused independent re-review then
  passed, and Sebastien explicitly accepted the corrected contract on
  2026-09-08.

This checkpoint authorizes documentation only. It does not authorize source
code, tests, construction of `torch.nn.Parameter` objects, production token
streams, context windows, training, or any later-phase work.

## Repository Phase 3 contract

**Goal:** Understand learned token and positional representations.

**Exit criteria:**

1. Token embeddings and an explicit positional representation are implemented.
2. Tensor shapes and parameter roles are explainable and tested.
3. Small examples demonstrate lookup, batching, and gradient flow.

The rest of this specification makes those criteria testable without adding
another exit criterion.

## Accepted conceptual architecture

### Vocabulary identity

The representation is bound to exactly:

| Field | Accepted value |
|---|---|
| Tokenizer ID | `shakespeare-code-point-v1` |
| Artifact schema version | `1` |
| Vocabulary size | `81` |
| Canonical relative path | `artifacts/tokenizers/shakespeare-code-point-v1/vocabulary.json` |
| Artifact SHA-256 | `9f4235f9dab3e0361221a90c5ae7ca7fee540760fcc1708964368e6f17d6d70e` |

Vocabulary size alone is not identity. Construction must fail rather than
truncate, pad, regenerate, reorder, remap, or substitute another mapping.

### Representation configuration

| Property | Accepted value |
|---|---|
| Embedding dimension | `32` |
| Maximum positional capacity | `256` positions, indexed `0` through `255` |
| Token table shape | `(81, 32)` |
| Position table shape | `(256, 32)` |
| Token parameters | `2,592` |
| Position parameters | `8,192` |
| Total parameters | `10,784` |
| Parameter device and dtype | CPU `torch.float32` |
| Token and position ID dtype | `torch.long` |
| Initialization seed | `1337` |
| Initialization distribution | Normal with mean `0` and standard deviation `1 / sqrt(32)` |

Both tables are learnable. Phase 3 uses one explicit `torch.nn.Module` with two
manually managed `torch.nn.Parameter` matrices. It does not use
`torch.nn.Embedding` as its implementation. Token IDs directly select token
rows. Learned absolute positions directly select position rows. The two vectors
are added with no scaling, projection, normalization, bias, dropout, or other
transformation.

`max_positions = 256` defines positional-table capacity only. It does not
select a training context length or authorize production windows or batches.

Sinusoidal positions are not selected. RoPE is not selected and belongs with
later attention concepts. Comparing `torch.nn.Embedding` is explanatory only,
not a Phase 3 exit requirement.

## Accepted vocabulary authority boundary

### Finding from the current implementation

The Phase 2 production verifier reconstructs authority from the Phase 1
manifest and accepted extracted works. That is correct for artifact creation,
but there is no existing lightweight runtime loader. Requiring model
construction to rerun extraction or possess corpus text would couple a model
primitive to the dataset pipeline and is not appropriate.

### Accepted boundary

The ownership chain is exactly:

```text
tokenizer authority layer
→ corpus-free runtime loader/verifier
→ immutable verified VocabularyBinding
→ Phase 3 TokenPositionEmbedding constructor
```

Use option B: the representation receives a small immutable, already-verified
`VocabularyBinding` from that tokenizer-owned runtime boundary.

The accepted public boundary is:

```text
load_accepted_vocabulary_binding(repository_root: pathlib.Path)
    -> VocabularyBinding

VocabularyBinding:
    tokenizer_id: str
    schema_version: int
    vocabulary_size: int
    artifact_sha256: str
    code_points: tuple[int, ...]
```

The loader always resolves the accepted canonical relative path beneath the
given repository root; callers do not supply an alternate artifact path. It
reads the artifact without accessing raw or processed prose, then performs
these checks in order:

1. The path exists as a regular file.
2. The SHA-256 of the exact bytes equals the accepted artifact SHA-256.
3. The bytes satisfy the complete accepted deterministic-serialization
   contract: strict UTF-8 without a BOM, canonical sorted-key/two-space JSON,
   LF line endings, one terminal LF, and a JSON-object root.
4. The root has exactly the accepted Phase 2 keys and values: schema version
   exact integer `1`, explicitly not Boolean; tokenizer ID
   `shakespeare-code-point-v1`; token unit
   `python_str_code_point`; normalization `none`; unknown policy `error`;
   special-token policy `none`; ID order `ascending_unicode_code_point`;
   contract decision `DEC-0014`; Phase 1 manifest SHA-256
   `157e324c41c6aee756b9c554ae465388a892ea9a0f8fb8296ef0d986a0c9f6fb`;
   implementation commit
   `de7a7f096fbbd8607c944412eaef30be9b686b56`; training-work records;
   vocabulary size; and vocabulary records. Unknown root keys are rejected.
5. The six training-work objects have exactly the accepted keys, identities,
   manifest order, and processed SHA-256 values from the accepted artifact.
   Every manifest order has exact integer type, explicitly excluding Boolean;
   validation and test records and additional keys are rejected.
6. `vocabulary_size` has exact JSON/Python integer type, explicitly excluding
   Boolean values, and equals 81 and the vocabulary-array length.
7. The vocabulary is exactly 81 objects with exactly `token_id`, `code_point`,
   and `code_point_uplus`. Both numeric fields have exact integer type and are
   not Booleans. Token IDs equal their positions `0..80`; code points are in
   `0..0x10FFFF`, strictly increasing, and therefore unique; every U+ value is
   the canonical uppercase form padded to at least four hexadecimal digits.
8. Every SHA-256 and Git-commit field satisfies its accepted grammar and exact
   pinned value; forbidden literals, Unicode-name fields, and additional
   record keys are rejected.

The loader verifies every schema and semantic rule in the accepted canonical
artifact contract in `docs/TOKENIZER_SPEC.md`, not merely the subset later
needed for table shape. The exact hash is also an ultimate same-bytes and
same-mapping check: a structurally valid 81-entry artifact containing even one
different code point fails. The loader never regenerates the vocabulary and
does not need the Phase 1 corpus or sealed test. Its verification logic is
tokenizer-owned and may refactor corpus-free checks out of the current Phase 2
verifier instead of duplicating the contract in the embedding layer.

Loader failures use a tokenizer-owned, deterministic, content-safe
`VocabularyArtifactError`, never an embedding exception. The loader produces
an immutable `VocabularyBinding` through a tokenizer-owned factory; direct
unverified binding construction is not public. The binding is an application
contract, not a security credential.

`TokenPositionEmbedding.__init__` consumes that verified binding and may check
only that the argument is a factory-produced `VocabularyBinding` before
parameter allocation. Neither its constructor nor `forward` reads, parses,
hashes, or independently validates JSON or repeats artifact semantic checks.
Artifact verification occurs exactly once at the tokenizer authority boundary,
never during embedding lookup.

## Accepted representation module and API

Use one module because token lookup, internally generated positions, addition,
and their shared output shape form one small concept. Separate modules would
add indirection without making either parameter role clearer.

```text
TokenPositionEmbedding(
    vocabulary: VocabularyBinding,
    *,
    embedding_dim: int,
    max_positions: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
)

public learnable attributes:
    token_embeddings: torch.nn.Parameter
    position_embeddings: torch.nn.Parameter

public method:
    forward(token_ids: torch.Tensor) -> torch.Tensor
```

Every constructor argument is required and has no default, so a call site makes
the architectural configuration visible. Phase 3 accepts only exact built-in
integers `32`, `256`, and `1337`, exact `torch.device("cpu")`, and
`torch.float32`; any other type or value fails before parameter allocation. The
arguments are explicit for learning and validation, not an authorization for
other configurations. A later phase would need a recorded decision before
generalizing them.

The module registers exactly two parameters in token-first order:

1. `token_embeddings`, shape `(81, 32)`, 2,592 elements
2. `position_embeddings`, shape `(256, 32)`, 8,192 elements

Both have `requires_grad=True`. There are no other parameters or registered
buffers. Vocabulary identity is immutable non-tensor metadata. There is no
bias, normalization, projection, scale parameter, dropout, or output head.

## Accepted token-input contract

`forward` accepts an object when `isinstance(token_ids, torch.Tensor)` is true.
It does not coerce lists, tuples, NumPy values, or other objects into a tensor.
A legitimate `torch.Tensor` subclass that satisfies every remaining invariant
is accepted rather than rejected for its concrete Python type.

For an accepted tensor:

- dtype is exactly `torch.long`;
- device is exactly CPU and equals the module parameter device; Phase 3 forward
  rejects a non-CPU input or parameters moved away from CPU;
- rank is exactly one `(T,)` or two `(B, T)`;
- `T` is the final dimension and satisfies `0 <= T <= 256`;
- every token ID satisfies `0 <= id < 81`;
- contiguous and non-contiguous tensors are both accepted;
- zero-length dimensions are accepted, including `(0,)`, `(B, 0)`, `(0, T)`,
  and `(0, 0)` when the other dimensions satisfy the contract;
- the input tensor is never mutated.

Validation order is type, rank, dtype, device, sequence length, then token-ID
range. Every check completes before token or position lookup begins. Empty
inputs skip value-range reduction safely. Raw tensor indexing does not define
validity; in particular, negative IDs are rejected rather than interpreted as
indices from the end.

The contract does not require a contiguous memory layout and must not create a
contiguous copy merely for validation.

## Accepted position contract

Callers cannot supply position IDs. `forward` derives them internally from
sequence length using the future equivalent of:

```text
torch.arange(T, dtype=torch.long, device=token_ids.device)
```

For both accepted ranks the position sequence is exactly `0, 1, ..., T - 1`.
For `(B, T)`, one `(T, 32)` position lookup is broadcast across the batch, so
every batch member uses the same position row at a given sequence index.

- `T = 0` is accepted and uses an empty position sequence.
- `T = 256` is accepted and uses rows `0..255`.
- `T = 257` is rejected before `arange` or either lookup.

Arbitrary offsets, caller-provided positions, position wrapping, truncation,
and extrapolation are outside Phase 3.

## Accepted output contract

After complete validation:

```text
token_vectors = token_embeddings[token_ids]
position_ids = torch.arange(
    T,
    dtype=torch.long,
    device=token_ids.device,
)
position_vectors = position_embeddings[position_ids]
output = token_vectors + position_vectors
```

The position term is broadcast only across the batch dimension when rank is
two. The results are:

| Input | Output |
|---|---|
| `(T,)` | `(T, 32)` |
| `(B, T)` | `(B, T, 32)` |

The output is a new CPU `torch.float32` tensor, does not alias or mutate the
input or either parameter, and remains connected to both parameters through
autograd. Its value equals exactly token lookup plus position lookup. There is
no hidden transformation. Moving the module to another device is outside the
Phase 3 forward contract; later MPS compatibility work requires separate
authorization and does not redefine initialization identity.

## Accepted initialization contract

Initialization is mechanical and occurs only after the vocabulary binding and
fixed configuration have passed validation:

1. Create one local `torch.Generator(device="cpu")`.
2. Seed it with `1337` using `manual_seed` on that generator.
3. Allocate an empty CPU `torch.float32` tensor of shape `(81, 32)`.
4. Fill it with `torch.nn.init.normal_`, passing `mean=0.0`,
   `std=1 / math.sqrt(32)`, and the local generator explicitly.
5. Wrap it as `token_embeddings = torch.nn.Parameter(...)`.
6. Allocate an empty CPU `torch.float32` tensor of shape `(256, 32)`.
7. Fill it with the same `torch.nn.init.normal_` arguments and the same local
   generator, continuing its state after the token table.
8. Wrap it as `position_embeddings = torch.nn.Parameter(...)`.

`torch.empty` and `torch.nn.Parameter` do not select initial values; every
element is overwritten by the explicit initializer before exposure. No
default `nn.Embedding` or global-RNG initialization is used.

Because both random fills receive the local generator, construction preserves
PyTorch's global CPU RNG state bitwise. Calls to the global RNG before, during,
or between separate module constructions cannot affect either table.

## Accepted determinism guarantee

Two independent constructions under the same supported runtime, platform, and
PyTorch build, using the same Phase 3 implementation, accepted vocabulary
identity, CPU float32 tensors, `torch.nn.init.normal_` primitive, local CPU
generator, seed 1337, and token-first then position-second generator
consumption guarantee:

- `torch.equal` bitwise identity of the initial token tables;
- `torch.equal` bitwise identity of the initial position tables;
- independence from unrelated global CPU RNG consumption; and
- unchanged global CPU RNG state across construction.

This bitwise promise is deliberately local. Phase 3 does not promise bitwise
equality across arbitrary PyTorch versions or builds, Python/runtime versions
whose behavior differs, operating systems, CPU architectures, devices, or
initialization algorithms. It is an initialization guarantee, not a guarantee
that future training under different devices or execution orders produces
bitwise-equal learned parameters. MPS does not define Phase 3 initialization
identity.

## Accepted gradient demonstrations

All gradient checks use a synthetic scalar sum solely to invoke `backward()`.
They create no optimizer, update no parameter, and define no language-model
loss.

### Token rows

For single-sequence IDs `[3, 1, 3]`, sum all output elements and backpropagate.
With an all-ones upstream gradient:

- token row 3 has gradient `2 * ones(32)`;
- token row 1 has gradient `ones(32)`;
- an unrelated token row has `zeros(32)`.

### Position rows

For a two-member synthetic batch of length three, summing the combined output
causes each used position row `0`, `1`, and `2` to receive
`2 * ones(32)`, one contribution from each batch member. An unused position
row receives zero.

### Combined representation

The same backward call demonstrates non-`None`, correctly accumulated
gradients in both tables. It must not call an optimizer or mutate parameter
values.

## Accepted one-hot equivalence demonstration

Use a separate tiny fixed synthetic table, not the production representation.
For each chosen synthetic token ID, construct a one-hot vector of the tiny
vocabulary size, convert it to the table's floating dtype, and prove with exact
equality that:

```text
one_hot(token_id) @ table == table[token_id]
```

One-hot vectors are never accepted by `TokenPositionEmbedding.forward` and are
not stored or used by the production representation.

## Accepted error model

Use three deterministic public exception types at their owning boundaries:

- tokenizer-owned `VocabularyArtifactError`, a `ValueError` subclass, for
  artifact path, bytes, schema, semantics, provenance, hash, or binding-factory
  failures;

- `EmbeddingTypeError`, a `TypeError` subclass, for a non-tensor `forward`
  argument;
- `EmbeddingContractError`, a `ValueError` subclass, for an invalid binding
  argument, configuration, rank, dtype, device, token range, or
  positional-capacity failure.

Each error exposes an immutable `details` mapping containing:

- a stable `invariant` name;
- only applicable expected and observed safe facts.

Permitted safe facts are tokenizer ID, schema version, vocabulary size,
artifact relative path and hashes, parameter or input shape, rank, dtype,
device, and numeric bounds. Token-range errors report the accepted half-open
range and input shape but do not render the token values. Messages are
deterministic sorted JSON derived from `details`.

Errors and representations must never contain source prose, decoded
characters, a token sequence or tensor representation, adjacent tokens,
sealed-test content, absolute machine-local paths, or parameter values.
Validation reports the first failed invariant in the specified validation
order and returns no partial representation.

Accepted tokenizer-owned stable invariant families are:

```text
vocabulary.path.regular_file
vocabulary.sha256
vocabulary.utf8
vocabulary.json_object
vocabulary.canonical_serialization
vocabulary.root_keys
vocabulary.tokenizer_id
vocabulary.schema_version
vocabulary.size
vocabulary.semantic_fields
vocabulary.provenance
vocabulary.training_works
vocabulary.records
vocabulary.code_point_uplus
```

Accepted embedding-owned invariant families are:

```text
vocabulary.binding
configuration.embedding_dim
configuration.max_positions
configuration.seed
configuration.device
configuration.dtype
input.tensor
input.rank
input.dtype
input.device
input.sequence_length
input.token_id_range
```

## Accepted exact test contract

Tests use synthetic tensors unless a test specifically verifies the canonical
tracked vocabulary artifact. They do not read processed prose or tokenize the
sealed test.

Critical expectations must be test-local accepted literals or independently
mathematically derived. Tests may import the implementation under test, but
must not obtain both actual and expected values from its constants. This rule
applies at least to the vocabulary SHA-256, vocabulary size 81, embedding
dimension 32, maximum positions 256, seed 1337, both table shapes, total
parameter count 10,784, initialization-order consequences, output shapes,
position IDs and vectors, valid token range, acceptance of `T=256`, rejection
of `T=257`, global CPU RNG isolation, repeated-construction equality, and
gradient accumulation.

### Vocabulary binding

1. The canonical artifact loads to the five exact accepted identity values and
   the 81 accepted code points.
2. Missing or non-regular canonical paths fail safely.
3. Invalid UTF-8 and invalid/non-object JSON fail safely in temporary roots.
4. Wrong tokenizer ID, schema version, or vocabulary size fails before a
   binding is returned.
5. A structurally valid same-size mapping with one changed code point is
   rejected by the pinned hash.
6. Otherwise-valid bytes with a different hash are rejected.
7. No loader error exposes artifact contents, literals, or an absolute path.
   Only the tokenizer factory can produce the verified binding accepted by the
   embedding constructor; constructor and forward tests prove they perform no
   artifact I/O or semantic revalidation.

### Parameters

8. `named_parameters()` contains exactly `token_embeddings` followed by
   `position_embeddings`; `named_buffers()` is empty.
9. Shapes are exactly `(81, 32)` and `(256, 32)`.
10. Element counts are exactly 2,592, 8,192, and 10,784 total.
11. Both parameters are CPU `torch.float32` with `requires_grad=True`.
12. No bias, normalization, projection, scale, dropout, or third parameter is
    present. Wrong type or value for any required configuration argument is
    rejected before either parameter is allocated.

### Initialization and determinism

13. A test-local generator and literal accepted configuration independently
    reproduce token-first then position-second values using
    `torch.nn.init.normal_`.
14. Repeated constructions are bitwise identical for both tables.
15. Consuming the global RNG before or between constructions has no effect.
16. Construction leaves a captured global CPU RNG state bitwise unchanged.
17. A test that reverses the two local-generator draws does not match the
    accepted tables, proving construction order matters.

### Token input and lookup

18. `(T,)` returns `(T, 32)` and `(B, T)` returns `(B, T, 32)`.
19. Repeated IDs retrieve equal token vectors; boundary IDs 0 and 80 succeed.
20. `-1` and `81` fail before lookup.
21. Rank-zero, rank-three, float, bool, and non-`torch.long` integer tensors
    fail safely. A wrong-device `meta` tensor is rejected at the device check
    before any token-value access, without requiring MPS.
22. Non-tensor inputs fail without coercion.
23. A non-contiguous valid tensor and a legitimate `torch.Tensor` subclass are
    accepted when they satisfy every mathematical invariant, with correct
    values and shapes.
24. The input remains byte-for-byte unchanged.

### Positions and empty dimensions

25. Generated positions are exactly `0..T-1` and are not caller-controlled.
26. Batch members share the same position vectors.
27. `T=256` succeeds using position row 255; `T=257` fails before lookup.
28. `(0,)`, `(B,0)`, `(0,T)`, and `(0,0)` produce the contract-defined empty
    shapes without range-reduction or lookup errors.

### Combination and gradients

29. Output equals an independently computed token lookup plus position lookup
    exactly, with CPU float32 dtype and no hidden transform.
30. Output is a new tensor, inputs are not mutated, and autograd connects to
    both parameters.
31. IDs `[3,1,3]` produce the exact selected, repeated, and unselected token-row
    gradients specified above.
32. A two-member batch produces the exact shared-position gradient
    accumulation specified above.
33. Backward reaches both tables without an optimizer or parameter update.

### One-hot and safety

34. A tiny fixed synthetic example proves one-hot multiplication equals direct
    lookup exactly.
35. Error strings, `repr`, and `details` contain only approved safe facts and
    never tensor contents or source text.

### Phase boundary review

36. Review the Phase 3 change set and import graph to confirm that it introduces
    no corpus windowing, persistent token stream, target construction,
    optimizer, training loop, language-model loss, QKV/attention, Transformer
    block, output head, or generation path. This is a checkpoint review rather
    than a brittle source-text unit test.

## Later-phase boundary

Phase 3 is representations only. It does not authorize production context
windows or batches, next-token targets, logits, probabilities, cross-entropy,
attention, QKV projections, causal masks, Transformer blocks, a model/output
head, an optimizer, parameter updates, a training loop, checkpointing, or
generation. A scalar synthetic reduction exists only to expose autograd.

## Accepted Phase 3 gates

1. **Learning/design discussion.** Complete.
2. **Conceptual architecture approval.** Complete through DEC-0015.
3. **Detailed contract documentation.** Complete in this specification.
4. **Independent detailed-contract review.** Complete: the first review
   returned FAIL with six required corrections while leaving the accepted
   conceptual architecture unchanged.
5. **Contract correction, focused independent re-review, and explicit
   acceptance.** Complete: all six findings were corrected, focused independent
   re-review returned PASS, master-chat adjudication returned PASS, and
   Sebastien explicitly accepted the corrected detailed contract.
6. **Accepted contract commit.** This explicitly authorized dedicated commit
   completes Gate 6.
7. **Contract push and remote verification.** Complete: local and remote `main`
   were verified at the Gate 6 commit.
8. **Explicit implementation authorization.** Complete: Sebastien authorized
   implementation strictly under this accepted contract.
9. **Implementation and deterministic synthetic tests.** Complete locally:
   the minimal runtime vocabulary binding, representation module, and contract
   checks are implemented and pass.
10. **Independent implementation review.** Complete: the independent review
    returned PASS with no must-fix issues, contract contradictions, or
    later-phase leakage.
11. **Implementation correction, focused independent re-review, and explicit
    acceptance.** Complete: no corrections or focused re-review were required;
    master-chat adjudication returned PASS, and Sebastien explicitly accepted
    the implementation.
12. **Accepted implementation commit.** This explicitly authorized dedicated
    implementation commit completes Gate 12.
13. **Implementation push and remote verification.** Not authorized.
14. **Separately justified educational demonstration or inspection, if
    required.** Not authorized and not currently required for an exit criterion
    beyond the synthetic demonstrations in Gate 9.
15. **Independent Phase 3 exit review against exactly the three roadmap
    criteria.** Not started.
16. **Phase 3 closure commit, if required.** Not authorized.
17. **Closure push and remote verification.** Not authorized.
18. **Explicit Phase 4 authorization.** Not authorized.

No major checkpoint may be collapsed into another, and later gate status does
not advance implicitly.
