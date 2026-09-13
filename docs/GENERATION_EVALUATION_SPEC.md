# Phase 9 Generation and Evaluation Specification

> Status: documentation-only detailed-contract draft under accepted DEC-0021.
> This draft authorizes no implementation, checkpoint load, model construction,
> generation, sampling, evaluation, artifact creation, or sealed-test access.

## Purpose and authority levels

Phase 9 has the roadmap goal:

> Generate text and evaluate model behavior with appropriately modest claims.

DEC-0021 is already accepted conceptual architecture. This document proposes
the exact mechanics needed to make that architecture independently
implementable and reviewable. Until fresh independent review, correction if
needed, and explicit Sebastien acceptance, every mechanic introduced here is a
proposal rather than accepted authority.

The authority levels are deliberately separate:

1. **Already accepted:** DEC-0021 and all inherited Phase 0–8 authority.
2. **Proposed here:** exact interfaces, records, validation, arithmetic,
   evidence, tests, and workflow mechanics.
3. **Future implementation:** source and tests, only after contract acceptance,
   publication, and separate implementation authorization.
4. **Future execution:** checkpoint loading, model construction, generation,
   metrics, baselines, and evidence creation, only after accepted implementation
   publication and separate run authorization.
5. **Future sealed-test event:** one explicitly authorized event after every
   required identity and policy is frozen.

## Accepted Phase 9 architecture preserved

This specification does not reopen DEC-0021. It preserves exactly:

- one Phase 9-owned read-only loader selecting canonical `best_validation`;
- complete catalog, hash, schema, and provenance validation;
- no change to Phase 8 `load_phase8_checkpoint` and no optimizer, ordering,
  progress, metric, resume-lineage, or RNG restoration for inference;
- nonempty encodable prompts of at most 256 tokens;
- an explicit requested number of generated tokens;
- newest-256 rolling context with full recomputation and no KV cache;
- deterministic greedy argmax;
- positive-temperature, explicit-top-k categorical sampling;
- `top_k=81` as the complete distribution;
- a dedicated seeded CPU sampling generator isolated from global RNG;
- validation-led character NLL/cross-entropy, perplexity, and secondary top-1
  accuracy;
- uniform-over-81 and training-only empirical-unigram baselines on equivalent
  transitions;
- fixed pre-registered qualitative prompts, lengths, settings, and seeds;
- complete recording of accepted samples without cherry-picking;
- strict separation of exploration from accepted evidence;
- a separately authorized one-shot *Twelfth Night* gate; and
- claims limited to this small character-level Shakespeare model.

## Consolidated proposed contract-level policies

DEC-0021 deliberately leaves detailed mechanics to contract review. The
following policies are proposed here as one coherent contract package; they are
not previously accepted model architecture and do not become authoritative
unless the corrected complete contract is accepted:

1. generated-token requests use the inclusive resource bound `0..1024`;
2. the training-only empirical unigram is explicitly Laplace-smoothed with one
   added count per vocabulary token as a simple robustness policy, not because
   zero production counts have been demonstrated;
3. accepted qualitative evidence uses the exact six-row prompt/settings/seed/
   length matrix defined below;
4. the public surface exposes small pure mechanics plus two thin governed
   runners, while corpus loading, production comparison, and sealed operations
   remain internal orchestration details;
5. the six-class error hierarchy and deterministic validation precedence below
   own Phase 9 failures;
6. accepted attempts use explicit evaluation IDs, append-only record grammars,
   durable no-clobber attempt/access markers, canonical evidence, and
   new-identity retry rules;
7. the sealed authorization is consumed by durable marker publication before
   the first test open, including interruption after that commit point; and
8. Phase 9 may close with the sealed test unused only through a later explicit
   Sebastien decision accepting validation-only evidence; otherwise an
   authorized one-shot test result must be reviewed before closure.

Reviewers should accept, reject, or correct this package at contract level
without fragmenting it into new model-architecture decisions. None of these
policies authorizes implementation or execution.

## Authoritative inherited identities

### Mandatory validation inputs

The following identities are correctness requirements. A mismatch fails closed;
it is not merely reported:

| Authority | Exact identity |
|---|---|
| Phase 8 closure | `6d086625cb293f61700bd59b551ea310a5b680b9` |
| Phase 8 result commit | `334119e63c716e4922cd0b67da4f5fc221faa886` |
| Accepted experiment | `EXP-20260912-01`, completed, result `PASS` |
| Phase 8 implementation anchor | `809834323d53407cb4a54ae539585bb3d78856eb` |
| Corrected Phase 8 contract commit | `c50d77ac935bdf924b9b429a5776c419982a5d11` |
| Corrected Phase 8 provenance implementation | `3e0b9c69963702718446901b2fb85e4f5e38aea9` |
| Phase 8 specification SHA-256 | `1630f9c7a113ff4af6db709ba2c356e8dab450eb4d918a54d64b734084e70efd` |
| Catalog path | `checkpoints/phase8/EXP-20260912-01/checkpoint-catalog.json` |
| Catalog SHA-256 | `6d590f0355ec950d770e77b4c542d65cda15012fbea56349b97b154a0d614241` |
| Selected role | exactly `best_validation` |
| Selected logical ID / epoch | `epoch-0010` / `10` |
| Selected validation loss | `0x1.3f94b678f5807p+1` |
| Selected object SHA-256 | `6990c89166d48732b0601aeae1edd57a9f0e22eec1c7c1160d5be1b8281c9148` |
| Selected object path | `objects/6990c89166d48732b0601aeae1edd57a9f0e22eec1c7c1160d5be1b8281c9148.pt` |
| Phase 7 closure | `33d4510421107848c4aa8a6014f4a7b1e391065a` |
| Phase 7 contract / implementation | `60b2a9cce55da79ccc9fbd03fad014cb2a939290` / `3139b1736f005fe903e2ea111d91934478b5a683` |
| MiniGPT specification SHA-256 | `3e1987658d4c9ece59bebbea7061f939c1138aaa73751a523f659f8b3217c022` |
| MiniGPT source / export SHA-256 | `6b8db96577a0ad1f5daa4649d7ca9375e59873d4b635c3f76e75a6751f74f12d` / `af056658e6d7ffe6de89b3ac0486929305655ea76029240d26869b705a2a5d86` |
| Tokenizer | `shakespeare-code-point-v1`, schema `1`, vocabulary size `81` |
| Tokenizer implementation | `de7a7f096fbbd8607c944412eaef30be9b686b56` |
| Vocabulary path / SHA-256 | `artifacts/tokenizers/shakespeare-code-point-v1/vocabulary.json` / `9f4235f9dab3e0361221a90c5ae7ca7fee540760fcc1708964368e6f17d6d70e` |
| Dataset | `shakespeare-eight-play` |
| Phase 1 manifest SHA-256 | `157e324c41c6aee756b9c554ae465388a892ea9a0f8fb8296ef0d986a0c9f6fb` |
| Processing manifest SHA-256 | `bbf938e565022dde72f26470e2fa7214f2fe1735afbc62ace320f1e6ebf372cc` |
| Dependency lock SHA-256 | `8ddf7a11bf67d36bce8ba58abffc5416a0fb778cdc20d03c077d318715122cfa` |

### Protected inherited executable bytes

Phase 9 uses exact live Git-blob equality against the Phase 8 closure commit as
its one protection strategy for inherited executable behavior. After proving
`6d086625cb293f61700bd59b551ea310a5b680b9` is an ancestor of live `HEAD`, the
loader or governed runner reads each live file and the exact blob at
`6d086625:<path>` and requires byte equality for this exhaustive protected set:

```text
src/sebgpt/__init__.py
src/sebgpt/tokenization/__init__.py
src/sebgpt/tokenization/code_point.py
src/sebgpt/tokenization/vocabulary_artifact.py
src/sebgpt/model/__init__.py
src/sebgpt/model/embeddings.py
src/sebgpt/model/self_attention.py
src/sebgpt/model/transformer_block.py
src/sebgpt/model/simple_language_model.py
src/sebgpt/model/mini_gpt.py
src/sebgpt/data/__init__.py
src/sebgpt/data/shakespeare_extract.py
src/sebgpt/data/shakespeare_preflight.py
src/sebgpt/data/shifted_examples.py
src/sebgpt/data/shakespeare_examples.py
src/sebgpt/model/phase4_corpus.py
src/sebgpt/model/phase4_experiment.py
src/sebgpt/training/phase8_types.py
src/sebgpt/training/phase8_checkpoint.py
```

This 19-path set covers the top-level `sebgpt` and data package initializers;
the tokenizer and accepted vocabulary loader; complete MiniGPT, every component
it composes, and the transitive modules eagerly imported by the accepted
`sebgpt.model` package; accepted cross-entropy; and the Phase 8
checkpoint/type implementation whose catalog/payload/provenance semantics Phase
9 validates. Phase 9's checkpoint reader is independently owned and must not
import or call the Phase 8 training package, which prevents optimizer-oriented
package initialization and preserves the Phase 8 interface unchanged. Phase 9
source is not in this set; its separately accepted implementation commit and
file hashes are bound by pre-registration.

Tests, documentation not used at runtime, and source modules neither transitively
imported by the accepted model package nor semantically referenced by the Phase
9 loader are intentionally excluded. Their accepted historical identities remain
review evidence rather than execution dependencies. A protected live file that
differs from its closure blob fails before checkpoint, corpus, or marker access
even when the worktree is otherwise clean and the closure commit is an ancestor.

The catalog's complete `latest` and `best_validation` graph must validate under
the accepted Phase 8 schema even though Phase 9 selects only
`best_validation`. Both roles currently reference the same immutable epoch-10
object. Equality of their bytes does not merge their semantics.

The checkpoint payload's complete schema, including optimizer, progress, RNG,
mode, metric, and lineage mappings, remains mandatory validation input. Phase 9
validates those fields because they establish artifact integrity; it does not
install, return, or otherwise restore their operational state.

The supported accepted-execution runtime remains CPython 3.14.4, PyTorch
2.14.0, Darwin arm64, CPU parameters, `torch.float32` model values, and
`torch.long` token IDs. Accepted evaluation additionally requires the complete
Phase 8 recorded deterministic runtime envelope and dependency-lock identity to
match without mutating caller runtime settings.

### Dataset and split authority

- Training is exactly manifest orders 1–6: *Hamlet*, *Romeo and Juliet*,
  *Macbeth*, *A Midsummer Night's Dream*, *Much Ado About Nothing*, and
  *Henry V*.
- Validation is exactly manifest order 7, *The Tempest*.
- Test is exactly manifest order 8, *Twelfth Night*.
- Training-only information may fit the proposed Laplace-smoothed empirical
  unigram baseline.
- Validation may guide development and is used for accepted development
  evaluation.
- Test cannot influence code, checkpoint selection, metrics, baselines,
  prompts, lengths, settings, seeds, interpretation policy, or reruns.
- A context window never crosses a work boundary.

### Informational provenance

The following facts are recorded for interpretation but are not independently
selected or tuned by Phase 9:

- final Phase 8 training loss `2.4611534265660575`;
- final/best validation loss `2.4967258539791177` at epoch 10;
- ten completed epochs and 3,880 optimizer updates;
- exact-resume result `PASS`;
- retained Phase 8 training-evidence SHA-256
  `28d5370501b9c97b0429477da9f679314a2337714f6e1524f7e757fe2f8eb784`;
- accepted MiniGPT focused-test SHA-256
  `95ec62d1164c48525e59e33f3f90fada6d482c84a31ca16edba38178b16b2cab`;
- historical epoch-1 through epoch-9 object identities.

The selected validation-loss hex is nevertheless mandatory where the catalog,
reference, payload metric state, and accepted result record cross-validate.

## Proposed module boundary and public interface

Phase 9 implementation will be owned by a new `sebgpt.inference` package. It
must not modify the public signatures or semantics of any accepted Phase 8
module. Proposed files are:

```text
src/sebgpt/inference/__init__.py
src/sebgpt/inference/phase9_types.py
src/sebgpt/inference/phase9_checkpoint.py
src/sebgpt/inference/phase9_generation.py
src/sebgpt/inference/phase9_evaluation.py
src/sebgpt/inference/phase9_experiment.py
```

Exact ownership is:

| Module | Public names owned |
|---|---|
| `phase9_types` | all six Phase 9 exceptions and all eight Phase 9 records |
| `phase9_checkpoint` | `load_phase9_inference_bundle` |
| `phase9_generation` | `select_greedy_token_id`, `build_categorical_probabilities`, `generate_phase9_text` |
| `phase9_evaluation` | `build_phase9_windows`, `fit_phase9_laplace_unigram`, `evaluate_phase9_model`, `evaluate_phase9_uniform`, `evaluate_phase9_laplace_unigram` |
| `phase9_experiment` | `run_fixed_phase9_development_evaluation`, `run_fixed_phase9_sealed_test_evaluation` |

Each module's literal `__all__` contains exactly its row in the order shown.
The package `__all__` contains exceptions in the order shown under **Error
model**, records in the order shown under **Proposed public records**, then
functions in signature order below.

The exact proposed public interfaces are:

```python
load_phase9_inference_bundle(
    repository_root: pathlib.Path,
) -> Phase9InferenceBundle

select_greedy_token_id(
    next_token_logits: torch.Tensor,
) -> int

build_categorical_probabilities(
    next_token_logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
) -> torch.Tensor

generate_phase9_text(
    bundle: Phase9InferenceBundle,
    prompt: str,
    *,
    generated_token_count: int,
    mode: typing.Literal["greedy", "categorical"],
    temperature: float | None,
    top_k: int | None,
    seed: int | None,
) -> Phase9Generation

build_phase9_windows(
    documents: tuple[Phase9TokenDocument, ...],
    *,
    vocabulary_size: int,
    context_length: int,
    stride: int,
) -> tuple[Phase9Window, ...]

fit_phase9_laplace_unigram(
    training_windows: tuple[Phase9Window, ...],
    *,
    vocabulary_size: int,
) -> Phase9LaplaceUnigramBaseline

evaluate_phase9_model(
    model: MiniGPT,
    windows: tuple[Phase9Window, ...],
    *,
    predictor: str,
    split: str,
) -> Phase9Metrics

evaluate_phase9_uniform(
    windows: tuple[Phase9Window, ...],
    *,
    vocabulary_size: int,
    split: str,
) -> Phase9Metrics

evaluate_phase9_laplace_unigram(
    baseline: Phase9LaplaceUnigramBaseline,
    windows: tuple[Phase9Window, ...],
    *,
    split: str,
) -> Phase9Metrics

run_fixed_phase9_development_evaluation(
    repository_root: pathlib.Path,
    *,
    evaluation_id: str,
) -> Phase9EvidenceReference

run_fixed_phase9_sealed_test_evaluation(
    repository_root: pathlib.Path,
    *,
    evaluation_id: str,
) -> Phase9EvidenceReference
```

All arguments shown without defaults are required. `temperature`, `top_k`, and
`seed` are required keyword arguments even though their values are nullable.
This makes greedy non-use explicit and prevents hidden defaults.

### Proposed public records

All records are exact frozen dataclasses. Tensor-, model-, tokenizer-,
vocabulary-, text-, token-, count-vector-, and probability-vector-bearing fields
are excluded from generated `repr`.

```python
@dataclass(frozen=True)
class Phase9CheckpointIdentity:
    run_id: str
    role: str
    logical_id: str
    epoch: int
    validation_loss_hex: str
    checkpoint_sha256: str
    checkpoint_relative_path: str
    catalog_sha256: str

@dataclass(frozen=True)
class Phase9InferenceBundle:
    model: MiniGPT
    tokenizer: CodePointTokenizer
    vocabulary: VocabularyBinding
    checkpoint: Phase9CheckpointIdentity

@dataclass(frozen=True)
class Phase9Generation:
    prompt: str
    generated_text: str
    full_text: str
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    mode: str
    generated_token_count: int
    temperature: float | None
    top_k: int | None
    seed: int | None
    context_capacity: int
    local_generator_final_state_sha256: str | None

@dataclass(frozen=True)
class Phase9Window:
    document_id: str
    document_order: int
    split: str
    start_index: int
    input_ids: tuple[int, ...]
    target_ids: tuple[int, ...]

@dataclass(frozen=True)
class Phase9TokenDocument:
    document_id: str
    document_order: int
    split: str
    token_ids: tuple[int, ...]

@dataclass(frozen=True)
class Phase9Metrics:
    predictor: str
    split: str
    window_sha256: str
    nll: float
    perplexity: float
    correct_count: int
    target_count: int
    top1_accuracy: float
    window_count: int

@dataclass(frozen=True)
class Phase9LaplaceUnigramBaseline:
    vocabulary_size: int
    training_target_count: int
    raw_counts: tuple[int, ...]
    smoothed_counts: tuple[int, ...]
    probabilities: tuple[float, ...]
    top_token_id: int

@dataclass(frozen=True)
class Phase9EvidenceReference:
    evaluation_id: str
    kind: str
    status: str
    relative_path: str
    byte_count: int
    sha256: str
```

Whenever a record or artifact says `evidence-reference mapping`, its exact JSON
keys in record-field order are `evaluation_id`, `kind`, `status`,
`relative_path`, `byte_count`, and `sha256`, matching
`Phase9EvidenceReference`. Paths are repository-relative fixed artifact paths;
byte count is positive; digest is lowercase SHA-256.

Exact field types are `str`, `str`, `str`, `str`, exact built-in `int`, and
`str` respectively. `evaluation_id` matches the Phase 9 ID grammar;
`byte_count > 0`; `sha256` is exactly 64 lowercase hexadecimal characters. The
only valid `kind`/`status`/path/size-limit combinations are:

| `kind` | `status` | Exact relative path suffix | Maximum bytes |
|---|---|---|---:|
| `phase9_development_attempt` | `consumed` | `artifacts/phase9-development-attempt.json` | 16,384 |
| `phase9_development` | `completed` | `artifacts/phase9-development-evidence.json` | 1,048,576 |
| `phase9_development_failure` | `failed` | `artifacts/phase9-development-failure.json` | 1,048,576 |
| `phase9_sealed_test_access` | `consumed` | `artifacts/phase9-sealed-test-access.json` | 16,384 |
| `phase9_sealed_test` | `completed` | `artifacts/phase9-sealed-test-evidence.json` | 262,144 |
| `phase9_sealed_test_failure` | `failed` | `artifacts/phase9-sealed-test-failure.json` | 262,144 |

Each full path is exactly
`experiments/<evaluation_id>/<suffix>`. No other kind or status exists;
cross-row combinations are invalid. An uncertain lifecycle has no artifact
kind/status and therefore uses JSON null rather than fabricating an evidence
reference.

The dataclass field order above is the validation/record-field order. When
nested in a canonical record or artifact, the mapping is serialized under that
container's `sort_keys=True` JSON rule; parsing first requires the exact six-key
set and exact values, then canonical reserialization equality.

`Phase9InferenceBundle` and `Phase9LaplaceUnigramBaseline` are factory-only: direct
construction raises `Phase9ContractError`. `Phase9TokenDocument` is deliberately
plain and constructible for small synthetic examples; pure functions fully
validate it before arithmetic. Other records are returned only after complete
function-level validation.

### Exact package exports

`sebgpt.inference` exports exactly the six Phase 9 error classes, the eight
records above, and the eleven public functions above. No private checkpoint
validator, sealed-test supplier, artifact writer, corpus-text helper, or RNG
helper is exported. The detailed implementation review must compare literal
module and package `__all__` values with this contract.

## Error model

The exact Phase 9-owned exceptions are:

```python
Phase9TypeError(TypeError)
Phase9ContractError(ValueError)
Phase9GovernanceError(ValueError)
Phase9CheckpointError(ValueError)
Phase9NumericalError(ArithmeticError)
Phase9PublicationError(RuntimeError)
```

Each exception stores an immutable `details` mapping and serializes a sorted,
ASCII-safe JSON object containing `invariant` plus structural safe facts. It
must not contain prompt text, generated text, token IDs, probabilities, logits,
parameters, corpus prose, arbitrary object representations, absolute paths, or
sealed-work content.

The stable invariant families are:

```text
phase9.type.argument
phase9.type.record
phase9.contract.bundle
phase9.contract.generation
phase9.contract.window
phase9.governance.repository
phase9.governance.runtime
phase9.governance.dataset
phase9.governance.sealed_test
phase9.checkpoint.path
phase9.checkpoint.catalog
phase9.checkpoint.hash
phase9.checkpoint.schema
phase9.checkpoint.provenance
phase9.checkpoint.cross_field
phase9.checkpoint.model_state
phase9.generation.prompt
phase9.generation.count
phase9.generation.mode
phase9.generation.settings
phase9.generation.context
phase9.generation.output
phase9.generation.mutation
phase9.sampling.logits
phase9.sampling.temperature
phase9.sampling.top_k
phase9.sampling.probabilities
phase9.sampling.generator
phase9.evaluation.split
phase9.evaluation.windows
phase9.evaluation.metric
phase9.evaluation.mutation
phase9.baseline.training_only
phase9.baseline.counts
phase9.baseline.probabilities
phase9.evidence.schema
phase9.evidence.amendment
phase9.evidence.publication
```

Wrong public argument or record types use `Phase9TypeError`. Well-typed values
violating local semantic rules use `Phase9ContractError`. Repository, runtime,
dataset, pre-registration, or sealed-access authority failures use
`Phase9GovernanceError`. Catalog, object, payload, or model-state failures use
`Phase9CheckpointError`. Nonfinite or unrepresentable arithmetic uses
`Phase9NumericalError`. No-clobber/durability failures after evidence publication
begins use `Phase9PublicationError`.

The accepted tokenizer owns unknown prompt characters: after Phase 9 prompt
type/empty/length validation, `CodePointTokenizer.encode` may raise its existing
`UnknownCodePointError` unchanged. Accepted MiniGPT and explicit-cross-entropy
exceptions likewise retain ownership after Phase 9 has completed its own
prevalidation. Phase 9 must not ambiguously translate one accepted child
failure into multiple possible Phase 9 errors.

## Read-only inference loading contract

### Repository preflight

`load_phase9_inference_bundle` performs no checkpoint or vocabulary open until:

1. `repository_root` is exact `pathlib.Path`;
2. it is absolute, physically resolved, exists, is not a symlink, and is the
   exact Git top level;
3. the branch is `main`;
4. local `HEAD == origin/main` and the tracked worktree/index are clean;
5. Phase 8 closure and result commits are ancestors of `HEAD`;
6. all mandatory inherited tracked identities match their accepted hashes;
7. the complete supported runtime and dependency-lock identity match; and
8. the accepted result record contains the exact catalog, role, object, loss,
   and experiment identities above.

Untracked and ignored retained project artifacts are allowed only at their
contracted locations. Any unexpected untracked file outside accepted ignored
roots fails repository cleanliness. The loader performs no network operation
and trusts the locally refreshed `origin/main`; real execution pre-registration
must independently require prior live-remote verification.

### Catalog and object validation

After repository preflight, loading uses the accepted Phase 8 local-project
trust model, path grammars, descriptor-relative no-follow traversal, catalog
limit 16,384 bytes, object limit 67,108,864 bytes, and same-opened-byte hashing.
It performs, in order:

1. open and identity-pin the repository, `checkpoints`, `phase8`, fixed run, and
   `objects` directories;
2. open the canonical catalog once, enforce regular-file type and size, read
   exact bytes, verify SHA-256, strict UTF-8 JSON, and canonical bytes;
3. validate the complete exact catalog schema and both role references;
4. require the `best_validation` reference to equal the accepted logical ID,
   epoch, validation-loss hex, digest, and relative path;
5. open the referenced object without following symlinks, enforce positive
   bounded size, read it once, and verify the digest over the same bytes;
6. deserialize those bytes with
   `torch.load(io.BytesIO(bytes), map_location="cpu", weights_only=True)`;
7. validate the complete Phase 8 payload schema and every nested exact type,
   key, tensor, configuration, authority, metric, mode, progress, optimizer,
   RNG, and lineage invariant;
8. validate catalog/reference/payload agreement under the semantic
   `best_validation` role; and
9. revalidate all retained directory and entry identities.

If `latest` and `best_validation` differ in a future accepted catalog, both
objects must be opened and validated to prove the graph, but only the exact
`best_validation` `model_state` may populate the inference model. The current
catalog legitimately permits one opened object to satisfy both roles because
both references name identical bytes.

### Model reconstruction and returned state

Only after complete artifact validation, the loader:

1. calls the accepted vocabulary loader and constructs one exact
   `CodePointTokenizer` from its code points;
2. constructs one exact Phase 7 `MiniGPT` with the accepted literal
   configuration and local deterministic construction streams;
3. loads only the selected payload's exact `model_state` with `strict=True`;
4. validates all 90 parameter names, objects, shapes, CPU devices, float32
   dtypes, finiteness, non-aliasing, and accepted values;
5. calls `model.eval()` once and requires every named module's training flag to
   be exact `False`;
6. requires every parameter's `.grad is None` while preserving its accepted
   `requires_grad=True` property; and
7. returns the exact factory-produced bundle and checkpoint identity.

The loader never constructs an optimizer and never installs checkpoint
optimizer state, progress, ordering state, global RNG state, dropout-generator
state, mode state, metric state, or resume lineage. Deserializing and validating
those payload fields is not restoration.

Caller global CPU RNG is snapshotted before any operation and must be bitwise
unchanged on success and every ordinary failure. Model construction already
uses accepted local generators; Phase 9 adds a final equality check before
return or re-raise. Loader calls create no gradient graph, alter no caller
module, write no file, and return no mutable checkpoint mapping.

The existing public signature and behavior of
`load_phase8_checkpoint(..., role: Literal["latest"])` remain byte-for-byte and
semantically unchanged.

### Loader validation precedence

The first failure is determined by this category-major order:

1. public argument type;
2. repository path shape and containment;
3. Git branch/synchronization/cleanliness/ancestry;
4. tracked inherited file identities and result record;
5. runtime identity;
6. checkpoint directory chain and catalog file safety;
7. catalog hash/canonical JSON/schema;
8. `latest`, then `best_validation`, reference schemas;
9. accepted `best_validation` identity;
10. object path/type/size/hash;
11. safe deserialization and complete payload schema;
12. catalog/payload cross-field and provenance agreement;
13. vocabulary artifact validation;
14. model construction and exact model-state load;
15. evaluation-mode/parameter/gradient guarantees; and
16. caller-global-RNG final equality.

No later-stage object allocation, model construction, parameter mutation, or
artifact content access occurs after an earlier failure.

## Generation contract

### Argument contract and precedence

`generate_phase9_text` validates, before tokenization or model arithmetic:

1. exact `Phase9InferenceBundle` type and factory identity;
2. bundle checkpoint identity, exact model/tokenizer/vocabulary object types,
   parameter structure, all-false evaluation modes, and all gradients `None`;
3. `prompt` is an instance of `str`;
4. prompt is nonempty;
5. Python length is at most 256; one Python string element is one accepted
   token unit, so this is also the pre-encoding token-capacity bound;
6. `generated_token_count` is exact built-in `int`, excluding `bool`;
7. count is in inclusive range `0..1024`;
8. `mode` is exact built-in `str` and exactly `greedy` or `categorical`;
9. mode-specific setting types and values in signature order
   `temperature`, `top_k`, `seed`; and
10. accepted tokenizer encoding, preserving its first-unknown behavior.

Greedy requires `temperature is None`, `top_k is None`, and `seed is None`.
Categorical requires exact built-in finite `float` temperature greater than
zero, exact built-in `int` top-k in `1..81`, and exact built-in `int` seed in
`0..9_223_372_036_854_775_807`. Booleans are rejected for integer fields.

Zero requested tokens is valid. After complete argument and prompt encoding
validation, it returns the prompt unchanged, an empty generated string and token
tuple, the validated configuration, `context_capacity=256`, and a null local
generator-state digest. It performs no model call and creates or consumes no
sampling generator.

### Autoregressive operation

For positive `generated_token_count`:

1. preserve the exact prompt string and immutable encoded prompt tuple;
2. copy token IDs into a private mutable history list;
3. for categorical mode only, construct one CPU `torch.Generator`, call
   `manual_seed(seed)` exactly once, and retain it for the whole generation;
4. for each requested new token, take
   `context = history[-256:]` without any other truncation or padding;
5. construct a new rank-one CPU `torch.long` tensor from that context;
6. under `torch.inference_mode()`, call the exact evaluation-mode MiniGPT once;
7. require finite CPU-float32 logits of shape `(len(context), 81)`;
8. use exactly `logits[-1, :]` as next-token logits;
9. select one token through the configured decoder;
10. append that exact integer ID to history and the generated-ID list; and
11. repeat until the exact requested count is reached.

Every call assigns accepted absolute positions `0..len(context)-1`. After a
roll, the oldest token is discarded from model context and the retained newest
256 tokens are reindexed `0..255`. The full returned history is still the
original prompt plus every generated token; rolling changes model visibility,
not the returned text.

After the loop, decode only the generated tuple and also decode the full token
tuple. Require:

```text
full_text == prompt + generated_text
len(generated_token_ids) == generated_token_count
len(prompt_token_ids) == len(prompt)
decode(prompt_token_ids) == prompt
decode(prompt_token_ids + generated_token_ids) == full_text
```

There is no EOS token and no content-dependent early stop. Completion occurs
only after the explicit count. No generated token is removed, normalized, or
replaced.

The call preserves parameter bytes and objects, gradients, module modes,
vocabulary/tokenizer state, input bundle identity, prompt value, and global RNG
on success and ordinary failure. Categorical local-generator advancement is
private to the call; its final state is hashed into the result. Greedy returns a
null generator digest and performs no random operation.

## Greedy decoding contract

`select_greedy_token_id` accepts exactly one CPU-float32 rank-one tensor of
shape `(81,)`. Validation order is tensor object, rank, shape, dtype, device,
then finiteness. It mutates nothing and consumes no RNG.

The selected ID is the smallest token ID whose logit equals the exact maximum.
This is the first-index behavior of `torch.argmax(next_token_logits).item()` on
the validated vector. The result is exact built-in `int` in `0..80`.

Greedy generation always applies this function to `logits[-1, :]`. Equal
maximum values therefore have deterministic lowest-ID resolution. There is no
temperature, softmax, probability construction, or random draw in the greedy
path.

## Stochastic decoding contract

### Probability construction

`build_categorical_probabilities` validates the logits exactly as the greedy
function, then validates exact finite positive built-in-float `temperature`,
then exact built-in-int `top_k` in `1..81`.

The arithmetic order is exact:

1. `scaled = next_token_logits / temperature` in CPU float32;
2. require every scaled value finite;
3. rank token IDs by descending scaled value, breaking exact ties by ascending
   token ID;
4. retain exactly the first `top_k` IDs;
5. set every non-retained effective logit to negative infinity;
6. subtract the maximum retained scaled logit;
7. exponentiate retained shifted values in float32 and assign exact zero weight
   to excluded IDs;
8. require all weights finite and nonnegative and their float32 sum finite and
   strictly positive;
9. divide all 81 weights by that sum in float32; and
10. require finite nonnegative probabilities, exact zeros outside top-k, exact
    shape/device/dtype, a positive retained probability for every retained ID,
    and a sum within `8 * torch.finfo(torch.float32).eps` of `1.0`.

The returned tensor is a new detached CPU-float32 `(81,)` tensor. It does not
alias logits and consumes no RNG. `top_k=81` retains every token and is exactly
the full temperature-scaled distribution. `top_k=1` yields probability one for
the same lowest-ID tie-resolved maximum that greedy selects.

A positive temperature may still be too small to keep float32 division finite;
that raises `Phase9NumericalError` rather than silently clipping, promoting
precision, or altering temperature.

### Categorical selection

Categorical generation calls the probability function on every final logit
row, then performs exactly one:

```python
torch.multinomial(
    probabilities,
    num_samples=1,
    replacement=True,
    generator=local_generator,
)
```

The sampled scalar is converted to exact built-in `int` and must lie in
`0..80`. The one local generator is never exposed to or replaced by caller
state. The global CPU generator is never passed, seeded, or consumed.

On the supported runtime, identical bundle parameters, prompt, count,
temperature, top-k, and seed must yield identical generated token IDs, text,
and final local-generator-state digest. Different seeds are not required to
produce different short outputs; tests prove state/control wiring rather than
making a probabilistic inequality claim.

No top-p, beam search, repetition/frequency/presence penalty, banned-token list,
minimum length, EOS rule, CFG, cache, or other decoder exists.

## Quantitative evaluation contract

### Pure mechanics versus governed production orchestration

The public functions in `phase9_evaluation` are corpus-neutral educational
mechanics. They know nothing about Git, filesystem paths, Shakespeare work IDs,
production counts, checkpoint catalogs, experiment records, artifact paths, or
sealed authorization. They accept small in-memory records and are directly
testable without fabricating production-sized data or patching production
constants.

The two fixed runners in `phase9_experiment` privately own production
governance: repository and record authority, accepted corpus loading and
tokenization, exact production identities/counts, inference loading, fixed
configuration, cross-predictor comparison, lifecycle markers, evidence, and
sealed protection. Private orchestration may compose the public pure functions;
pure functions never discover or select an experiment record.

### Pure token documents and windows

`build_phase9_windows` accepts an exact nonempty tuple of
`Phase9TokenDocument`, plus explicit exact built-in integers
`vocabulary_size`, `context_length`, and `stride`. It validates arguments in
that order, then documents in tuple order.

Pure-mechanics constraints are:

- `vocabulary_size >= 1`;
- `1 <= context_length <= 256`;
- `stride == context_length`, preserving exact-once nonoverlapping coverage;
- every document ID is a nonempty string and unique within the call;
- every document order is an exact nonnegative integer and strictly increases;
- every split is the same nonempty string supplied by the records and is not
  the reserved exact value `test`;
- every token tuple has length at least two; and
- every token is exact built-in `int` in `0..vocabulary_size-1`.

The function imposes no production split name, work name, work count, window
count, or target count. A direct synthetic document is valid when it satisfies
these local rules.

For each independent token sequence `z` of length `N`, require `N >= 2` and use
the caller's equal context/stride, no padding, and retained tails:

```text
for s in 0, stride, 2 * stride, ... while s < N - 1:
    T = min(context_length, N - 1 - s)
    input_ids  = z[s : s + T]
    target_ids = z[s + 1 : s + T + 1]
```

This covers each within-document next-token transition exactly once. Every
window has `1 <= T <= 256`, equal input/target lengths, and
`input_ids[1:] == target_ids[:-1]`. Canonical order is manifest order followed
by increasing start index. Windows never cross works and are never shuffled.

The governed development runner separately requires context/stride 256, the
exact manifest-authorized six training works and one validation work, and these
production counts before any metric or baseline work:

| Split | Works | Windows | Targets |
|---|---:|---:|---:|
| train | 6 | 3,099 | 792,699 |
| validation | 1 | 384 | 98,295 |

The production runner's private corpus loader validates the accepted tracked
Phase 1 manifest and ignored processing manifest before opening permitted
processed text. It verifies exact six-plus-one order; path containment;
regular-file/no-symlink type; byte count and SHA-256 over the same opened bytes;
strict UTF-8; LF-only normalization; terminal LF; code-point counts; and exact
work identities. It converts each work into a `Phase9TokenDocument`, never
joins documents, and then calls the pure builder.

Development orchestration categorically refuses test selection and performs no
test path stat/open/hash/decode/tokenization. The sealed runner owns a private
test-document path reachable only after its durable access marker. Test counts
may be derived from approved manifest numeric metadata before that event and are
revalidated against content only afterward.

### Common metric semantics

All three predictors are evaluated on the identical supplied window tuple in
canonical order. Before arithmetic, each pure evaluator canonically serializes
the complete window tuple as ASCII JSON with sorted mapping keys, compact
separators, and one final LF. Each window mapping contains exactly
`document_id`, `document_order`, `split`, `start_index`, `input_ids`, and
`target_ids`; token tuples serialize as JSON arrays. SHA-256 of those exact
bytes becomes `window_sha256`. The bytes are ephemeral and never persisted.

`Phase9Metrics` uses:

```text
nll            = total negative log likelihood / target_count
perplexity     = math.exp(nll)
top1_accuracy  = correct_count / target_count
```

All counts are exact built-in nonnegative integers; `target_count > 0` and
`window_count > 0`. NLL, perplexity, and accuracy are finite built-in floats;
`0.0 <= top1_accuracy <= 1.0` and `perplexity >= 1.0`. Evidence stores every
float together with its canonical lowercase `float.hex()` spelling.

`math.fsum` aggregates per-window or per-target Python-float contributions in
canonical order. No unweighted mean of window means is permitted. Perplexity is
computed exactly once from the final aggregate NLL with `math.exp`; overflow or
nonfinite output fails rather than saturating.

### Trained-model metrics

`evaluate_phase9_model` is pure with respect to production governance. It
requires one exact evaluation-mode `MiniGPT`, an exact nonempty window tuple,
and nonempty exact predictor/split strings matching every window; public split
`test` is reserved and rejected. It does not
require an inference bundle, production checkpoint identity, Shakespeare split
name, or production-sized count. Small accepted synthetic MiniGPT/windows are
valid inputs.

Before arithmetic it requires the model and every named module in evaluation
mode, all parameter gradients `None`, unchanged accepted parameters, and
canonical windows. It snapshots parameter bytes/objects, modes, gradients,
dropout-generator states, and global RNG.

For each window, under `torch.inference_mode()`:

1. create CPU-long input and target tensors;
2. call the model once and require finite CPU-float32 `(T, 81)` logits;
3. call accepted `explicit_cross_entropy(logits, targets)` once;
4. convert the finite scalar mean loss to Python float and append
   `mean_loss * T` to the aggregate list;
5. use `torch.argmax(logits, dim=1)`, whose first-index rule resolves ties to
   the lowest token ID; and
6. add the exact elementwise target-match count.

The final NLL is `math.fsum(weighted_losses) / target_count`. The governed
development runner separately passes the loaded best-validation model and
production validation windows, requires predictor `mini_gpt_best_validation`,
and requires loss hex `0x1.3f94b678f5807p+1` before evidence publication. The
pure evaluator contains none of those production literals.

On success and ordinary failure, parameter bytes/objects, modes, gradients,
dropout states, and global RNG must equal their entry state. The function never
constructs an optimizer, calls backward, or updates anything.

## Baseline contract

### Uniform-over-81 baseline

The uniform predictor assigns every token exact conceptual probability `1/81`.
For each target its negative log likelihood is `math.log(81)`. Aggregation uses
the same canonical targets, target count, window count, `math.fsum`, perplexity
calculation, and metric record as the trained model.

All 81 classes tie for top prediction, so the exact top-1 token is ID `0`. The
correct count is therefore the number of target IDs equal to zero. The
predictor label is exactly `uniform_81`.

`evaluate_phase9_uniform` validates explicit `vocabulary_size`, non-test split,
and all windows before arithmetic. It supports small synthetic vocabulary sizes and
reads no model or fitting statistics. The governed runners require size 81.

### Training-only Laplace-smoothed empirical unigram baseline

`fit_phase9_laplace_unigram` is a pure mechanic. It accepts any completely
validated nonempty window tuple whose common split is exactly `train`, plus an
explicit positive vocabulary size. It counts only target IDs from those
windows. The governed development runner separately requires the production
counts 3,099 and 792,699 before fitting.

For token ID `i` in `0..80`:

```text
raw_count[i]      = number of training target transitions equal to i
smoothed_count[i] = raw_count[i] + 1
denominator       = training_target_count + vocabulary_size
probability[i]    = smoothed_count[i] / denominator
```

This fixed add-one rule is a chosen simple robustness policy; it is not based on
a demonstrated zero count in the production corpus. Counts use Python integers;
probabilities are Python float division in token-ID order. Require
all raw counts nonnegative, all smoothed counts positive, their exact sums to
equal the expected totals, every probability finite and positive, and
`math.fsum(probabilities)` within `8 * sys.float_info.epsilon` of `1.0`.

The top unigram token is the greatest raw count, breaking ties by lowest token
ID. No validation/test count, smoothing choice, or observed model output can
affect fitting. The predictor label is exactly
`training_laplace_unigram_add_one`.

`evaluate_phase9_laplace_unigram` validates the factory-produced baseline,
nonempty non-test split string, and locally valid requested windows. For every target it accumulates
`-math.log(probabilities[target_id])` and counts equality to the fixed top token.
It uses the common aggregation and metric record. The same frozen baseline is
applied unchanged to validation and, only through the private sealed runner,
test transitions.

### Metric equivalence and interpretation

The private shared comparison operation owned by `phase9_experiment` receives
the three completed production metric records and the canonical window digest.
It requires identical split, target count, window count, and window digest
before a governed runner may publish evidence. Their NLLs, perplexities, and
accuracies may differ, but no predictor may receive additional history, skip
tails, cross documents, or use a different target set. Pure evaluator calls do
not look for sibling predictors or perform global experiment comparison.

Beating either baseline is an observation, not a Phase 9 success prerequisite.
A worse result is recorded without tuning or suppressing it. The historical
Phase 4 validation loss may be discussed only as a non-equivalent contextual
observation because its model, context length, optimizer, and training regime
differ.

## Qualitative evaluation contract

### Exploratory versus accepted output

Exploratory generation may occur only after accepted implementation publication
and separate authorization. It must be labeled `exploratory`, stored outside
accepted evidence, and cannot be renamed or copied into accepted evidence.
Exploration cannot change the fixed matrix below; a desired change requires an
explicit contract amendment, new review, new contract acceptance/publication,
and a fresh pre-registration before accepted generation.

Accepted qualitative generation is exactly this six-sample matrix:

| Sample ID | Prompt | New tokens | Mode | Temperature | Top-k | Seed |
|---|---|---:|---|---:|---:|---:|
| `romeo-greedy` | `"ROMEO:\n"` | 256 | `greedy` | null | null | null |
| `romeo-focused` | `"ROMEO:\n"` | 256 | `categorical` | `0.8` | 20 | 9001 |
| `romeo-full` | `"ROMEO:\n"` | 256 | `categorical` | `1.0` | 81 | 9002 |
| `to-be-greedy` | `"To be"` | 256 | `greedy` | null | null | null |
| `to-be-focused` | `"To be"` | 256 | `categorical` | `0.8` | 20 | 9003 |
| `to-be-full` | `"To be"` | 256 | `categorical` | `1.0` | 81 | 9004 |

The accepted vocabulary artifact contains every code point in both prompts.
The matrix deliberately compares greedy, narrower top-k/lower-temperature, and
full-distribution sampling for two fixed non-test prompts. Every output crosses
the 256-token total-history boundary and therefore exercises rolling context.

The development runner executes rows in table order using one freshly created
generation call per row. It records all six outputs, including repetition,
malformed structure, empty-looking whitespace, or other poor behavior. It may
not retry a row for a more attractive seed or output. A technical failure makes
the run failed; it does not permit omission of earlier completed rows.

### Sample provenance and allowed claims

Each accepted sample records exact sample ID, prompt and UTF-8 prompt SHA-256,
checkpoint role/digest, mode, requested length, temperature hex or null, top-k
or null, seed or null, generated token count, generated text, generated-text
UTF-8 SHA-256, full-text UTF-8 SHA-256, and categorical final-generator-state
SHA-256 or null.

Full generated text is retained only in ignored immutable machine-readable
evidence under the accepted generated-artifact policy. Tracked
`EXPERIMENT_LOG.md` records every sample ID, settings, byte/count facts, and
hashes plus a complete evidence-artifact digest; it does not duplicate generated
text. No generated sample is silently edited, normalized, truncated, excerpted
as if complete, or hand-selected.

Qualitative evidence may support statements such as “this fixed sample repeats
speaker-like labels” or “coherence degraded after the rolling boundary.” It
cannot establish factuality, authorship, understanding, general language
ability, memorization, or broad generalization. Observations must identify the
specific fixed sample; interpretations and speculation must remain labeled.

## Unified governed-publication state machine

Every governed artifact publication—development attempt marker, sealed access
marker, development completed/failure evidence, and sealed completed/failure
evidence—uses the same four exact states:

| State | Exact meaning |
|---|---|
| `not_published` | Initial in-memory state before any final-path publication operation is attempted. It makes no filesystem claim. |
| `known_absent` | The operation is proven to have failed before the final target could exist, or a descriptor-relative no-follow inspection proves final-path absence while the retained parent identity remains stable. |
| `known_present` | Exclusive final-path publication, exact entry validation, file durability, parent-directory durability, and retained-identity revalidation all completed successfully during this invocation. |
| `unknown` | Publication or durability may have occurred but cannot be proven completely; this includes ambiguous link/rename outcome, observed final-path presence without confirmed durability, file-fsync success followed by directory-fsync failure, substitution/identity uncertainty, or interruption obscuring the commit point. |

Each publisher begins `not_published`, validates expected absence and no-clobber
authority, creates/writes/fsyncs its exclusive temporary file, publishes the
final name without overwrite, validates the final entry, fsyncs the parent, and
revalidates retained identities. Only that complete success reaches
`known_present`. A proven pre-final failure reaches `known_absent`; every other
escaping publication failure reaches `unknown`. No state transition deletes,
repairs, overwrites, or renames an existing final artifact.

For terminal pairs—development completed versus development failure, and sealed
completed versus sealed failure—the publisher performs descriptor-relative
read-only inspection of both final paths before either publication. The exact
opposite-kind rule is:

1. if either relevant final path is confirmed present, publish no opposite-kind
   artifact;
2. if both are confirmed absent and lifecycle policy permits the intended kind,
   attempt its normal exclusive publication;
3. if either path's state is `unknown`, publish neither kind; and
4. after an attempted terminal publication, inspect both paths again on any
   non-`known_present` result before deciding tracked status.

A confirmed matching completed artifact yields tracked `completed`; a confirmed
matching failure artifact yields tracked `failed`. A computed failure with both
terminal artifacts `known_absent` may yield tracked `failed` with null evidence
reference and the normalized failure directly in the result record. Any
ambiguous target/opposite state yields tracked `uncertain`, retains the consumed
ID, and forbids opposite publication. A path that exists but is malformed or
hash/schema-mismatched blocks its opposite and yields `uncertain`; existence is
never treated as permission to overwrite.

The state machine is proportional and local to the two fixed runners. It is not
a general transaction framework.

## Development-attempt lifecycle

`run_fixed_phase9_development_evaluation` requires an explicit
`evaluation_id`; it never discovers “the current” or “latest” record. After all
repository, runtime, record, predecessor, configuration, artifact-absence, and
authorization preflight succeeds—but before checkpoint open, corpus open,
model construction, metric work, baseline fitting, or generation—it publishes:

```text
experiments/<evaluation_id>/artifacts/phase9-development-attempt.json
```

The canonical, immutable marker has exactly:

```text
schema_version: int                     # 1
evaluation_id: str
kind: str                               # "phase9_development_attempt"
development_plan_record_sha256: str
development_pre_registration_commit: str
development_authorization_record_sha256: str
development_authorization_commit: str
phase9_contract_sha256: str
phase9_implementation_commit: str
checkpoint_sha256: str
created_at_utc: str
consumed: bool                          # true
```

Publication is descriptor-relative, exclusive, no-follow, no-clobber, and
durable through file then parent-directory `fsync`. Competing processes using
the same ID race only at exclusive marker creation: exactly one may commit the
marker; every loser fails before governed work. Different eligible IDs remain
independent.

Marker `known_present` is the development attempt's commit point. Before a
publication attempt, preflight failure leaves state `not_published`, consumes no
ID, and may be retried only with unchanged accepted plan/authorization bytes.
A marker failure ending `known_absent` likewise permits such a retry. Marker
`known_present` permanently consumes the ID and permits governed work.

Marker state `unknown` conservatively consumes the ID but permits no governed
work. Read-only inspection may refine `unknown` to `known_absent` only by proving
the final path absent with stable parent identity; an observed path cannot
retroactively prove missing directory durability and remains `unknown`. Normal
failure after commit, process death, machine interruption, or missing final
evidence requires a new evaluation ID. Existing marker bytes are never treated
as idempotent runner permission. Inspection never deletes, replaces, repairs,
or fsyncs marker state.

After the marker, the runner retains safely completed metric and sample values
in memory and publishes completed or failed evidence when possible. Abrupt
termination may leave only the marker; subsequent inspection classifies that ID
as `uncertain` and appends an uncertain development result before a successor is
planned. No general transaction coordinator, lock service, or recovery daemon
is introduced.

## Development evidence artifact

### Location, canonical bytes, and publication

The development runner uses a pre-registered ID matching
`EXP-[0-9]{8}-[0-9]{2}` and publishes successful evidence at:

```text
experiments/<evaluation_id>/artifacts/phase9-development-evidence.json
```

A normal caught failure after attempt-marker publication instead targets:

```text
experiments/<evaluation_id>/artifacts/phase9-development-failure.json
```

Both files are ignored by Git, immutable, mutually exclusive for one ID, and
limited to 1,048,576 bytes. Each is an exact built-in mapping serialized with
UTF-8, `ensure_ascii=True`, sorted keys, indent 2, `allow_nan=False`, and one
final LF. SHA-256 is over those exact raw bytes.

Publication follows repository-contained descriptor-relative no-follow,
exclusive temporary creation, file `fsync`, no-clobber final publication,
parent-directory `fsync`, and identity revalidation. An existing exact artifact
may be verified only by a read-only evidence reader; a runner never reuses or
overwrites it. Any existing final evidence path raises
`Phase9PublicationError` because the attempt ID is already consumed.

### Development evidence schema version 1

The top-level mapping has exactly:

```text
schema_version: int                     # exactly 1
evaluation_id: str
kind: str                               # exactly "phase9_development"
status: str                             # exactly "completed"
recorded_at_utc: str                    # second-resolution UTC ...Z
attempt_marker_sha256: str
authority: mapping
configuration: mapping
model_validation: metric mapping | None
uniform_validation: metric mapping | None
laplace_unigram_baseline: unigram mapping | None
laplace_unigram_validation: metric mapping | None
samples: list[sample mapping]
sealed_test_access: str                 # exactly "none"
failure: None
limitations: list[str]
```

`authority` has exactly:

```text
phase8_closure_commit: str
phase8_result_commit: str
phase8_experiment_id: str
phase8_implementation_commit: str
phase8_contract_commit: str
phase8_spec_sha256: str
phase9_contract_commit: str
phase9_contract_sha256: str
phase9_implementation_commit: str
development_plan_record_sha256: str
development_pre_registration_commit: str
development_authorization_record_sha256: str
development_authorization_commit: str
catalog_sha256: str
checkpoint_role: str                    # best_validation
checkpoint_sha256: str
vocabulary_sha256: str
dataset_manifest_sha256: str
processing_manifest_sha256: str
requirements_lock_sha256: str
```

Future contract and implementation commit fields are external authorities fixed
after their respective accepted commits; they are never self-referential values
inside those commits. The development pre-registration commit contains the
already accepted plan record and is derived externally at launch.

`configuration` has exactly:

```text
context_length: int                     # 256
stride: int                             # 256
retain_unpadded_tail: bool               # true
maximum_generated_tokens: int            # 1024
metric_names: list[str]                  # nll, perplexity, top1_accuracy
baseline_names: list[str]                # uniform_81, training_laplace_unigram_add_one
unigram_smoothing: str                   # add_one
qualitative_matrix: list[configuration-row mapping]
```

A metric mapping has exactly:

```text
predictor: str
split: str
window_sha256: str
nll: float
nll_hex: str
perplexity: float
perplexity_hex: str
correct_count: int
target_count: int
top1_accuracy: float
top1_accuracy_hex: str
window_count: int
```

An unigram mapping has exactly:

```text
vocabulary_size: int
training_target_count: int
raw_counts: list[int]                   # length 81, token-ID order
smoothed_counts: list[int]              # length 81
probabilities: list[float]              # length 81
probabilities_hex: list[str]            # length 81
top_token_id: int
```

A tracked development result never embeds that full 81-element mapping. Its
`Laplace unigram baseline` value is exactly one
`laplace-unigram-result-reference` mapping or JSON null:

```text
schema_version: int                     # 1
baseline_kind: str                      # "training_laplace_unigram_add_one"
vocabulary_size: int                    # 81
training_target_count: int              # 792699
raw_counts_sha256: str
smoothed_counts_sha256: str
probabilities_hex_sha256: str
top_token_id: int                       # 0..80
baseline_sha256: str
evidence: evidence-reference mapping
```

`raw_counts_sha256`, `smoothed_counts_sha256`, and
`probabilities_hex_sha256` hash the exact canonical JSON array plus one LF for
the corresponding evidence field. `baseline_sha256` hashes the exact canonical
JSON bytes plus one LF of the complete unigram mapping shown above. `evidence`
must be the same-ID development-completed or development-failure reference that
actually contains that mapping. All digests are lowercase SHA-256; the evidence
reference digest must equal the enclosing artifact's digest. The exact key set,
types, fixed values, hashes, and canonical reserialization are required. No
alternative count list, partial mapping, bare digest, or “mapping/hash” union is
valid.

A qualitative configuration row has exactly:

```text
sample_id: str
prompt: str
generated_token_count: int
mode: str
temperature: float | None
temperature_hex: str | None
top_k: int | None
seed: int | None
```

Nullability must match mode. A sample mapping has exactly those fields plus:

```text
prompt_sha256: str
checkpoint_role: str                   # best_validation
checkpoint_sha256: str
generated_text: str
generated_byte_count: int
generated_code_point_count: int
generated_text_sha256: str
full_text_sha256: str
local_generator_final_state_sha256: str | None
```

The tracked development result's `Qualitative samples` value is exactly a list
of six `sample-reference` mappings. Each mapping has exactly:

```text
sample_index: int                       # 0..5
sample_id: str
configuration_sha256: str
prompt_sha256: str
checkpoint_sha256: str
mode: str                               # "greedy" or "categorical"
temperature_hex: str | None
top_k: int | None
seed: int | None
generated_token_count: int              # 256
generated_byte_count: int               # positive
generated_code_point_count: int         # 256
generated_text_sha256: str
full_text_sha256: str
local_generator_final_state_sha256: str | None
evidence: evidence-reference mapping
```

`configuration_sha256` hashes the exact canonical qualitative configuration row
including its prompt, so the tracked reference identifies prompt/configuration
without copying prompt content. `prompt_sha256`, `generated_text_sha256`, and
`full_text_sha256` hash exact UTF-8 bytes. `checkpoint_sha256` is the accepted
best-validation object. `evidence` is the same-ID development-completed or
development-failure artifact containing the full sample and must match its raw
artifact digest.

The list is in exact fixed-matrix order with indices `0,1,2,3,4,5` and IDs
`romeo-greedy`, `romeo-focused`, `romeo-full`, `to-be-greedy`,
`to-be-focused`, `to-be-full`. IDs, indices, and configuration digests are
unique. Greedy references require null temperature/top-k/seed/final-generator
digest. The two focused rows require temperature hex
`0x1.999999999999ap-1`; the two full rows require
`0x1.0000000000000p+0`. Categorical references require those exact values,
their fixed top-k/seed, and a lowercase generator-state SHA-256. No missing, duplicate,
reordered, extra, partial-success promotion, or reference to a different
artifact/evaluation ID is valid. A failed development result contains exactly
the completed prefix; completed requires all six.

Accepted evidence contains exactly this ordered `limitations` list:

```text
1. "small character-level Shakespeare corpus"
2. "CPU-float32 model with maximum 256-token context"
3. "teacher-forced metrics do not measure free-running coherence"
4. "qualitative samples are fixed observations, not broad generalization evidence"
5. "rolling context discards history older than 256 tokens"
6. "no BOS, EOS, unknown, or padding token"
7. "validation guided development; sealed test remains separately gated and one-shot"
```

Completed evidence requires all four development metric/baseline fields, the
complete six samples in fixed order, null failure, and the fixed limitations.
A completed or failed artifact never changes kind or status under the same ID.

The successful fixed development operation order is repository/run/ID preflight,
durable attempt-marker publication, inference-bundle load, private governed
development-corpus load, token-document conversion, pure train/validation
window builds, Laplace-unigram fit, model validation metrics, uniform validation
metrics, Laplace-unigram validation metrics, six qualitative rows in fixed
order, governed cross-predictor comparison, complete evidence validation, then
one final evidence publication. Failure evidence records the first escaping
stage. No later computational or governed evaluation stage executes; normalized
failure or uncertainty evidence publication may still be attempted as lifecycle
handling and does not violate first-failure ordering.

### Exact normalized development failures

Development failure evidence has exactly:

```text
schema_version: int                     # 1
evaluation_id: str
kind: str                               # "phase9_development_failure"
status: str                             # "failed"
recorded_at_utc: str
started_at_utc: str
finished_at_utc: str
attempt_marker_sha256: str
authority: mapping
configuration: mapping
model_validation: metric mapping | None
uniform_validation: metric mapping | None
laplace_unigram_baseline: unigram mapping | None
laplace_unigram_validation: metric mapping | None
samples: list[sample mapping]           # completed prefix in fixed order
sealed_test_access: str                 # "none"
failure: failure mapping
limitations: list[str]                  # exact ordered seven-item list
```

The nested `failure` has exactly `stage`, `exception_type`, `invariant`, and
`safe_facts`. No extra nested key is permitted. The runner catches only known
project exceptions and normalizes them according to this exhaustive table:

| Allowed stage | Allowed exception/invariant family | Exact `safe_facts` keys and types |
|---|---|---|
| `repository_preflight` | `Phase9TypeError` / `phase9.type.argument`; `Phase9GovernanceError` / `phase9.governance.repository` or `.runtime` | `field: str` |
| `inference_load` | `Phase9GovernanceError` / `phase9.governance.repository` or `phase9.governance.runtime`; `Phase9CheckpointError` / `phase9.checkpoint.path`, `.catalog`, `.hash`, `.schema`, `.provenance`, `.cross_field`, or `.model_state` | `field: str` |
| `development_corpus` | `Phase9GovernanceError` / `phase9.governance.dataset` | `field: str`, `position: int | None` |
| `training_window_build` or `validation_window_build` | `Phase9TypeError` / `phase9.type.argument` or `.record`; `Phase9ContractError` / `phase9.contract.window` | `field: str` |
| `laplace_unigram_fit` | `Phase9TypeError` / `phase9.type.argument` or `.record`; `Phase9ContractError` / `phase9.baseline.training_only` or `.counts`; `Phase9NumericalError` / `phase9.baseline.probabilities` | `field: str` |
| `model_validation`, `uniform_validation`, or `laplace_unigram_validation` | `Phase9TypeError` / `phase9.type.argument` or `.record`; `Phase9ContractError` / `phase9.evaluation.split`, `.windows`, `.metric`, `.mutation`, `phase9.baseline.counts`, or `.probabilities`; `Phase9NumericalError` / `phase9.evaluation.metric` or `phase9.baseline.probabilities`; exact accepted MiniGPT/cross-entropy class and invariant from their accepted specifications | `field: str`, `split: str` |
| `sample_romeo_greedy`, `sample_romeo_focused`, `sample_romeo_full`, `sample_to_be_greedy`, `sample_to_be_focused`, or `sample_to_be_full` | `Phase9TypeError` / `phase9.type.argument` or `.record`; `Phase9ContractError` / `phase9.contract.bundle`, `.generation`, `phase9.generation.*`, or `phase9.sampling.*`; `Phase9NumericalError` / `phase9.sampling.*`; exact accepted tokenizer/MiniGPT class and invariant from their accepted specifications | `sample_id: str`, `field: str`, `completed_token_count: int` |
| `cross_predictor_comparison` | `Phase9ContractError` / `phase9.evaluation.metric` | `field: str`, `split: str` |
| `development_evidence_publication` | `Phase9PublicationError` / `phase9.evidence.publication` | `operation: str`, `publication_state: str` where value is `known_absent` or `unknown` |

`exception_type` is the exact class name. `invariant` must match the listed
family and the normalized stage. `field`, `operation`, `publication_state`,
`sample_id`, and `split` values come from closed contract vocabularies;
`position` is nonnegative or null; `completed_token_count` is in
`0..generated_token_count`. No original exception text or arbitrary child
`details` mapping is copied.

The exact `field` vocabulary is
`repository_root`, `evaluation_id`, `branch`, `synchronization`, `clean`,
`ancestor`, `protected_path`, `record`, `configuration`, `runtime`, `catalog`,
`reference`, `checkpoint`, `payload`, `authority`, `vocabulary`, `model`, `rng`,
`manifest`, `work_count`, `work_identity`, `path`, `bytes`, `utf8`,
`normalization`, `documents`, `vocabulary_size`, `context_length`, `stride`,
`token`, `counts`, `probabilities`, `split`, `window_sha256`, `window_count`,
`target_count`, `prompt`, `generated_token_count`, `mode`, `temperature`,
`top_k`, `seed`, `logits`, `output`, `mutation`, or `evidence`. The exact
`publication_state` uses only the four unified state strings. The exact
`operation` vocabulary is `open_parent`, `create_temporary`, `write`,
`fsync_file`, `publish_no_clobber`, `fsync_parent`, or `revalidate_identity`.
Sample IDs are exactly the six fixed IDs; split is `train` or `validation` in
development failures.

Process death or uncertain state that cannot publish evidence is represented
only by a later tracked `uncertain` result with stage `attempt_uncertain`,
exception type `ProcessInterruption`, invariant
`phase9.evidence.amendment`, and exact safe facts
`{"attempt_marker_present": bool, "final_evidence_present": bool}`. It is not
invented by the interrupted process.

## Append-only experiment records and run authority

### Canonical record grammar

Both governed runners require explicit exact built-in-string `evaluation_id`
matching `EXP-[0-9]{8}-[0-9]{2}`. They never choose an ID through file order,
recency, “latest,” or a singleton assumption.

Every Phase 9 record uses exactly one heading from the schemas below, then its
labels with no omission, duplicate, reordering, or extra label. Each field line
is exactly:

```text
- **<Label>:** <canonical-json-value>
```

Values are one-line canonical JSON: `ensure_ascii=True`, sorted object keys,
compact separators, `allow_nan=False`, lowercase `true`/`false`/`null`, and
JSON strings in double quotes. One empty line terminates a record. Record
SHA-256 is computed over the UTF-8 bytes from the heading through that single
terminating empty line. Heading ID, `Evaluation ID`, scope, kind, and status
must agree. Unknown Phase 9 headings or labels make the named ID ineligible.

Common JSON objects—`Phase 8 authority`, `Checkpoint authority`, `Phase 9
contract authority`, `Phase 9 implementation authority`, `Runtime identity`,
`Dataset authority`, `Tokenizer authority`, `Model authority`, `Evaluation
configuration`, `Qualitative matrix`, and `Evidence policy`—use the exact
nested fields already defined by this specification and are compared by exact
canonical bytes, not permissive subset matching.

Their exact roots are:

```text
Phase 8 authority:
  phase8_closure_commit, phase8_result_commit, phase8_experiment_id,
  phase8_implementation_commit, phase8_contract_commit, phase8_spec_sha256

Checkpoint authority:
  catalog_path, catalog_sha256, role, logical_id, epoch,
  validation_loss_hex, checkpoint_sha256, checkpoint_relative_path

Phase 9 contract authority:
  decision, specification_path, specification_sha256, contract_commit

Phase 9 implementation authority:
  implementation_commit, source_sha256s, test_sha256s

Attempt marker policy:
  schema_version, relative_path, maximum_bytes, no_clobber, commit_point

Evidence policy:
  schema_version, completed_relative_path, failure_relative_path,
  maximum_bytes, canonical_json,
  generated_text_storage, tracked_record_storage
```

`decision` is `DEC-0021`; paths and inherited identities equal this contract;
future accepted contract/implementation commits and source/test mappings are
literal pre-registration values. `Runtime identity`, `Dataset authority`,
`Tokenizer authority`, and `Model authority` are exact canonical copies of the
same-named mappings in the accepted completed `EXP-20260912-01` record, with no
extra sealed-work content. `Evaluation configuration` equals the development
artifact configuration mapping. `Qualitative matrix` equals its exact six-row
list. Attempt/access marker maximum size is 16,384 bytes; development evidence
maximum is 1,048,576; sealed result/failure maximum is 262,144.

### Development plan record

Heading:

```text
### <evaluation_id> — Phase 9 development plan
```

Exact labels in order:

```text
Evaluation ID
Entry kind                         # "development_plan"
Scope                              # "development"
Status                             # "planned"
Question
Predecessor evaluation             # null or another evaluation ID
Phase 8 authority
Checkpoint authority
Phase 9 contract authority
Phase 9 implementation authority
Runtime identity
Dataset authority
Tokenizer authority
Model authority
Evaluation configuration
Qualitative matrix
Attempt marker policy
Evidence policy
Sealed-test access                 # "none"
Success policy
Next gate
```

`Question` is exactly `Does the accepted Phase 9 best-validation model produce
the fixed development metrics, baseline comparisons, and complete qualitative
matrix under the accepted contract?` `Success policy` states exactly that
contracted execution/evidence completion—not score, baseline victory, or prose
quality—determines PASS.

### Development authorization record

Heading:

```text
### <evaluation_id> — Phase 9 development authorization
```

Exact labels in order:

```text
Evaluation ID
Entry kind                         # "development_authorization"
Scope                              # "development"
Status                             # "authorized_once"
Development plan record SHA-256
Development pre-registration commit
Phase 9 implementation commit
Checkpoint SHA-256
Authorization                     # fixed one-run authorization sentence
Next gate
```

The record does not contain the commit that will contain itself. Its containing
commit becomes external `development_authorization_commit` after separate
commit, push, and live-remote verification. The fixed authorization sentence is
`Authorize exactly one governed development attempt for this evaluation ID;
sealed-test access remains unauthorized.`

### Development result record

Heading:

```text
### <evaluation_id> — Phase 9 development result
```

Exact labels in order:

```text
Evaluation ID
Entry kind                         # "development_result"
Scope                              # "development"
Status                             # "completed", "failed", or "uncertain"
Recorded at UTC
Started at UTC
Finished at UTC
Predecessor evaluation
Development plan record SHA-256
Development pre-registration commit
Development authorization record SHA-256
Development authorization commit
Attempt marker                     # evidence-reference mapping, or null only for uncertain publication
Development evidence               # evidence-reference mapping or null
Model validation                   # metric mapping or null
Uniform validation                 # metric mapping or null
Laplace unigram baseline            # exact laplace-unigram-result-reference mapping or null
Laplace unigram validation          # metric mapping or null
Qualitative samples                # complete sample-reference list
Sealed-test access                 # "none"
Failure                            # normalized failure mapping or null
Observations                       # list[str]
Conclusions                        # list[str]
Limitations                        # exact ordered limitations
Next gate
```

Completed requires all metrics, six samples, completed evidence, and null
failure. Failed requires a durable marker, normalized failure, every safely
completed value/sample, and failed evidence when publication succeeds.
Uncertain requires a durable or possibly committed marker, no claim that later
work did not occur, null final evidence when absent, and conservative failure
stage `attempt_uncertain`. Times are exact second-resolution UTC strings ending
in `Z`, with monotonic ordering when known.

Status `completed` additionally requires completed evidence `known_present` and
failure evidence `known_absent`. Status `failed` requires completed evidence
`known_absent` and failure evidence either `known_present` or `known_absent`; the
latter uses a null evidence reference and retains normalized failure in this
record. Any `unknown`, malformed existing terminal path, or inability to prove
opposite absence requires `uncertain`. No valid lifecycle contains both terminal
references.

### Development supersession record

Before a successor plan may name a failed or uncertain predecessor, append:

```text
### <prior_evaluation_id> — Phase 9 development supersession
```

with exact labels:

```text
Evaluation ID
Entry kind                         # "development_supersession"
Scope                              # "development"
Status                             # "superseded"
Prior terminal status              # "failed" or "uncertain"
Prior result record SHA-256
Successor evaluation ID
Reason                             # "corrected_implementation", "corrected_contract", or "operational_replacement"
Recorded at UTC
Next gate
```

A completed ID cannot be superseded or rerun. A failed/uncertain ID can name
exactly one successor. The successor plan must point back to that ID. Cycles,
forks, missing reverse references, self-predecessors, and successors that
already have an attempt marker are ineligible.

### Sealed plan record

The sealed stage uses the same evaluation ID as its accepted completed
development result. Heading:

```text
### <evaluation_id> — Phase 9 sealed-test plan
```

Exact labels in order:

```text
Evaluation ID
Entry kind                         # "sealed_test_plan"
Scope                              # "sealed_test"
Status                             # "planned"
Accepted development result commit
Development result record SHA-256
Development evidence SHA-256
Frozen authority
Frozen evaluation configuration
Frozen qualitative matrix
Access marker policy
Sealed evidence policy
Interpretation policy
Sealed-test access                 # "pending_explicit_authorization"
Next gate
```

The frozen objects must be byte-equivalent to accepted development authority,
metrics/baselines/prompts/settings/seeds/evidence policy and the exact sealed
schemas. No test-derived value appears.

### Sealed authorization record

Heading:

```text
### <evaluation_id> — Phase 9 sealed-test authorization
```

Exact labels in order:

```text
Evaluation ID
Entry kind                         # "sealed_test_authorization"
Scope                              # "sealed_test"
Status                             # "authorized_once"
Sealed plan record SHA-256
Sealed pre-registration commit
Development evidence SHA-256
Phase 9 implementation commit
Checkpoint SHA-256
Authorization                     # fixed one-shot sentence
Next gate
```

The fixed sentence is `Authorize one durable-marker-consumed Twelfth Night
metric evaluation for this evaluation ID with no generation, tuning, or retry.`
The containing, pushed, live-verified commit is external
`sealed_authorization_commit`; it is not self-embedded.

### Sealed result record

Heading:

```text
### <evaluation_id> — Phase 9 sealed-test result
```

Exact labels in order:

```text
Evaluation ID
Entry kind                         # "sealed_test_result"
Scope                              # "sealed_test"
Status                             # "completed", "failed", or "uncertain"
Recorded at UTC
Started at UTC
Finished at UTC
Sealed plan record SHA-256
Sealed pre-registration commit
Sealed authorization record SHA-256
Sealed authorization commit
Access marker                     # evidence-reference mapping, or null only for uncertain publication
Sealed evidence                   # evidence-reference mapping or null
Model test                        # metric mapping or null
Uniform test                      # metric mapping or null
Laplace unigram test              # metric mapping or null
Sealed-test access                # "one_shot_completed", "one_shot_consumed_failed", or "one_shot_consumed_uncertain"
Failure                           # normalized sealed failure or null
Observations                      # list[str]
Conclusions                        # list[str]
Limitations                       # exact ordered limitations
Next gate
```

Sealed status `completed` requires access marker and completed evidence
`known_present`, failure evidence `known_absent`, all three test metrics, null
failure, and access state `one_shot_completed`. Status `failed` requires access
marker `known_present`, completed evidence `known_absent`, failure evidence
either `known_present` or `known_absent`, a normalized failure, and access state
`one_shot_consumed_failed`; null failure-evidence reference is permitted only in
the proven-absent case. Status `uncertain` requires an unknown marker or terminal
publication state, publishes no opposite artifact, uses only references proven
valid, records normalized `attempt_uncertain` or
`sealed_access_marker_publication`, and sets
`one_shot_consumed_uncertain`. No valid sealed lifecycle contains both terminal
references.

### Deterministic eligibility and predecessor rules

Primary selection starts only from the explicit runner argument. The parser
reads at most 1,048,576 committed `EXPERIMENT_LOG.md` bytes, validates the exact
argument grammar, and locates records by exact `(evaluation_id, entry_kind)`.
For the selected ID it requires exactly one valid development plan and exactly
one valid development authorization, exact cross-record hashes, and zero
development result, supersession owned by that ID, attempt marker, or terminal
artifact. Zero or multiple matches for any required/forbidden selected record
are deterministic `evaluation_id` governance failures. It requires live
`HEAD == origin/main == development_authorization_commit`, with the plan's
pre-registration commit and accepted Phase 9 implementation commit as
ancestors.

Predecessor validation is this exact bounded backward traversal:

1. Set `current_id` to the explicit argument, `visited` to the empty set, and
   edge count to zero.
2. Reject if `current_id` is already visited; otherwise add it.
3. Read the already uniquely selected development plan for `current_id` and its
   `Predecessor evaluation` value.
4. If the value is null, terminate successfully. That terminal plan must have
   zero incoming supersession edge.
5. Otherwise require a valid ID `predecessor_id`, reject self-reference and any
   visited ID, increment the edge count, and reject an edge count above 64.
6. Locate exactly one development plan, one development result, and one
   development supersession record for `predecessor_id`. Zero or multiple
   matches fail. The result status must be `failed` or `uncertain` and its hash
   must equal the supersession's `Prior result record SHA-256`.
7. Require the supersession's owner/`Evaluation ID` to equal `predecessor_id`,
   its `Successor evaluation ID` to equal `current_id`, and the current plan's
   predecessor to equal `predecessor_id`.
8. Perform the only global uniqueness scan: inspect heading identity plus the
   owner, successor, and prior-result-hash fields of every development
   supersession record. Require exactly one outgoing edge from
   `predecessor_id`, exactly one incoming edge to `current_id`, and no second
   record—valid or malformed—claiming either edge endpoint. This rejects forks
   and merges without validating unrelated chain bodies.
9. Set `current_id = predecessor_id` and repeat from step 2.

Traversal touches only the selected ID and the explicit predecessors reached by
these edges, except for the minimal global supersession-edge uniqueness scan in
step 8. A cycle, self-edge, fork, merge, broken reciprocal link, invalid terminal
status, missing/null mismatch, duplicate record, or chain longer than 64 edges
fails before marker/checkpoint/corpus access. Unrelated evaluation chains do not
affect eligibility unless they claim an endpoint used by this chain.

The sealed runner requires the same explicit ID; one accepted completed
development result; one valid sealed plan; one valid sealed authorization;
exact record/evidence hashes; `HEAD == origin/main ==
sealed_authorization_commit`; and no sealed result, access marker, sealed
evidence, or sealed failure evidence. It accepts no predecessor or different
evaluation ID. It performs no predecessor traversal; it validates only that
same-ID accepted development chain plus the global uniqueness of that chain's
already accepted predecessor edges.

Missing, malformed, unknown, duplicate, conflicting, already-completed,
already-superseded, marker-consumed, or otherwise ineligible IDs raise
`Phase9GovernanceError` with invariant `phase9.governance.repository` and
safe field `evaluation_id` before checkpoint/corpus/artifact access. Low-level
pure mechanics never parse `EXPERIMENT_LOG.md`.

`Next gate` is a JSON string from this exact record/status mapping:

```text
development plan          -> "independent_pre_registration_review"
development authorization -> "execute_authorized_development_attempt"
development completed     -> "independent_development_result_review"
development failed        -> "adjudicate_failure_and_plan_successor"
development uncertain     -> "adjudicate_failure_and_plan_successor"
development supersession  -> "review_successor_plan"
sealed-test plan           -> "independent_sealed_plan_review"
sealed-test authorization  -> "execute_authorized_sealed_test_once"
sealed-test completed      -> "independent_sealed_result_review"
sealed-test failed         -> "independent_sealed_result_review"
sealed-test uncertain      -> "independent_sealed_result_review"
```

The sealed plan's `Interpretation policy` is exact JSON string
`report_observation_without_threshold_or_post_test_tuning`.

All planned, authorization, result, and supersession records are append-only.
No score or unattractive sample may be amended away. Development success means
contracted execution/evidence completion, not a score threshold. After marker
publication, all retries require a new eligible ID and predecessor chain.

## Sealed-test gate

### Frozen prerequisites

Drafting, accepting, or implementing this contract does not authorize test
access. Before a one-shot authorization may be requested, all of the following
must be accepted, committed, pushed, and remotely verified or byte-frozen in an
accepted pre-registration:

1. Phase 9 contract commit and exact specification SHA-256;
2. Phase 9 implementation commit and exact source/test identities;
3. Phase 8 closure/result/experiment identities;
4. catalog SHA-256, semantic `best_validation` role, checkpoint object SHA-256,
   logical ID, epoch, and validation-loss hex;
5. tokenizer/vocabulary, MiniGPT, dataset manifests, and runtime identities;
6. context 256, stride 256, retained-tail and aggregation definitions;
7. NLL, perplexity, and top-1 definitions and numerical behavior;
8. uniform and Laplace-smoothed training-only empirical-unigram definitions
   plus the already fitted frozen counts/probabilities hash;
9. the exact six prompts, output lengths, modes, temperatures, top-k values,
   and seeds, even though no test prompt generation is performed;
10. development evidence schema, accepted development result, and artifact hash;
11. sealed access-marker and result/failure evidence schemas;
12. one-shot success/failure and no-rerun policy; and
13. interpretation policy: test scores are observations, no threshold is a
    Phase 9 pass condition, and no post-test tuning or replacement run follows.

The one-shot request must explicitly name the evaluation ID, implementation
commit, checkpoint digest, contract digest, pre-registration commit, and exact
test scope. Authorization is recorded append-only and committed/pushed before
execution. General Phase 9 implementation or evaluation authorization is
insufficient.

### Access consumption marker

`run_fixed_phase9_sealed_test_evaluation` validates its explicit
`evaluation_id` and every matching prerequisite using Git, tracked records,
development evidence, and approved manifest metadata without statting or
opening the test file. It then exclusively and durably publishes:

```text
experiments/<evaluation_id>/artifacts/phase9-sealed-test-access.json
```

with exact schema:

```text
schema_version: int                     # 1
evaluation_id: str
kind: str                               # "phase9_sealed_test_access"
sealed_plan_record_sha256: str
sealed_pre_registration_commit: str
sealed_authorization_record_sha256: str
sealed_authorization_commit: str
phase9_contract_sha256: str
phase9_implementation_commit: str
checkpoint_sha256: str
development_evidence_sha256: str
created_at_utc: str
consumed: bool                          # true
```

Before marker publication, the sealed runner proves the marker and both sealed
terminal paths absent. It then applies the unified publication state machine.
Only `known_present` consumes the authorization and permits the exact
manifest-authorized *Twelfth Night* file to be opened. `known_absent` permits no
test access and leaves the unchanged authorization eligible for another marker
attempt. `unknown` permits no test access, is conservatively and irreversibly
treated as authorization consumed, requires tracked sealed status `uncertain`,
and forbids reuse of the same authorization even if later inspection observes
no marker. Failure to prove directory durability can never be used as a reason
to access sealed content.

Existing marker, result, failure evidence, or uncertain marker state refuses all
later access. After marker `known_present`, every subsequent failure permits no
automatic retry.

The sealed operation opens the approved test file once, verifies its exact
manifest path/type/size/hash and strict text invariants over the same bytes,
tokenizes it once, constructs canonical test windows, and evaluates the already
loaded model plus both already frozen baselines. It performs no test-prompt
generation, sample selection, parameter update, baseline fitting, configuration
change, or second file open.

### Sealed evidence

Successful execution publishes immutable
`phase9-sealed-test-evidence.json`; failure after marker publication publishes
immutable `phase9-sealed-test-failure.json` when safe publication remains
possible. Both are ignored, canonically serialized, bounded to 262,144 bytes,
mutually exclusive under the unified publication state machine, and hashed into
a later append-only result record. A non-`known_present` success-publication
outcome triggers read-only inspection before any failure-artifact attempt;
`unknown` forbids the opposite artifact and yields tracked `uncertain`. The same
rule applies symmetrically if failure-evidence publication is ambiguous.

Sealed `authority` has exactly:

```text
phase8_closure_commit: str
phase8_result_commit: str
phase8_experiment_id: str
phase8_implementation_commit: str
phase8_contract_commit: str
phase8_spec_sha256: str
phase9_contract_commit: str
phase9_contract_sha256: str
phase9_implementation_commit: str
development_plan_record_sha256: str
development_pre_registration_commit: str
development_authorization_record_sha256: str
development_authorization_commit: str
development_result_record_sha256: str
development_result_commit: str
development_evidence_sha256: str
sealed_plan_record_sha256: str
sealed_pre_registration_commit: str
sealed_authorization_record_sha256: str
sealed_authorization_commit: str
catalog_sha256: str
checkpoint_role: str                    # "best_validation"
checkpoint_sha256: str
vocabulary_sha256: str
dataset_manifest_sha256: str
processing_manifest_sha256: str
requirements_lock_sha256: str
```

Every inherited/configuration value must remain byte-equivalent to the accepted
development authority. The top-level marker digest cross-validates the separate
marker bytes.

Successful sealed evidence has exactly:

```text
schema_version: int                     # 1
evaluation_id: str
kind: str                               # "phase9_sealed_test"
status: str                             # "completed"
recorded_at_utc: str
authority: mapping                      # exact sealed authority defined above
access_marker_sha256: str
development_evidence_sha256: str
model_test: metric mapping
uniform_test: metric mapping
laplace_unigram_test: metric mapping
sealed_test_access: str                 # "one_shot_completed"
failure: None
limitations: list[str]
```

Sealed failure evidence has exactly:

```text
schema_version: int                     # 1
evaluation_id: str
kind: str                               # "phase9_sealed_test_failure"
status: str                             # "failed"
recorded_at_utc: str
started_at_utc: str
finished_at_utc: str
authority: mapping                      # exact sealed authority above
access_marker_sha256: str
development_evidence_sha256: str
model_test: metric mapping | None
uniform_test: metric mapping | None
laplace_unigram_test: metric mapping | None
sealed_test_access: str                 # "one_shot_consumed_failed"
failure: failure mapping
limitations: list[str]
```

The exact allowed sealed failure stages and normalized mappings are:

| Allowed stage | Allowed exception/invariant family | Exact `safe_facts` keys and types |
|---|---|---|
| `sealed_access_marker_publication` | `Phase9PublicationError` / `phase9.evidence.publication` | `operation: str`, `publication_state: str` where value is `known_absent` or `unknown` |
| `sealed_corpus_open` or `sealed_corpus_validation` | `Phase9GovernanceError` / `phase9.governance.dataset` or `.sealed_test` | `field: str` |
| `sealed_tokenization` | accepted tokenizer error | `field: str`, `position: int | None` |
| `sealed_window_build` | `Phase9TypeError` / `phase9.type.argument` or `.record`; `Phase9ContractError` / `phase9.contract.window` | `field: str` |
| `sealed_model_evaluation`, `sealed_uniform_evaluation`, or `sealed_laplace_unigram_evaluation` | `Phase9TypeError` / `phase9.type.argument` or `.record`; `Phase9ContractError` / `phase9.evaluation.split`, `.windows`, `.metric`, `.mutation`, `phase9.baseline.counts`, or `.probabilities`; `Phase9NumericalError` / `phase9.evaluation.metric` or `phase9.baseline.probabilities`; exact accepted MiniGPT/cross-entropy class and invariant | `field: str`, `split: str` where split is `test` |
| `sealed_cross_predictor_comparison` | `Phase9ContractError` / `phase9.evaluation.metric` | `field: str`, `split: str` where split is `test` |
| `sealed_evidence_publication` | `Phase9PublicationError` / `phase9.evidence.publication` | `operation: str`, `publication_state: str` where value is `known_absent` or `unknown` |

`failure` contains exactly `stage`, exact class-name `exception_type`, matching
`invariant`, and only the stage's exact `safe_facts`. No original message,
traceback, arbitrary details, test prose, token IDs, logits, probabilities, or
excerpt is persisted. The limitations list is the same exact ordered seven-item
list required for development evidence.

The access-marker-publication row is used in tracked lifecycle reporting, not a
sealed failure artifact: `known_absent` consumes no authorization and produces
no terminal record, while `unknown` produces tracked `uncertain` with null
access/evidence references and permanently consumes the authorization. All
post-marker rows require marker `known_present`.

Abrupt termination after the marker but before failure evidence is represented
by the tracked sealed result status `uncertain`, access state
`one_shot_consumed_uncertain`, and normalized `attempt_uncertain` mapping; it
never fabricates a sealed failure artifact after the fact.

The later tracked result record stores metrics, marker/evidence hashes, and
interpretation but no test text. Any failure after the access marker is an
observed final-test event and must be reported. A later proposal to inspect or
rerun the already unsealed test requires a new explicit governance decision and
cannot be described as a first or independent test.

## Public-interface validation precedence

### Sampling helpers

Both helpers validate logits by object, rank, shape, dtype, device, and
finiteness. Categorical probability construction then validates temperature
type/value followed by top-k type/value before arithmetic. Greedy never inspects
sampling settings. No helper changes global RNG or input storage.

### Generation

Generation uses the exact order already listed: bundle type/factory, bundle
state, prompt type/empty/length, count type/range, mode type/value,
temperature/top-k/seed mode contract, tokenizer encoding, zero-count return or
model loop, output cross-checks, mutation/RNG checks. A simultaneous invalid
prompt length and categorical seed therefore reports prompt length first; an
unknown prompt character is reported only after all structural settings pass.

### Corpus and windows

The pure window builder validates document container/records, explicit
vocabulary size, context, stride, and token IDs before constructing windows; it
has no production identities or counts. The governed runner's private corpus
layer separately validates root/Git authority, manifests, exact permitted work
sequence, each file in manifest order, tokenization, production counts, and
test exclusion before passing small records into pure mechanics.

### Baselines and metrics

Pure Laplace-unigram fitting validates explicit vocabulary size and local
training-window structure before counts/smoothing/probabilities. Pure evaluators
validate their predictor/model or baseline, local windows, split, IDs, and
nonmutation invariants before arithmetic. The governed runner separately owns
production counts, accepted validation-loss replay, and the final three-record
window-digest/count equivalence comparison.

### Runners and publication

Both fixed runners validate root then explicit evaluation-ID type/grammar,
deterministic record eligibility/predecessors, repository/Git/runtime, accepted
contract and implementation authority, checkpoint/catalog, dataset authority,
fixed configuration, marker/evidence absence, and global nonmutation state.
The sealed runner additionally validates every frozen prerequisite and absence
of prior access, publishes the durable marker, and only then accesses test.

Repository/governance errors precede checkpoint errors; checkpoint validation
precedes corpus/model evaluation; all development inputs precede evidence
publication; and sealed authorization/marker rules precede sealed content.

## Exact implementation and execution-evidence obligations

All Phase 9 implementation tests use small deterministic synthetic objects
unless a test explicitly validates an accepted tracked identity or retained
checkpoint graph. No test may read *Twelfth Night*. A sealed-protection test uses
guarded suppliers/sentinel paths and proves zero supplier/path access.

Obligations 20 and 22 are authorized-development-execution checks, not ordinary
implementation-test permission to read real corpus or checkpoint content.
Obligation 34's implementation evidence is guarded and synthetic; its real-file
counterpart runs only during the separately authorized sealed event.

The later implementation must satisfy at least these exact grouped obligations:

1. Literal public signatures, annotations, required/default status, record field
   order/frozen/repr policy, exception inheritance/details, module ownership,
   and package/module exports match this contract. Every one of the six allowed
   evidence-reference kind/status/path combinations passes and every cross-row,
   unknown, wrong-type, malformed-ID/size/hash, and noncanonical combination
   fails.
2. Static import/tree checks prove no pretrained/hosted model, high-level
   generation framework, top-p, beam search, penalties, KV cache, model/training
   redesign, Phase 10 work, or unauthorized sealed path exists.
3. Loader rejects wrong root types, symlinks, non-root paths, branches, dirty or
   unsynchronized tracked state, missing ancestry, and every independent
   mutation of all 19 paths in the exhaustive protected live/closure-blob set—including
   top-level and data package initializers—before checkpoint access; actual
   transitive import closure is independently enumerated and unrelated paths are
   not falsely pinned.
4. Catalog bytes, canonical JSON, size/type/path/no-follow rules, fixed digest,
   both reference schemas, and semantic `best_validation` selection are tested,
   including a synthetic distinct-latest/distinct-best graph.
5. Object same-buffer hashing/deserialization, size bounds, `weights_only=True`,
   complete payload schema, all nested exact fields, and catalog/payload/
   accepted-result cross-validation reject independent mutations.
6. The loader constructs exactly one accepted MiniGPT, loads only best model
   state, constructs no optimizer, restores no progress/metric/lineage/RNG/mode
   state, returns all modules in eval, and leaves all gradients `None`.
7. Loader success and every injected failure preserve caller global RNG; Phase
   8 public loader signature/behavior and accepted Phase 7/8 files regress.
8. Prompt tests cover non-string, empty, one token, 255, 256, 257, unknown first
   code point, preserved exact whitespace, and no silent truncation.
9. Count tests cover `bool`, non-int, `-1`, `0`, `1`, `1024`, and `1025`, with
   zero proving no model call or generator construction.
10. Greedy settings reject every non-null sampling field; categorical settings
    test exact float/int types, nonfinite/zero/negative temperature, top-k 0/1/
    80/81/82, and seed boundaries including booleans.
11. Synthetic greedy logits prove final-row use, lowest-ID exact tie handling,
    no softmax, no RNG draw, correct built-in-int result, and input nonmutation.
12. Synthetic categorical logits independently prove temperature division,
    descending-score/ascending-ID top-k ties, exact excluded zeros, stable
    subtract-max softmax, normalization, non-aliasing, and `top_k=1`/`81`.
13. Tiny positive temperature that overflows float32 scaling and every injected
    nonfinite weight/probability path fail with the exact numerical invariant
    before sampling.
14. Categorical selection uses exactly one persistent local CPU generator and
    one multinomial draw per token; same seed/input/settings reproduce tokens,
    text, and final state digest without requiring different seeds to diverge.
15. Greedy and categorical success/failure leave global RNG bitwise unchanged;
    no call uses global, CUDA, MPS, Python, or NumPy randomness.
16. Generation call instrumentation proves one full model recomputation per new
    token, contexts of lengths up to 256, exact newest-256 rolling slices,
    position restart `0..255`, no cache, and exact final-row decoder handoff.
17. Generation output tests prove original prompt identity, exact token count,
    encode/decode round trips, full-text concatenation, no EOS stop, and no
    parameter/gradient/mode/dropout/global-RNG mutation.
18. Development corpus tests prove exact six-plus-one order, manifest-before-
    text validation, same-opened-byte facts, independent documents, and zero
    test supplier/path access for every success and failure stage.
19. Window tests prove the exact formula for lengths 2, 256, 257, 258, 512,
    and 513; retained one-token tails; no padding; exact-once transitions;
    canonical document/start order; and no cross-document transition.
20. Production development-window counts independently equal 3,099/792,699 and
    384/98,295 without reading test.
21. Synthetic model metrics independently derive target-weighted NLL via
    `math.fsum`, perplexity, first-index top-1 counts/accuracy, shapes, and
    canonical ordering; an unequal-tail example disproves mean-of-window-means.
22. Accepted validation re-evaluation from selected checkpoint equals exact loss
    hex `0x1.3f94b678f5807p+1` before accepted result publication.
23. Evaluation uses eval plus inference mode, invokes accepted cross-entropy
    once per window, and preserves parameters, objects, gradients, modes,
    dropout states, and global RNG on success and injected failure.
24. Pure uniform-baseline tests use small explicit vocabulary sizes to derive
    `math.log(V)`, `math.exp` perplexity, token-ID-0 tie prediction, and
    independence from fitting statistics; governed evidence separately checks
    production `V=81` and transition identity.
25. Pure Laplace-smoothed empirical-unigram tests prove training-target-only counting, chosen
    add-one smoothing, exact count sums, all-positive normalized probabilities,
    lowest-ID top-token ties, immutability, and rejection of non-training inputs
    without production counts.
26. Pure Laplace-smoothed empirical-unigram evaluation independently derives per-target
    `-math.log(p)`, NLL, perplexity, accuracy, and exact reuse of one frozen
    baseline on small validation and non-reserved synthetic records; separate
    tests prove every public exact-`test` request is rejected.
27. The `phase9_experiment` shared comparison tests require identical split,
    window digest, window count, and target count across all three production
    metrics and prove a mismatch prevents evidence publication.
28. The fixed qualitative matrix oracle proves exact row order, prompt bytes,
    counts, modes, float hex/settings/seeds, successful vocabulary membership,
    six calls, rolling-boundary exercise, and complete output retention.
29. Exploration cannot be promoted, accepted rows cannot be retried/omitted,
    and partial failure retains all earlier completed sample evidence.
30. Development lifecycle/artifact tests cover explicit ID validation, exact
   attempt-marker schema, exclusive concurrent no-clobber, marker-before-work
   ordering, all four publication states, fsync commit point, permanent
   consumption, abrupt interruption, tracked uncertain classification, final
   success/failure publication uncertainty, opposite-kind inspection, zero
   contradictory terminal artifacts, final evidence schemas/canonical bytes/
   size/hash, no runner reuse, and content-safe normalized failures.
31. Parsers independently cover every exact development plan/authorization/
   result/supersession and sealed plan/authorization/result heading, label,
   canonical value, hash, eligibility, exact Laplace-result reference, exact six
   sample references/order/uniqueness, zero/multiple matches, bounded explicit
   predecessor traversal, termination, reciprocal supersession, cycle,
   self-reference, fork, merge, global endpoint uniqueness, consumed-ID,
   non-circular commit, and new-ID rule; unrelated chains are not traversed and
   tracked result sample references contain no prompt/generated/sealed text
   beyond the separately pre-registered fixed prompt matrix.
32. Sealed preflight tests independently mutate every frozen prerequisite and
    prove refusal before test stat/open/supplier invocation.
33. Sealed access-marker tests prove `known_present`, `known_absent`, and
    `unknown` outcomes; durable exclusive publication before the first test
    open; no access for absent/unknown; irreversible authorization consumption
    for unknown/present; existing-marker refusal; concurrent no-clobber; later
    failure consumption; tracked uncertain state; and no automatic retry.
34. Guarded sealed tests prove exactly one approved test-file open/read, exact
    manifest verification, one tokenization/window construction, model plus two
    frozen-baseline evaluations, no generation/fitting/tuning, and no persisted
    test prose/tokens/logits.
35. Sealed success/failure artifact and append-only result tests prove exact
    schemas, marker/evidence hashes, all four publication states, success and
    failure publication uncertainty, opposite-kind read-only inspection,
    prohibition on contradictory terminal evidence, completed/failed/uncertain
    tracked statuses, one-shot consumption, safe failure facts, no amendment,
    and honest recording regardless of scores.
36. Full accepted Phase 1–8 regression suites remain passing; retained
    checkpoint/catalog/evidence bytes and all sealed-test boundaries remain
    unchanged by implementation tests.

Critical expected values must be test-local literals or independently derived,
not imported from the production constant being tested. Failure-injection tests
must prove both first-failure ownership and absence of later work.

## Phase 9 exit-criteria mapping

| Roadmap exit criterion | Contract requirement | Later acceptance evidence |
|---|---|---|
| Autoregressive generation and sampling controls are implemented and explained. | Exact rolling-context loop, final-logit selection, greedy and temperature/top-k categorical contracts, no EOS/KV cache/advanced decoding. | Accepted source review, focused obligations 8–17, and educational walkthrough of one generation trace. |
| Deterministic and stochastic generation paths are tested. | Lowest-ID greedy ties, no RNG; dedicated seeded CPU categorical generator, same-seed reproduction, global isolation. | Focused obligations 10–17 plus full regressions and independent implementation review. |
| Quantitative metrics and qualitative samples are recorded. | Equivalent-window model/uniform/Laplace-smoothed empirical-unigram metrics, fixed six-sample matrix, canonical evidence and append-only records. | Accepted development artifact/result and, only if separately authorized, sealed artifact/result; obligations 18–31 and 32–35 as applicable. |
| Limitations, failure modes, and comparisons to simple baselines are documented. | Uniform and Laplace-smoothed empirical-unigram comparisons, failure artifacts, fixed limitations, evidence-limited qualitative interpretation, no threshold/cherry-picking. | Independent result review confirms complete metrics, failures, limitations, observations versus interpretation, and modest claims. |

The sealed test is not intrinsically required to prove implementation
correctness. If Sebastien chooses not to authorize it, Phase 9 may close only
with an explicit decision that accepted validation evidence satisfies the
roadmap criterion while the final reserved test remains unused. If authorized,
its one-shot result must be reviewed before Phase 9 exit acceptance.

## Proposed consolidated Phase 9 workflow

The workflow uses meaningful approval boundaries without reproducing Phase 8's
microscopic gate count:

1. **Contract draft:** create this documentation-only proposal and reconcile
   current continuity. No implementation or execution.
2. **Contract review:** Master Chat sanity review plus fresh independent Codex
   review of DEC-0021 fidelity, inherited authority, mechanics, tests, evidence,
   and sealed boundary.
3. **Correction and acceptance:** authorize only required documentation fixes,
   obtain focused independent re-review, then explicitly accept the exact
   specification bytes.
4. **Contract publication:** separately authorize the exact documentation
   commit; commit; separately authorize push; push; independently verify remote
   identity and cleanliness.
5. **Implementation authorization and work:** explicitly authorize the accepted
   source/test scope; implement without execution against real checkpoint or
   corpus; run synthetic, permitted-identity, and regression tests.
6. **Implementation review and acceptance:** fresh independent review,
   authorized corrections and focused re-review if needed, then explicit
   acceptance of exact source/test identities.
7. **Implementation publication:** separately authorize commit and push, perform
   them, and independently verify remote authority and clean state.
8. **Development pre-registration:** append, independently review, accept,
   commit, push, and verify the exact evaluation plan and fixed sample matrix.
9. **Development execution and result:** separately authorize one fixed run;
   execute validation metrics/baselines/samples; independently review all
   evidence and claims; accept and publish the append-only result.
10. **Optional sealed event:** freeze and verify every prerequisite, append and
    publish its plan, obtain explicit one-shot authorization, execute once,
    independently review, and publish the result without tuning or rerun.
11. **Exit and closure:** independently map evidence to all four roadmap exit
    criteria, obtain Sebastien's technical acceptance, reconcile continuity,
    then separately authorize/perform/verify the closure commit and push before
    Phase 10 learning/design authorization.

Implementation, commit, push, accepted generation, evaluation, and sealed-test
actions remain distinct authorizations even when grouped into these eleven
workflow stages.

## Explicit exclusions and current boundary

This contract introduces no model training, optimizer, scheduler, warmup,
gradient update, checkpoint continuation, architecture/tokenizer/dataset
redesign, special token, padding, batch-shaped MiniGPT call, GPU/MPS path, mixed
precision, KV cache, top-p, beam search, penalty decoder, web/API model,
pretrained model, deployment, conversational product, or Phase 10
interpretability work.

The specification itself does not access checkpoint bytes, construct MiniGPT,
read processed train/validation/test prose, tokenize a prompt, generate text,
fit a baseline, calculate a metric, write an experiment record, create an
artifact, or authorize *Twelfth Night*. Those remain future gated actions.
