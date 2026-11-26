# Data preparation

This directory contains the maintained raw-data pipeline for KuaiSAR, JDsearch,
and Qilin. It is independent of the model package and writes one common session
schema directly; there is no intermediate session format.

Run commands from the repository root:

```bash
python -m data_proc self-check
python -m data_proc --help
```

## Canonical outputs

Every JSONL object contains exactly these fields:

```text
schema_version, record_id, user_id, dataset, chunk_index,
item_token_count, query_slot_count, semantic_ids, items_per_session,
query_token_ids, query_lengths, interaction_labels, session_attributes,
item_categories, visibility_codes, query_scatter_positions,
item_scatter_positions, semantic_target_positions, item_identifiers
```

`item_identifiers` is always present and never null. Semantic ID files must have
this exact header:

```text
item_identifier,semantic_id_1,semantic_id_2,semantic_id_3
```

The CSV values are zero-based RQ cluster indices. The JSONL writer converts them
to the model's one-based, level-offset ranges. Missing item mappings become
`[0, 0, 0]` while retaining the source item identifier.

Each query reserves 12 dense coordinates (start, ten term slots, end), and each
item reserves four (marker and three SIDs). `semantic_target_positions` points to
the actual three SID coordinates after the item marker. Visibility metadata packs
the one-based session ordinal into the high 32 bits and the item ordinal into the
low 32 bits; item zero denotes the query. This is collision-free for every record
the token limit can represent.

The writer splits a request larger than the record token budget into safe pieces
and repeats its query in each piece. It then chunks chronologically, applies
`--minimum-sessions` (default 2), and only then assigns continuous `record_id`
values. `chunk_index` remains the chronological chunk number for that user.

## Raw to semantic IDs

The stages are explicit so frozen artifacts can be reused independently:

1. Extract item text.
2. Optionally train integer-term BPE for KuaiSAR or JDsearch.
3. Embed item text with the production GME backend.
4. Train three-level residual quantization.
5. Build canonical session JSONL.

Example:

```bash
python -m data_proc train-bpe \
  --dataset kuaisar \
  --source raw/kuaisar/src_inter.csv \
  --item-source raw/kuaisar/item_features.csv \
  --output artifacts/kuaisar/integer_bpe.json

python -m data_proc items \
  --dataset kuaisar \
  --source raw/kuaisar/item_features.csv \
  --bpe-model artifacts/kuaisar/integer_bpe.json \
  --output artifacts/kuaisar/items.csv

python -m data_proc embed \
  --input artifacts/kuaisar/items.csv \
  --embeddings-output artifacts/kuaisar/embeddings.npy \
  --identifiers-output artifacts/kuaisar/item_identifiers.npy \
  --backend gme

python -m data_proc rq \
  --embeddings artifacts/kuaisar/embeddings.npy \
  --identifiers artifacts/kuaisar/item_identifiers.npy \
  --output artifacts/kuaisar/semantic_ids.csv \
  --codebooks-output artifacts/kuaisar/rq_codebooks.npz
```

`--backend` is mandatory. `gme` never falls back when CUDA, the model, or an
optional dependency is missing. `hash-smoke` is a deterministic CPU fixture
backend and must not be used for paper results. Qilin multimodal embedding uses
`embed --modality with_image --image-root ...`; behavioral `--scope` and
`--sid-modality` are deliberately separate controls.

## Session adapters

KuaiSAR preserves the real integer user ID and groups search sessions in timestamp
order. The five source tables and their fields are documented in
[`kuaisar/README.md`](kuaisar/README.md):

```bash
python -m data_proc sessions kuaisar \
  --source raw/kuaisar/src_inter.csv \
  --sid-file artifacts/kuaisar/semantic_ids.csv \
  --bpe-model artifacts/kuaisar/integer_bpe.json \
  --output processed/kuaisar/session_sequences.jsonl
```

JDsearch has one user per raw behavior TSV row. The one-based physical data-row
number is its stable `user_id`; all chunks from the row keep that ID and receive
successive `chunk_index` values. The original behavior and product-metadata
formats are documented in [`jdsearch/README.md`](jdsearch/README.md):

```bash
python -m data_proc sessions jdsearch \
  --source raw/jdsearch/user_behavior_data.txt \
  --sid-file artifacts/jdsearch/semantic_ids.csv \
  --bpe-model artifacts/jdsearch/integer_bpe.json \
  --output processed/jdsearch/session_sequences.jsonl
```

Qilin `all_requests` scope requires and combines search, DQA, and recommendation inputs.
Rows are aggregated by the real user identity, mapped to a deterministic
non-negative integer, and sorted across sources. Recommendation ordering uses
`request_timestamp` from the first result detail, with the row-level value only as
a fallback. The optional user map records the reversible input identity mapping
used for an output run. The source-table fields and nested result structures are
documented in [`qilin/README.md`](qilin/README.md).

```bash
python -m data_proc sessions qilin \
  --scope all_requests \
  --search 'raw/qilin/search/**/*.parquet' \
  --dqa 'raw/qilin/dqa/**/*.parquet' \
  --recommendation 'raw/qilin/recommendation/**/*.parquet' \
  --sid-modality text \
  --sid-file artifacts/qilin/semantic_ids.csv \
  --query-backend huggingface \
  --tokenizer models/query-tokenizer \
  --user-map processed/qilin/users.csv \
  --output processed/qilin/session_sequences.jsonl
```

Use `--sid-modality with_image` with SIDs produced from multimodal embeddings;
this does not silently change the behavioral scope. CSV and JSONL inputs use the
standard library. Parquet requires `pandas` and `pyarrow`; Hugging Face query
tokenization and GME embedding require `transformers`, and image embedding also
requires Pillow.

All readers detect Git-LFS pointer stubs and fail with an actionable error instead
of parsing them as data.
