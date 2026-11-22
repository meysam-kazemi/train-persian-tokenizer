# Persian Tokenizer Lab

A reproducible project for collecting Persian text from Hugging Face Datasets, normalizing and filtering it, training a byte-level BPE tokenizer, and evaluating tokenization efficiency.

> This repository trains a tokenizer. It does **not** automatically adapt Qwen or train a language model. A tokenizer's vocabulary IDs must be aligned with a model's embedding and output layers before the tokenizer can be used with that model.

## Features

- Streams datasets from the Hugging Face Hub (no need to prepare local `.txt` files).
- Supports multiple sources with per-source document limits.
- Conservative Persian normalization (Arabic Yeh/Kaf normalization; preserves ZWNJ).
- Basic quality filtering and exact-document deduplication.
- Saves a reusable text corpus so tokenizer training can be repeated offline.
- Trains byte-level BPE using Hugging Face `tokenizers`.
- Saves a `PreTrainedTokenizerFast` directory.
- Produces JSON dataset statistics and tokenizer evaluation.
- Optional token-count comparison with a Qwen tokenizer.

## Project structure

```text
persian-tokenizer-lab/
├── configs/
│   └── default.yaml
├── src/
│   └── train.py
├── data/                  # generated locally; do not commit
├── artifacts/             # generated tokenizer; do not commit by default
├── logs/                  # generated logs
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

- Python 3.10 or newer
- Internet access to Hugging Face Hub
- Enough disk space for the selected datasets and processed corpus
- A GPU is not required: tokenizer training runs on CPU

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/persian-tokenizer-lab.git
cd persian-tokenizer-lab

python -m venv .venv
```

Activate the environment:

**Linux/macOS**
```bash
source .venv/bin/activate
```

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train

From the repository root:

```bash
python src/train.py
```

The first run streams the configured Hugging Face datasets, filters and deduplicates documents, and writes:

```text
data/processed/persian_corpus.txt
data/processed/dataset_statistics.json
```

It then trains the tokenizer and writes:

```text
artifacts/persian-bpe-32k/
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
└── evaluation.json
```

The exact files can vary slightly by Transformers version.

### Rebuild the corpus

If you change dataset settings or normalization, rebuild the corpus:

```bash
python src/train.py --rebuild-corpus
```

### Compare against Qwen

```bash
python src/train.py --compare-qwen
```

This downloads only Qwen's tokenizer, not its model weights. The comparison measures token counts; it does not measure language-model quality.

## Configure datasets and tokenizer

Edit `configs/default.yaml`.

For each dataset source:

- `name`: Hugging Face dataset repository ID.
- `configs`: dataset configuration names to try, in order. Use `[null]` for a dataset's default configuration.
- `split`: usually `train`.
- `text_fields`: preferred text columns. The loader falls back to any string column if these are absent.
- `max_documents`: maximum rows to inspect for that source.
- `enabled`: set to `false` to skip it.

The project logs dataset load failures and continues to the next source. A dataset can still fail if it is gated, removed, requires authentication, has no compatible streaming support, or has changed its schema.

### Dataset sources configured by default

| Source | Role | Notes |
|---|---|---|
| `wikimedia/wikipedia` | Encyclopedia | Config `20231101.fa`; formal Persian |
| `oscar-corpus/OSCAR-2301` | Web crawl | Persian deduplicated config |
| `uonlp/CulturaX` | Web crawl | Persian config `fa`; verify current Hub availability |
| `allenai/mc4` | Web crawl | Persian config `fa`; verify current Hub availability |
| `SLPL/naab` | Persian corpus | Very large; capped by document count |
| `ali619/corpus-dataset-normalized-for-persian-farsi` | Blog/web text | Normalized Persian collection |
| `mshojaei77/PersianCorpus_merged` | Aggregated Persian corpus | May overlap with other sources |
| `taesiri/TinyStories-Farsi` | Short stories | Adds story-like text |

The list is intentionally configurable. Some corpora aggregate other public datasets, so overlap can remain even after exact duplicate removal. The current script removes exact normalized-document duplicates, not near-duplicates.

## Important data and licensing note

Dataset availability, configuration names, licenses, and terms can change. Review the dataset card and license of **every source** before using its content.

This project's source code license (if you add one) does not grant rights to the underlying datasets. Do not redistribute the downloaded corpus unless the source licenses permit it. Some datasets have attribution, non-commercial, or other restrictions.

## Persian normalization

The current normalizer:

- applies Unicode NFC;
- maps Arabic Yeh (`ي`) to Persian Yeh (`ی`);
- maps Arabic Kaf (`ك`) to Persian Kaf (`ک`);
- removes tatweel;
- normalizes whitespace;
- preserves Persian ZWNJ (`‌`) and punctuation.

Normalization is a modeling decision. Keep the same normalization policy for tokenizer training, evaluation, and later model training.

## Tokenizer evaluation

The script reports:

- vocabulary size;
- token IDs and token strings for sample text;
- token count per example;
- approximate whitespace word count;
- decode round-trip result.

Fewer tokens per sentence are not automatically better. Evaluate coverage, readability, round-trip behavior, and downstream language-model quality too.

## Important: using this tokenizer with Qwen

Do not simply load this tokenizer with an existing Qwen checkpoint.

Qwen's embedding matrix and language-model output head were trained for Qwen's original token IDs. A completely new tokenizer assigns different meanings to IDs. Calling `resize_token_embeddings()` only changes matrix size; it does not transfer semantic knowledge to the new vocabulary.

There are two separate research paths:

1. **Vocabulary extension:** retain all original Qwen token IDs and append new tokens. Initialize and train the new embedding/output rows, then continue pretraining.
2. **Full tokenizer replacement:** train or initialize a new vocabulary, rebuild/initialize the embedding and output layers, and perform substantial continued pretraining before instruction tuning.

The tokenizer produced here is a standalone tokenizer for experimentation. It is not a drop-in replacement for Qwen.

## Reproducibility

- Set the seed in `configs/default.yaml`.
- Keep the generated `dataset_statistics.json`.
- Record the dataset revisions and licenses used in each experiment.
- Keep a copy of the exact YAML config with each trained tokenizer.
- For serious experiments, pin dataset revisions/commits rather than relying on a moving `main` branch.

## Troubleshooting

### A dataset cannot be loaded

Open `logs/training.log`. Check the dataset card and update its `configs` or `text_fields` in YAML. Some older dataset repositories use loading scripts that recent versions of `datasets` no longer execute.

### No documents were accepted

Lower `min_characters`, `min_persian_characters`, or `min_persian_ratio` cautiously. Check whether the dataset's text column is named differently.

### Not enough disk space

Reduce each source's `max_documents`, reduce `max_total_documents`, or disable large sources. Streaming avoids downloading a full dataset archive up front, but the project writes the accepted corpus locally.

### Hugging Face authentication

For gated datasets, log in with:

```bash
hf auth login
```

Only use datasets you are authorized to access.

## Citation and attribution

Cite each dataset according to its own dataset card when publishing research or derived work. Also cite the Hugging Face `datasets` and `tokenizers` libraries as appropriate.

## License

Choose a license for this repository's code separately from the licenses and terms of the datasets. Dataset content is not bundled with this repository.
