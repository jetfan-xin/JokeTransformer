<p align="center">
  <img src="https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExNjYwdnJldTl6MzRocTV3b3d1bXlud2k3emM3OG9lZmtwZTI0amNkeiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/wKSpqpmltIyPsIA1IM/giphy.gif" alt="jokes" width="200"/>
</p>

# Efficient Joke Transformer (Jana Branch)

This branch contains an experimental pipeline for **efficient joke generation with small Transformer models**, designed to run on a laptop / single GPU and trained **from scratch** on curated joke datasets.

The focus of the Jana branch is on:

- A fully scripted data pipeline (collection, cleaning, topic extraction).
- Training and comparing different **tokenizers** (BPE vs Unigram, various vocab sizes).
- Decoder-only Transformer architectures, including an **ALiBi** variant.
- A detailed **evaluation suite** (topic recall, GPT‑2 perplexity, BLEU, semantic similarity, and diversity).

This README documents *what is currently implemented in the Jana branch* and how to reproduce the full workflow from raw data to trained and evaluated models.

---

## 0. What Has Been Done So Far

At the time of this README, the Jana branch has completed the following:

1. **End‑to‑end experiment pipeline**
   - Implemented a full pipeline covering:
     - Data collection and cleaning (`data/combine_datasets.py`, `data/clean.py`, `data/add_topics.py`).
     - Tokenizer training (`models/bpe_tokenizer.py`, `models/unigram_tokenizer.py`).
     - Decoder‑only model training (`models/train.py`).
     - Automatic evaluation (`eval/run_eval.py`, `eval/metrics.py`).

2. **Three main model runs (different tokenizers)**
   - **Unigram 30k tokenizer model**
     - Trained for ~50k optimization steps (checkpoint at iteration 49,500) using a 30k‑vocab Unigram tokenizer.
     - Checkpoint: `models/chekpoints/checkpoint_49500.pt` (legacy path).
   - **BPE 4k tokenizer model**
     - Trained for 3k steps with a 4k‑vocab BPE tokenizer.
     - Run directory (config, loss curves, etc.): `models/checkpoints/bpe4k_h6_e384_L6_nopad_run1/`.
   - **BPE 8k tokenizer model**
     - Trained for 3k steps with an 8k‑vocab BPE tokenizer.
     - Run directory: `models/checkpoints/bpe8k_h6_e384_L6_nopad_run1/`.
   - Every training run saves a `config.json`, `loss_history.csv`, and loss curve plots under `models/checkpoints/*`, so all experimental settings are reproducible.

3. **Evaluation of all three models**
   - All three models have been evaluated on the same topic‑based evaluation setup using `eval/run_eval.py`.
   - Aggregated summaries are stored under `eval/eval_outputs/`, for example:
     - `summary_decoder_50k.json`      — Unigram 30k model (legacy run, named “decoder_50k”).
     - `summary_decoder_bpe_4k_3ksteps.json` — BPE 4k model.
     - `summary_decoder_bpe_8k_3ksteps.json` — BPE 8k model.
   - These files contain mean/median metrics, diversity scores, and per‑topic‑count breakdowns.

4. **ALiBi implementation**
   - Implemented an ALiBi‑based decoder‑only model in `models/decoder_alibi.py`.
   - This variant is ready to be wired into a training script (by adapting `models/train.py`) for direct comparison against the standard positional‑embedding model.

---

## 1. Repository Overview (Jana Branch)

Only the most relevant parts for the Jana branch are listed here:

```text
uhh-ias-ml/
├── data/
│   ├── raw/                 # Raw datasets (placed locally, not committed)
│   ├── raw_combined/        # Output of combine_datasets.py
│   ├── combined_data/       # Intermediate cleaned data from clean.py
│   └── processed/           # Final data and tokenizers for training/eval
│
├── models/
│   ├── decoder_only.py      # Standard GPT-style decoder-only Transformer
│   ├── decoder_alibi.py     # Decoder-only variant using ALiBi positional bias
│   ├── encoder_decoder.py   # Early encoder–decoder architecture (not main focus)
│   ├── bpe_tokenizer.py     # Script to train a BPE tokenizer (default 4k vocab)
│   ├── unigram_tokenizer.py # Script to train a Unigram tokenizer
│   ├── train.py             # New decoder-only training script (streaming LM)
│   ├── checkpoints/         # Experiment checkpoints, loss curves, configs
│   └── chekpoints/          # Legacy directory with a 30k-vocab checkpoint
│
├── eval/
│   ├── metrics.py           # Topic recall, GPT‑2 ppl, BLEU, similarity, diversity
│   ├── eval_prompts.py      # Build topic-based evaluation prompts
│   ├── run_eval.py          # Load checkpoint, generate jokes, compute metrics
│   └── eval_outputs/        # CSV and JSON summaries for trained models
│
├── utils/                   # Early/simple pipeline utilities
│   ├── preprocessing.py
│   ├── train_utils.py
│   ├── evaluation.py
│   ├── inference.py
│   └── config.py, tokenizer.py
│
├── tests/
│   ├── test_data_loading.py
│   ├── test_generation.py
│   ├── test_generate.py
│   └── test_evaluation.py
│
├── main.py                  # Early CLI demo (decoder / enc-dec, simple pipeline)
├── requirements.txt
└── README.md                # This file
```

The **Jana pipeline** mainly uses `data/`, `models/train.py`, and `eval/`.  
The `main.py` + `utils/` path is an older minimal example and not the recommended entry point for experiments on this branch.

---

## 2. Data Pipeline (from raw to `final_clean_jokes.csv`)

All data processing scripts live in `data/`. The pipeline is fully scripted and designed to be reproducible.

### 2.1 Combining Raw Datasets — `data/combine_datasets.py`

This script collects multiple joke / humor sources into a single CSV:

- Kaggle short jokes.
- Humor detection datasets (positive / humorous samples).
- rJokesData (`train.tsv` / `train.tsv.gz`).
- `shuttie/dadjokes`.
- `Amirkid/jokes` (via local CSV or HuggingFace `datasets`).

The result is:

- `data/raw_combined/combined_raw.csv` with a single column:
  - `raw_text`: one joke / humorous text per row.

Run:

```bash
cd uhh-ias-ml
python data/combine_datasets.py
```

> See: `data/combine_datasets.py`

### 2.2 Cleaning and Deduplication — `data/clean.py`

`clean.py` performs aggressive normalization and filtering on `raw_text`, aiming to get short, reasonably clean jokes suitable for tokenization and LM training.

Key actions:

- Remove HTML / Markdown / URLs / e‑mails / code fences.
- Normalize unicode, quotes, dashes, emojis, and repeated punctuation.
- Enforce length constraints:
  - Drop extremely short or very long entries.
  - Optionally drop multi-paragraph stories.
- Deduplicate using normalized text.
- Further tokenize-oriented cleaning (lowercasing, handling mentions, punctuation spacing).
- Drop strings that are “mostly symbols” (ASCII art, Morse, etc).

Outputs (in `data/combined_data/`):

- `clean_jokes.csv` with columns:
  - `joke` — normalized text.
  - `joke_cleaned` — further cleaned, tokenization-ready text.
  - `word_count` — token count for `joke_cleaned`.
  - `stable_id` — stable hash for joining and dedup tracking.

Run:

```bash
python data/clean.py --min_len 10
```

> See: `data/clean.py`

### 2.3 Topic Extraction with spaCy — `data/add_topics.py`

To evaluate topic conditioning, we extract a small list of topic words per joke.

`add_topics.py`:

- Loads `combined_data/clean_jokes.csv`.
- Uses:
  - `joke_cleaned` if available, otherwise `joke`.
- Applies a spaCy English model (`en_core_web_sm`):
  - Extracts nouns / proper nouns, with heuristics to filter:
    - stopwords, numbers, non-alpha tokens;
    - laugh patterns (`haha`, `loooool`, etc) and nonsense strings;
    - words without vowels, very short or malformed tokens.
  - Falls back to verbs/adjectives when no nouns are usable.
- Builds a comma-separated `topic` field (up to 3 terms per joke).

Outputs (in `data/processed/`):

- `final_clean_jokes.csv`
- `final_clean_jokes.parquet` (if supported)

Required before running:

```bash
python -m spacy download en_core_web_sm
python data/add_topics.py
```

> See: `data/add_topics.py`

### 2.4 Processed Artifacts in the Repo

The Jana branch already contains several large processed files that you can reuse directly:

- `data/processed/final_clean_jokes.csv` — main training CSV (`joke_cleaned`, `topic`, etc.).
- `data/processed/train.csv`, `val.csv`, `test.csv` — earlier split versions.
- `data/processed/tokenizer.json` — a 30k Unigram tokenizer (see below).
- `data/processed/train_embeddings.npz` — sentence embeddings for all training jokes (used by evaluation scripts).

---

## 3. Tokenizers (BPE and Unigram)

The branch uses HuggingFace `tokenizers` for fast subword tokenization. Two training scripts are provided.

### 3.1 BPE Tokenizer (default 4k vocab) — `models/bpe_tokenizer.py`

Configuration (in the script):

- Input CSV: `data/processed/final_clean_jokes.csv`
  - Column: `TEXT_COL = "joke_cleaned"`.
- Model: `BPE(unk_token="[UNK]")`.
- Vocabulary size: `VOCAB_SIZE = 4_000`.
- Special tokens:
  - `[S]`, `[/S]`, `[PAD]`, `[UNK]`, `[MASK]`, `[USER]`, `[JOKE]`.
- Pre-tokenizer: `ByteLevel`.
- Decoder: `ByteLevel`.
- Post-processing template:
  - Single: `[S] $A [/S]`
  - Pair:   `[S] $A [JOKE] $B [/S]`
- Truncation and padding:
  - `max_length = 256`, right truncation + padding to `[PAD]`.

Default output path:

- `data/processed/tokenizer_bpe_4k.json`

Run:

```bash
python -m models.bpe_tokenizer
```

### 3.2 Unigram Tokenizer — `models/unigram_tokenizer.py`

Configuration:

- Input CSV: `data/processed/final_clean_jokes.csv`
- Model: `Unigram`.
- Vocabulary size: `VOCAB_SIZE = 30_000`.
- Special tokens and templates analogous to the BPE tokenizer.
- Uses ByteLevel pre-tokenizer and decoder.

Default output path:

- `data/processed/tokenizer_unigram.json`

In addition, the repo already has:

- `data/processed/tokenizer.json`  
  A 30k-vocab Unigram tokenizer that can be used directly in experiments.

---

## 4. Decoder-Only Models and Training

### 4.1 Standard Decoder-Only Transformer — `models/decoder_only.py`

`TransformerDecoder` implements a GPT-style decoder-only model:

- Components:
  - Token embedding: `nn.Embedding(vocab_size, emb_dim)`
  - Learned position embedding: `nn.Embedding(context_size, emb_dim)`
  - A stack of `Block`s (default 6 layers), each with:
    - Multi-head self-attention with causal masking.
    - Feed-forward network with 4× hidden size.
    - LayerNorm and residual connections.
  - Final LayerNorm and linear head to vocab logits.
- Forward:
  - Input: `idx` of shape `(B, T)` with token IDs (`T <= context_size`).
  - Optional `targets` for training (teacher forcing).
  - Optional `attn_mask` for padding masks.
  - Returns `(logits, loss)` where `loss` is cross-entropy over flattened tokens.
- Generation:
  - `generate(idx, max_new_tokens)` implements autoregressive decoding:
    - Repeatedly crops the context to `context_size`.
    - Samples the next token from the softmax over logits.

Default hyperparameters (as used by `models/train.py`):

- `emb_dim = 384`
- `context_size = 256`
- `num_heads = 6`
- `num_layers = 6`
- `dropout = 0.2`

### 4.2 ALiBi Variant — `models/decoder_alibi.py`

`decoder_alibi.py` defines a structurally similar decoder-only model, but **without explicit position embeddings**. Instead, it uses **ALiBi (Attention with Linear Biases)**:

- For each head, a slope is computed and a bias matrix is added to attention logits:
  - Encourages attending to recent tokens more strongly.
  - Can generalize better across sequence lengths.
- The rest of the architecture (blocks, feed-forward, residuals) mirrors the standard decoder.

At the moment there is no dedicated `train_alibi.py`, but you can:

- Copy `models/train.py` and import `TransformerDecoder` from `decoder_alibi.py` instead of `decoder_only.py`.
- Keep the same hyperparameters to isolate the effect of positional encoding.

### 4.3 Training Script (Streaming LM) — `models/train.py`

`models/train.py` is the main training script for the Jana branch. It implements an efficient **streaming language model** training loop without per-sequence padding.

Key steps:

1. **Configuration (top of the file)**  
   Example (current defaults):

   ```python
   TOKENIZER_PATH = ROOT / "../data/processed/tokenizer_bpe_4k.json"
   CLEAN_CSV_PATH = ROOT / "../data/processed/final_clean_jokes.csv"
   TEXT_COL       = "joke_cleaned"

   CHECKPOINT_ROOT = ROOT / "../models/checkpoints"
   RUN_NAME        = "bpe4k_h6_e384_L6_nopad_run1"

   VOCAB_SIZE   = 4_000
   EMB_DIM      = 384
   CONTEXT_SIZE = 256
   NUM_HEADS    = 6
   NUM_LAYERS   = 6
   DROPOUT      = 0.2

   BATCH_SIZE   = 32
   MAX_ITERS    = 3000
   EVAL_INTERVAL = 200
   EVAL_ITERS   = 20
   LEARNING_RATE = 3e-4
   ```

2. **Tokenizer and data loading**
   - Loads tokenizer from `TOKENIZER_PATH`.
   - Reads `CLEAN_CSV_PATH` and uses `TEXT_COL` as the text column.

3. **Packed token stream construction**
   - Tokenizes every joke and concatenates all token IDs into a single 1D tensor.
   - Splits into `train_ids` and `val_ids` (e.g., 90% / 10%).

4. **Batch sampling**
   - `get_batch` samples random starting positions in the token stream.
   - Builds input `x` windows of length `CONTEXT_SIZE`.
   - Builds targets `y` as the next-token shifted version.

5. **Training loop**
   - For each step:
     - Periodically (`EVAL_INTERVAL`) calls `estimate_loss` on train/val:
       - Averages multiple mini-batches to reduce variance.
     - Logs and stores:
       - `step`, `train_loss`, `val_loss`.
     - Saves a checkpoint:
       - `checkpoint_<step>.pt` with:
         - `model_state_dict`, `optimizer_state_dict`, `iteration`, and a `config` dict.
     - Performs a single optimization step with `AdamW`.

6. **Run metadata and plots**
   - `save_run_config` writes `config.json` into `models/checkpoints/<RUN_NAME>/`.
   - `save_loss_history_and_plot` writes:
     - `loss_history.csv` (step, train_loss, val_loss).
     - `loss_curve*.png` for quick visual inspection.

7. **Resume training**
   - `RESUME_FROM_STEP` can be set to a checkpoint step (e.g., `2000`) to resume training from `checkpoint_002000.pt`.

Run:

```bash
cd uhh-ias-ml
python -m models.train
```

#### Existing Training Runs

The repo already includes results for at least two runs:

- `models/checkpoints/bpe4k_h6_e384_L6_nopad_run1/`
- `models/checkpoints/bpe8k_h6_e384_L6_nopad_run1/`

Each directory contains:

- `config.json` — complete configuration used.
- `loss_history.csv` — recorded train/val losses.
- `loss_curve*.png` — training and validation loss curves.
- (Checkpoints themselves may or may not be committed depending on `.gitignore`.)

### 4.4 Legacy 30k-Vocab Checkpoint

There is a legacy decoder-only checkpoint with a larger vocabulary:

- Path: `models/chekpoints/checkpoint_49500.pt`  
  (note the misspelling `chekpoints`).

Its evaluation summary lives in:

- `eval/eval_outputs/summary_decoder_50k.json`

Some scripts (e.g. `tests/test_generate.py`) refer to `../models/checkpoints/checkpoint_49500.pt`; adjust these paths to match your local placement of the file.

---

## 5. Evaluation (`eval/`)

### 5.1 Metrics — `eval/metrics.py`

The evaluation module provides a rich set of metrics tailored to **topic-conditioned joke generation**.

**Topic relevance**

- `topic_recall(generated_joke, requested_topics)`  
  - Uses spaCy POS tagging and lemmatization to extract content nouns from the generated joke.
  - Returns:
    - Hard recall: fraction of requested topics present as exact lemma matches.
    - Full-hit flag: 1 if all topics are covered, else 0.

- `topic_soft_recall(generated_joke, requested_topics, sim_threshold=0.4)`  
  - Uses spaCy vectors (e.g. `en_core_web_md`) for soft matching.
  - A topic is considered covered if some noun in the joke has cosine similarity ≥ threshold.

**Fluency**

- `gpt2_perplexity(text)`  
  - Loads GPT‑2 via HuggingFace `transformers`.
  - Computes perplexity on the text; lower is better.

**Copying / memorization**

- `max_bleu_to_training(generated_joke, training_jokes, max_refs=None)`  
  - Computes the maximum BLEU‑4 score (using `nltk`) between the generated joke and a subset of training jokes.
- `is_copied_from_training(max_bleu, threshold=0.8)`  
  - Returns 1 if `max_bleu` ≥ threshold.

**Semantic similarity**

- `encode_sentences(texts)`  
  - Uses `sentence-transformers/all-MiniLM-L6-v2` to produce normalized embeddings.
- `max_embedding_similarity_to_training(generated_joke, train_embeddings)`  
  - Encodes the generated joke and computes maximum cosine similarity against all training embeddings.
- `is_semantic_duplicate(max_sim, threshold=0.9)`  
  - Flags near-duplicates at the semantic level.

**Diversity**

- `distinct_n(texts, n)` and `diversity_metrics(texts)`  
  - Distinct‑1 and Distinct‑2 metrics over an entire set of generated jokes.

Dependencies:

- `transformers` (for GPT‑2).
- `spacy` + model `en_core_web_md` (vector-based similarity).
- `sentence-transformers`.
- `nltk` (for BLEU).

Before first use:

```bash
python -m spacy download en_core_web_md
python -m spacy download en_core_web_sm   # also used by data/add_topics.py
```

### 5.2 Evaluation Driver — `eval/run_eval.py`

`run_eval.py` provides a full evaluation loop for a trained checkpoint.

High-level flow:

1. **Load model and tokenizer** — `load_model_and_tokenizer(model_ckpt, device)`
   - Loads the PyTorch checkpoint.
   - Reads configuration from `ckpt["config"]`.
   - Locates the tokenizer:
     - Uses `config["tokenizer_path"]` if present;
     - Otherwise falls back to a default (e.g. BPE 4k).
   - Reconstructs `TransformerDecoder` with the saved hyperparameters.

2. **Load evaluation prompts** — `load_eval_prompts(eval_csv)`
   - Expects CSV with columns: `eval_id`, `topic_1`, `topic_2`, `topic_3`.
   - Each row defines a set of topics to condition on.

3. **Format prompts** — `format_prompt_from_topics(topics)`
   - E.g. `"tell me a joke about cat, rain [JOKE]"`.

4. **Load training jokes and embeddings**
   - Training jokes: from `data/processed/final_clean_jokes.csv` (`joke_cleaned` column).
   - Embeddings:
     - If `train_embeddings.npz` exists, load it.
     - Otherwise compute with `encode_sentences` and save.

5. **Generate and evaluate** — `evaluate_model_on_prompts(...)`
   - For each eval row:
     - Builds prompt, generates a joke with `generate_joke(...)`.
     - Computes:
       - Topic recall (hard/soft).
       - GPT‑2 perplexity.
       - Max BLEU vs training jokes.
       - Max embedding similarity vs training jokes.
       - Semantic duplicate and copy flags.
     - Stores all metrics in a DataFrame.
   - Adds diversity metrics as `df.attrs["diversity"]`.

6. **Write outputs**
   - Per-example results:

     - `eval/eval_outputs/results_<model_name>.csv`

   - Summary JSON:

     - `eval/eval_outputs/summary_<model_name>.json`
     - Contains:
       - Overall mean / median for all metrics.
       - Diversity scores (distinct‑1, distinct‑2).
       - Metrics grouped by number of topics (1, 2, 3).

Example usage (modify `model_ckpt` and `model_name` inside `run_eval.py` as needed):

```bash
cd uhh-ias-ml
python -m eval.run_eval
```

Existing summaries include:

- `eval/eval_outputs/summary_decoder_50k.json`
- `eval/eval_outputs/summary_decoder_bpe_4k_3ksteps.json`
- `eval/eval_outputs/summary_decoder_bpe_8k_3ksteps.json`

These can be compared to choose the best tokenizer / model configuration.

---

## 6. Testing and Utilities

### 6.1 Generation Sanity Check — `tests/test_generate.py`

This script loads a chosen checkpoint and generates jokes for a few hardcoded topic sets:

- Uses:
  - `eval.run_eval.load_model_and_tokenizer`
  - `eval.run_eval.generate_joke`
- Prints prompts and generated jokes at different sampling temperatures and `top_k` values.
- Intended for quick qualitative inspection of:
  - Topic coverage.
  - Style and coherence.

Run:

```bash
cd uhh-ias-ml
python tests/test_generate.py
```

You may need to adjust `model_ckpt` inside the script to point to an existing checkpoint on your machine.

### 6.2 Metric Sanity Check — `tests/test_evaluation.py`

`test_evaluation.py` is a small harness that:

- Constructs a handful of “fake training jokes” and “fake generations”.
- Runs all major metrics from `eval.metrics`:
  - Topic recall (hard and soft).
  - GPT‑2 perplexity.
  - Max BLEU to training.
  - Max embedding similarity + semantic duplicate flag.
  - Distinct‑1 and Distinct‑2 across all generated jokes.
- Prints human-readable summaries for each example.

Run:

```bash
cd uhh-ias-ml
python tests/test_evaluation.py
```

> Note: These are research utilities rather than strict unit tests; they’re meant to confirm environment and dependencies are correctly set up.

---

## 7. Legacy Simple CLI — `main.py` and `utils/`

The repository also contains an earlier, simplified pipeline:

- `main.py`:
  - CLI with:
    - `--mode {train,infer}`
    - `--arch {decoder, encdec}`
  - Uses `utils.tokenizer.Tokenizer` (currently just a stub) and older versions of the models.
- `utils/`:
  - `preprocessing.py`, `train_utils.py`, `evaluation.py`, `inference.py`, `config.py`.

This path is useful as an educational example but does **not** reflect the more advanced and robust data/ training/ evaluation flow implemented in the Jana branch. For serious experiments, use:

- `data/*.py`
- `models/train.py`
- `eval/*.py`

---

## 8. Environment Setup and Quickstart

### 8.1 Install Dependencies

```bash
cd uhh-ias-ml
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -U pip
pip install -r requirements.txt

# Required spaCy models
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
```

For full evaluation, you also need:

- Network access to download:
  - GPT‑2 (via `transformers`).
  - SentenceTransformer models (via `sentence-transformers`).

### 8.2 Typical Experiment Flow

1. **Prepare raw datasets**
   - Place the various Kaggle / HF / TSV files into `data/raw/` according to `combine_datasets.py`.

2. **Combine and clean**

   ```bash
   python data/combine_datasets.py
   python data/clean.py
   python data/add_topics.py
   ```

3. **Train a tokenizer** (optional if you reuse `data/processed/tokenizer.json`)

   ```bash
   # Train BPE 4k tokenizer
   python -m models.bpe_tokenizer

   # Or train a Unigram tokenizer
   python -m models.unigram_tokenizer
   ```

4. **Train a decoder-only model**

   ```bash
   python -m models.train
   ```

5. **Evaluate a checkpoint**

   - Edit `eval/run_eval.py` to:
     - Point `model_ckpt` to the desired checkpoint.
     - Set `model_name` to something informative.
     - Ensure `eval_csv` (e.g. `data/processed/eval_prompts.csv`) exists.

   ```bash
   python -m eval.run_eval
   ```

6. **Inspect results**
   - Training:
     - `models/checkpoints/<RUN_NAME>/loss_history.csv`
     - `models/checkpoints/<RUN_NAME>/loss_curve*.png`
   - Evaluation:
     - `eval/eval_outputs/results_<model_name>.csv`
     - `eval/eval_outputs/summary_<model_name>.json`

---

## 9. Possible Next Steps Based on Current Code

Given what is already implemented in the Jana branch, natural next experiments include:

- **Tokenizer size sweep**
  - Systematically compare vocab sizes: 4k vs 8k vs 30k.
  - Keep model size (emb_dim / layers / heads) fixed to isolate tokenizer effects.

- **Positional encoding comparison**
  - Train models with standard learned position embeddings vs ALiBi.
  - Evaluate on the same topic-based evaluation set.

- **Attention head sweep**
  - Fix total embedding dimension (e.g. 384).
  - Compare `n_head = 4, 6, 8` (head sizes 96, 64, 48 respectively).

- **Config and experiment management**
  - Externalize training and evaluation hyperparameters into YAML / JSON configs.
  - Add small scripts to launch consistent sweeps.

- **Result aggregation**
  - Collect all `summary_*.json` into a single table or notebook for comparison.

This README should be sufficient for a new contributor to start from raw data, reproduce the existing pipeline, and extend experiments in a controlled way.
