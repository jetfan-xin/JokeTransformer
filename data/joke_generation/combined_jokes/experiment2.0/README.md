# Clean & Filtered Jokes Pipeline – Complete Documentation

## Overview

This experiment implements a comprehensive **multi-stage data cleaning and quality filtering pipeline** for creating a high-quality jokes dataset suitable for training and evaluation. The pipeline combines:

- **Multi-source joke collection** (3 human/web datasets + 2 LLM-generated datasets)
- **Text cleaning & normalization** (unicode, punctuation, URLs, deduplication)
- **Perplexity (PPL) analysis** using distilgpt2 to measure text fluency and identify low-quality jokes
- **Toxicity evaluation & filtering** using Detoxify to remove offensive content
- **Quality refinement** (PPL-based percentile filtering: 40–80%)
- **Semantic enrichment** (noun extraction via spaCy for topic annotation)

**Final Output:** Three high-quality joke datasets with different formats, suitable for LLM training and evaluation.

---

## Project Structure

```
experiment2.0/
├── README.md                           # This file
│
├── data_source/                        # Input datasets
│   ├── raw/                            # Original datasets (6 sources)
│   │   ├── amirkid_jokes.csv
│   │   ├── shuttie_dadjokes.csv
│   │   ├── ysharma_short_jokes.csv
│   │   ├── train.tsv.gz (rJokes)
│   │   ├── deepseek_jokes_dataset_500k.csv  # LLM-generated
│   │   └── gemini_jokes_dataset_500k.csv    # LLM-generated
│   └── raw_combined/                   # Intermediate combined files
│
├── outputs/                            # Pipeline outputs
│   ├── preprocessed/                   # Cleaned & deduplicated jokes
│   │   ├── clean_jokes.csv
│   │   ├── clean_jokes_llm.csv
│   │   └── debug/
│   ├── perplexity/                     # PPL analysis results
│   │   ├── train_quality_metrics.csv
│   │   └── train_quality_metrics_llm.csv
│   ├── detox/                          # Toxicity filtering results
│   │   ├── clean_jokes_detox.csv
│   │   ├── clean_jokes_detox_safe.csv
│   │   ├── clean_jokes_llm_detox.csv
│   │   └── clean_jokes_llm_detox_safe.csv
│   └── final/                          # Final high-quality datasets
│       ├── clean_good_jokes.csv
│       ├── clean_good_jokes_all_nouns.csv
│       └── clean_good_jokes_single_topic.csv
│
├── stats_plots/                        # Visualization outputs
│   ├── detox/                          # Toxicity analysis
│   │   ├── joke_toxicity/
│   │   │   ├── toxicity_distribution.html
│   │   │   ├── toxicity_distribution.png
│   │   │   ├── toxicity_dims_distribution.html
│   │   │   └── toxicity_dims_distribution.png
│   │   └── detox_before_after/
│   │       ├── ppl_samples_*.json
│   │       └── safe_filter_summary.json
│   └── perplexity/                     # PPL analysis
│       ├── joke_sources/
│       │   ├── ppl_joke_sources.html
│       │   └── ppl_joke_sources.png
│       └── ppl_before_after/
│           ├── ppl_before_after.html
│           ├── ppl_before_after.png
│           └── ppl_samples_*.json
│
└── utils/                              # 15 Python pipeline scripts
    ├── preprocessing_jingfan.py        # Clean, combine, deduplicate
    ├── training_data_eval.py           # Compute perplexity
    ├── plot_ppl_joke_sources.py        # Visualize PPL distributions
    ├── extract_ppl_samples.py          # Sample at percentiles
    ├── safety_filter.py                # Toxicity scoring
    ├── filter_safe_toxicity.py         # Apply threshold
    ├── plot_toxicity.py                # Plot toxicity
    ├── plot_toxicity_dims.py           # Plot dimensions
    ├── add_ppl_to_safe.py              # Merge PPL into safe data
    ├── plot_ppl_before_after.py        # Compare before/after
    ├── extract_mid_ppl.py              # PPL percentile filter (40–80%)
    ├── prepare_clean_good.py           # Prepare final dataset
    ├── add_all_nouns.py                # Extract topics via spaCy
    ├── llm_jokes_prepare.py            # Format for LLM training
    └── logger.py                       # Logging utility
```

---

## Data Sources

### Original Datasets

| Dataset | Type | File | Size | Purpose |
|---------|------|------|------|---------|
| AmirKid | Human | `amirkid_jokes.csv` | ~23k | Diversify human-created content |
| Shuttie | Human | `shuttie_dadjokes.csv` | ~10k | Dad jokes subgenre |
| Y. Sharma | Human | `ysharma_short_jokes.csv` | ~23k | Short jokes collection |
| rJokes | Human | `train.tsv.gz` | ~195k | Reddit community jokes |
| DeepSeek | LLM | `deepseek_jokes_dataset_500k.csv` | ~500k | LLM-generated variety |
| Gemini | LLM | `gemini_jokes_dataset_500k.csv` | ~500k | LLM-generated quality |

**Total: ~1.2M jokes before filtering**

---

## Pipeline Stages

### Stage A: Dataset Collection & Combination

**Input:** 6 separate CSV/TSV files in `data_source/raw/`  
**Output:** Combined raw pool (~1.2M jokes)  
**Script:** `utils/preprocessing_jingfan.py` (Class: `DatasetCombiner`)

Processing:
- Load each dataset and extract the joke column(s)
- Handle multiple formats (CSV, TSV, gzipped)
- Normalize column names
- Remove empty rows

---

### Stage B: Cleaning, PPL Analysis & Sampling

#### B1) Text Cleaning & Deduplication

**Script:** `utils/preprocessing_jingfan.py` (Class: `PreprocessorCleaner`)

**Cleaning operations:**
- **Normalize Unicode:** Use ftfy to fix encoding issues
- **Text normalization:** Remove control characters, normalize whitespace
- **Remove URLs:** Filter HTTP/HTTPS links
- **Remove markup:** Strip HTML, XML, markdown syntax
- **Length filtering:** Keep jokes with 10–1000 characters
- **Newline filtering:** Disallow >4 newlines per joke
- **Deduplication:** Remove exact duplicates via hash-based approach

**Configuration:**
```python
MAX_LEN = 1000      # Max characters after normalization
MIN_LEN = 10        # Min characters
MAX_LINES = 4       # Max newlines allowed
```

**Output:** Cleaned joke CSVs
- `outputs/preprocessed/clean_jokes.csv` (non-LLM sources)
- `outputs/preprocessed/clean_jokes_llm.csv` (LLM sources)

Typical statistics after cleaning: ~95% of original jokes retained

#### B2) Perplexity (PPL) Computation & Visualization

**Goal:** Measure text fluency and identify low-quality jokes.

**Script:** `utils/training_data_eval.py`

**Model:** DistilGPT2
- Fast, lightweight transformer-based language model
- Good proxy for text naturalness
- Runs on CPU/GPU (configurable via `--device`)

**Metrics computed per joke:**
- **Perplexity:** How "surprised" the model is by the text
  - Lower PPL = more fluent, natural text
  - Higher PPL = unusual, nonsensical, or low-quality text
- **Loss:** Cross-entropy loss averaged over tokens

**Output:** CSV with PPL scores
- `outputs/perplexity/train_quality_metrics.csv` (non-LLM)
- `outputs/perplexity/train_quality_metrics_llm.csv` (LLM)

**Columns:**
- `rid` (or `id`): Row ID
- `text`: Original joke text
- `perplexity`: PPL score
- `loss`: Model loss
- `num_tokens`: Token count

**Usage:**
```bash
cd /uhh-ias-ml/data/joke_generation/combined_jokes/experiment2.0

# Compute PPL for non-LLM jokes (GPU)
python -m utils.training_data_eval \
  --data-csv outputs/preprocessed/clean_jokes.csv \
  --text-col joke_cleaned \
  --out-csv outputs/perplexity/train_quality_metrics.csv \
  --device cuda

# Compute PPL for LLM jokes
CUDA_VISIBLE_DEVICES=1 python -m utils.training_data_eval \
  --data-csv outputs/preprocessed/clean_jokes_llm.csv \
  --text-col joke_cleaned \
  --out-csv outputs/perplexity/train_quality_metrics_llm.csv \
  --device cuda \
  --batch-size 8 \
  --max-length 256
```

**Visualization:** `utils/plot_ppl_joke_sources.py`

Generates interactive histograms showing PPL distribution per source to identify which sources produce more/less fluent jokes.

**Output plots:**
- `stats_plots/perplexity/joke_sources/ppl_joke_sources.html` (interactive)
- `stats_plots/perplexity/joke_sources/ppl_joke_sources.png` (static)

#### B3) PPL Percentile Sampling

**Script:** `utils/extract_ppl_samples.py`

**Goal:** Extract representative jokes at each 10% PPL percentile for manual inspection.

**Sampling:**
- Divide jokes into 10 percentile bins: [0–10%), [10–20%), ..., [90–100%]
- Extract representative samples from each bin
- Save as JSON with text and metadata

**Purpose:** Manual inspection to understand what jokes look like across PPL spectrum
- 0–40th percentile: Trivial, repetitive, over-generalized jokes
- 40–80th percentile: **Balanced fluency + novelty** (target zone)
- 80–100th percentile: Nonsensical, low-quality jokes

---

### Stage C: Toxicity Evaluation & Filtering

#### C1) Toxicity Scoring with Detoxify

**Script:** `utils/safety_filter.py`

**Toxicity model:** Detoxify (multi-label classification)
- Fast CPU/GPU inference
- Outputs 6 toxicity dimensions:
  - `toxicity` (general offensive language)
  - `severe_toxicity` (extreme vulgarities)
  - `obscene` (obscene language)
  - `threat` (threats)
  - `insult` (insults/derogatory)
  - `identity_attack` (attacks based on identity)

**Optional:** LlamaGuard-7b for stricter review on flagged rows

**Usage:**
```bash
# Stage 1: Fast Detoxify scan
python -m utils.safety_filter \
  --data-csv outputs/preprocessed/clean_jokes.csv \
  --text-col joke_cleaned \
  --out-csv outputs/detox/clean_jokes_detox.csv \
  --detox-threshold 0.5 \
  --device cuda

# Stage 1 + Stage 2 (with LlamaGuard on flagged rows)
python -m utils.safety_filter \
  --data-csv outputs/preprocessed/clean_jokes.csv \
  --text-col joke_cleaned \
  --out-csv outputs/detox/clean_jokes_detox.csv \
  --detox-threshold 0.5 \
  --use-llamaguard \
  --llama-max 128 \
  --device cuda
```

**Output:** CSV with toxicity scores
- `outputs/detox/clean_jokes_detox.csv`
- `outputs/detox/clean_jokes_llm_detox.csv`

**Columns:**
- Original columns + 6 toxicity score columns ([0.0–1.0] each)
- `flagged_by_detox` (boolean): True if any dimension ≥ threshold
- `flagged_by_llamaguard` (boolean, optional): Strict review result

#### C2) Toxicity Visualization

**Scripts:**
- `utils/plot_toxicity.py`: Overlay histograms by source
- `utils/plot_toxicity_dims.py`: Separate plots for each dimension

**Usage:**
```bash
# Plot toxicity distribution
python -m utils.plot_toxicity \
  --files outputs/detox/clean_jokes_detox.csv outputs/detox/clean_jokes_llm_detox.csv \
  --out stats_plots/detox/joke_toxicity/toxicity_distribution.html \
  --png stats_plots/detox/joke_toxicity/toxicity_distribution.png

# Plot all 6 dimensions
python -m utils.plot_toxicity_dims \
  --files outputs/detox/clean_jokes_detox.csv outputs/detox/clean_jokes_llm_detox.csv \
  --out stats_plots/detox/joke_toxicity/toxicity_dims_distribution.html
```

**Output plots:**
- `stats_plots/detox/joke_toxicity/toxicity_distribution.{html,png}`
- `stats_plots/detox/joke_toxicity/toxicity_dims_distribution.{html,png}`

#### C3) Toxicity Threshold & Safe Dataset Creation

**Script:** `utils/filter_safe_toxicity.py`

**Decision threshold:** Keep only jokes with **all toxicity dimensions < 0.1**

**Rationale:**
- 0.1 threshold empirically removes obvious toxic content
- Balances safety with dataset utility

**Output:** Safe joke subsets
- `outputs/detox/clean_jokes_detox_safe.csv` (non-LLM safe)
- `outputs/detox/clean_jokes_llm_detox_safe.csv` (LLM safe)
- Summary JSON: `stats_plots/detox/detox_before_after/safe_filter_summary.json`
  - Per-source removal ratios
  - Overall filtering statistics

**Typical filtering results:**
- Non-LLM: ~95% retention
- LLM: ~98% retention

---

### Stage D: Safe Data Re-check & Quality Refinement

#### D1) Add PPL Scores to Safe Data

**Script:** `utils/add_ppl_to_safe.py`

**Goal:** Recompute PPL only on toxicity-filtered "safe" subset.

**Join mechanism:**
- Merges toxicity-safe CSVs with PPL metrics (join on `rid` or `id`)
- Preserves row order and all columns from safe CSV
- Adds `perplexity` column from metrics CSV

#### D2) Before-After PPL Comparison

**Script:** `utils/plot_ppl_before_after.py`

**Comparison:**
- Plot PPL histograms before toxicity filtering
- Plot PPL histograms after toxicity filtering
- Visualize whether toxicity filtering changes PPL distribution

**Observation:** PPL distribution remains similar after toxicity removal
- Safe subset preserves fluency/quality patterns
- Toxicity filtering doesn't bias toward specific PPL ranges

**Output plots:**
- `stats_plots/perplexity/ppl_before_after/{html,png}`
- Sample files: `stats_plots/perplexity/ppl_samples_after_detox_*.json`

#### D4) Apply PPL Percentile Filter (40–80%)

**Script:** `utils/extract_mid_ppl.py`

**Decision:** Keep only jokes in **40th–80th PPL percentile** of safe data

**Rationale:**
- 0–40th percentile: Trivial, repetitive, over-generalized jokes
- 40–80th percentile: **Balanced fluency + novelty** (target zone)
- 80–100th percentile: Nonsensical, low-quality jokes

**Processing:**
1. Load safe joke CSVs (after toxicity filtering)
2. Merge non-LLM + LLM safe datasets
3. Compute global 40th and 80th PPL percentiles
4. Filter to keep only jokes within this range
5. Write final high-quality dataset

**Output:** Merged safe + filtered dataset
- `outputs/final/clean_good_jokes.csv`

**Typical statistics after PPL filtering:**
- ~40% of safe jokes retained (40–80% percentile range)
- Combined with toxicity filter: ~38% of original jokes

---

### Stage E: Final Dataset Preparation

#### E1) Prepare Clean Good Jokes

**Script:** `utils/prepare_clean_good.py`

**Processing:**
- Validate columns and data integrity
- Remove redundant/debug columns
- Standardize column order and naming
- Add metadata (source, processing flags)

**Output:** `outputs/final/clean_good_jokes.csv`

**Columns:**
- `id` or `rid`: Unique identifier
- `joke_text` or similar: Final cleaned joke
- `source`: Which dataset (amirkid, deepseek, etc.)
- `is_llm`: Boolean flag for LLM-generated vs. human
- `perplexity`: PPL score
- Toxicity dimensions (optional)

#### E2) Noun-Based Topic Extraction

**Script:** `utils/add_all_nouns.py`

**Processing:**
1. Load spaCy English model: `en_core_web_sm`
2. For each joke, extract:
   - All NOUN and PROPN tokens
   - Lemmatize (convert to base form)
   - Filter custom stopwords (e.g., "joke", "thing", "people")
3. Rank nouns by frequency and length
4. Produce comma-separated string: `topic_all_nouns`

**Topic stopwords configuration:**
```python
TOPIC_STOPWORDS = {
    "joke", "jokes", "thing", "things", "one", "ones",
    "way", "time", "day", "guy", "people", "someone",
    ...
}
```

**Usage:**
```bash
python -m utils.add_all_nouns \
  --input outputs/final/clean_good_jokes.csv \
  --output outputs/final/clean_good_jokes_all_nouns.csv
```

**Output:**
- `outputs/final/clean_good_jokes_all_nouns.csv`

**New columns:**
- `topic_all_nouns`: Comma-separated extracted nouns
- `topic_single`: First noun (for single-topic format)

#### E3) Derived Datasets

Two final formats are generated:

| Dataset | File | Format | Use Case |
|---------|------|--------|----------|
| Base | `clean_good_jokes.csv` | ID, text, source, PPL, toxicity | Foundational dataset |
| All-nouns | `clean_good_jokes_all_nouns.csv` | Base + multi-noun topics | Topic-based filtering |
| Single-topic | `clean_good_jokes_single_topic.csv` | Base + single topic | Topic-aware training |

---

### Stage F: Training Data Evaluation

**Script:** `utils/llm_jokes_prepare.py`

**Goal:** Format final dataset for LLM training/fine-tuning.

**Processing:**
- Load final clean good jokes
- Format into standard training structures:
  - Hugging Face `datasets` format
  - OpenAI JSONL format (one JSON per line)
  - Plain CSV with standardized headers
- Optional: Split into train/val/test sets
- Generate metadata (e.g., data statistics, label distributions)

**Output:** Training-ready dataset formats
- JSONL files: `data_train.jsonl`, `data_val.jsonl`
- CSV: `data_all_formatted.csv`
- Metadata: `data_stats.json`

---

## Installation & Requirements

### Python Version
- Python 3.8+

### Core Dependencies

```bash
pip install -r requirements.txt
```

**Key packages:**
- **pandas** ≥2.1.0 – Data manipulation
- **spacy** ≥3.7.0 – NLP (noun extraction)
- **transformers** – LLM models (DistilGPT2, LlamaGuard)
- **torch** – PyTorch for GPU inference
- **detoxify** – Toxicity scoring
- **plotly** – Interactive visualizations
- **ftfy** – Text encoding fixes
- **nltk** – BLEU score computation

### GPU Setup (Recommended)

All PPL and toxicity scoring scripts support GPU acceleration via PyTorch/CUDA.

**Check GPU availability:**
```bash
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

**Set GPU device:**
```bash
# Use GPU 2
CUDA_VISIBLE_DEVICES=2 python -m utils.training_data_eval \
  --device cuda ...
```

### spaCy Model

Download English model for noun extraction:
```bash
python -m spacy download en_core_web_sm
```

---

## Running the Complete Pipeline

### Step-by-Step Execution (Recommended)

```bash
cd /uhh-ias-ml/data/joke_generation/combined_jokes/experiment2.0

# Stage A+B1: Combine & clean datasets
python -m utils.preprocessing_jingfan

# Stage B2: Compute PPL
CUDA_VISIBLE_DEVICES=0 python -m utils.training_data_eval \
  --data-csv outputs/preprocessed/clean_jokes.csv \
  --text-col joke_cleaned \
  --out-csv outputs/perplexity/train_quality_metrics.csv \
  --device cuda

CUDA_VISIBLE_DEVICES=1 python -m utils.training_data_eval \
  --data-csv outputs/preprocessed/clean_jokes_llm.csv \
  --text-col joke_cleaned \
  --out-csv outputs/perplexity/train_quality_metrics_llm.csv \
  --device cuda

# Stage B2: Plot PPL distributions
python -m utils.plot_ppl_joke_sources

# Stage B3: Extract samples
python -m utils.extract_ppl_samples

# Stage C1: Compute toxicity
CUDA_VISIBLE_DEVICES=0 python -m utils.safety_filter \
  --data-csv outputs/preprocessed/clean_jokes.csv \
  --text-col joke_cleaned \
  --out-csv outputs/detox/clean_jokes_detox.csv \
  --detox-threshold 0.5 \
  --device cuda

CUDA_VISIBLE_DEVICES=1 python -m utils.safety_filter \
  --data-csv outputs/preprocessed/clean_jokes_llm.csv \
  --text-col joke_cleaned \
  --out-csv outputs/detox/clean_jokes_llm_detox.csv \
  --detox-threshold 0.5 \
  --device cuda

# Stage C1: Apply toxicity threshold
python -m utils.filter_safe_toxicity

# Stage C2: Plot toxicity
python -m utils.plot_toxicity
python -m utils.plot_toxicity_dims

# Stage D1: Merge PPL into safe CSVs
python -m utils.add_ppl_to_safe

# Stage D1: Plot before-after
python -m utils.plot_ppl_before_after

# Stage D4: Apply PPL percentile filter
python -m utils.extract_mid_ppl

# Stage E2: Extract topics
python -m utils.add_all_nouns

# Stage F: Prepare for training
python -m utils.llm_jokes_prepare
```

### Batch Processing with Parallelization

For large-scale processing, parallelize independent stages:

```bash
# In parallel terminals:
# Terminal 1: Clean + compute non-LLM PPL
python -m utils.preprocessing_jingfan && \
  CUDA_VISIBLE_DEVICES=0 python -m utils.training_data_eval \
    --data-csv outputs/preprocessed/clean_jokes.csv ...

# Terminal 2: Compute LLM PPL + toxicity
CUDA_VISIBLE_DEVICES=1 python -m utils.training_data_eval \
  --data-csv outputs/preprocessed/clean_jokes_llm.csv ... && \
  python -m utils.safety_filter \
    --data-csv outputs/preprocessed/clean_jokes_llm.csv ...

# Then run final stages serially
python -m utils.add_ppl_to_safe && \
  python -m utils.extract_mid_ppl && \
  python -m utils.add_all_nouns
```

---

## Output Summary

### Final Datasets (Ready to Use)

Three high-quality joke CSV files ready for use:

1. **`clean_good_jokes.csv`**
   - Base high-quality jokes (toxicity < 0.1, PPL 40–80%)
   - ~380k jokes (38% of original ~1.2M)
   - Columns: ID, text, source, is_llm, perplexity, toxicity dimensions
   - **Best for:** General-purpose training

2. **`clean_good_jokes_all_nouns.csv`**
   - Same as above + `topic_all_nouns` column
   - Multiple topics per joke (comma-separated nouns)
   - **Best for:** Topic-based filtering, conditional generation

3. **`clean_good_jokes_single_topic.csv`**
   - Same as above + `topic_single` column
   - Single topic per joke (first noun)
   - **Best for:** Topic-aware LLM training, topic classification

### Analysis Artifacts

- **PPL histograms:** `stats_plots/perplexity/` (interactive HTML + static PNG)
- **Toxicity analysis:** `stats_plots/detox/` (distributions, before-after)
- **Quality samples:** JSON files with representative jokes at each percentile
- **Summary statistics:** `stats_plots/detox/detox_before_after/safe_filter_summary.json`

---

## Troubleshooting

### Issue: "Out of Memory" during PPL/toxicity computation

**Solution:** Reduce batch size
```bash
python -m utils.training_data_eval \
  --batch-size 4 \  # Instead of default 8
  --max-length 128 \  # Instead of 256
  --device cuda ...
```

### Issue: spaCy model not found

**Solution:** Download the model
```bash
python -m spacy download en_core_web_sm
```

### Issue: GPU not detected

**Solution:** Check CUDA installation
```bash
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True (if GPU installed)
# If False, install pytorch with CUDA support:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: LlamaGuard model slow to load

**Solution:** First run downloads the model (~15GB). Subsequent runs are fast. If timeout occurs, increase `--timeout 300` or skip LlamaGuard on first run.

### Issue: Plots not generating

**Solution:** Ensure kaleido is installed for PNG export
```bash
pip install kaleido
```

---

## Advanced Configuration

### Adjusting Toxicity Threshold

Edit threshold in `utils/filter_safe_toxicity.py`:
```python
TOXICITY_THRESHOLD = 0.1  # Change to 0.15 for more permissive
```

### Adjusting PPL Percentile Range

Edit range in `utils/extract_mid_ppl.py`:
```python
PERCENTILE_LOW = 40   # Change to 30
PERCENTILE_HIGH = 80  # Change to 85
```

### Adjusting Cleaning Parameters

Edit in `utils/preprocessing_jingfan.py`:
```python
class PreprocessConfig:
    MAX_LEN = 1000    # Max characters
    MIN_LEN = 10      # Min characters
    MAX_LINES = 4     # Max newlines
```

---

## Performance Notes

### Typical Runtime

On a single GPU (NVIDIA A100):
- Cleaning (1.2M jokes): ~5 minutes
- PPL computation (1.2M jokes, batch_size=8): ~30 minutes
- Toxicity scoring (1.2M jokes): ~20 minutes
- Visualization & topic extraction: ~10 minutes
- **Total: ~1–2 hours for complete pipeline**

### Memory Requirements

- GPU: ~20GB (for batch_size=8, max_length=256)
- RAM: ~16GB (for data loading)
- Disk: ~50GB (raw + outputs + temp files)

### Parallelization Opportunities

- PPL computation for non-LLM and LLM subsets (parallel on different GPUs)
- Toxicity scoring for both subsets (parallel)
- Visualization generation (parallel)
- Topic extraction (parallel via multiprocessing)

---

## References & Justification

### Perplexity-Based Quality Filtering

Perplexity via language models is a well-established proxy for text quality:
- Low PPL: Fluent, coherent text
- High PPL: Unusual, low-quality, or domain-shifted text
- 40–80% percentile empirically balances creativity and coherence for joke generation

### Detoxify Scoring

Detoxify multi-label classification:
- Fast inference (CPU/GPU)
- 6 independent toxicity dimensions
- 0.1 threshold: Conservative, removes obvious toxic content while preserving utility
- LlamaGuard: More principled but slower; use on flagged samples for validation

### Topic Extraction via Nouns

Extracting noun lemmas:
- Represents joke subjects/entities
- Useful for topic-aware training and conditional generation
- spaCy NLP: Efficient, accurate, handles edge cases

---

## Related Documentation
- Individual script docstrings for detailed parameter descriptions
