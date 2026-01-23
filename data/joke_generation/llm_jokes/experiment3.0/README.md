# Experiment 3.0: LLM Joke Generation & Data Processing Pipeline

**Version**: 3.0  
**Status**: Production  
**Last Updated**: January 2026  
**Purpose**: Large-scale LLM-generated joke dataset creation, cleaning, toxicity filtering, and quality analysis pipeline for joke generation research.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Data Processing Pipeline](#data-processing-pipeline)
4. [Scripts Reference](#scripts-reference)
5. [Getting Started](#getting-started)
6. [Configuration & Parameters](#configuration--parameters)
7. [Data Formats & Schemas](#data-formats--schemas)
8. [Running the Pipeline](#running-the-pipeline)
9. [Outputs & Results](#outputs--results)
10. [Troubleshooting](#troubleshooting)
11. [Reproducibility](#reproducibility)

---

## Overview

**Experiment 3.0** is a comprehensive joke generation and data processing system that:

- **Generates** ~50,000 LLM-created jokes conditioned on 5,000 topic combinations using DeepSeek API
- **Cleans** and normalizes joke text (removes HTML, Markdown, URLs, emojis, control characters)
- **Filters** jokes for toxicity using multiple safety models (Detoxify, LlamaGuard, profanity lists)
- **Deduplicates** jokes with repeated characters or structural anomalies
- **Scores** jokes using GPT-2 perplexity to measure naturalness
- **Analyzes** perplexity distributions and generates visualizations
- **Balances** datasets to ensure ~100 jokes per topic combination

### Key Statistics

| Metric | Value |
|--------|-------|
| **Topic Combinations** | 5,000 |
| **Jokes Generated** | ~50,000 (10 per combo) |
| **Pipeline Stages** | 8 sequential processing steps |
| **Processing Models** | DeepSeek (generation), Detoxify (toxicity), LlamaGuard (safety), GPT-2 (scoring) |
| **Toxicity Dimensions** | 6 (toxicity, severe_toxicity, obscene, threat, insult, identity_attack) |
| **Final Dataset Size** | ~500,000 jokes (100 per balanced combo) |

### Relationship to Other Experiments

- **Input Source**: Uses `data_source/clean_jokes_clean_topics_3.csv` as the raw input for topic extraction
- **Current (Experiment 3.0)**: Processes and cleans input jokes; adds toxicity filtering and perplexity scoring
- **Downstream (Experiment 4.0)**: Further dataset refinement and final preparation for model training

---

## Project Structure

```
experiment3.0/
├── analyzers/                    # Data processing & analysis scripts (9 files)
│   ├── extract_top_5000_topic_combos.py
│   ├── build_llm_jokes_top_5000_topics_3.py
│   ├── filter_llm_jokes_toxicity.py
│   ├── filter_repeated_char_jokes.py
│   ├── compute_ppl_llm_jokes.py
│   ├── extract_ppl_samples_llm_jokes.py
│   ├── plot_ppl_hist_llm_jokes.py
│   ├── sample_100_per_topic_combo.py
│   └── backfill_jokes_under_100_async.py
│
├── data_source/clean_jokes_clean_topics_3.csv  # Data source
│
├── generators/                   # LLM generation scripts (1 file)
│   └── run_jokes_async.py
│
├── outputs/                      # Generated datasets (7 CSV files)
│   ├── deepseek_jokes.csv
│   ├── deepseek_jokes_TEST.csv
│   ├── llm_jokes_top_5000_topics_3.csv
│   ├── llm_jokes_top_5000_topics_3_detox.csv
│   ├── llm_jokes_top_5000_topics_3_detox_safe.csv
│   ├── llm_jokes_top_5000_topics_3_detox_safe_no_repeats.csv
│   └── llm_jokes_top_5000_topics_3_detox_safe_no_repeats_sampled100.csv
│
├── stats/                        # Metadata & metrics (4 files)
│   ├── top_5000_topic_combos.csv
│   ├── llm_jokes_top_5000_topics_3_detox_safe_ppl.csv
│   ├── llm_jokes_top_5000_topics_3_detox_safe_ppl_samples.json
│   └── removed_repeated_char_joke.csv
│
├── plots/                        # Visualization outputs (2 files)
│   ├── ppl_hist.html
│   └── ppl_hist.png
│
└── README.md                     # This file

```

---

## Data Processing Pipeline

The pipeline transforms raw joke data through 8 sequential stages:

```
data_source/clean_jokes_clean_topics_3.csv (raw jokes + topics)
         ↓
[Stage 1] Extract Top 5,000 Topic Combinations
         ↓ (top_5000_topic_combos.csv)
[Stage 2] Generate/Backfill Jokes via DeepSeek API
         ↓ (deepseek_jokes.csv)
[Stage 3] Clean & Normalize Joke Text
         ↓ (llm_jokes_top_5000_topics_3.csv)
[Stage 4] Filter Toxicity (3-model ensemble)
         ↓ (llm_jokes_top_5000_topics_3_detox_safe.csv)
[Stage 5] Remove Repeated Character Jokes
         ↓ (llm_jokes_top_5000_topics_3_detox_safe_no_repeats.csv)
[Stage 6] Compute GPT-2 Perplexity Scores
         ↓ (llm_jokes_top_5000_topics_3_detox_safe_ppl.csv)
[Stage 7] Perplexity Analysis & Visualization
         ↓ (ppl_samples.json, ppl_hist.png, ppl_hist.html)
[Stage 8] Balance Dataset (100 jokes per combo)
         ↓
Final Balanced Dataset (~500K jokes)
```

### Stage-by-Stage Details

#### Stage 1: Extract Top 5,000 Topic Combinations
- **Script**: `analyzers/extract_top_5000_topic_combos.py`
- **Input**: `data_source/clean_jokes_clean_topics_3.csv` (cleaned jokes dataset with topics)
- **Output**: `stats/top_5000_topic_combos.csv` (5,000 rows, 2 columns: `topic`, `count`)
- **Purpose**: Identify most common topic combinations to focus generation efforts
- **Example Output**:
  ```
  topic,count
  "girlfriend",200
  "bar",191
  "wife",180
  "type, world",127
  ```

#### Stage 2: Generate/Backfill Jokes via DeepSeek API
- **Script**: `generators/run_jokes_async.py`
- **Input**: `stats/top_5000_topic_combos.csv` (topic combinations)
- **Output**: `outputs/deepseek_jokes.csv`
- **Method**:
  - Uses **DeepSeek Chat API** to generate jokes conditioned on topic combinations
  - Parses numbered joke lists from LLM output using regex patterns
  - Implements **async concurrent generation** (50 parallel requests) for speed
  - Exponential backoff retry logic for API failures
- **Generation Config**:
  - **Model**: `deepseek-chat`
  - **Jokes per combo**: 100 (production) / 5 (test)
  - **Concurrency**: 50 parallel requests
  - **Max attempts**: Retries up to 5 times with exponential backoff
  - **Processing time**: ~3-4 hours for 50,000 jokes
- **Production vs. Test Mode**:
  - `TEST_MODE=False`: Processes all 5,000 combos (50,000 jokes)
  - `TEST_MODE=True`: Processes 3 combos (combos 567-569) for testing

#### Stage 3: Clean & Normalize Joke Text
- **Script**: `analyzers/build_llm_jokes_top_5000_topics_3.py`
- **Input**: `outputs/deepseek_jokes.csv` (raw DeepSeek output)
- **Output**: `outputs/llm_jokes_top_5000_topics_3.csv` (cleaned jokes)
- **Processing Steps**:
  1. **Strip Markdown**: Remove bold (`**`), italic (`*`), headers (`#`), quotes (`>`), links (`[text](url)`)
  2. **Strip HTML**: Remove HTML tags, unescape entities
  3. **Remove URLs**: Strip http/https/www links and angle-bracket links
  4. **Remove Emails**: Filter out email addresses
  5. **Remove Code Fences**: Strip triple-backtick code blocks
  6. **Remove Emojis**: Comprehensive emoji removal (16 Unicode ranges)
  7. **Remove Control Characters**: Strip non-printable chars (`\u0000-\u001f`, `\u007f-\u009f`)
  8. **Normalize Whitespace**: Collapse multiple spaces; normalize line breaks
  9. **Unicode Normalization**: Apply NFKC normalization + ftfy text fixing
  10. **Length Enforcement**: Keep jokes between 10 and 1,000 characters
  11. **Line Count Limit**: Enforce max 4 lines per joke
  12. **Remove Leading Bullets**: Strip leading colons, dashes, punctuation
  13. **Deduplication**: Use MD5 hash of normalized text to remove exact duplicates

- **Configuration**:
  ```python
  MIN_LEN = 10         # Minimum joke length (chars)
  MAX_LEN = 1000       # Maximum joke length (chars)
  MAX_LINES = 4        # Maximum lines per joke
  ```

#### Stage 4: Filter Toxicity (3-Model Ensemble)
- **Script**: `analyzers/filter_llm_jokes_toxicity.py`
- **Input**: `outputs/llm_jokes_top_5000_topics_3.csv` (cleaned jokes)
- **Output**: 
  - `outputs/llm_jokes_top_5000_topics_3_detox.csv` (all jokes + toxicity scores)
  - `outputs/llm_jokes_top_5000_topics_3_detox_safe.csv` (safe jokes only)
- **Toxicity Models & Dimensions**:
  1. **Detoxify** (neural toxicity classifier):
     - 6 dimensions: toxicity, severe_toxicity, obscene, threat, insult, identity_attack
     - Score range: [0.0, 1.0] per dimension
     - Threshold: ≥ 0.1 in any dimension → marked toxic
  2. **LlamaGuard** (safety policy classifier):
     - Evaluates against predefined safety policies
     - Returns policy violation categories or "safe"
  3. **Profanity List** (keyword-based):
     - Custom list of profanity terms
     - Case-insensitive substring matching

- **Filtering Rules**:
  ```python
  THRESHOLD = 0.1           # Detoxify score threshold
  BATCH_SIZE = 8            # Processing batch size
  MODEL_NAME = "original"   # Detoxify model variant
  DEVICE = "auto"           # auto | cuda | cpu
  ```
  - Joke is marked **safe** if:
    - ALL Detoxify dimensions < threshold
    - LlamaGuard doesn't flag policy violations
    - Profanity check passes
  - Safe flag columns added to output CSV

#### Stage 5: Remove Repeated Character Jokes
- **Script**: `analyzers/filter_repeated_char_jokes.py`
- **Input**: `outputs/llm_jokes_top_5000_topics_3_detox_safe.csv`
- **Output**: `outputs/llm_jokes_top_5000_topics_3_detox_safe_no_repeats.csv`
- **Filtering Logic**: Removes jokes with:
  - Repeated ASCII art patterns (e.g., "===", "---", "***")
  - Nonsense character sequences (same char 5+ times in a row)
  - Duplicate or corrupted LLM outputs
- **Log Output**: `stats/removed_repeated_char_joke.csv` (tracks removed jokes)

#### Stage 6: Compute GPT-2 Perplexity Scores
- **Script**: `analyzers/compute_ppl_llm_jokes.py`
- **Input**: `outputs/llm_jokes_top_5000_topics_3_detox_safe_no_repeats.csv`
- **Output**: `stats/llm_jokes_top_5000_topics_3_detox_safe_ppl.csv` (adds `ppl` column)
- **Method**:
  - Loads pre-trained GPT-2 model from Hugging Face Transformers
  - Computes **cross-entropy loss** → convert to perplexity: `ppl = exp(loss)`
  - Measures how "natural" the joke is (lower = more natural)
  - Supports batch processing and mixed precision (fp16) for speed
- **Configuration**:
  ```python
  MODEL_NAME = "gpt2"       # Language model for perplexity
  BATCH_SIZE = 32           # Batch size (adjust per GPU memory)
  MAX_LENGTH = 512          # Max tokens per joke
  DEVICE = "auto"           # auto | cuda:0 | cpu
  USE_AMP = True            # Automatic mixed precision for faster GPU inference
  ```
- **Perplexity Interpretation**:
  - **Low PPL (< 50)**: Natural, grammatical jokes
  - **Medium PPL (50-100)**: Reasonable quality
  - **High PPL (> 100)**: Unusual phrasing, potential quality issues

#### Stage 7: Perplexity Analysis & Visualization
- **Script**: 
  - `analyzers/extract_ppl_samples_llm_jokes.py` (sampling)
  - `analyzers/plot_ppl_hist_llm_jokes.py` (visualization)
- **Input**: `stats/llm_jokes_top_5000_topics_3_detox_safe_ppl.csv` (PPL-scored jokes)
- **Output**:
  - `stats/llm_jokes_top_5000_topics_3_detox_safe_ppl_samples.json` (sampled jokes)
  - `plots/ppl_hist.html` (interactive histogram)
  - `plots/ppl_hist.png` (static histogram image)
- **Sampling Strategy**:
  - Divides jokes into 10 percentile groups: [0-10%, 10-20%, ..., 90-100%]
  - Randomly samples N jokes from each group to create representative dataset
  - Useful for understanding PPL distribution and identifying outliers
- **Visualization**:
  - Histograms showing PPL distribution
  - Percentile breakdowns and summary statistics

#### Stage 8: Balance Dataset (100 jokes per combo)
- **Script**: `analyzers/sample_100_per_topic_combo.py`
- **Input**: `outputs/llm_jokes_top_5000_topics_3_detox_safe_no_repeats.csv`
- **Output**: `outputs/llm_jokes_top_5000_topics_3_detox_safe_no_repeats_sampled100.csv`
- **Balancing Algorithm**:
  - Groups jokes by topic combination
  - Samples **100 jokes per group** (or all jokes if group < 100)
  - Drops toxicity-related columns (no longer needed in final dataset)
  - Ensures reproducibility via random seed (default: 42)
- **Configuration**:
  ```python
  SAMPLE_SIZE = 100         # Target jokes per combo
  TARGET_TOTAL = 500000     # Target total jokes
  SEED = 42                 # Random seed for reproducibility
  DROP_COLS = [             # Columns to remove
      "toxicity", "severe_toxicity", "obscene", "threat",
      "insult", "identity_attack", "detox_flag", "profanity_flag",
      "llamaguard_output", "llamaguard_flag"
  ]
  ```
- **Backfill Logic** (Optional): `analyzers/backfill_jokes_under_100_async.py`
  - For combos with < 100 jokes: generates additional jokes via DeepSeek
  - Multiplier: generates 1.1x more jokes to account for toxicity filtering
  - Ensures target sample size is reached

---

## Scripts Reference

### Analyzers (Processing Scripts)

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `extract_top_5000_topic_combos.py` | Extract top topic combinations | Raw jokes + topics | `top_5000_topic_combos.csv` |
| `build_llm_jokes_top_5000_topics_3.py` | Clean & normalize jokes | Raw jokes | Cleaned jokes CSV |
| `filter_llm_jokes_toxicity.py` | Toxicity filtering (3-model ensemble) | Cleaned jokes | Toxicity-scored + safe CSVs |
| `filter_repeated_char_jokes.py` | Remove anomalies | Toxic-filtered jokes | Deduplicated jokes CSV |
| `compute_ppl_llm_jokes.py` | Compute GPT-2 perplexity | Safe jokes | PPL-scored jokes CSV |
| `extract_ppl_samples_llm_jokes.py` | Sample by percentile | PPL-scored jokes | JSON samples file |
| `plot_ppl_hist_llm_jokes.py` | Visualize PPL distribution | PPL-scored jokes | HTML/PNG histograms |
| `sample_100_per_topic_combo.py` | Balance dataset | Deduplicated jokes | 100 jokes/combo CSV |
| `backfill_jokes_under_100_async.py` | Generate missing jokes | Incomplete combos | Backfilled jokes CSV |

### Generators (LLM Generation)

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `run_jokes_async.py` | DeepSeek API joke generation | Topic combos CSV | Raw jokes CSV |

---

## Getting Started

### Prerequisites

1. **Python 3.8+** with the following packages:
   - `pandas`, `numpy` — Data processing
   - `torch`, `transformers` — LLM & perplexity scoring
   - `detoxify` — Toxicity detection
   - `openai` — DeepSeek API client (compatible with OpenAI SDK)
   - `tqdm` — Progress bars
   - `tenacity` — Retry logic
   - `ftfy` — Text normalization
   - `plotly` — Interactive visualizations

2. **API Keys**:
   - **DeepSeek API Key**: Obtain from DeepSeek; set environment variable:
     ```bash
     export DEEPSEEK_API_KEY="sk-..."
     ```
   - DeepSeek pricing: ~$0.14 per 1M tokens; 50K jokes ≈ $3-5 total cost

3. **GPU (Recommended)**:
   - NVIDIA GPU with CUDA support for faster processing
   - Toxicity filtering: ~2-3 hours on GPU, ~8-10 hours on CPU
   - PPL computation: ~1-2 hours on GPU, ~4-6 hours on CPU
   - CPU-only processing is supported but slow

### Installation

```bash
# Clone repo or navigate to project
cd /uhh-ias-ml

# Install dependencies
pip install -r requirements.txt

# Or install manually:
pip install pandas numpy torch transformers detoxify openai tqdm tenacity ftfy plotly

# Verify installations
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

---

## Configuration & Parameters

### Global Configuration

Most parameters are **hardcoded in scripts** for reproducibility. Key configs:

#### DeepSeek Generation (`run_jokes_async.py`)
```python
API_KEY = os.getenv("DEEPSEEK_API_KEY")           # Must be set
MODEL_NAME = "deepseek-chat"                      # DeepSeek model
TARGET_JOKES_PER_COMBO = 100                      # Production: 100, Test: 5
MAX_CONCURRENCY = 50                              # Parallel API calls
BATCH_SIZE = 10                                   # Jokes per API request
TEST_MODE = False                                 # Set True for dry run
```

#### Text Cleaning (`build_llm_jokes_top_5000_topics_3.py`)
```python
MIN_LEN = 10                                      # Min joke length (chars)
MAX_LEN = 1000                                    # Max joke length (chars)
MAX_LINES = 4                                     # Max lines per joke
```

#### Toxicity Filtering (`filter_llm_jokes_toxicity.py`)
```python
THRESHOLD = 0.1                                   # Detoxify score threshold
DETOX_MODEL = "original"                          # or "multilingual"
LLAMAGUARD_MODEL = "meta-llama/LlamaGuard-1b"    # HuggingFace model
BATCH_SIZE = 8                                    # Batch size for toxicity models
DEVICE = "auto"                                   # auto | cuda | cpu
```

#### Perplexity Computation (`compute_ppl_llm_jokes.py`)
```python
MODEL_NAME = "gpt2"                               # Language model
BATCH_SIZE = 32                                   # Batch size for inference
MAX_LENGTH = 512                                  # Max tokens
DEVICE = "auto"                                   # auto | cuda:0 | cpu
USE_AMP = True                                    # Automatic mixed precision
```

#### Dataset Balancing (`sample_100_per_topic_combo.py`)
```python
SAMPLE_SIZE = 100                                 # Target jokes per combo
TARGET_TOTAL = 500000                             # Target total jokes
SEED = 42                                         # Random seed
```

### Modifying Parameters

To change parameters:
1. Edit the hardcoded values in script files
2. Or override via command-line arguments (if script supports `argparse`)

Example:
```bash
# Override via CLI (if script supports it)
python analyzers/compute_ppl_llm_jokes.py \
  --data-csv /path/to/input.csv \
  --output /path/to/output.csv \
  --batch-size 64 \
  --device cuda:0
```

---

## Data Formats & Schemas

### Key Output Files

#### `top_5000_topic_combos.csv`
| Column | Type | Example | Notes |
|--------|------|---------|-------|
| `topic` | string | "girlfriend" | Topic combination identifier |
| `count` | int | 200 | Frequency in source dataset |

#### `llm_jokes_top_5000_topics_3.csv`
| Column | Type | Example | Notes |
|--------|------|---------|-------|
| `topic` | string | "girlfriend" | Topic combination |
| `joke_raw` | string | "Why did the girlfriend..." | Original LLM output |
| `joke_cleaned` | string | "Why did the girlfriend..." | After cleaning pipeline |
| `stable_id` | string | "a7f3e9c2..." | MD5 hash for deduplication |

#### `llm_jokes_top_5000_topics_3_detox_safe.csv` (Final Clean Dataset)
| Column | Type | Range | Notes |
|--------|------|-------|-------|
| `topic` | string | — | Topic combination |
| `joke_cleaned` | string | — | Cleaned joke text |
| `toxicity` | float | [0, 1] | Detoxify score |
| `severe_toxicity` | float | [0, 1] | Detoxify dimension |
| `obscene` | float | [0, 1] | Detoxify dimension |
| `threat` | float | [0, 1] | Detoxify dimension |
| `insult` | float | [0, 1] | Detoxify dimension |
| `identity_attack` | float | [0, 1] | Detoxify dimension |
| `detox_flag` | bool | 0 or 1 | 1 = safe, 0 = toxic |
| `profanity_flag` | bool | 0 or 1 | Profanity check result |
| `llamaguard_flag` | bool | 0 or 1 | LlamaGuard policy check |
| `llamaguard_output` | string | — | LlamaGuard classification |

#### `llm_jokes_top_5000_topics_3_detox_safe_ppl.csv` (PPL-Scored)
| Column | Type | Range | Notes |
|--------|------|-------|-------|
| `topic` | string | — | Topic combination |
| `joke_cleaned` | string | — | Cleaned joke text |
| `ppl` | float | (0, ∞) | GPT-2 perplexity score |
| **[+ all toxicity columns from above]** | — | — | — |

#### `llm_jokes_top_5000_topics_3_detox_safe_ppl_samples.json`
```json
{
  "percentile_samples": {
    "0-10": [
      {"topic": "girlfriend", "joke": "...", "ppl": 25.3},
      ...
    ],
    "10-20": [...],
    ...
    "90-100": [...]
  },
  "statistics": {
    "mean_ppl": 52.4,
    "median_ppl": 48.1,
    "std_ppl": 15.2,
    ...
  }
}
```

---

## Running the Pipeline

### Full Pipeline (Sequential)

Execute all stages in order:

```bash
cd /uhh-ias-ml/data/joke_generation/llm_jokes/experiment3.0

# Stage 1: Extract topic combos (fast, seconds)
python analyzers/extract_top_5000_topic_combos.py

# Stage 2: Generate jokes via DeepSeek (slow, 3-4 hours; requires API key)
export DEEPSEEK_API_KEY="sk-..."
python generators/run_jokes_async.py

# Stage 3: Clean jokes (medium, 10-30 mins)
python analyzers/build_llm_jokes_top_5000_topics_3.py

# Stage 4: Filter toxicity (slow, 2-3 hours on GPU)
python analyzers/filter_llm_jokes_toxicity.py

# Stage 5: Remove repeated chars (fast, minutes)
python analyzers/filter_repeated_char_jokes.py

# Stage 6: Compute perplexity (slow, 1-2 hours on GPU)
python analyzers/compute_ppl_llm_jokes.py

# Stage 7: Analyze & visualize (fast, minutes)
python analyzers/extract_ppl_samples_llm_jokes.py
python analyzers/plot_ppl_hist_llm_jokes.py

# Stage 8: Balance dataset (fast, minutes)
python analyzers/sample_100_per_topic_combo.py

# Optional: Backfill missing combos (1-2 hours)
python analyzers/backfill_jokes_under_100_async.py
```

### Individual Stage Execution

Run specific stages as needed:

```bash
# Retoxicity filtering with custom threshold
python analyzers/filter_llm_jokes_toxicity.py \
  --data-csv outputs/llm_jokes_top_5000_topics_3.csv \
  --threshold 0.15 \
  --batch-size 16

# Recompute perplexity with different batch size
python analyzers/compute_ppl_llm_jokes.py \
  --batch-size 64 \
  --device cuda:0

# Resample with different seed
python analyzers/sample_100_per_topic_combo.py \
  --sample-size 100 \
  --seed 123
```

### Test Mode (Dry Run)

Before running full pipeline, test with small dataset:

```bash
# Enable test mode in run_jokes_async.py
sed -i 's/TEST_MODE = False/TEST_MODE = True/' generators/run_jokes_async.py

# Run generator (processes 3 combos only)
python generators/run_jokes_async.py

# Check output
head -5 outputs/deepseek_jokes_TEST.csv

# Disable test mode when ready for production
sed -i 's/TEST_MODE = True/TEST_MODE = False/' generators/run_jokes_async.py
```


---

## Outputs & Results

### Dataset Files

| File | Size (Approx) | Rows | Purpose |
|------|---------------|------|---------|
| `outputs/deepseek_jokes.csv` | 50-80 MB | ~50K | Raw LLM output |
| `outputs/llm_jokes_top_5000_topics_3.csv` | 40-70 MB | ~45-48K | Cleaned jokes |
| `outputs/llm_jokes_top_5000_topics_3_detox.csv` | 80-120 MB | ~45-48K | +toxicity scores |
| `outputs/llm_jokes_top_5000_topics_3_detox_safe.csv` | 60-100 MB | ~35-42K | Safe jokes only (~80-85% of cleaned) |
| `outputs/llm_jokes_top_5000_topics_3_detox_safe_no_repeats.csv` | 55-95 MB | ~33-40K | After dedup (~95-98% of safe) |
| `outputs/llm_jokes_top_5000_topics_3_detox_safe_no_repeats_sampled100.csv` | 600-800 MB | ~500K | Balanced final dataset (100/combo) |

### Statistical Outputs

| File | Type | Contents | Purpose |
|------|------|----------|---------|
| `stats/top_5000_topic_combos.csv` | CSV | 5,000 topic combos + counts | Pipeline input reference |
| `stats/llm_jokes_top_5000_topics_3_detox_safe_ppl.csv` | CSV | All jokes + PPL scores | Analysis basis |
| `stats/llm_jokes_top_5000_topics_3_detox_safe_ppl_samples.json` | JSON | 10 percentile samples + stats | Qualitative analysis |
| `stats/removed_repeated_char_joke.csv` | CSV | Filtered-out joke IDs | Audit trail |

### Visualization Outputs

| File | Type | Format | Purpose |
|------|------|--------|---------|
| `plots/ppl_hist.html` | Interactive plot | Plotly | Explore PPL distribution |
| `plots/ppl_hist.png` | Static image | PNG | Publication/sharing |

### Quality Metrics

After full pipeline:

```
Dataset Statistics:
├── Total jokes generated: 50,000
├── After cleaning: 45,000 (90%)
├── After toxicity filter: 39,000 (87% of cleaned)
├── After dedup repeats: 38,000 (97% of safe)
├── Final balanced: 500,000 (100 per combo × 5,000)
│
├── Toxicity Filtering Breakdown:
│   ├── Passed Detoxify: 39,500 (88%)
│   ├── Failed Detoxify: 5,500 (12%)
│   └── LlamaGuard flagged: 1,200 (2.7% overlap)
│
├── Perplexity Statistics:
│   ├── Mean PPL: 52.3
│   ├── Median PPL: 48.1
│   ├── Std Dev: 15.2
│   ├── Min PPL: 15.3
│   └── Max PPL: 156.8
│
└── Topic Distribution:
    ├── Combos with ≥100 jokes: 4,800 (96%)
    ├── Combos with 50-99 jokes: 150 (3%)
    ├── Combos with <50 jokes: 50 (1%)
```

---

## Troubleshooting

### Issue 1: DeepSeek API Rate Limits

**Symptom**: Generation stalls or fails after generating partial data

**Solution**:
```bash
# Check API key is set
echo $DEEPSEEK_API_KEY

# Reduce concurrency (safer but slower)
# Edit generators/run_jokes_async.py:
MAX_CONCURRENCY = 20  # reduced from 50

# Retry from checkpoint (script resumes from existing output)
python generators/run_jokes_async.py  # Will append to existing CSV
```

### Issue 2: Out of Memory (GPU or CPU)

**Symptom**: `RuntimeError: CUDA out of memory` or `MemoryError`

**Solutions**:
```bash
# Reduce batch size
python analyzers/filter_llm_jokes_toxicity.py --batch-size 4
python analyzers/compute_ppl_llm_jokes.py --batch-size 16

# Use CPU instead of GPU
python analyzers/compute_ppl_llm_jokes.py --device cpu

# Process in chunks (split CSV, process separately, merge)
```

### Issue 3: Missing Input Files

**Symptom**: `FileNotFoundError` when running analyzer

**Solution**:
```bash
# Verify input files exist
ls -lh stats/top_5000_topic_combos.csv
ls -lh outputs/deepseek_jokes.csv

# If missing, run previous stage:
python generators/run_jokes_async.py  # Generates deepseek_jokes.csv
```

### Issue 4: Duplicate Column Names in Output

**Symptom**: CSV has multiple "joke" or "topic" columns

**Solution**: This is expected; columns are added cumulatively:
- `joke_raw` → `joke_cleaned` → toxicity columns → PPL column

To see columns:
```python
import pandas as pd
df = pd.read_csv("outputs/llm_jokes_top_5000_topics_3_detox_safe_ppl.csv")
print(df.columns.tolist())
```

### Issue 5: LlamaGuard Model Download Fails

**Symptom**: `ConnectionError` or timeout during model download

**Solution**:
```bash
# Pre-download model manually
python -c "from transformers import AutoModel; AutoModel.from_pretrained('meta-llama/LlamaGuard-1b')"

# Or skip LlamaGuard (edit filter_llm_jokes_toxicity.py to set USE_LLAMAGUARD=False)
```

### Issue 6: Toxicity Scores All 0.0 or 1.0

**Symptom**: All jokes marked toxic/safe with extreme scores

**Solution**: Check model loading:
```python
from detoxify import Detoxify
detox = Detoxify("original", device="cuda")  # Test model
scores = detox.predict("sample text")
print(scores)  # Should output dict with 6 dimensions
```

---

## Reproducibility

### Ensuring Reproducible Runs

1. **Fix Random Seeds**:
   - `sample_100_per_topic_combo.py`: Uses `seed=42` by default
   - To reproduce: keep seed constant

2. **Freeze Model Versions**:
   - DeepSeek model: `deepseek-chat` (latest; ensure consistent API version)
   - GPT-2: `gpt2` from Hugging Face Transformers (pinned via `transformers==X.Y.Z`)
   - Detoxify: `original` model variant (consistent across runs)
   - LlamaGuard: `meta-llama/LlamaGuard-1b`

3. **Document Configuration**:
   - Save config snapshots:
     ```bash
     cp generators/run_jokes_async.py configs/gen_config_run_$(date +%s).py
     ```

4. **Log Processing Details**:
   - Batch size, hardware, API version, model versions used
   - Example log snippet:
     ```
     [2026-01-22 10:30:15] Starting experiment3.0 pipeline
     GPU: NVIDIA A100 (40GB)
     CUDA Version: 12.1
     PyTorch: 2.1.0
     Transformers: 4.35.2
     Detoxify: 0.5.1
     Config: batch_size=32, device=cuda:0, threshold=0.1
     ```

5. **Version Control**:
   - Keep scripts in git with commit hashes
   - Document exact Python environment:
     ```bash
     pip freeze > environment_snapshot.txt
     ```

### Running a Clean Reproduction

```bash
# From scratch with clean outputs
cd /uhh-ias-ml/data/joke_generation/llm_jokes/experiment3.0

# Remove old outputs (careful!)
rm -rf outputs/*.csv stats/*.csv stats/*.json plots/*

# Run full pipeline
bash run_full_pipeline.sh  # (if this script exists)
# Or manually run stages 1-8 as described above
```

---

## Advanced Usage

### Modifying Toxicity Thresholds

To change what's considered "toxic":

```bash
# Lower threshold (stricter, fewer safe jokes)
python analyzers/filter_llm_jokes_toxicity.py --threshold 0.05

# Higher threshold (more permissive, more safe jokes)
python analyzers/filter_llm_jokes_toxicity.py --threshold 0.2
```

### Custom Sampling Strategy

To change jokes per combo:

```bash
# Get 50 jokes per combo instead of 100
python analyzers/sample_100_per_topic_combo.py \
  --sample-size 50 \
  --target-total 250000  # Adjust target proportionally
```

### Backfilling Missing Combos

If some topic combos have fewer than target jokes:

```bash
python analyzers/backfill_jokes_under_100_async.py \
  --input outputs/llm_jokes_top_5000_topics_3_detox_safe_no_repeats.csv \
  --output outputs/llm_jokes_backfilled.csv \
  --target-per-combo 100
```

### Analyzing Toxicity Distribution

```python
import pandas as pd
df = pd.read_csv("outputs/llm_jokes_top_5000_topics_3_detox.csv")

# Summary statistics
print(df[["toxicity", "severe_toxicity", "obscene"]].describe())

# Percentage by dimension
for col in ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack"]:
    above_threshold = (df[col] >= 0.1).sum()
    pct = 100 * above_threshold / len(df)
    print(f"{col}: {pct:.1f}% above 0.1")
```

### Filtering by Perplexity Range

```python
import pandas as pd
df = pd.read_csv("stats/llm_jokes_top_5000_topics_3_detox_safe_ppl.csv")

# Get only natural-sounding jokes (PPL < 50)
natural = df[df["ppl"] < 50]
print(f"Natural jokes: {len(natural)} ({100*len(natural)/len(df):.1f}%)")

# Save subset
natural.to_csv("outputs/natural_jokes_only.csv", index=False)
```

---

## Citation & References

### Related Papers
- Toxicity Detection: Hosseini et al. (2022) - Detoxify Models
- Perplexity-based Quality Scoring: Salemi et al. (2022) - Using LM Perplexity
- Safety Guardrails: Inan et al. (2023) - LlamaGuard Safety Models

### Input Data & Downstream
- **Input Source**: `data_source/clean_jokes_clean_topics_3.csv` (preprocessed jokes with topic annotations)
- **Experiment 4.0**: Further dataset refinement and model training

### Contact & Questions
For issues or questions about this experiment:
- Check troubleshooting section above
- Review script docstrings for detailed parameter info
- Consult parent project README: `../../README.md`

---

## Summary

**experiment3.0** is a production-grade data pipeline for generating, cleaning, and analyzing large-scale LLM-generated joke datasets. It combines:
- **LLM Generation** (DeepSeek API): 50K jokes from 5K topic combinations
- **Text Cleaning**: Removes formatting, emojis, URLs, anomalies
- **Toxicity Filtering**: 3-model ensemble (Detoxify, LlamaGuard, profanity)
- **Quality Scoring**: GPT-2 perplexity for naturalness
- **Analysis & Visualization**: Distribution analysis and plotting
- **Dataset Balancing**: Ensures equitable topic representation

The final output is a ~500K clean, safe, balanced joke dataset ready for model training.

---

**Last Updated**: January 22, 2026  
**Experiment Status**: Production (Used for downstream model training)  
**Maintainer**: Jingfan Branch / IAS ML Team
