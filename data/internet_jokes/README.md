# Internet Jokes Dataset

> A comprehensive multi-source English joke corpus with cleaning, analysis, and visualization

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Directory Structure](#directory-structure)
- [Dataset Information](#dataset-information)
- [Scripts & Tools](#scripts--tools)
- [Analysis & Visualizations](#analysis--visualizations)
- [Data Schema](#data-schema)
- [Usage Examples](#usage-examples)
- [Statistics & Insights](#statistics--insights)
- [Integration Guide](#integration-guide)

---

## Overview

**Internet Jokes** is a large-scale, multi-source English joke corpus containing **724,507 unique jokes** after comprehensive cleaning and deduplication. This dataset serves as the baseline foundation for joke generation experiments and humor research.

### Key Features

- ✅ **724,507 unique jokes** from 5 public datasets
- ✅ **Comprehensive cleaning** (Unicode normalization, deduplication, filtering)
- ✅ **Multi-source diversity** (one-liners, Reddit jokes, dad jokes, etc.)
- ✅ **Rich metadata** (source attribution, cleaned text, topic tags)
- ✅ **Statistical analysis** with 20+ visualizations
- ✅ **Ready-to-use scripts** for preprocessing and analysis

### Dataset Summary

| Metric | Value |
|--------|-------|
| **Final Corpus Size** | 724,507 unique jokes |
| **Raw Collection** | 1,305,211 items (5 sources) |
| **Retention Rate** | 55.51% |
| **Deduplication Loss** | 540,855 rows (41.43%) |
| **Avg Joke Length** | 135 characters (29 words) |
| **Sources** | ShortJokes, rJokesData, AmirkidJokes, HumorDetection200k, Dadjokes |

### Research Context

This dataset represents **Phase 1** of an iterative study design:

- **Phase 1 (Current):** Large internet-sourced joke corpus (~725K jokes) for establishing baselines
- **Phase 2 (Future):** Cleaner synthetic LLM-generated dataset to study data quality effects

---

## Quick Start

### Install Dependencies

```bash
pip install pandas jupyter plotly spacy numpy tqdm
python -m spacy download en_core_web_sm
```

### Load & Explore Data

```python
import pandas as pd

# Load the main dataset
df = pd.read_csv('data_source/final_combined_jokes.csv')
print(f"Loaded {len(df):,} jokes")

# View sample
print(df.head())
print(df.columns)
```

### Run Scripts

```bash
# Navigate to directory
cd /ltstorage/home/4xin/uhh-ias-ml/data/internet_jokes

# Launch analysis notebook
cd plot_data
jupyter notebook plot_data.ipynb
```

---

## Directory Structure

```
internet_jokes/
├── README.md                                    # This documentation
│
├── data_source/                                 # Dataset files
│   ├── deepcontractor_200k_dataset.csv         # DeepContractor base dataset
│   ├── deepcontractor_200k_humor_true.csv      # Humor-only subset
│   └── final_combined_jokes.csv                # ⭐ Main dataset (724K jokes)
│
├── preprocessing_stats/                        # Processing logs & metrics
│   ├── human_merge_source_stats.csv            # Source merging statistics
│   ├── human_preprocess_source_drop_stats.csv  # Data filtering statistics
│   └── human_preprocess_stats.json             # Detailed processing metrics
│
├── plot_data/                                   # Analysis & visualization
│   ├── plot_data.ipynb                         # Main analysis notebook
│   ├── PLOT_DATA.md                            # Notebook documentation
│   └── outputs/                                # Generated visualizations (~20 files)
│       ├── *.csv                               # Statistical summaries
│       ├── *.html                              # Interactive Plotly charts
│       └── *.png                               # Static PNG exports
│
└── preprocessing.py                            # Data cleaning pipeline
```

**Key Files:**
- **Main Dataset:** `data_source/final_combined_jokes.csv` (recommended starting point)
- **Analysis Notebook:** `plot_data/plot_data.ipynb` (statistical analysis & visualizations)
- **Scripts:** `preprocessing.py` (data processing tool)

---

## Dataset Information

### 1. Data Sources

This corpus combines 5 publicly available joke datasets:

| Source | Original Size | Final Count | Retention | Description |
|--------|---------------|-------------|-----------|-------------|
| **ShortJokes** (Kaggle) | 231,657 | 230,137 | 99.34% | Short one-line jokes |
| **rJokesData** (GitHub) | 345,965 | 276,863 | 80.03% | Reddit r/Jokes collection |
| **AmirkidJokes** (HuggingFace) | 574,189 | 157,857 | 27.51% | General humor texts |
| **HumorDetection200k** (Kaggle) | 100,000 | 9,135 | 9.14% | Humor detection dataset |
| **Dadjokes** (Reddit/Kaggle) | 53,400 | 50,515 | 94.60% | Dad jokes (Q&A format) |
| **TOTAL** | **1,305,211** | **724,507** | **55.51%** | **All sources combined** |

**Note:** High-quality sources (ShortJokes, Dadjokes) have 94-99% retention, while others required heavy deduplication.

### 2. Data Cleaning Pipeline

A comprehensive 7-step cleaning process reduces the raw collection from 1.3M to 724K unique jokes:

#### Cleaning Steps

1. **Unicode Normalization** – Standardize character encoding (NFKC)
2. **Encoding Artifact Repair** – Fix common encoding issues (e.g., `Ã©` → `é`)
3. **Structural Noise Removal** – Remove HTML/Markdown, URLs, emails, emojis, control characters
4. **Punctuation & Whitespace Normalization** – Standardize formatting
5. **Length Filtering** – Remove jokes < 10 or > 1000 characters
6. **Symbolic Content Filtering** – Remove entries with >40% digits/symbols
7. **Exact Deduplication** – Remove duplicates using normalized text as key

#### Cleaning Results

| Filter Step | Dropped Rows | % of Input |
|-------------|--------------|-----------|
| Empty after cleaning | 18 | 0.00% |
| Highly symbolic / near-nontext | 38 | 0.00% |
| Too short (< 10 chars) | 150 | 0.01% |
| Too long (> 1000 chars) | 39,643 | 3.04% |
| **Deduplication** | **540,855** | **41.43%** |
| **Final Cleaned Corpus** | **724,507** | **55.51%** ✅ |

**Key Insight:** Deduplication is the largest reduction factor (41.43%), indicating heavy cross-source duplication.

### 3. Final Dataset Characteristics

**Size & Distribution:**
- **Total:** 724,507 unique jokes
- **Average length:** 135 characters (29 words)
- **Length range:** 10-1000 characters (enforced)

**Source Breakdown:**
- rJokesData: 276,863 jokes (38.3%)
- ShortJokes: 230,137 jokes (31.8%)
- AmirkidJokes: 157,857 jokes (21.8%)
- Dadjokes: 50,515 jokes (7.0%)
- HumorDetection200k: 9,135 jokes (1.3%)

**Quality Guarantees:**
- ✓ Unicode normalized (NFKC)
- ✓ No HTML/Markdown artifacts
- ✓ No URLs, emails, emojis
- ✓ No duplicates
- ✓ No overly short/long entries
- ✓ 100% coverage on critical fields (id, text, source)

### 4. Available Data Files

#### `data_source/final_combined_jokes.csv` ⭐ **[Main Dataset]**

**Size:** 724,507 unique jokes

**Columns:**
- `id` / `joke_id` – Unique record identifier
- `joke_cleaned` – Cleaned joke text (10-1000 chars)
- `source` – Original source dataset name
- *(varies)* – Source-specific metadata

**Use Cases:**
- Model training for joke generation
- Humor classification experiments
- Topic modeling and analysis
- Baseline dataset for experiments

#### `data_source/deepcontractor_200k_*.csv` (Optional)

Original HumorDetection200k dataset files (may be used for humor detection tasks).

---

## Scripts & Tools

### 1. preprocessing.py

**Purpose:** Automated data cleaning and preprocessing pipeline

**Features:**
- Data loading and validation
- Text cleaning and normalization
- Duplicate detection and removal
- Statistics and quality reporting

**Usage:**

```bash
cd /ltstorage/home/4xin/uhh-ias-ml/data/internet_jokes
python preprocessing.py
```

**Outputs:**
- Processed CSV files → `data_source/`
- Statistics → `preprocessing_stats/`

---

### 2. plot_data/plot_data.ipynb

**Purpose:** Comprehensive statistical analysis and visualization notebook

**Features:**
- 📊 7 analysis sections with 20+ visualizations
- 📈 Distribution plots (joke length, tag frequency, POS tags)
- 🏷️ Topic analysis (top tags, tag combinations)
- 🔢 Interactive Plotly charts + static PNG exports

**Technology Stack:**
- Pandas (data processing)
- spaCy (NLP/POS tagging)
- Plotly (interactive visualizations)
- NumPy (statistics)

**How to Run:**

```bash
# Install dependencies
pip install pandas jupyter plotly spacy numpy
python -m spacy download en_core_web_sm

# Navigate and launch
cd /ltstorage/home/4xin/uhh-ias-ml/data/internet_jokes/plot_data
jupyter notebook plot_data.ipynb

# In Jupyter: Kernel → Restart & Run All
```

**Runtime:** 5-15 minutes (first run: +2-5 min for spaCy download)

**Outputs:** All visualizations saved to `plot_data/outputs/` (CSV + HTML + PNG)

---

## Analysis & Visualizations

The `plot_data.ipynb` notebook generates 7 comprehensive analysis sections:

### Analysis Sections

| # | Analysis | Output Files |
|---|----------|--------------|
| 1 | **POS Tag Distribution** | `tag_type_count.*` |
| 2 | **Joke Length Distribution** | `joke_length_chars.*` |
| 3 | **Tags per Joke Distribution** | `tag_count_distribution.*` |
| 4 | **Top N Most Frequent Tags** | `tag_frequencies_top{30,100,500,1000}.*` |
| 5 | **Tag Frequency Distribution (Log-binned)** | `tag_frequency_distribution_log.*` |
| 6 | **All-Nouns Tag Distribution** | `tag_count_distribution_all_nouns.*` |
| 7 | **Tag Count vs. Joke Length** | `tag_count_vs_length_with_std_clean.*` |

Each analysis produces **3 file types:**
- `.csv` – Statistical summaries
- `.html` – Interactive Plotly charts
- `.png` – Static PNG exports

### Key Visualizations

#### 1. POS Tag Distribution

Shows distribution of part-of-speech categories in joke topics:
- NOUN: ~50%
- PROPN: ~20%
- VERB: ~15%
- ADJ: ~10%
- MISC: ~5%

#### 2. Joke Length Distribution

Histogram showing joke length in characters:
- Mean: ~135 characters
- Median: ~85 characters
- Most jokes: 50-150 characters
- Power-law distribution (long tail)

#### 3. Tags per Joke

Distribution of topic tags assigned per joke:
- 1 tag: ~6%
- 2 tags: ~15%
- 3 tags: ~79% (standard "top-3 nouns" extraction)

#### 4. Top Tags Frequency

Most frequent topic tags across the corpus:
- Logarithmic scale shows power-law distribution
- Top tags: humor, comedy, animals, wordplay, etc.
- Multiple n-values (30, 100, 500, 1000) for different analysis depths

#### 5. Tag Frequency Distribution

Long-tail analysis showing:
- Thousands of rare tags (appear 1-10 times)
- Few common tags (1000+ appearances)
- Classic power-law/Zipf distribution

#### 6. All-Nouns Tag Distribution

Shows distribution when extracting ALL nouns (not just top-3):
- 1-5 nouns: ~20%
- 6-10 nouns: ~35%
- 11-15 nouns: ~30%
- 16+ nouns: ~15%

#### 7. Tag Count vs. Length

Correlation analysis revealing:
- More tags → longer jokes (correlation ~0.6)
- Standard deviation increases with tag count
- Most jokes: 6-10 tags, 80-120 characters

### Viewing Outputs

```bash
# Open interactive HTML plots in browser
open plot_data/outputs/joke_length_chars.html

# Or view static PNG exports
open plot_data/outputs/joke_length_chars.png

# All files are in plot_data/outputs/
ls plot_data/outputs/
```

---

## Data Schema

### Core Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `id` / `joke_id` | Integer | Unique record identifier | `12345` |
| `joke_cleaned` | String | Cleaned joke text (10-1000 chars) | "Why did the chicken cross..." |
| `source` | String | Original source dataset | "ShortJokes" |
| `topic_all_nouns` | String | Comma-separated noun topics (if generated) | "chicken, road" |

### Data Types & Encoding

| Column | Type | Encoding | Nullable |
|--------|------|----------|----------|
| `id` | Int64 | N/A | No |
| `joke_cleaned` | String | UTF-8 | No |
| `source` | String | UTF-8 | No |
| `topic_all_nouns` | String | UTF-8 | Yes (if not generated) |

### Source Values

| Source ID | Dataset Name | Count |
|-----------|--------------|-------|
| `ShortJokes` | Kaggle Short Jokes | 230,137 |
| `rJokesData` | Reddit r/Jokes | 276,863 |
| `AmirkidJokes` | HuggingFace Amirkid | 157,857 |
| `HumorDetection200k` | Kaggle Humor Detection | 9,135 |
| `Dadjokes` | Reddit Dad Jokes | 50,515 |

---

## Usage Examples

### Loading Data

```python
import pandas as pd

# Load main dataset
df = pd.read_csv('data_source/final_combined_jokes.csv')
print(f"Total jokes: {len(df):,}")

# Basic statistics
print(df.info())
print(df.describe())

# View samples
print(df.head(10))

# Analyze lengths
df['length'] = df['joke_cleaned'].str.len()
print(f"Mean length: {df['length'].mean():.1f} characters")
```

### Filtering by Source

```python
# Get jokes from specific source
reddit_jokes = df[df['source'] == 'rJokesData']
print(f"Reddit jokes: {len(reddit_jokes):,}")

# Get short jokes only
short_jokes = df[df['joke_cleaned'].str.len() < 100]
print(f"Short jokes (< 100 chars): {len(short_jokes):,}")
```

### Analyzing Topics

```python
# Load dataset with noun topics (if generated)
df_topics = pd.read_csv('data_source/final_clean_jokes_with_all_nouns.csv')

# Count jokes with specific topic
chicken_jokes = df_topics[df_topics['topic_all_nouns'].str.contains('chicken', na=False)]
print(f"Chicken jokes: {len(chicken_jokes):,}")

# Most common topics
all_topics = ','.join(df_topics['topic_all_nouns'].dropna()).split(',')
topic_counts = pd.Series(all_topics).value_counts()
print(topic_counts.head(20))
```

### Training Split

```python
from sklearn.model_selection import train_test_split

# Split into train/val/test
train, temp = train_test_split(df, test_size=0.2, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

print(f"Train: {len(train):,}, Val: {len(val):,}, Test: {len(test):,}")

# Save splits
train.to_csv('train.csv', index=False)
val.to_csv('val.csv', index=False)
test.to_csv('test.csv', index=False)
```

---

## Statistics & Insights

### Dataset Composition

| Metric | Value |
|--------|-------|
| Raw Merged Data | 1,305,211 items |
| Final Cleaned Corpus | 724,507 unique jokes |
| Overall Retention Rate | 55.51% |
| Deduplication Loss | 540,855 jokes (41.43%) |
| Length Filtering Loss | 39,643 jokes (3.04%) |

### Source Quality Analysis

**Best Quality (Minimal Cleaning Needed):**
- ShortJokes: 99.34% retention
- Dadjokes: 94.60% retention

**Moderate Quality:**
- rJokesData: 80.03% retention

**Heavy Deduplication Required:**
- AmirkidJokes: 27.51% retention (heavy cross-source duplication)
- HumorDetection200k: 9.14% retention (designed for classification, not generation)

### Text Characteristics

**Length Distribution:**
- Min: 10 characters (enforced)
- Max: 1000 characters (enforced)
- Mean: 135 characters (29 words)
- Median: ~85 characters
- Mode: 50-100 character range

**Stylistic Coverage:**
- Short one-liners (ShortJokes)
- Reddit-style jokes (rJokesData)
- General humor texts (AmirkidJokes)
- Dad jokes with Q&A format (Dadjokes)
- Humor detection texts (HumorDetection200k)

---

## Integration Guide

### Use Cases

**1. Humor Generation**
- Pre-training language models on joke data
- Topic-conditioned joke generation
- Style transfer across joke types

**2. Humor Understanding**
- Humor detection/classification
- Topic extraction and analysis
- Semantic humor analysis

**3. Data Quality Research**
- Studying effects of data cleaning on model performance
- Comparing internet-sourced vs. LLM-generated data
- Source quality impact on generation quality

**4. Benchmark & Baselines**
- Establishing perplexity baselines
- Evaluating joke generation models
- Comparing with Phase 2 LLM-generated dataset

### Downstream Integration

**As Training Data:**

```python
from transformers import AutoTokenizer
import pandas as pd

# Load dataset
df = pd.read_csv('data_source/final_combined_jokes.csv')
texts = df['joke_cleaned'].tolist()

# Tokenize for training
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenized = tokenizer(texts, truncation=True, padding=True)
```

**For Evaluation:**

```python
# Load test set
test_df = pd.read_csv('test.csv')

# Generate jokes and compare
# ... (your generation code)

# Evaluate with metrics
from eval.metrics import calculate_diversity, calculate_coherence
diversity = calculate_diversity(generated_jokes)
coherence = calculate_coherence(generated_jokes)
```

### Related Files in Repository

**Training Scripts:**
- `build_model/train.py` – Main training script
- `build_model/model/decoder_only.py` – Model architecture

**Evaluation Scripts:**
- `eval/run_eval.py` – Evaluation runner
- `eval/metrics.py` – Evaluation metrics
- `eval/data/final_clean_jokes.csv` – Evaluation reference data

**Experiment Runs:**
- `data/runs/` – Saved model checkpoints and logs

---

## File Reference

### Key Scripts Summary

| Script | Purpose | Command |
|--------|---------|---------|
| `preprocessing.py` | Data cleaning pipeline | `python preprocessing.py` |
| `plot_data/plot_data.ipynb` | Statistical analysis | `jupyter notebook plot_data.ipynb` |

### Output Files Summary

| Location | File Type | Contents |
|----------|-----------|----------|
| `data_source/` | `.csv` | Dataset files |
| `preprocessing_stats/` | `.csv`, `.json` | Processing statistics |
| `plot_data/outputs/` | `.csv`, `.html`, `.png` | Visualizations (~30 files) |

### Important Paths

```bash
# Main dataset
data_source/final_combined_jokes.csv

# Visualizations
plot_data/outputs/
```

---

## Summary

### Quick Facts

- 📊 **724,507 unique jokes** from 5 public datasets
- 🧹 **41.43% deduplication** removed 540K duplicate jokes
- 📏 **Average 135 characters** (29 words) per joke
- 🎯 **5 diverse sources** (one-liners, Reddit, dad jokes, etc.)
- 📈 **20+ visualizations** for comprehensive analysis
- ⚙️ **2 ready-to-use scripts** (preprocessing, analysis)

### Getting Started Checklist

1. ✓ Install dependencies: `pip install pandas jupyter plotly spacy numpy tqdm`
2. ✓ Download spaCy model: `python -m spacy download en_core_web_sm`
3. ✓ Load data: `pd.read_csv('data_source/final_combined_jokes.csv')`
4. ✓ (Optional) Run analysis: `jupyter notebook plot_data/plot_data.ipynb`

### Documentation Status

- ✅ **Complete dataset documentation**
- ✅ **Script usage guides**
- ✅ **Analysis methodology**
- ✅ **Integration examples**
- ✅ **Ready for research and experimentation**

---

**Last Updated:** March 7, 2026  
**Status:** ✅ Production Ready  
**Version:** Phase 1 (Internet Dataset)  
**Contact:** See repository for maintainer information

---

*For more information on the LLM-generated dataset (Phase 2), see `data/joke_generation/` directory.*
