# Data Directory

## Overview

This directory contains all datasets, processing scripts, and experimental outputs for the **Joker Transformer** project—a study of topic-conditioned joke generation with a decoder-only Transformer trained from scratch.

## Directory Structure

```
data/
├── internet_jokes/          # Internet-sourced joke corpus (Phase 1)
│   ├── data_source/         # Raw and cleaned datasets (724K jokes)
│   ├── preprocessing.py     # Data cleaning pipeline
│   ├── plot_data/          # Statistical analysis & visualizations
│   └── README.md           # Detailed documentation
│
├── joke_generation/         # LLM-generated jokes & combined datasets
│   ├── llm_jokes/          # LLM generation experiments
│   │   ├── experiment1.0/  # Initial LLM joke generation (7 models screened)
│   │   ├── experiment3.0/  # Large-scale generation (500K+ jokes)
│   │   └── experiment4.0/  # Non-topic generation exploration (abandoned)
│   └── combined_jokes/     # Merged internet + LLM datasets
│       └── experiment2.0/  # Combined dataset processing
│
└── runs/                    # Training experiment outputs
    ├── jana_alibi_*/       # Model training runs with different configs
    └── [experiment_name]/  # Checkpoints, logs, and evaluation results
```

## Quick Navigation

### 📊 Datasets

| Dataset | Location | Size | Description |
|---------|----------|------|-------------|
| **Internet Jokes (Baseline)** | `internet_jokes/data_source/` | 724K jokes | Multi-source corpus from 5 public datasets, cleaned & deduplicated |
| **LLM Screening Results** | `joke_generation/llm_jokes/experiment1.0/` | 78K jokes | Preliminary LLM quality comparison (7 models tested) |
| **Combined Dataset** | `joke_generation/combined_jokes/experiment2.0/` | Variable | Merged internet + early LLM jokes with quality filtering |
| **Large-scale LLM Jokes** | `joke_generation/llm_jokes/experiment3.0/` | 500K jokes | DeepSeek-generated topic-conditioned jokes with 2-stage post-filtering |
| **Non-topic Exploration** | `joke_generation/llm_jokes/experiment4.0/` | ~14K jokes | Non-topic generation test (86% duplicates → abandoned) |

### 🔬 Key Experiments

The project follows a **staged dataset strategy** with sequential experiments:

#### **Baseline: Internet Jokes Corpus** (`internet_jokes/`)
- **Goal**: Establish baseline training data from public sources
- **Sources**: 5 datasets (1.3M raw jokes) → 724K after cleaning
  - ShortJokes (Kaggle), rJokesData (GitHub), AmirkidJokes (HuggingFace)
  - HumorDetection200k (Kaggle), Dad Jokes (Reddit/Kaggle)
- **Processing**: Multi-step cleaning pipeline (deduplication: 41.43% dropped)
- **Topic Extraction**: Scoring-based method using spaCy POS tagging
  - Ranks candidates by: position, POS category, length, frequency
  - Yields 1-3 topic keywords per joke
- **Key Finding**: Baseline experiments revealed data quality as major bottleneck

#### **experiment1.0**: LLM Screening & Initial Generation
- **Goal**: Compare LLM generators for topic-conditioned joke generation
- **Models tested**: 7 LLMs (Llama-3.1-8B/70B, Qwen3-30B, DeepSeek-Chat, Gemini variants)
- **Collection**: ~78K jokes across all models
- **Evaluation**: Two anonymous LLM judges (GPT-5.2-Thinking, Gemini-3-Pro)
- **Winner**: DeepSeek-Chat (7.5/10 quality) + Qwen3-30B-A3B (7.5/10)
  - Selected DeepSeek for large-scale generation: no rate limits, cost-effective
- **Methods**: Noun combination analysis, generation approach exploration

#### **experiment2.0**: Combined Dataset Creation
- **Goal**: Merge internet + early LLM-generated jokes, apply quality control
- **Pipeline**:
  - Merge internet jokes + experiment1.0 LLM outputs
  - Toxicity filtering (Detoxify: 6 dimensions, threshold < 0.1)
  - Perplexity-based quality scoring (DistilGPT2)
  - Dataset balancing and preparation for training
- **Outputs**: `clean_good_jokes.csv` (combined & filtered corpus)
- **Use case**: Training data for baseline model experiments

#### **experiment3.0**: Large-scale Synthetic Generation
- **Goal**: Generate large-scale, high-quality topic-conditioned jokes
- **Method**: Two-stage pipeline
  1. **Generation**: DeepSeek-Chat with top 5,000 three-topic combinations
     - 100 jokes per topic combination (10 jokes/call, temperature=1.3)
     - Raw output: 526,308 jokes
  2. **Post-filtering**: Multi-step quality control
     - Basic cleaning (reuse internet pipeline steps)
     - Toxicity filtering (Detoxify, 6 dimensions < 0.1)
     - Repeated-character filtering (removes low-PPL artifacts)
     - Balancing to 100 jokes per topic combination
- **Final corpus**: 500,000 high-quality jokes
- **Quality improvement**: Cleaner data → better training stability & topic adherence

#### **experiment4.0**: Non-topic Generation Exploration (Abandoned)
- **Goal**: Test non-topic LLM joke generation (no input keywords)
- **Method**: DeepSeek-Chat async generation without topic conditioning
- **Result**: 100K generated → 14K after deduplication
- **Key Finding**: 86% duplicate rate (vs. ~5% for topic-conditioned)
- **Conclusion**: Non-topic generation produces repetitive output → abandoned approach
- **Impact**: Confirmed necessity of topic-conditioning for diverse joke generation

### 🛠️ Processing Scripts

All scripts use **relative paths** for cross-platform compatibility. Key processing tools:

- `internet_jokes/preprocessing.py` - Clean and deduplicate internet jokes
- `joke_generation/llm_jokes/experiment3.0/generators/` - Async LLM joke generation
- `joke_generation/llm_jokes/experiment3.0/analyzers/` - Post-processing, filtering, analysis
- `joke_generation/combined_jokes/experiment2.0/utils/` - Dataset merging and quality control

## Data Pipeline

The project follows a **staged dataset strategy** with iterative refinement:

```
┌──────────────────────────────────────────────────────────────────┐
│  BASELINE: Internet Jokes Corpus (internet_jokes/)              │
│  ─────────────────────────────────────────────────────────────   │
│  Raw: 1,305,211 jokes from 5 public datasets                    │
│    ↓ Multi-step cleaning pipeline                               │
│       • Unicode normalization (NFKC)                             │
│       • HTML/Markdown removal, URL/email filtering               │
│       • Length filtering (10-1000 chars)                         │
│       • Symbolic content removal (>40% digits/symbols)           │
│       • Exact deduplication (41.43% dropped)                     │
│    ↓                                                             │
│  Cleaned: 724,507 unique jokes                                   │
│    ↓ Topic extraction (spaCy POS tagging + scoring)             │
│  Result: Baseline corpus with 1-3 topics per joke               │
│                                                                  │
│  Key Finding: Data noise limits generation quality              │
└──────────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────────┐
│  EXPERIMENT 1.0: LLM Screening (llm_jokes/experiment1.0/)       │
│  ────────────────────────────────────────────────────────────    │
│  Goal: Compare LLM generators for joke generation               │
│    • Test 7 models (Llama, Qwen, DeepSeek, Gemini variants)     │
│    • Generate ~78K topic-conditioned jokes                       │
│    • Judge with GPT-5.2-Thinking + Gemini-3-Pro                 │
│    ↓                                                             │
│  Result: DeepSeek-Chat selected (7.5/10, no rate limits)        │
│                                                                  │
│  Additional analysis:                                            │
│    • Extract noun combinations from baseline                     │
│    • Explore generation approaches                               │
│    • Topic vs. non-topic generation comparison                   │
│      (Non-topic: 86% duplicates → use topic-conditioned)         │
└──────────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────────┐
│  EXPERIMENT 2.0: Combined Dataset (combined_jokes/exp2.0/)      │
│  ────────────────────────────────────────────────────────────    │
│  Goal: Merge internet + early LLM jokes with quality control    │
│    ↓ Merge internet jokes + experiment1.0 outputs               │
│    ↓ Toxicity filtering (Detoxify, 6-dim, threshold < 0.1)      │
│    ↓ Perplexity scoring (DistilGPT2) for quality assessment     │
│    ↓ Dataset balancing & formatting                             │
│    ↓                                                             │
│  Output: clean_good_jokes.csv (training-ready dataset)          │
│                                                                  │
│  Use: Baseline model training experiments                       │
└──────────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────────┐
│  EXPERIMENT 3.0: Large-scale Synthetic (llm_jokes/exp3.0/)      │
│  ────────────────────────────────────────────────────────────    │
│  Stage 1: LLM-based Generation                                   │
│    • Select top 5,000 three-topic combinations from baseline     │
│    • DeepSeek-Chat: 100 jokes per combination                    │
│      (10 jokes/call, temperature=1.3, async pipeline)            │
│    • Raw output: 526,308 jokes                                   │
│    ↓                                                             │
│  Stage 2: Post-filtering Pipeline                                │
│    • Basic cleaning  → 522,347 jokes (3,961 dropped)            │
│    • Toxicity filter → 509,087 jokes (13,260 dropped)           │
│    • Repeated-char   → 509,013 jokes (74 dropped)               │
│    • Balance to 100/topic → 500,000 jokes (9,013 dropped)       │
│    ↓                                                             │
│  Final: 500,000 high-quality topic-conditioned jokes            │
│                                                                  │
│  Quality improvement: Cleaner data → better training stability  │
└──────────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────────┐
│  EXPERIMENT 4.0: Non-topic Generation (llm_jokes/exp4.0/)       │
│  ────────────────────────────────────────────────────────────    │
│  Exploratory: Test non-topic generation (no keyword input)       │
│    • DeepSeek-Chat: Generate 100K jokes without topics           │
│    • Aggressive deduplication pipeline                           │
│    ↓                                                             │
│  Result: 14K unique jokes (86% duplicates removed)               │
│                                                                  │
│  ❌ Conclusion: Abandoned approach                               │
│    • 86% duplicate rate (vs. ~5% for topic-conditioned)          │
│    • Non-topic generation too repetitive                         │
│    • Confirms necessity of topic-conditioning                    │
└──────────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────────┐
│  MODEL TRAINING (runs/)                                          │
│  ────────────────────────────────────────────────────────────    │
│  Train decoder-only Transformers on various datasets:            │
│    • Architecture: learned pos emb vs. ALiBi                     │
│    • Sequence: padded vs. packed training                        │
│    • Tokenization: BPE vs. Unigram (4K/10K vocab)                │
│    • Datasets: internet baseline vs. synthetic (quality study)   │
│    ↓                                                             │
│  Outputs: Model checkpoints, logs, evaluation metrics           │
└──────────────────────────────────────────────────────────────────┘
```

## Getting Started

### Quick Start: Using the Data in Python

```python
import pandas as pd
from pathlib import Path

# Define data root (portable across platforms)
DATA_DIR = Path(__file__).parent

# Load internet jokes baseline (724K jokes)
internet_jokes = pd.read_csv(
    DATA_DIR / 'internet_jokes/data_source/final_combined_jokes.csv'
)
print(f"Internet jokes corpus: {len(internet_jokes):,} samples")
print(f"Columns: {list(internet_jokes.columns)}")
print(f"Topic coverage: {internet_jokes['topics'].nunique()} unique topics")

# Load LLM screening results (experiment1.0, ~78K jokes across 7 models)
screening_jokes = pd.read_csv(
    DATA_DIR / 'joke_generation/llm_jokes/experiment1.0/final_llm_jokes/combined_llm_jokes_78k.csv'
)
print(f"\nLLM screening: {len(screening_jokes):,} samples")

# Load combined dataset (experiment2.0, internet + screening)
combined_jokes = pd.read_csv(
    DATA_DIR / 'joke_generation/combined_jokes/experiment2.0/processed_data/combined_internet_llm_jokes.csv'
)
print(f"\nCombined dataset: {len(combined_jokes):,} samples")

# Load large-scale synthetic jokes (experiment3.0, 500K DeepSeek-Chat)
synthetic_jokes = pd.read_csv(
    DATA_DIR / 'joke_generation/llm_jokes/experiment3.0/llm_jokes_final/final_llm_jokes_500k.csv'
)
print(f"\nLarge-scale synthetic: {len(synthetic_jokes):,} samples")
print(f"Topic combinations: {synthetic_jokes['topics'].nunique()}")
```

### Schema Overview

All final datasets share a common schema:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `joke` | str | The joke text | "Why did the chicken cross the road?" |
| `topics` | str | Comma-separated nouns | "chicken,road,crossing" |
| `source` | str | Dataset origin | "ShortJokes", "DeepSeek-Chat" |
| `ppl` | float | DistilGPT2 perplexity | 15.23 (lower = more fluent) |
| `toxicity_*` | float | Detoxify scores (6 dims) | 0.02 (< 0.1 = clean) |

### Command Line Usage

#### 1. Explore Internet Jokes Dataset

```bash
cd internet_jokes
# Read the comprehensive documentation
cat README.md

# Launch analysis notebook
cd plot_data
jupyter notebook plot_data.ipynb
```

#### 2. Work with LLM-Generated Jokes

```bash
cd joke_generation/llm_jokes/experiment3.0

# View generated jokes
head outputs/llm_jokes_top_5000_topics_3_detox_safe.csv

# Run analysis scripts (from experiment3.0 directory)
python analyzers/compute_ppl_llm_jokes.py
python analyzers/plot_ppl_hist_llm_jokes.py
```

#### 3. Access Training Data

```bash
cd joke_generation/combined_jokes/experiment2.0

# View final combined dataset
head outputs/final/clean_good_jokes.csv

# Check preprocessing utilities
ls utils/
```

### Recommended Usage

**For training language models**:
- Start with `internet_jokes/` baseline (724K jokes)
- Or use `experiment3.0/` synthetic (500K jokes, higher uniformity)
- Check `runs/` for tokenization and split configurations

**For joke generation research**:
- See `experiment1.0/` for LLM comparison methodology
- Use `experiment2.0/` combined dataset for diversity
- Inspect `experiment3.0/analyzers/` for quality control pipelines

**For topic-controlled generation**:
- Extract topics from `internet_jokes/stats/topic_keyword.json`
- Use `experiment3.0/build_llm_jokes_top_5000_topics_3.py` pattern
- Constrain generation with 1-3 noun combinations

For detailed documentation on each subdirectory, see the respective README files.

## Key Statistics

### Internet Jokes Corpus (Baseline)

**Raw Collection**: 1,305,211 jokes from 5 public datasets

| Source | Original Size | After Cleaning | Retention |
|--------|--------------|----------------|-----------|
| ShortJokes (Kaggle) | 231,657 | 230,137 | 99.34% |
| rJokesData (GitHub) | 345,965 | 276,863 | 80.03% |
| AmirkidJokes (HuggingFace) | 574,189 | 157,857 | 27.51% |
| HumorDetection200k (Kaggle) | 100,000 | 9,135 | 9.14% |
| Dad Jokes (Reddit/Kaggle) | 53,400 | 50,515 | 94.60% |
| **Total** | **1,305,211** | **724,507** | **55.51%** |

**Cleaning Pipeline Results**:

| Filter Step | Dropped Rows | % of Input |
|-------------|--------------|-----------|
| Empty after cleaning | 18 | 0.00% |
| Highly symbolic / near-nontext | 38 | 0.00% |
| Too short (< 10 chars) | 150 | 0.01% |
| Too long (> 1000 chars) | 39,643 | 3.04% |
| **Deduplication** | **540,855** | **41.43%** |
| **Final Corpus** | **724,507** | **55.51%** ✅ |

**Text Characteristics**:
- Average length: 135 characters (29 words)
- Length range: 10-1000 characters
- Topic keywords: 1-3 per joke
- Quality: Unicode normalized, deduplicated, filtered

### LLM Screening (experiment1.0)

**Total Collection**: ~78,000 jokes across 7 models

| Model | Type | Jokes | Quality Score* | Notes |
|-------|------|-------|----------------|-------|
| Llama-3.1-8B | Open-source | 3,879 | 3/10 | Fast but lower quality |
| Llama-3.1-70B | Open-source | 32,531 | 2.5/10 | Poorest quality |
| Qwen3-30B-A3B | Open-source | 7,231 | **7.5/10** | Strong quality |
| **DeepSeek-Chat** | API | 23,567 | **7.5/10** | ⭐ Selected for scale |
| Gemini-2.5-Flash-Lite | API | 2,128 | 3.5/10 | Rate limited |
| Gemini-2.5-Flash | API | 6,177 | 4.5/10 | Rate limited |
| Gemini-2.5-Pro | API | 2,487 | 7/10 | Good but expensive |

*Quality judged by GPT-5.2-Thinking + Gemini-3-Pro on 50 random samples per model

### Large-scale Synthetic Generation (experiment3.0)

**Two-Stage Pipeline Results**:

| Stage | Dropped Rows | Remaining |
|-------|--------------|-----------|
| Raw LLM-generated jokes | -- | 526,308 |
| After basic cleaning | 3,961 | 522,347 |
| After Detoxify filtering | 13,260 | 509,087 |
| After repeated-char filtering | 74 | 509,013 |
| After balancing (100/topic) | 9,013 | 500,000 |
| **Final Synthetic Corpus** | -- | **500,000** ✅ |

**Generation Configuration**:
- Topic combinations: 5,000 (top three-noun combinations)
- Jokes per combination: 100 (target)
- LLM: DeepSeek-Chat
- Temperature: 1.3 (encourage creativity)
- Batch size: 10 jokes per API call
- Total API calls: ~52,600

**Post-filtering Quality Control**:
- Toxicity dimensions: 6 (all < 0.1 threshold)
  - toxicity, severe_toxicity, obscene, threat, insult, identity_attack
- PPL scoring: DistilGPT2 for quality diagnostics
- Repeated-char filter: Remove low-PPL "humor" artifacts
- Balancing: Ensure 100 jokes per topic for uniform coverage

### Training Runs (runs/)

**Multiple experimental configurations**:
- Positional encoding: Learned embeddings vs. ALiBi
- Sequence construction: Padded vs. packed
- Tokenization: BPE vs. Unigram (4K/10K vocab)
- Dataset variants: Internet baseline vs. synthetic
- Sample sizes: 25K subsets for rapid iteration

**Evaluation Framework**: 5-dimensional
1. Language modeling quality (NLL, PPL, accuracy)
2. Topic adherence (recall, precision)
3. Memorization detection (n-gram overlap)
4. Diversity metrics (distinct n-grams, entropy)
5. Humor likelihood (qualitative inspection)

## Data Quality Assurance

All datasets undergo rigorous quality control:

1. **Text Cleaning**: Unicode normalization, HTML/Markdown removal, whitespace standardization
2. **Deduplication**: Exact and near-duplicate removal
3. **Length Filtering**: Min 10 chars, max 1000 chars
4. **Toxicity Screening**: Detoxify-based filtering with dimension-specific thresholds
5. **Perplexity Scoring**: GPT-2 based quality metrics
6. **Manual Sampling**: Regular quality checks on random samples

### Environment Requirements

```bash
# Core dependencies
pip install pandas numpy torch transformers
pip install plotly jupyter spacy tqdm ftfy

# NLP models
python -m spacy download en_core_web_sm

# Optional: Toxicity filtering
pip install detoxify
```

**Last Updated**: March 7, 2026  
**Status**: ✅ Production Ready  
**Cross-platform**: ✅ Compatible (Linux/Windows/macOS)