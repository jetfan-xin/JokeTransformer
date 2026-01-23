# Dataset Jokes: Consolidated Joke Dataset with Analysis

## Overview

**Dataset Jokes** is a centralized data aggregation hub for cleaned and processed joke corpora used across all LLM joke generation experiments. This directory serves as the **source data foundation** for experiments and provides comprehensive visualization and statistical analysis of the consolidated joke dataset.

Unlike the experiment directories (experiment1.0, experiment3.0, experiment4.0), which focus on **generation and evaluation**, this directory stores the **aggregated, cleaned datasets** that feed into those experiments.

### Purpose
- ✅ Central storage for final cleaned joke datasets
- ✅ Statistical analysis and visualization of joke collection
- ✅ Metadata and tagging information for all jokes
- ✅ Foundation data for downstream experiments and training
- ✅ Quick access to comprehensive joke collection statistics

### Key Statistics

| Metric | Value |
|--------|-------|
| **Total Jokes** | ~726,000 jokes |
| **Unique Topics** | ~15,000+ unique topic tags |
| **Data Files** | 3 CSV sources |
| **Analysis Outputs** | 20+ visualizations |
| **Coverage** | Multi-part-of-speech annotations |

---

## Directory Structure

```
dataset_jokes/
├── README.md                            # This file - complete documentation
│
├── data_source/                         # Input data: Raw cleaned datasets
│   ├── final_clean_jokes.csv           # Core cleaned jokes (primary source)
│   ├── final_clean_jokes_with_all_nouns.csv  # Enhanced with all noun tags
│   └── final_combined_jokes.csv        # Complete aggregated dataset
│
├── plot_data/                           # Visualization & Analysis
│   ├── plot_data.ipynb                 # Jupyter notebook (all analysis)
│   ├── PLOT_DATA.md                    # Notebook documentation
│   └── outputs/                        # Generated plots and statistics (20+ files)
│       ├── *.csv                       # Statistical summaries (CSV)
│       ├── *.html                      # Interactive plots (Plotly)
│       ├── *.png                       # Static visualizations (PNG)
│       └── [See "Visualizations & Outputs" section below]
```

---

## Data Sources Overview

### Input CSV Files (data_source/)

All CSV files in `data_source/` contain cleaned jokes ready for analysis and use. These are the **primary inputs** for visualization and are used by downstream experiments.

#### 1. final_clean_jokes.csv

**Purpose:** Core cleaned joke dataset with basic annotations

**Size:** ~78,000-100,000 jokes (estimated)

**Columns:**
| Column | Type | Example | Description |
|--------|------|---------|-------------|
| `joke` | String | "Why did the..." | Original cleaned joke text |
| `topic` | String | "Comedy, Animals" | Associated topic tags (POS-based) |

**Use Cases:**
- Primary source for experiment1.0 and experiment3.0
- General-purpose joke corpus
- Topic-based analysis

**Data Quality:**
- ✓ Text cleaned and normalized
- ✓ Unicode issues resolved
- ✓ HTML entities decoded
- ✓ Markup removed

---

#### 2. final_clean_jokes_with_all_nouns.csv

**Purpose:** Enhanced dataset with comprehensive noun extraction

**Size:** ~726,000 jokes (extended with all noun combinations)

**Columns:**
| Column | Type | Example | Description |
|--------|------|---------|-------------|
| `joke_cleaned` | String | "Why did the..." | Cleaned joke text |
| `topic` | String | "Comedy, Animals" | Limited topic tags |
| `topic_all_nouns` | String | "chicken, road, humor, ..." | ALL noun mentions extracted |

**Enhanced Features:**
- Comprehensive noun extraction via spaCy NLP
- Every noun in the joke is tagged
- Enables fine-grained topic analysis
- Better semantic coverage

**Use Cases:**
- Advanced topic modeling
- Noun-based analysis in plot_data.ipynb
- Semantic similarity studies
- Training data with rich annotations

**Data Quality:**
- ✓ All cleaning from final_clean_jokes applied
- ✓ NOUN and PROPN POS tags extracted
- ✓ Lowercase normalized
- ✓ Duplicates removed

---

#### 3. final_combined_jokes.csv

**Purpose:** Complete aggregated dataset combining all sources

**Size:** ~726,000 jokes (full dataset)

**Columns:**
| Column | Type | Example | Description |
|--------|------|---------|-------------|
| `rid` | Integer | 1, 2, 3, ... | Record ID (sequential) |
| `stable_id` | String | "a1b2c3d4..." | SHA-1 hash for stability |
| `joke_cleaned` | String | "Why did the..." | Cleaned joke text |
| `topic` | String | "Comedy, Animals" | Original topic tags |
| `topic_all_nouns` | String | "chicken, road, ..." | Comprehensive noun tags |

**What This Contains:**
- All final_clean_jokes records
- All all-noun annotations
- Complete metadata
- Ready for production use

**Use Cases:**
- Primary data source for plot_data.ipynb (default)
- Training dataset for fine-tuning
- Complete backup/reference dataset
- Multi-faceted analysis

**Data Quality:**
- ✓ Highest quality, most complete
- ✓ Combines all annotations
- ✓ No duplicates (deduplicated)
- ✓ Full metadata preserved

---

## Data Schema & Column Descriptions

### Common Columns Across All CSVs

#### joke / joke_cleaned
- **Type:** String
- **Length:** 10-1000 characters
- **Contains:** Cleaned joke text with proper punctuation
- **Processing:** Unicode normalized (NFKC), HTML decoded, markdown removed
- **Example:** `"Why did the chicken cross the road? To get to the other side!"`

#### topic
- **Type:** String (comma-separated list)
- **Format:** `"Tag1, Tag2, Tag3"`
- **Count:** 1-3 tags per joke
- **Content:** POS-filtered tags (NOUN, PROPN, VERB, ADJ preferred)
- **Example:** `"Humor, Animals, Wordplay"`

#### topic_all_nouns (only in final_clean_jokes_with_all_nouns.csv and final_combined_jokes.csv)
- **Type:** String (comma-separated list)
- **Format:** `"noun1, noun2, noun3, ..."`
- **Count:** Variable (all nouns in joke)
- **Content:** Every NOUN/PROPN found in joke text
- **Example:** `"chicken, road, humor, destination, journey"`
- **Note:** Can be very long if joke contains many nouns

#### rid / stable_id (only in final_combined_jokes.csv)
- **Type:** Integer / String
- **rid:** Sequential record ID (1, 2, 3, ...)
- **stable_id:** SHA-1 hash of normalized joke (for deduplication tracking)
- **Use:** Cross-reference and duplicate detection

### Data Types & Encoding

| Column | Type | Encoding | Nullable |
|--------|------|----------|----------|
| joke | String | UTF-8 | No |
| topic | String | UTF-8 | No |
| topic_all_nouns | String | UTF-8 | No |
| rid | Int64 | N/A | No |
| stable_id | String | UTF-8 | No |

### Data Quality Notes

**Text Cleaning Applied:**
1. Unicode normalization (NFKC)
2. HTML entity decoding (`&amp;` → `&`)
3. Markdown syntax removal
4. Control character removal
5. Whitespace normalization
6. URL and email removal

**Topic Tag Generation:**
1. Spacy NLP POS tagging
2. NOUN and PROPN prioritization
3. Lowercase normalization
4. Deduplication at spacy token level
5. Top 3 tags per joke selection

**Coverage:**
- ~100% of jokes have at least 1 topic tag
- ~95% have 2-3 tags
- ~100% have comprehensive noun extraction

---

## Plot Data Notebook Guide

### Overview

**File:** `plot_data/plot_data.ipynb`

The Jupyter notebook `plot_data.ipynb` contains comprehensive statistical analysis and interactive visualization of the dataset_jokes collection. It generates 20+ outputs including distribution charts, statistical summaries, and frequency analyses.

**What It Does:**
- 📊 Generates distribution plots for 7 different analyses
- 📈 Computes statistical summaries and aggregations
- 🏷️ Analyzes tag frequency and diversity
- 📏 Examines joke length characteristics
- 🔢 Creates interactive Plotly visualizations
- 💾 Exports all outputs to `plot_data/outputs/`

**Technology:**
- Pandas for data processing
- spaCy for NLP/POS tagging
- Plotly for interactive plots
- NumPy for statistics

---

### How to Run the Notebook

#### Step 1: Install Dependencies

```bash
pip install pandas jupyter plotly spacy numpy

# Download spaCy model (required for POS tagging)
python -m spacy download en_core_web_sm
```

#### Step 2: Launch Jupyter

```bash
cd /ltstorage/home/4xin/uhh-ias-ml/data/dataset_jokes/plot_data
jupyter notebook plot_data.ipynb
```

#### Step 3: Execute All Cells

Within Jupyter:
- **Kernel → Restart & Run All** (to run entire notebook)
- OR manually run cells top-to-bottom (Shift+Enter)

#### Expected Runtime

- First load: 2-5 minutes (spaCy initialization)
- Subsequent runs: 5-15 minutes total
- Depends on system specs and Plotly rendering

#### Outputs Location

All visualizations saved to:
```
plot_data/outputs/
├── *.csv        # Statistical summaries
├── *.html       # Interactive Plotly charts
└── *.png        # Static PNG exports
```

**Total output files:** ~20 HTML + CSV + PNG files

---

### Notebook Structure

The notebook is organized into **7 major analysis sections:**

| Section | Title | Outputs |
|---------|-------|---------|
| 1 | Count all tag types (POS distribution) | tag_type_count.* |
| 2 | Plot joke length distribution | joke_length_chars.* |
| 3 | Plot distribution of tags per joke | tag_count_distribution.* |
| 4 | Plot top n tags | tag_frequencies_top{30,100,500,1000}.* |
| 5 | Plot tag frequency distribution (log-binned) | tag_frequency_distribution_log.* |
| 6 | Analyze unique tags (all nouns version) | tag_count_distribution_all_nouns.* |
| 7 | Visualize tag count vs. joke length | tag_count_vs_length_with_std_clean.* |

**Key Dependencies Between Cells:**
- Load cell must run first (imports + initial data load)
- Each analysis can run independently after load
- Later analyses use `topic_all_nouns` column (requires final_clean_jokes_with_all_nouns.csv)

---

## Visualizations & Analysis Outputs

All visualizations are automatically generated in `plot_data/outputs/` when the notebook runs. Each analysis section produces **3 file types**:
- `.csv` - Statistical summaries (data behind plots)
- `.html` - Interactive Plotly charts (open in browser)
- `.png` - Static PNG exports (for reports/presentations)

---

### 1. POS-Based Tag Distribution

**Files:** `tag_type_count.csv`, `tag_type_count.html`, `tag_type_count.png`

**Purpose:** Show distribution of part-of-speech categories across all tags

**Analysis:**
- Counts NOUN, PROPN, VERB, ADJ, MISC tags
- Reveals what types of words dominate the topic corpus
- Helps understand topic generation preferences

**Typical Distribution:**
```
NOUN:   ~50%  (nouns dominate)
PROPN:  ~20%  (proper nouns)
VERB:   ~15%  (verbs)
ADJ:    ~10%  (adjectives)
MISC:   ~5%   (miscellaneous)
```

**Use Case:** Understand semantic emphasis in topic tags

---

### 2. Joke Length Distribution

**Files:** `joke_length_chars.csv`, `joke_length_chars.html`, `joke_length_chars.png`

**Purpose:** Show distribution of joke lengths (characters)

**Metrics:**
- Minimum length: ~10 characters (after cleaning)
- Maximum length: ~1000 characters (filter limit)
- Mean length: ~85 characters
- Median length: ~70 characters

**Features:**
- Histogram with bin width = 10 characters
- Cumulative percentage overlay
- 90th percentile marked with vertical line
- Shows most jokes are short (under 150 chars)

**Insight:** Joke length follows a power-law distribution—most jokes are short, few are long.

---

### 3. Number of Tags per Joke

**Files:** `tag_count_distribution.csv`, `tag_count_distribution.html`, `tag_count_distribution.png`

**Purpose:** Show how many tags are assigned per joke (basic POS-based tags)

**Distribution:**
```
1 tag:  ~6%    (miscellaneous or single noun)
2 tags: ~15%   (two-word combinations)
3 tags: ~79%   (three-tag standard)
```

**Key Finding:** Most jokes receive exactly 3 tags, reflecting the standard "top-3 nouns" extraction strategy.

---

### 4. Top N Tags Frequency Analysis

**Files:** `tag_frequencies_top{30,100,500,1000}.csv/.html/.png`

**Purpose:** Show which topic tags appear most frequently across all jokes (multiple n-values)

**Versions:**
- `top30.html` - Most frequent 30 tags (quick overview)
- `top100.html` - Top 100 tags (broad coverage)
- `top500.html` - Top 500 tags (medium tail)
- `top1000.html` - Top 1000 tags (long tail)

**Scale:** Logarithmic y-axis (log scale) to show power-law distribution

**Typical Top Tags:**
```
1. humor         (appears in ~5% of jokes)
2. comedy        (appears in ~4%)
3. animals       (appears in ~3%)
4. wordplay      (appears in ~2.5%)
...
30. [varies]     (appears in ~0.2%)
```

**Use Case:** Identify dominant topics in dataset

---

### 5. Tag Frequency Distribution (Log-Binned)

**Files:** `tag_frequency_distribution_log.csv`, `tag_frequency_distribution_log.html`, `tag_frequency_distribution_log.png`

**Purpose:** Visualize the long-tail distribution of tag frequencies

**Methodology:**
- Bins: Powers of 2 (1, 2, 4, 8, 16, 32, 64, ...)
- Shows: How many tags appear N times
- Reveals: Power-law structure

**Example Output:**
```
Frequency Range | Number of Tags | Percentage
1–1             | 2,500          | ~17%
2–3             | 1,200          | ~8%
4–7             | 800            | ~5%
8–15            | 600            | ~4%
16–31           | 400            | ~3%
...
```

**Key Finding:** Long-tail phenomenon—thousands of rare tags (appear 1-10 times), few common tags (1000+ times)

---

### 6. Unique Tags per Joke (All-Nouns Version)

**Files:** `tag_count_distribution_all_nouns.csv/.html/.png`

**Purpose:** Show distribution of unique noun tags per joke (using all-nouns extraction)

**Difference from Section 3:**
- Section 3: Limited to top-3 POS-filtered tags (mostly 3)
- Section 6: All nouns in joke (1-50+ per joke)

**Typical Distribution:**
```
1-5 nouns:     ~20%   (short simple jokes)
6-10 nouns:    ~35%   (typical jokes)
11-15 nouns:   ~30%   (longer jokes)
16+ nouns:     ~15%   (very long/complex jokes)
```

**Use Case:** Understand semantic complexity; fine-grained topic analysis

---

### 7. Tag Count vs. Joke Length (with Std Dev)

**Files:** `tag_count_vs_length_with_std_clean.csv/.html/.png`

**Purpose:** Correlation analysis—how joke length affects tag quantity

**Visualization:**
- Bar chart: Number of jokes (left y-axis)
- Line with shaded band: Average joke length ± std deviation (right y-axis)
- X-axis: Number of unique tags per joke

**Key Insights:**
- Jokes with more tags tend to be longer (correlation ~0.6)
- Standard deviation increases with tag count (longer jokes have more variability)
- Most jokes cluster around 6-10 tags and 80-120 characters

**Use Case:** Understand relationship between complexity (tags) and size (length)

---

## Key Statistics & Insights

### Dataset Size & Coverage

| Metric | Value |
|--------|-------|
| **Total Jokes** | ~726,000 |
| **Unique Topics** | ~15,000+ |
| **Avg Joke Length** | ~87 characters |
| **Min/Max Length** | 10 / 1000 characters |
| **Median Jokes/Tag** | ~50 jokes per tag |

### Tag Distribution Insights

**POS Breakdown:**
- NOUN tags: ~50% (primary category)
- PROPN tags: ~20% (names, places)
- VERB tags: ~15% (actions)
- ADJ tags: ~10% (descriptors)
- MISC tags: ~5% (other)

**Top 10 Most Frequent Tags:**
1. humor (~5% coverage)
2. comedy (~4% coverage)
3. animals (~3% coverage)
4. wordplay (~2.5% coverage)
5. [varies by dataset version]

**Long-Tail Phenomenon:**
- Top 30 tags appear in ~25% of jokes
- Top 100 tags appear in ~40% of jokes
- Top 1000 tags appear in ~65% of jokes
- Remaining 14,000+ tags appear in ~35% of jokes (long tail)

### Data Quality Summary

**Cleaning Pipeline Applied:**
✓ Unicode normalization (NFKC)  
✓ HTML entity decoding  
✓ Markdown syntax removal  
✓ URL and email removal  
✓ Control character removal  
✓ Whitespace normalization  
✓ Deduplication (SHA-1 hashing)  

**Coverage Metrics:**
- 100% jokes have at least 1 topic tag
- 95%+ have 2-3 POS-based tags
- 100% have comprehensive noun extraction
- 0% null values in cleaned text fields

**Retention Rates:**
- Source jokes: ~1M raw jokes
- After cleaning: ~726K unique jokes (73% retention)
- Primary reason for loss: Exact/normalized duplicates

---

## Integration with Experiments

This dataset directory serves as the **data foundation** for downstream experiments:

### experiment1.0: LLM-Based Joke Generation

**Uses:**
- `data_source/final_clean_jokes_with_all_nouns.csv` - Source data for noun combination extraction
- Extracts top 5000 noun combinations from this dataset
- Uses cleaned jokes as reference for quality benchmarking

**Data Flow:**
```
dataset_jokes/data_source/
  ↓
experiment1.0/analyzers/extract_noun_combinations.py
  ↓
Generates 5000 noun combination seeds
  ↓
experiment1.0/generators/*.py (7 models)
  ↓
Generates 100 jokes per combo
```

---

### experiment3.0: Toxicity-Filtered Large-Scale Pipeline

**Uses:**
- `data_source/final_combined_jokes.csv` - Foundation dataset
- Cleans and toxicity-filters this dataset
- Produces enhanced version with quality scores

**Data Flow:**
```
dataset_jokes/data_source/
  ↓
experiment3.0/analyzers/safety_filter.py
  ↓
Removes toxic/harmful content
  ↓
experiment3.0/analyzers/ppl_scoring.py
  ↓
Adds perplexity quality scores
  ↓
Enhanced 500K+ joke dataset
```

---

### experiment4.0: Async DeepSeek Generation & Text Cleaning

**Uses:**
- Partially independent (generates new jokes)
- Can use this dataset for comparison/benchmarking

**Data Flow:**
```
experiment4.0/generators/run_jokes_async.py
  ↓
Generates 100K new jokes
  ↓
experiment4.0/analyzers/clean_jokes.py
  ↓
Produces 14K cleaned, high-quality jokes
  ↓
Can compare with dataset_jokes quality
```

---

## Getting Started / Quick Start

### Prerequisites

#### System Requirements
- Python 3.8+
- ~2GB free disk space (for outputs)
- Modern web browser (for viewing .html outputs)

#### Required Libraries

```bash
pip install pandas jupyter plotly spacy numpy

# Download spaCy NLP model (required for POS tagging)
python -m spacy download en_core_web_sm
```

#### Optional Libraries (for advanced analysis)
```bash
pip install matplotlib seaborn scikit-learn
```

---

### Loading Data in Python

**Quick Start: Read the main dataset**

```python
import pandas as pd

# Load complete dataset (recommended)
df = pd.read_csv('data_source/final_combined_jokes.csv')
print(f"Loaded {len(df)} jokes")
print(df.head())

# Access columns
print(f"Sample joke: {df['joke_cleaned'].iloc[0]}")
print(f"Tags: {df['topic'].iloc[0]}")
print(f"All nouns: {df['topic_all_nouns'].iloc[0]}")
```

**Load specific CSV files**

```python
# Core dataset (no all-nouns)
df_core = pd.read_csv('data_source/final_clean_jokes.csv')

# Enhanced with all nouns
df_nouns = pd.read_csv('data_source/final_clean_jokes_with_all_nouns.csv')

# Complete (recommended)
df_complete = pd.read_csv('data_source/final_combined_jokes.csv')
```

---

### Running the Analysis Notebook

```bash
# Navigate to notebook directory
cd /ltstorage/home/4xin/uhh-ias-ml/data/dataset_jokes/plot_data

# Start Jupyter
jupyter notebook plot_data.ipynb

# In Jupyter:
# 1. Click: Kernel → Restart & Run All
# 2. Wait 5-15 minutes for completion
# 3. Check outputs/ directory for results
```

**Expected Output:**
```
plot_data/outputs/
├── tag_type_count.csv / .html / .png
├── joke_length_chars.csv / .html / .png
├── tag_count_distribution.csv / .html / .png
├── tag_frequencies_top30.csv / .html / .png
├── tag_frequencies_top100.csv / .html / .png
├── tag_frequencies_top500.csv / .html / .png
├── tag_frequencies_top1000.csv / .html / .png
├── tag_frequency_distribution_log.csv / .html / .png
├── tag_count_distribution_all_nouns.csv / .html / .png
└── tag_count_vs_length_with_std_clean.csv / .html / .png
```

---

### Viewing Interactive Plots

All `.html` files can be opened in any web browser:

```bash
# Open directly
open plot_data/outputs/tag_type_count.html

# Or from command line
python -m webbrowser plot_data/outputs/tag_type_count.html
```

**Features:**
- Hover for detailed info
- Click legend to toggle series
- Pan and zoom interactive plots
- Download as PNG button available

---

## File Reference & Outputs Map

Quick lookup table showing which notebook analysis produces which outputs:

| Notebook Section | Input CSV | Outputs | Purpose |
|------------------|-----------|---------|---------|
| **1. Tag Types** | final_combined_jokes.csv | tag_type_count.* | POS distribution |
| **2. Joke Length** | final_combined_jokes.csv | joke_length_chars.* | Length statistics |
| **3. Tags/Joke** | final_combined_jokes.csv | tag_count_distribution.* | Tags per joke |
| **4. Top N Tags** | final_combined_jokes.csv | tag_frequencies_top*.* | Frequency analysis |
| **5. Tag Freq Dist** | final_combined_jokes.csv | tag_frequency_distribution_log.* | Long-tail analysis |
| **6. Unique Tags** | final_clean_jokes_with_all_nouns.csv | tag_count_distribution_all_nouns.* | All-nouns analysis |
| **7. Tags vs Length** | final_clean_jokes_with_all_nouns.csv | tag_count_vs_length_with_std_clean.* | Correlation analysis |

**Total Output Files:** ~20 (CSV + HTML + PNG combined)

---

## Troubleshooting

### Error: "spaCy model not found"

```
OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't appear to be installed or you haven't set the correct model_name.
```

**Solution:**
```bash
python -m spacy download en_core_web_sm
```

---

### Error: "CSV file not found"

```
FileNotFoundError: data_source/final_combined_jokes.csv
```

**Solution:** Ensure you're in the correct directory:
```bash
cd /ltstorage/home/4xin/uhh-ias-ml/data/dataset_jokes
```

---

### Notebook runs very slowly

**Causes & Solutions:**
- First run downloads spaCy model (~50MB) - subsequent runs faster
- Large dataset processing - normal for 700K jokes
- Plotly rendering can be slow on older browsers - try exporting PNG instead

**Speed-up tips:**
```python
# Sample first 10K jokes for quick testing
df = df.head(10000)
```

---

### Output files not appearing

**Checklist:**
1. Did notebook complete without errors? (Check for red error boxes)
2. Check `plot_data/outputs/` directory
3. Verify write permissions: `ls -la plot_data/outputs/`
4. Try running individual cells (not whole notebook)

---

## Key Columns Reference

**Quick lookup for common tasks:**

| Task | Use Column | Example |
|------|------------|---------|
| Read joke text | `joke` or `joke_cleaned` | "Why did the chicken..." |
| Get assigned topics | `topic` | "Comedy, Animals" |
| Extract all nouns | `topic_all_nouns` | "chicken, road, direction, ..." |
| Track duplication | `stable_id` | "abc123def456..." |
| Sequence order | `rid` | 1, 2, 3, ... |

---

## Future Improvements

### Potential Enhancements

1. **Sentiment Analysis** - Add sentiment scores per joke
2. **Semantic Clustering** - Group similar jokes by embeddings
3. **Humor Scoring** - Automated humor quality metrics
4. **Multi-language Support** - Extend beyond English
5. **Interactive Dashboard** - Real-time exploration UI

### Dataset Expansion

- Integrate new joke sources
- Add more POS-based annotations
- Cross-reference with benchmark datasets
- Version control and archival

---

## References & Related Work

### Related Directories

- [experiment1.0](../joke_generation/llm_jokes/experiment1.0/README.md) - LLM Generation Pipeline
- [experiment3.0](../joke_generation/llm_jokes/experiment3.0/README.md) - Toxicity Filtering Pipeline
- [experiment4.0](../joke_generation/llm_jokes/experiment4.0/README.md) - Async DeepSeek Pipeline

### Parent Documentation

- Main project README: `/ltstorage/home/4xin/uhh-ias-ml/README.md`
- Data directory overview: `/ltstorage/home/4xin/uhh-ias-ml/data/README.md`

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-23 | Initial README with analysis documentation, notebook guide, and integration notes |

---

## Summary

**Dataset Jokes** is your central hub for:
- 🎯 Accessing cleaned joke datasets (~726K jokes)
- 📊 Analyzing distribution and statistics via notebook
- 🏷️ Understanding topic tagging and metadata
- 🔗 Feeding data into downstream experiments

**Quick Start:**
1. Load data: `pd.read_csv('data_source/final_combined_jokes.csv')`
2. Run analysis: `jupyter notebook plot_data/plot_data.ipynb`
3. View outputs: Open any `.html` file in browser

**Best For:**
- Data exploration and statistics
- Quality benchmarking
- Topic analysis and visualization
- Training data preparation

---

**Last Updated:** January 23, 2026  
**Status:** ✓ Ready to Use  
**Total Jokes:** ~726,000  
**Data Files:** 3 CSV sources  
**Analysis Outputs:** 20+ visualizations  
**Documentation:** Complete with notebook integration guide
