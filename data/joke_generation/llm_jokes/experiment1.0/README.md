# Experiment 1.0: LLM-Based Joke Generation Pipeline

## Overview

**Experiment 1.0** is a comprehensive pipeline for generating high-quality jokes using multiple Large Language Models (LLMs) on top 5000 noun combinations. The experiment evaluates different LLMs' ability to generate creative, contextually relevant jokes and compares their quality through systematic evaluation.

### Objective

- Generate **100 jokes per noun combination** from **5000 unique noun combinations**
- Evaluate **7 different LLM models** for joke generation quality
- Analyze consistency, originality, and comedic quality across models
- Produce a dataset of **~78,000 LLM-generated jokes** for training or evaluation

### Key Results Summary

| Model | Group | Type | Jokes Generated | Quality Score | Notes |
|-------|-------|------|-----------------|---------------|-------|
| Llama-3.1-70B | A | Open-source | 32,531 | 2.5/10 | Fast but lower quality |
| Llama-3.1-8B | B | Open-source | 3,879 | 3/10 | Poorest quality |
| **Qwen3-30B-A3B** | C | **Open-source** | 7,231 | **7.5/10** | **Strong quality** |
| **DeepSeek-Chat** | D | **API** | 23,567 | **7.5/10** | **Best value** |
| Gemini-2.5-Flash | E | API | 6,177 | 4.5/10 | Rate limited |
| Gemini-2.5-Flash-Lite | F | API | 2,128 | 3.5/10 | Rate limited |
| Gemini-2.5-Pro | G | API | 2,487 | 7/10 | Good but expensive |
| **TOTAL** | | | **78,000** | | |

---

## Directory Structure

```
experiment1.0/
├── README.md                          # This file - complete documentation
├── LLM_Jokes.md                       # Evaluation results and model comparisons
├── PATH_FIX_SUMMARY.md                # Documentation of path fixes
│
├── generators/                        # LLM joke generation scripts (8 files)
│   ├── deepseek_chat.py              # DeepSeek-Chat API generator
│   ├── gemini_apis.py                # Gemini API generator
│   ├── llama3_1_70b.py               # Llama 3.1 70B with NF4 quantization
│   ├── llama3_1_8b.py                # Llama 3.1 8B full precision
│   ├── Qwen3-30B-A3B-Instruct-2507_para0.py
│   ├── Qwen3-30B-A3B-Instruct-2507_para1.py
│   ├── Qwen3-30B-A3B-Instruct-2507_para2.py
│   └── Qwen3-30B-A3B-Instruct-2507_para3.py
│
├── analyzers/                         # Data analysis and aggregation (6 files)
│   ├── merge_jokes.py                # Merge all raw outputs into single CSV
│   ├── analyze_jokes_sorted_by_index.py  # Analyze stats by combo index
│   ├── extract_noun_combinations.py  # Extract top noun combinations
│   ├── joke_stats.py                 # Generate detailed statistics JSON
│   ├── model_joke_samples.py         # Sample 50 jokes per model
│   └── top_5000_combos_extracter.py  # Extract top 5000 combos
│
├── data_source/                       # Input data
│   └── final_clean_jokes_with_all_nouns.csv  # Source jokes with noun tags
│
├── stats/                             # Statistical outputs (10 files)
│   ├── combos_k1_stats.csv           # 1-word noun combos frequency
│   ├── combos_k2_stats.csv           # 2-word noun combos frequency
│   ├── combos_k3_stats.csv           # 3-word noun combos frequency
│   ├── combos_k4_stats.csv           # 4-word noun combos frequency
│   ├── combos_k5_stats.csv           # 5-word noun combos frequency
│   ├── combos_top5000_across_k1_to_k5.csv  # Final top 5000 combos
│   ├── merged_llm_jokes.csv          # All jokes merged and deduplicated
│   ├── jokes_stats_summary.csv       # Summary statistics per combo
│   ├── llm_jokes_stats.json          # Detailed JSON statistics
│   └── model_joke_samples.json       # 50 random jokes per model
│
└── raw_outputs/                       # Raw generation outputs (13+ files)
    ├── deepseek_jokes.csv
    ├── gemini_jokes.csv
    ├── llama3_1_70b_nf4_jokes.csv
    ├── llama3_1_8b_full_jokes.csv
    ├── llama3_70b_manual_combos_jokes_para*.csv
    ├── qwen3-30b_*.csv
    └── qwen3_nf4_jokes_from_43_fixed.csv
```

---

## Complete Workflow & Pipeline

### Phase 1: Data Preparation

#### Step 1: Extract Noun Combinations from Source

**Script:** `analyzers/extract_noun_combinations.py`

This script processes the source data to extract all noun combinations (1 to 5 word phrases) and computes frequency statistics.

```bash
cd /uhh-ias-ml/data/joke_generation/llm_jokes/experiment1.0
python3 analyzers/extract_noun_combinations.py
```

**Input:**
- `data_source/final_clean_jokes_with_all_nouns.csv` - Source jokes with noun annotations

**Output:**
- `stats/combos_k1_stats.csv` through `stats/combos_k5_stats.csv`
- Additional CSV files with top combos and percentile-based selections

**Key Functions:**
- `parse_tags()` - Parse comma-separated noun strings
- `build_combination_index()` - Create frequency index
- `extract_combos_and_jokes_top5_p75()` - Extract top 5 and p75 combos per k value

**Path Configuration:** Uses relative paths (`BASE_DIR / "stats"` and `BASE_DIR / "data_source"`)

---

#### Step 2: Create Master Top 5000 Combos Index

**Script:** `analyzers/top_5000_combos_extracter.py`

Aggregates all k values (1-5 words) and creates the master combo list by frequency.

```bash
python3 analyzers/top_5000_combos_extracter.py
```

**Input:**
- `stats/combos_k*.csv` (5 files for k=1 to k=5)

**Output:**
- `stats/combos_top5000_across_k1_to_k5.csv` - 5,001 rows including header

**Processing:**
- Loads all frequency files
- Combines frequencies across all k values
- Sorts by frequency (descending)
- Selects top 5000 combinations
- Formats with double quotes for CSV consistency

---

### Phase 2: Joke Generation (7 Models)

All generator scripts follow the same pattern and support parallelization via `START_INDEX`/`END_INDEX`.

#### Common Features in All Generators

```python
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE_DIR / "stats/combos_top5000_across_k1_to_k5.csv"
OUTPUT_CSV = BASE_DIR / "raw_outputs/{model_name}.csv"

TARGET_JOKES_PER_COMBO = 100      # Target per combo
BATCH_SIZE = 20                    # Jokes per request
START_INDEX = 0                    # Index range start
END_INDEX = 5000                   # Index range end
```

All generators:
1. Read combo index from `stats/combos_top5000_across_k1_to_k5.csv`
2. Process configurable index ranges
3. Generate jokes with metadata (combo, prompt, text, model name)
4. Support resume from failed runs
5. Implement batch processing and rate limiting

---

#### Generator 1: DeepSeek-Chat (Group D) - BEST VALUE

**Script:** `generators/deepseek_chat.py`

OpenAI-compatible API for DeepSeek's chat model. **Recommended for best value**.

```bash
export DEEPSEEK_API_KEY="sk-your_key_here"
python3 generators/deepseek_chat.py
```

**Configuration:**
```python
API_KEY = "xxx"              # or set DEEPSEEK_API_KEY env var
MODEL_NAME = "deepseek-chat"
BATCH_SIZE = 10              # 10 jokes per request
START_INDEX = 567
END_INDEX = 3000
```

**Results:**
- **Jokes Generated:** 23,567
- **Quality Score:** 7.5/10 ⭐ (Tied best)
- **Pricing:** ~€1 ≈ 24,000 jokes (cheapest)
- **Speed:** Variable (API-dependent)
- **Rate Limit:** None - unlimited continuous processing

**API Setup:**
1. Get token: https://platform.deepseek.com/api_keys
2. Create account and add credits
3. Set environment variable: `export DEEPSEEK_API_KEY="sk-xxx"`

**Advantages:**
- Best quality-to-price ratio globally
- No rate limiting or request caps
- Consistent, witty wordplay
- Excellent misdirection and joke structure
- Professional-level comedic writing

**Evaluation:** Smart, polished wordplay with lightly absurd edge—like a professional writer having fun.

---

#### Generator 2: Gemini (Groups E, F, G)

**Script:** `generators/gemini_apis.py`

Google's Generative AI API with three model tiers.

```bash
export GOOGLE_API_KEY="AIzaSXXXXXXXXXXXXX"
python3 generators/gemini_apis.py
```

**Configuration:**
```python
API_KEY = "xxx"              # or set GOOGLE_API_KEY env var
MODEL_NAME = "gemini-2.5-pro"     # Options: gemini-2.5-flash, gemini-2.5-flash-lite
BATCH_SIZE = 50
START_INDEX = 567
END_INDEX = 3000
```

**Model Tier Comparison:**

| Model | Group | Generated | Score | Limit | Price |
|-------|-------|-----------|-------|-------|-------|
| Gemini-2.5-Pro | G | 2,487 | 7/10 | 50 reqs/day | Paid |
| Gemini-2.5-Flash | E | 6,177 | 4.5/10 | 250 reqs/day | Free |
| Gemini-2.5-Flash-Lite | F | 2,128 | 3.5/10 | 1,000 reqs/day | Free |

**API Setup:**
1. Get token: https://makersuite.google.com/app/apikey
2. Enable Generative Language API in Google Cloud Console
3. Set: `export GOOGLE_API_KEY="AIza..."`

**Safety Settings:**
Script uses permissive safety settings to avoid content filtering:
```python
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}
```

**Rate Limiting:**
- Free tier has strict limits
- Script implements automatic retry with sleep on 429 errors
- Consider paid tier for higher throughput

**Evaluation Summary:**
- **Pro (7/10):** Witty set with creative narrative elements
- **Flash (4.5/10):** Mixed quality with topic whiplash
- **Flash-Lite (3.5/10):** Tired clichés—not recommended

---

#### Generator 3: Llama 3.1 70B (Group A) - OPEN SOURCE FASTEST

**Script:** `generators/llama3_1_70b.py`

Meta's Llama 3.1 70B Instruct with 4-bit NF4 quantization via Hugging Face.

```bash
export HF_TOKEN="hf_your_token_here"
python3 generators/llama3_1_70b.py
```

**Configuration:**
```python
MODEL_NAME = "meta-llama/Llama-3.1-70B-Instruct"
HF_TOKEN = "xxx"             # Hugging Face token
BATCH_SIZE = 20
START_INDEX = 200
END_INDEX = 300
```

**Results:**
- **Jokes Generated:** 32,531 (highest volume)
- **Quality Score:** 2.5/10 ❌ (lowest quality)
- **Speed:** ~270 seconds per 100 jokes (fastest)
- **GPU Memory:** 80GB+ required (A100, H100)

**Quantization:**
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)
```

**Hugging Face Setup:**
1. Create account at https://huggingface.co
2. Request access to Llama 3.1 model
3. Generate token: https://huggingface.co/settings/tokens
4. Set: `export HF_TOKEN="hf_xxx"`

**Advantages:**
- Fully open-source (no API calls)
- Fastest generation speed
- No licensing costs

**Disadvantages:**
- Generic Q&A puns with weak craftsmanship
- Repetitive templates
- Limited originality
- Not recommended for quality

**Evaluation:** Generic puns with weak craftsmanship—mostly stock templates repeated without cleverness.

---

#### Generator 4: Llama 3.1 8B (Group B) - OPEN SOURCE SMALL

**Script:** `generators/llama3_1_8b.py`

Smaller Llama variant for faster inference on modest GPUs.

```bash
export HF_TOKEN="hf_your_token_here"
python3 generators/llama3_1_8b.py
```

**Configuration:**
```python
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
HF_TOKEN = "xxx"
BATCH_SIZE = 20
START_INDEX = 100
END_INDEX = 200
```

**Results:**
- **Jokes Generated:** 3,879
- **Quality Score:** 3/10 ❌ (poorest quality)
- **Speed:** ~60 seconds per 100 jokes (2.5x faster than 70B)
- **GPU Memory:** 24-32GB sufficient

**GPU Requirements:**
- Minimum: 24GB VRAM (RTX 4090)
- Recommended: 32GB+

**Evaluation:** Bland and uninspired—aggressive lack of originality with ancient recycled templates. **Not recommended**.

---

#### Generator 5: Qwen3-30B (Group C) - BEST OPEN SOURCE

**Script:** `generators/Qwen3-30B-A3B-Instruct-2507_para*.py` (4 parallel shards)

Alibaba's Qwen3-30B with 4-bit quantization. **Recommended for best open-source quality**.

```bash
# Run all 4 parallel instances
python3 generators/Qwen3-30B-A3B-Instruct-2507_para0.py &
python3 generators/Qwen3-30B-A3B-Instruct-2507_para1.py &
python3 generators/Qwen3-30B-A3B-Instruct-2507_para2.py &
python3 generators/Qwen3-30B-A3B-Instruct-2507_para3.py &
wait
```

**Index Ranges (parallelization):**
- `para0.py`: indices 805-1000
- `para1.py`: indices 1200-1400
- `para2.py`: indices 1400-1600
- `para3.py`: indices 805-1000 (can override)

**Configuration Template:**
```python
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
BATCH_SIZE = 20
START_INDEX = 805              # Modify per shard
END_INDEX = 1000               # Modify per shard
TARGET_JOKES_PER_COMBO = 100
SKIP_THRESHOLD = 85            # Skip if already >85 jokes
```

**Results:**
- **Total Generated:** 7,231 jokes
- **Quality Score:** 7.5/10 ⭐ (Tied best)
- **Speed:** ~15 minutes per 100 jokes
- **GPU Memory:** 40-48GB required
- **Quantization:** 4-bit NF4 with bfloat16

**GPU Requirements:**
- Minimum: 40GB VRAM (A6000, L40S)
- Recommended: 48GB+ or multi-GPU

**Advantages:**
- Competitive quality with DeepSeek (7.5/10)
- Distinct creative voice
- Surreal, clever humor
- Good consistency
- Open-source (reproducible)
- Parallelizable design

**Evaluation:** Quirky, introspective, slightly dark comedian—uneven but with real style and genuinely clever turns.

---

### Phase 3: Results Aggregation & Analysis

After all generation completes, run these analyzers in order:

#### Step 1: Merge All Raw Outputs

**Script:** `analyzers/merge_jokes.py`

Combines all individual generator outputs into single CSV with deduplication.

```bash
python3 analyzers/merge_jokes.py
```

**Input:**
- All CSV files in `raw_outputs/` (13+ files from generators)

**Output:**
- `stats/merged_llm_jokes.csv` (78,000+ rows after merge)

**Processing:**
- Globbing: Finds all `*.csv` in `raw_outputs/`
- Encoding: Handles UTF-8 and Latin-1 fallback
- Deduplication: Removes exact duplicate rows
- Logging: Reports file counts and row statistics

---

#### Step 2: Analyze Statistics by Combo Index

**Script:** `analyzers/analyze_jokes_sorted_by_index.py`

Aligns jokes back to original combo index order for coverage analysis.

```bash
python3 analyzers/analyze_jokes_sorted_by_index.py
```

**Output:**
- `stats/jokes_stats_summary.csv` with columns:
  - `Index` - Original combo index (0-4999)
  - `Combo` - Combo text
  - `Total_Jokes` - Total jokes for combo
  - `{model_name}` - Per-model joke count

**Use Cases:**
- Identify combos with complete coverage (100 jokes)
- Find missing combos (< 100 jokes)
- See per-model breakdown per combo
- Detect generation imbalances

---

#### Step 3: Generate Detailed JSON Statistics

**Script:** `analyzers/joke_stats.py`

Creates comprehensive nested JSON with detailed metrics.

```bash
python3 analyzers/joke_stats.py
```

**Output:**
- `stats/llm_jokes_stats.json` - Nested statistics

**Structure:**
```json
{
  "deepseek-chat": {
    "total_model_count": 23567,
    "batch_size_10": {
      "total_jokes_generated": 23567,
      "unique_combos_count": 2356,
      "combos_detail": {
        "dog, cat": {
          "original_index": 0,
          "generated_count": 100
        }
      }
    }
  }
}
```

---

#### Step 4: Sample Jokes Per Model

**Script:** `analyzers/model_joke_samples.py`

Generates representative samples (50 jokes per model).

```bash
python3 analyzers/model_joke_samples.py
```

**Output:**
- `stats/model_joke_samples.json` - 50 random jokes per model

**JSON Structure:**
```json
{
  "deepseek-chat": [
    {
      "combo": "dog, cat",
      "joke_text": "Why do dogs and cats never play cards?..."
    }
  ]
}
```

**Use:** Quick qualitative review without full dataset loading.

---

## Running the Complete Pipeline

### Quick Start Script

```bash
#!/bin/bash
cd /uhh-ias-ml/data/joke_generation/llm_jokes/experiment1.0

echo "=== Experiment 1.0: Full Pipeline ==="

# Phase 1: Preparation
echo "[Phase 1] Extracting noun combinations..."
python3 analyzers/extract_noun_combinations.py
python3 analyzers/top_5000_combos_extracter.py

# Phase 2: Generation (example with DeepSeek)
echo "[Phase 2] Generating jokes..."
export DEEPSEEK_API_KEY="your_key"
python3 generators/deepseek_chat.py

# Phase 3: Analysis
echo "[Phase 3] Aggregating results..."
python3 analyzers/merge_jokes.py
python3 analyzers/analyze_jokes_sorted_by_index.py
python3 analyzers/joke_stats.py
python3 analyzers/model_joke_samples.py

echo "Pipeline complete! Results in stats/"
```

---

## Environment Configuration

### API Keys & Credentials

```bash
# DeepSeek
export DEEPSEEK_API_KEY="sk-your_key"

# Google Gemini
export GOOGLE_API_KEY="AIza..."

# Hugging Face (for Llama/Qwen)
export HF_TOKEN="hf_..."
```

### GPU Requirements

| Model | VRAM | GPU Type |
|-------|------|----------|
| Llama 70B | 80GB | A100, H100 |
| Qwen3-30B | 40-48GB | A6000, L40S |
| Llama 8B | 24GB | RTX 4090 |
| DeepSeek (API) | N/A | Cloud |
| Gemini (API) | N/A | Cloud |

### Python Dependencies

```bash
pip install torch transformers bitsandbytes peft
pip install openai google-generativeai
pip install pandas numpy tqdm
```

---

## Quality Evaluation Results

### Evaluation Methodology

Each model group evaluated by GPT-4 and Gemini using comprehensive rubric:

**Scoring Criteria (1-10):**
1. **Craftsmanship** - Setup/punchline clarity, word economy, misdirection
2. **Originality** - Fresh punchlines, avoids recycled templates
3. **Consistency** - Fewer dead jokes, coherent voice
4. **Voice & Tone** - Distinct persona, steady tone
5. **Edgy Material** - Clever if edgy (not just shocking)

**Score Interpretation:**
- 9-10: Professional-level writing
- 7-8: Strong set with consistent voice
- 5-6: Mixed quality
- 3-4: Mostly weak
- 1-2: Fundamentally broken

### Results Hierarchy

**Tier 1: Excellent (7.5/10)**
- **DeepSeek-Chat** - Smart, polished wordplay
- **Qwen3-30B** - Quirky, distinct creative voice

**Tier 2: Good (7/10)**
- **Gemini-2.5-Pro** - Witty with narrative elements

**Tier 3: Mediocre (4.5/10)**
- **Gemini-2.5-Flash** - Mixed quality

**Tier 4: Poor (2.5-3.5/10)**
- **Llama 70B** - Generic puns
- **Llama 8B** - Bland, uninspired
- **Gemini Flash-Lite** - Tired clichés

### Key Findings

**High-Quality Indicators:**
✓ Clear joke structure (setup → misdirection → punchline)
✓ Wordplay and linguistic creativity
✓ Consistent comedic voice
✓ Wit applied to topics
✓ Minimal dead jokes

**Low-Quality Indicators:**
✗ Generic Q&A templates
✗ Lazy puns without cleverness
✗ Lack of internal logic
✗ Many fragments
✗ Repetitive templates

**Recommendations:**
1. **Best Overall:** DeepSeek-Chat (quality + pricing)
2. **Best Open-Source:** Qwen3-30B (competitive quality)
3. **Avoid:** Llama 3.1 8B (poorest)

---

## Troubleshooting

### API Key Not Found
```
ValueError: API Key not found
```
**Solution:**
```bash
export DEEPSEEK_API_KEY="your_key"
# Or edit script: API_KEY = "sk_xxx"
```

### GPU Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solutions:**
- Reduce `BATCH_SIZE`
- Use smaller model
- Upgrade GPU

### Rate Limiting
```
Error 429: Too Many Requests
```
**Solutions:**
- Wait for window reset
- Upgrade to paid tier
- Reduce request frequency

### Files Not Found
```
FileNotFoundError: stats/combos_top5000_across_k1_to_k5.csv
```
**Solution:**
- Run Phase 1 first: `extract_noun_combinations.py` → `top_5000_combos_extracter.py`

---

## Advanced Usage

### Custom Index Ranges (Parallelization)

Split 5000 combos across multiple nodes:

```python
# Node 1
START_INDEX = 0
END_INDEX = 1250

# Node 2
START_INDEX = 1250
END_INDEX = 2500

# etc.
```

### Resume from Checkpoints

All generators automatically support resuming:
- Reads existing output CSV
- Counts jokes per combo
- Skips completed combos
- Continues from last processed

To restart clean:
```bash
rm raw_outputs/deepseek_jokes.csv
python3 generators/deepseek_chat.py
```

---

## Output Formats

### CSV Format (all generators)

**Columns:**
- `combo` - Noun combination
- `batch_prompt` - Full prompt sent
- `joke_text` - Generated joke
- `model_version` - Model identifier

### JSON Statistics

```json
{
  "model_name": {
    "total_model_count": 23567,
    "batch_size_10": {
      "total_jokes_generated": 23567,
      "unique_combos_count": 2356
    }
  }
}
```

---

## References & Further Reading

### Related Documentation

- [LLM_Jokes.md](LLM_Jokes.md) - Evaluation results and judge feedback
- [PATH_FIX_SUMMARY.md](PATH_FIX_SUMMARY.md) - Path configuration changes

### Parent Documentation

- `/ltstorage/home/4xin/uhh-ias-ml/README.md` - Main project docs
- `/ltstorage/home/4xin/uhh-ias-ml/data/info/README.md` - Data directory overview

---

## Future Work

### Potential Improvements

1. **Finetuning:** Train models on high-quality jokes
2. **Ensemble:** Combine multiple model outputs
3. **Feedback Loops:** Use scores to improve prompts
4. **Human Evaluation:** Extensive human-in-the-loop
5. **Automated Filtering:** Remove non-jokes

### Dataset Reuse

- Training generation models
- Evaluating humor detection
- Linguistic comedy analysis
- Crowdsourced benchmarking

---

## Citation

```bibtex
@misc{experiment_1_0_2026,
  title={LLM-Based Joke Generation Pipeline: Experiment 1.0},
  author={Contributors},
  year={2026},
  howpublished={\url{https://github.com/user/repo}},
  note={78,000 jokes from 7 models on 5000 noun combinations}
}
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-23 | Complete pipeline documentation with 7 models |

---

**Last Updated:** January 23, 2026  
**Status:** ✓ Complete (78,000 jokes generated)  
**Total Models:** 7  
**Master Combos:** 5,000  
**Best Recommendation:** DeepSeek-Chat (7.5/10) or Qwen3-30B (7.5/10)  
