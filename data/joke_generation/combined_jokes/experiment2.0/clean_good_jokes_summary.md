# Clean & Filtered Jokes Pipeline (Diagram‑Aligned Summary)

This document describes the full pipeline designed for obtaining a high‑quality jokes dataset. It combines:

- multi‑source joke collection (human + LLM),
- cleaning + perplexity (PPL) analysis,
- toxicity evaluation and filtering,
- re‑checking PPL on safe data,
- and final topic extraction.

---

## 0. Data Sources

### Original datasets (non‑LLM)

Start from several human / web joke datasets:

1. `amirkiid-jokes.csv`
2. `shuttie-dadjokes.csv`
3. `ysharma-short-jokes.csv`
4. `train.tsv.gz` (rJokes)
5. `kaggle-dataset.csv` → **removed** (duplicated with dataset #3)

### LLM‑generated datasets

6. `deepseek-jokes-dataset.csv`
7. `gemini-....csv`

After noticing that `kaggle-dataset.csv` is effectively the same as dataset #3, drop it and keep the remaining 6 datasets.

---

## A. Collect datasets (6 datasets collected)

- Aggregate all remaining usable datasets into a single **collected pool**.
- Because `kaggle-dataset.csv` is removed as a duplicate, the pool contains **6 datasets** in total (3 + 1 human/web + 2 LLM).

**Output of stage A**

- A unified raw pool of **6 datasets collected**.

---

## B. Cleaning + PPL Analysis + Sampling + Toxicity Evaluation

Starting from the collected pool:

### B1) Clean the dataset

- Run a cleaning pipeline (using `utils/preprocessing_jingfan.py`):
  - normalize text (unicode, punctuation),
  - remove URLs, emails, markup,
  - filter by length (too short / too long),
  - deduplicate jokes.

**Output**

- A **cleaned dataset** of jokes in a standardized format.

### B2) Compute PPL and plot PPL histograms

- For each cleaned joke, compute **perplexity (PPL)** using distilgpt2 (`utils/training_data_eval.py`).
- Plot **PPL histograms** (`utils/plot_ppl_joke_sources.py`) to understand the distribution of difficulty/fluency across jokes.

**Output**

- PPL histogram(s)
  - `data/joke_sources/ppl_joke_sources.html`
  - `data/joke_sources/ppl_joke_sources.png`
- CSVs with PPL scores:
  - `data/combined_data/train_quality_metrics.csv`
  - `data/combined_data/train_quality_metrics_llm.csv`

### B3) Extract samples at every 10 percentiles (by PPL)

- Extract representative samples for inspection at each **10% PPL percentile bin**:
  - 0–10%, 10–20%, …, 90–100%.
- This is implemented via tools like `utils/extract_ppl_samples.py`.

**Purpose**

- To qualitatively inspect what jokes look like across the PPL spectrum, and to decide which PPL regions correspond to low‑quality jokes.

### B4) Evaluate toxicity

- Evaluate **toxicity** of the cleaned dataset:
  - run Detoxify models via `utils/safety_filter.py`,
  - obtain scores for `toxicity`, `severe_toxicity`, `obscene`, `threat`, `insult`, `identity_attack`.

### B5) Decision from stage B

Based on PPL histograms, samples, and toxicity scores, decide it is necessary to filter out:

1. **Extreme PPL regions**  
   - jokes in the **lowest 10%** PPL percentile (< 10%), and  
   - jokes in the **highest 90–100%** PPL percentile (> 90%).  
   These regions tend to correspond to poor quality (too trivial or nonsensical).

2. **Toxic data**  
   - jokes that are clearly toxic according to Detoxify scores should be removed.
   - The specific toxicity threshold is formalized in stage C.

---

## C. Detox Step + Toxicity Distribution + Toxicity Threshold

Starting from the cleaned dataset with PPL and toxicity scores:

### C1) Detox

- Apply a **detox filtering step** based on Detoxify outputs:
  - script: `utils/filter_safe_toxicity.py`.

### C2) Plot toxicity histograms

- Plot **toxicity histograms** to visualize the distribution and confirm where most data lies (`utils/plot_toxicity.py`, `utils/plot_toxicity_dims.py`).

### C3) Keep only low‑toxicity jokes

- From these histograms and analyses, choose a clear numeric threshold:

> **keep only jokes with toxicity < 0.1**

- In practice, drop jokes where **any** Detoxify dimension (toxicity, severe_toxicity, obscene, threat, insult, identity_attack) is ≥ 0.1.

**Output**

- A **safe data** subset:
  - `data/combined_data/clean_jokes_detox_safe.csv`
  - `data/combined_data/clean_jokes_llm_detox_safe.csv`
  - plus a summary JSON: `data/combined_data/detox_before_after/safe_filter_summary.json` (per‑source and overall removal ratios).

---

## D. Safe‑Data Re‑check with PPL + Sampling + “GPT as a Judge”

Now take the **safe data** from stage C and repeat the PPL analysis to confirm the quality patterns.

### D1) Compute PPL and plot PPL histograms again

- Using the safe datasets (`*_detox_safe.csv`), attach PPL back (via `utils/add_ppl_to_safe.py`) and:
  - compute/align PPL scores,
  - plot new PPL histograms (`utils/plot_ppl_before_after.py`).

### D2) Extract samples at every 10 percentiles again

- Again extract samples from **each 10% PPL percentile** to inspect quality on the safe data (`utils/extract_ppl_samples.py`).

### D3) Observation

- Observe that:

> The PPL patterns saw earlier still hold, but now **without toxic jokes**.

- In other words, the safe subset has similar PPL distribution and quality patterns as in stage B, just cleaned of toxicity.

### D4) GPT as a judge → final PPL‑based recommendation

- Then ask GPT (and my own judgement) to decide **which PPL range corresponds to truly high‑quality jokes**.
- GPT recommends keeping jokes whose PPL percentile lies in:

> **40% – 80% percentile**

- Therefore, **final PPL‑based selection rule** becomes:
  - **retain jokes whose PPL percentile is within [40%, 80%]**.

This rule is implemented in `utils/extract_mid_ppl.py`, which:

- merges `clean_jokes_detox_safe.csv` and `clean_jokes_llm_detox_safe.csv`,
- computes global 40th and 80th PPL percentiles,
- keeps all jokes whose PPL ∈ [40th, 80th] percentile,
- and writes the result as a combined CSV.

---

## E. Final High‑Quality Dataset

After applying:

- the **toxicity filter** (toxicity < 0.1, plus other Detoxify dimensions), and
- the **PPL percentile filter** (40%–80% on safe data),

Obtain final **clean + high‑quality** jokes dataset:

- `data/processed/clean_good_jokes.csv`

This dataset has:

- diverse topics from multiple sources (human + LLM),
- low toxicity (as defined by Detoxify thresholds),
- moderate PPL (avoiding trivial and extreme examples),
- and is thus suitable as a high‑quality training/evaluation corpus.

---

## F. Post‑processing: Noun Extraction & Derived Outputs

Starting from `data/processed/clean_good_jokes.csv`, add semantic topic information.

### F1) Extract nouns

- Run noun extraction on the jokes:
  - script: `utils/prepare_clean_good.py` (and related `add_all_nouns.py` logic),
  - spaCy model: `en_core_web_sm`.
- For each joke:
  - extract lemmas of all NOUN/PROPN tokens,
  - filter stopwords and a small custom stop list,
  - rank by frequency and length,
  - join into a comma‑separated string `topic_all_nouns`.

### F2) Derived datasets

Produce two final derived datasets:

1. **All‑nouns version**
   - `data/processed/clean_good_jokes_all_nouns.csv`
   - Contains all columns from `data/processed/clean_good_jokes.csv` plus:
     - `topic_all_nouns`

2. **Single‑topic version**
   - `data/processed/clean_good_jokes_single_topic.csv`
   - Contains:
     - everything in the all‑nouns version, and
     - `topic_single` (the first noun from `topic_all_nouns`).
   - This mirrors the format of earlier `*_single_topic` datasets and is convenient for models that expect a single topic field.

---

## Final Outputs (Checklist)

- **Safe, mid‑PPL joke pool**
  - `data/processed/clean_good_jokes.csv`
- **With topic annotations**
  - `data/processed/final_clean_good_jokes_all_nouns.csv`
  - `data/processed/final_clean_good_jokes_single_topic.csv`
- **Supporting artifacts**
  - PPL metrics: `data/combined_data/train_quality_metrics*.csv`
  - Toxicity metrics and safe subsets: `data/combined_data/clean_jokes*_detox_safe.csv`, summary `data/combined_data/detox_before_after/safe_filter_summary.json`
  - PPL plots: source overlay `data/joke_sources/ppl_joke_sources.{html,png}`, before/after `data/combined_data/ppl_before_after/*`
  - Toxicity plots: `toxicity_distribution*.html/png`, `toxicity_dims_distribution*.html/png`
  - Sampled PPL examples: `ppl_samples*.json`

This summarizes the complete workflow described in the diagram:  
**from multi‑source raw jokes → cleaned & de‑duplicated → PPL‑scored → toxicity‑filtered → PPL‑refined (40–80% percentile) → topic‑annotated, high‑quality final joke datasets.**
