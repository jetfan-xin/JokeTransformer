"""
Sample jokes by perplexity deciles (and extremes) into a JSON file.

Default inputs (run on either pre-filter or post-filter datasets; use --files to switch):
  - outputs/detox/clean_jokes_detox_safe.csv
  - outputs/detox/clean_jokes_llm_detox_safe.csv

For each percentile target (0, 10, ..., 100):
  - 0%: pick lowest N rows
  - 100%: pick highest N rows
  - others: pick N rows whose PPL is closest to that percentile value.

Output JSON structure: list of dicts with keys
  - source
  - id
  - text
  - perplexity
  - group   (e.g., "pct_10", "pct_90", "pct_0_lowest", "pct_100_highest")

Usage (from repo root):
  python -m utils.extract_ppl_samples
  # custom outputs
  python -m utils.extract_ppl_samples --out stats_plots/detox/detox_before_after/ppl_samples.json --sample-size 50

  python -m utils.extract_ppl_samples \
  --files outputs/detox/clean_jokes_detox_safe.csv outputs/detox/clean_jokes_llm_detox_safe.csv \
  --out stats_plots/detox/detox_before_after/ppl_samples_safe.json
"""
import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def load_files(files: List[Path], text_cols: List[str]) -> pd.DataFrame:
    dfs = []
    for p in files:
        if not p.exists():
            print(f"[WARN] skip missing {p}")
            continue
        df = pd.read_csv(p, engine="python", on_bad_lines="skip")
        if "perplexity" not in df.columns:
            print(f"[WARN] missing 'perplexity' in {p}, skipping")
            continue
        text_col = None
        for c in text_cols:
            if c in df.columns:
                text_col = c
                break
        if text_col is None:
            print(f"[WARN] missing text column {text_cols} in {p}, skipping")
            continue
        df = df.rename(columns={text_col: "text"})
        if "source" not in df.columns:
            if "source_file" in df.columns:
                df["source"] = df["source_file"]
            else:
                df["source"] = p.name
        # choose id column if present
        for cand in ["id", "rid", "stable_id"]:
            if cand in df.columns:
                df["chosen_id"] = df[cand]
                break
        else:
            df["chosen_id"] = range(len(df))
        dfs.append(df[["source", "chosen_id", "text", "perplexity"]])
    if not dfs:
        raise RuntimeError("No valid input files.")
    return pd.concat(dfs, ignore_index=True)


def sample_percentiles(df: pd.DataFrame, sample_size: int) -> List[dict]:
    results = []
    targets = list(range(0, 101, 10))  # 0,10,...,100
    ppl_values = df["perplexity"].values
    for pct in targets:
        if pct == 0:
            subset = df.nsmallest(sample_size, "perplexity")
            group = "pct_0_lowest"
        elif pct == 100:
            subset = df.nlargest(sample_size, "perplexity")
            group = "pct_100_highest"
        else:
            target_val = np.percentile(ppl_values, pct)
            df["__diff"] = (df["perplexity"] - target_val).abs()
            subset = df.nsmallest(sample_size, "__diff")
            group = f"pct_{pct}"
        for _, row in subset.iterrows():
            results.append(
                {
                    "source": row["source"],
                    "id": row["chosen_id"],
                    "text": row["text"],
                    "perplexity": float(row["perplexity"]),
                    "group": group,
                }
            )
    if "__diff" in df.columns:
        df.drop(columns="__diff", inplace=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--files",
        nargs="+",
        type=Path,
        default=[
            Path("outputs/detox/clean_jokes_detox_safe.csv"),
            Path("outputs/detox/clean_jokes_llm_detox_safe.csv"),
        ],
        help="Input CSVs with columns: perplexity, text, and source/source_file.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("stats_plots/detox/detox_before_after/ppl_samples_safe.json"),
        help="Output JSON path.",
    )
    ap.add_argument(
        "--text-cols",
        nargs="+",
        default=["text", "joke_cleaned"],
        help="Candidate text column names (first found will be used).",
    )
    ap.add_argument("--sample-size", type=int, default=50, help="Samples per percentile target.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (for reproducible shuffles if any).")
    args = ap.parse_args()

    df = load_files(args.files, args.text_cols)
    samples = sample_percentiles(df, args.sample_size)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"[WRITE] {args.out} (records={len(samples)})")


if __name__ == "__main__":
    main()
