"""
Extract rows whose perplexity is between given percentiles (default 40%-80%)
from the merged data of clean_jokes_detox_safe.csv and clean_jokes_llm_detox_safe.csv,
keeping all columns and writing a CSV.

Usage (from repo root):
  python -m utils.extract_mid_ppl

Customize percentiles or output:
  python -m utils.extract_mid_ppl --low 40 --high 80 \
    --files outputs/detox/clean_jokes_detox_safe.csv outputs/detox/clean_jokes_llm_detox_safe.csv \
    --out outputs/final/clean_good_jokes.csv
"""
import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def load_and_merge(files: List[Path]) -> pd.DataFrame:
    dfs = []
    for p in files:
        if not p.exists():
            print(f"[WARN] skip missing {p}")
            continue
        df = pd.read_csv(p, engine="python", on_bad_lines="skip")
        if "perplexity" not in df.columns:
            print(f"[WARN] missing 'perplexity' in {p}, skipping")
            continue
        if "source" not in df.columns:
            if "source_file" in df.columns:
                df["source"] = df["source_file"]
            else:
                df["source"] = p.name
        dfs.append(df)
    if not dfs:
        raise RuntimeError("No valid input files.")
    return pd.concat(dfs, ignore_index=True)


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
        help="Input CSVs with 'perplexity' column.",
    )
    ap.add_argument("--low", type=float, default=40.0, help="Low percentile")
    ap.add_argument("--high", type=float, default=80.0, help="High percentile")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/final/clean_good_jokes.csv"),
        help="Output CSV path for filtered rows.",
    )
    args = ap.parse_args()

    if args.low < 0 or args.high > 100 or args.low >= args.high:
        raise ValueError("Percentiles must satisfy 0 <= low < high <= 100")

    df = load_and_merge(args.files)
    ppl = df["perplexity"].values
    low_val = np.percentile(ppl, args.low)
    high_val = np.percentile(ppl, args.high)
    mid_df = df[(df["perplexity"] >= low_val) & (df["perplexity"] <= high_val)]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mid_df.to_csv(args.out, index=False)
    print(f"[WRITE] {args.out} (rows={len(mid_df)}, range=[{low_val:.4f}, {high_val:.4f}])")


if __name__ == "__main__":
    main()
