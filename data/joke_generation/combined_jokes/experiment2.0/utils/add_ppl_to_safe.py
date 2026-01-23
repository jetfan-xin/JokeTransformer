"""
Merge perplexity scores into safe CSVs based on rid (or chosen key).

Default inputs:
  Metrics:
    - outputs/perplexity/train_quality_metrics.csv
    - outputs/perplexity/train_quality_metrics_llm.csv
  Safe files:
    - outputs/detox/clean_jokes_detox_safe.csv
    - outputs/detox/clean_jokes_llm_detox_safe.csv

By default joins on column 'rid'. If your metrics use 'id' instead, pass --key id.

Output:
  Adds a 'perplexity' column to each safe file and writes back (or with a suffix if provided).

Usage (from repo root):
  python -m utils.add_ppl_to_safe
  # custom key and suffix (write new files)
  python -m utils.add_ppl_to_safe --key id --out-suffix "_with_ppl"
"""
import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_metrics(paths: List[Path], key: str) -> Dict:
    series_list = []
    for p in paths:
        if not p.exists():
            print(f"[WARN] skip missing metrics: {p}")
            continue
        df = pd.read_csv(p, engine="python", on_bad_lines="skip")
        # choose effective key
        if key in df.columns:
            use_key = key
        elif key == "rid" and "id" in df.columns:
            use_key = "id"
            print(f"[INFO] metrics file {p} missing 'rid', using 'id' instead for join")
        else:
            print(f"[WARN] metrics file {p} missing '{key}' (and no fallback); skipping")
            continue
        if "perplexity" not in df.columns:
            print(f"[WARN] metrics file {p} missing 'perplexity'; skipping")
            continue
        tmp = df[[use_key, "perplexity"]].rename(columns={use_key: key})
        series_list.append(tmp)
    if not series_list:
        raise RuntimeError("No valid metrics files.")
    merged = pd.concat(series_list, ignore_index=True)
    # drop duplicate keys keeping first occurrence
    merged = merged.drop_duplicates(subset=key, keep="first")
    return pd.Series(merged["perplexity"].values, index=merged[key]).to_dict()


def process_safe_file(path: Path, key: str, ppl_map: Dict, out_suffix: str):
    if not path.exists():
        print(f"[WARN] skip missing safe file: {path}")
        return
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    if key not in df.columns:
        raise ValueError(f"{path} missing key column '{key}'")

    df["perplexity"] = df[key].map(ppl_map)
    matched = df["perplexity"].notna().sum()
    out_path = path if not out_suffix else path.with_name(path.stem + out_suffix + path.suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[WRITE] {out_path} (matched {matched}/{len(df)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metrics",
        nargs="+",
        type=Path,
        default=[
            Path("outputs/perplexity/train_quality_metrics.csv"),
            Path("outputs/perplexity/train_quality_metrics_llm.csv"),
        ],
        help="Metric CSVs containing key and perplexity.",
    )
    ap.add_argument(
        "--safe-files",
        nargs="+",
        type=Path,
        default=[
            Path("outputs/detox/clean_jokes_detox_safe.csv"),
            Path("outputs/detox/clean_jokes_llm_detox_safe.csv"),
        ],
        help="Safe CSVs to receive the perplexity column.",
    )
    ap.add_argument(
        "--key",
        type=str,
        default="rid",
        help="Key column to join on (e.g., rid or id).",
    )
    ap.add_argument(
        "--out-suffix",
        type=str,
        default="",
        help="Optional suffix for output files; if empty, overwrite input safe files.",
    )
    args = ap.parse_args()

    ppl_map = load_metrics(args.metrics, args.key)
    for safe_path in args.safe_files:
        process_safe_file(safe_path, args.key, ppl_map, args.out_suffix)


if __name__ == "__main__":
    main()
