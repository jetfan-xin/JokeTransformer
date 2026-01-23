"""
Filter jokes by Detoxify toxicity dimensions and write safe subsets + summary.

Rule:
  Drop any row where *any* of the following dimensions >= threshold (default 0.1):
    - toxicity
    - severe_toxicity
    - obscene
    - threat
    - insult
    - identity_attack

For each input CSV:
  - Read full data (e.g., clean_jokes_detox.csv, clean_jokes_llm_detox.csv).
  - Keep only rows where all dimensions < threshold.
  - Write a new CSV with suffix "_safe" in the same directory.

Summary JSON:
  - For each source (from 'source' or 'source_file' column, or filename fallback):
      * total_rows
      * removed_rows
      * removed_ratio
  - Overall totals across all inputs.

Usage (from repo root):
  python -m utils.filter_safe_toxicity

or with custom inputs/threshold/output:
  python -m utils.filter_safe_toxicity \
    --files outputs/detox/clean_jokes_detox.csv outputs/detox/clean_jokes_llm_detox.csv \
    --threshold 0.1 \
    --summary-out stats_plots/detox/detox_before_after/safe_filter_summary.json
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

DIMENSIONS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
]


def detect_source_column(df: pd.DataFrame, default_name: str) -> str:
    if "source" in df.columns:
        return "source"
    if "source_file" in df.columns:
        return "source_file"
    df["source"] = default_name
    return "source"


def process_file(
    path: Path,
    threshold: float,
    summary_totals: Dict[str, int],
    summary_removed: Dict[str, int],
) -> Path:
    if not path.exists():
        print(f"[WARN] skip missing {path}")
        return path

    print(f"[READ] {path}")
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")

    source_col = detect_source_column(df, path.name)

    missing_dims = [c for c in DIMENSIONS if c not in df.columns]
    if missing_dims:
        raise ValueError(f"{path} is missing Detoxify columns: {missing_dims}")

    # update totals by source before filtering
    for src, cnt in df[source_col].value_counts().items():
        summary_totals[src] = summary_totals.get(src, 0) + int(cnt)

    # mask of rows to drop (any dim >= threshold)
    toxic_mask = (df[DIMENSIONS] >= threshold).any(axis=1)
    removed_df = df[toxic_mask]
    safe_df = df[~toxic_mask]

    for src, cnt in removed_df[source_col].value_counts().items():
        summary_removed[src] = summary_removed.get(src, 0) + int(cnt)

    out_path = path.with_name(path.stem + "_safe" + path.suffix)
    print(f"[WRITE] safe subset -> {out_path} (kept {len(safe_df)}/{len(df)})")
    safe_df.to_csv(out_path, index=False)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--files",
        nargs="+",
        type=Path,
        default=[
            Path("outputs/detox/clean_jokes_detox.csv"),
            Path("outputs/detox/clean_jokes_llm_detox.csv"),
        ],
        help="Input CSV files with Detoxify columns.",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Threshold for dropping rows (drop if any dimension >= threshold).",
    )
    ap.add_argument(
        "--summary-out",
        type=Path,
        default=Path("stats_plots/detox/detox_before_after/safe_filter_summary.json"),
        help="Where to write JSON summary.",
    )
    args = ap.parse_args()

    totals: Dict[str, int] = {}
    removed: Dict[str, int] = {}

    for path in args.files:
        process_file(path, args.threshold, totals, removed)

    overall_total = sum(totals.values())
    overall_removed = sum(removed.values())

    per_source = {}
    for src, total_cnt in totals.items():
        rem_cnt = removed.get(src, 0)
        ratio = rem_cnt / total_cnt if total_cnt else 0.0
        per_source[src] = {
            "total_rows": int(total_cnt),
            "removed_rows": int(rem_cnt),
            "removed_ratio": ratio,
        }

    summary = {
        "threshold": args.threshold,
        "per_source": per_source,
        "overall": {
            "total_rows": int(overall_total),
            "removed_rows": int(overall_removed),
            "removed_ratio": (overall_removed / overall_total) if overall_total else 0.0,
        },
    }

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[WRITE] summary -> {args.summary_out}")


if __name__ == "__main__":
    main()
