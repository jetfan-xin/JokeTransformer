import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Get the script directory and construct relative paths
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent

DEFAULT_INPUT = BASE_DIR / "stats" / "llm_jokes_top_5000_topics_3_detox_safe_ppl.csv"
DEFAULT_OUTPUT = BASE_DIR / "stats" / "llm_jokes_top_5000_topics_3_detox_safe_ppl_samples.json"


def load_file(path: Path, text_cols: List[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    if "perplexity" not in df.columns:
        raise ValueError(f"{path} missing 'perplexity' column.")
    text_col = None
    for c in text_cols:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        raise ValueError(f"{path} missing text column {text_cols}.")
    df = df.rename(columns={text_col: "text"})
    if "source" not in df.columns:
        if "source_file" in df.columns:
            df["source"] = df["source_file"]
        else:
            df["source"] = path.name
    for cand in ["id", "rid", "stable_id"]:
        if cand in df.columns:
            df["chosen_id"] = df[cand]
            break
    else:
        df["chosen_id"] = range(len(df))
    return df[["source", "chosen_id", "text", "perplexity"]]


def sample_percentiles(df: pd.DataFrame, sample_size: int) -> List[dict]:
    results = []
    targets = list(range(0, 101, 10))
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
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument(
        "--text-cols",
        nargs="+",
        default=["text", "joke_cleaned"],
        help="Candidate text column names (first found will be used).",
    )
    ap.add_argument("--sample-size", type=int, default=50)
    args = ap.parse_args()

    df = load_file(args.input, args.text_cols)
    samples = sample_percentiles(df, args.sample_size)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"[WRITE] {args.out} (records={len(samples)})")


if __name__ == "__main__":
    main()
