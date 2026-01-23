import argparse
import re
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path(
    "/ltstorage/home/4xin/uhh-ias-ml/data/llm_jokes/experiment3.0/outputs/llm_jokes_top_5000_topics_3_detox_safe.csv"
)
DEFAULT_OUTPUT = Path(
    "/ltstorage/home/4xin/uhh-ias-ml/data/llm_jokes/experiment3.0/outputs/llm_jokes_top_5000_topics_3_detox_safe_no_repeats.csv"
)


def build_repeat_pattern(min_repeat: int) -> re.Pattern:
    repeat_len = max(2, min_repeat)
    return re.compile(rf"([A-Za-z])\1{{{repeat_len - 1},}}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--text-col", default="joke_cleaned")
    ap.add_argument(
        "--min-repeat",
        type=int,
        default=5,
        help="Min consecutive repeated letters to flag (default: 5)",
    )
    ap.add_argument("--save-removed", type=Path, help="Optional CSV for removed rows")
    args = ap.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    df = pd.read_csv(args.input, engine="python", on_bad_lines="skip")
    if args.text_col not in df.columns:
        raise ValueError(f"text column '{args.text_col}' not found in CSV")

    pattern = build_repeat_pattern(args.min_repeat)
    text_series = df[args.text_col].astype(str)
    mask_repeat = text_series.str.contains(pattern, regex=True, na=False)

    removed_df = df[mask_repeat]
    kept_df = df[~mask_repeat]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    kept_df.to_csv(args.output, index=False)
    print(f"[WRITE] kept -> {args.output}")

    if args.save_removed:
        args.save_removed.parent.mkdir(parents=True, exist_ok=True)
        removed_df.to_csv(args.save_removed, index=False)
        print(f"[WRITE] removed -> {args.save_removed}")

    print(f"Total rows: {len(df)}")
    print(f"Removed rows: {len(removed_df)}")
    print(f"Kept rows: {len(kept_df)}")


if __name__ == "__main__":
    main()
    '''
    [WRITE] kept -> /ltstorage/home/4xin/uhh-ias-ml/data/llm_jokes/experiment3.0/outputs/llm_jokes_top_5000_topics_3_detox_safe_no_repeats.csv
    [WRITE] removed -> /ltstorage/home/4xin/uhh-ias-ml/data/llm_jokes/experiment3.0/stats/removed_repeated_char_joke.csv
    Total rows: 509087
    Removed rows: 74
    Kept rows: 509013
    '''