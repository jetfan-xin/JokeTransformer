import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Get the script directory and construct relative paths
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent

DEFAULT_INPUT = BASE_DIR / "outputs" / "llm_jokes_top_5000_topics_3_detox_safe_no_repeats.csv"
DEFAULT_OUTPUT = BASE_DIR / "outputs" / "llm_jokes_top_5000_topics_3_detox_safe_no_repeats_sampled100.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--topic-col", default="topic")
    ap.add_argument("--sample-size", type=int, default=100)
    ap.add_argument("--target-total", type=int, default=500000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    df = pd.read_csv(args.input, engine="python", on_bad_lines="skip")
    if args.topic_col not in df.columns:
        raise ValueError(f"Input CSV missing column '{args.topic_col}'.")
    if "stable_id" not in df.columns:
        raise ValueError("Input CSV missing column 'stable_id'.")

    drop_cols = [
        "toxicity",
        "severe_toxicity",
        "obscene",
        "threat",
        "insult",
        "identity_attack",
        "detox_flag",
        "profanity_flag",
        "llamaguard_output",
        "llamaguard_flag",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    rng = np.random.RandomState(args.seed)
    sampled_groups = []
    selected_indices = []
    small_groups = 0
    total_groups = 0

    for combo, group in df.groupby(args.topic_col, sort=False):
        total_groups += 1
        if len(group) <= args.sample_size:
            sampled_groups.append(group)
            selected_indices.extend(group.index.tolist())
            if len(group) < args.sample_size:
                small_groups += 1
            continue
        idx = rng.choice(group.index.values, size=args.sample_size, replace=False)
        selected_indices.extend(idx.tolist())
        sampled_groups.append(df.loc[idx])

    sampled_df = pd.concat(sampled_groups, ignore_index=True)
    if "rid" in sampled_df.columns:
        sampled_df = sampled_df.drop(columns=["rid"])

    added_rows = 0
    if args.target_total and len(sampled_df) < args.target_total:
        gap = args.target_total - len(sampled_df)
        remaining_df = df.loc[~df.index.isin(selected_indices)].copy()
        selected_ids = set(sampled_df["stable_id"].tolist())
        remaining_df = remaining_df[~remaining_df["stable_id"].isin(selected_ids)]
        remaining_df = remaining_df.drop_duplicates(subset=["stable_id"])
        if len(remaining_df) > 0:
            add_n = min(gap, len(remaining_df))
            extra_df = remaining_df.sample(n=add_n, replace=False, random_state=rng)
            if "rid" in extra_df.columns:
                extra_df = extra_df.drop(columns=["rid"])
            sampled_df = pd.concat([sampled_df, extra_df], ignore_index=True)
            added_rows = len(extra_df)

    sampled_df = sampled_df.sort_values("stable_id").reset_index(drop=True)
    sampled_df.insert(0, "rid", range(1, len(sampled_df) + 1))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sampled_df.to_csv(args.output, index=False)

    print(f"Total rows: {len(df)}")
    print(f"Total combos: {total_groups}")
    print(f"Combos below {args.sample_size}: {small_groups}")
    if args.target_total:
        print(f"Target total: {args.target_total}")
    print(f"Added rows to fill gap: {added_rows}")
    print(f"Output rows: {len(sampled_df)}")
    print(f"[WRITE] {args.output}")


if __name__ == "__main__":
    main()
    '''
    (/mnt/data1/users/4xin/conda_envs/ML) 4xin@ltgpu3:~/uhh-ias-ml$ python /ltstorage/home/4xin/uhh-ias-ml/data/llm_jokes/experiment3.0/analyzers/sample_100_per_topic_combo.py
    Total rows: 509013
    Total combos: 5000
    Combos below 100: 114
    Target total: 500000
    Added rows to fill gap: 538
    Output rows: 500000
    '''
