import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

SAVE_PATH = Path("/ltstorage/home/4xin/uhh-ias-ml/data/ouputs")
SAVE_PATH.mkdir(parents=True, exist_ok=True)

def parse_tags(topic_str):
    """Parse topic_all_nouns into sorted list of unique nouns."""
    if not isinstance(topic_str, str):
        return []
    s = topic_str.strip()
    if s == "" or s.lower() == "nan":
        return []
    tags = [t.strip().lower() for t in s.split(",") if t.strip()]
    return sorted(set(tags))


def build_combination_index(df, topic_col="topic_all_nouns", max_k=5):
    combos_counts_by_k = {k: Counter() for k in range(1, max_k + 1)}
    combos_jokes_by_k = {k: defaultdict(list) for k in range(1, max_k + 1)}

    for idx, topic_str in df[topic_col].items():
        tags = parse_tags(topic_str)
        k = len(tags)
        if 1 <= k <= max_k:
            combo_key = ", ".join(tags)
            combos_counts_by_k[k][combo_key] += 1
            combos_jokes_by_k[k][combo_key].append(idx)

    return combos_counts_by_k, combos_jokes_by_k


def extract_jokes_for_combos(df, combos, combo_to_indices, freq_map):
    """Return DataFrame of jokes for selected combination keys."""
    rows = []
    for combo in combos:
        for idx in combo_to_indices[combo]:
            row = df.loc[idx].copy()
            row["combo_key"] = combo
            row["combo_freq"] = freq_map[combo]
            rows.append(row)

    return pd.DataFrame(rows)


def extract_combos_and_jokes_top5_p75(df, topic_col="topic_all_nouns", max_k=5):
    combos_counts_by_k, combos_jokes_by_k = build_combination_index(
        df, topic_col=topic_col, max_k=max_k
    )

    for k in range(1, max_k + 1):
        counts_counter = combos_counts_by_k[k]
        combo_to_indices = combos_jokes_by_k[k]

        if not counts_counter:
            print(f"[k={k}] No combos found")
            continue

        # Convert to DataFrame
        df_freq = (
            pd.DataFrame(counts_counter.items(), columns=["combo", "freq"])
            .sort_values("freq", ascending=False)
            .reset_index(drop=True)
        )

        # -----------------------------
        # A. TOP 5 MOST FREQUENT COMBOS
        # -----------------------------
        top5 = df_freq.head(5)["combo"].tolist()
        df_top5_jokes = extract_jokes_for_combos(
            df, top5, combo_to_indices, counts_counter
        )

        df_freq.to_csv(SAVE_PATH / f"combos_k{k}_stats.csv", index=False)
        df_top5_jokes.to_csv(SAVE_PATH / f"jokes_top5_combo_k{k}.csv", index=False)

        # -----------------------------
        # B. TOP 5 COMBOS NEAREST TO 75TH PERCENTILE
        # -----------------------------
        p75 = df_freq["freq"].quantile(0.75)

        # sort by distance to 75th percentile, then by freq (descending)
        df_freq["abs_diff"] = (df_freq["freq"] - p75).abs()
        df_p75 = df_freq.sort_values(["abs_diff", "freq"])  # closest to 75%
        p75_top5 = df_p75.head(5)["combo"].tolist()

        df_p75_jokes = extract_jokes_for_combos(
            df, p75_top5, combo_to_indices, counts_counter
        )
        df_p75_jokes.to_csv(SAVE_PATH / f"jokes_p75_top5_combo_k{k}.csv", index=False)

        print(
            f"[k={k}] Total combos={len(df_freq)} | Top5={len(top5)} | P75-Top5={len(p75_top5)}"
        )



df = pd.read_csv("/ltstorage/home/4xin/uhh-ias-ml/data/processed/final_clean_jokes_with_all_nouns.csv", dtype=str).fillna("")
extract_combos_and_jokes_top5_p75(df, topic_col="topic_all_nouns", max_k=5)