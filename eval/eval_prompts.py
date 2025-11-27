import pandas as pd
from pathlib import Path



INPUT_PATH  = Path("../data/processed/final_clean_jokes.csv")
OUTPUT_PATH = Path("../data/processed/eval_prompts.csv")
N1, N2, N3  = 150, 150, 150
SEED        = 42

TOPIC_COLS = ["topic_1", "topic_2", "topic_3"]



def split_topic_column(df):
    def clean_topic_string(x):
        if pd.isna(x):
            return []
        # split by comma
        items = [t.strip() for t in str(x).split(",")]
        return [t for t in items if t]

    # Create new columns
    topics_list = df["topic"].apply(clean_topic_string)

    df["topic_1"] = topics_list.apply(lambda lst: lst[0] if len(lst) >= 1 else pd.NA)
    df["topic_2"] = topics_list.apply(lambda lst: lst[1] if len(lst) >= 2 else pd.NA)
    df["topic_3"] = topics_list.apply(lambda lst: lst[2] if len(lst) >= 3 else pd.NA)

    return df


def sample_group(df, n, name):
    """
    Sample N rows; if dataset is smaller, take all.
    """
    count = len(df)
    if count == 0:
        print(f"[WARN] No rows with: {name}")
        return df

    if count <= n:
        print(f"[INFO] {name}: only {count} rows available. Using all.")
        return df.sample(count, random_state=SEED)
    else:
        print(f"[INFO] {name}: sampling {n} of {count} rows.")
        return df.sample(n, random_state=SEED)


# Main Script
def main():
    print(f"[LOAD] Reading {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    df["topic_list"] = df["topic"].apply(lambda x: [] if pd.isna(x) else [t.strip() for t in str(x).split(",") if t.strip()])
    df["topic_count"] = df["topic_list"].apply(len)

    print("Max number of topics in dataset:", df["topic_count"].max())
    print(df["topic_count"].value_counts().sort_index())


    if "topic" not in df.columns:
        raise ValueError(f"Input CSV missing required column 'topic'.")

    df = split_topic_column(df)

    # Remove duplicate topic combinations
    before = len(df)
    df = df.drop_duplicates(subset=["topic_1", "topic_2", "topic_3"])
    after = len(df)
    print(f"[INFO] Removed {before - after} duplicated topic combinations.")

    #Categorize by topic count
    df["topic_size"] = df[["topic_1", "topic_2", "topic_3"]].notna().sum(axis=1)

    df1 = df[df["topic_size"] == 1]
    df2 = df[df["topic_size"] == 2]
    df3 = df[df["topic_size"] == 3]

    print(f"[INFO] Found:")
    print(f"1-topic rows: {len(df1)}")
    print(f"2-topic rows: {len(df2)}")
    print(f"3-topic rows: {len(df3)}")

    # Sample per group
    df1s = sample_group(df1, N1, "1-topic")
    df2s = sample_group(df2, N2, "2-topic")
    df3s = sample_group(df3, N3, "3-topic")

    # Combine & shuffle
    eval_df = pd.concat([df1s, df2s, df3s], ignore_index=True)
    eval_df = eval_df.sample(len(eval_df), random_state=SEED).reset_index(drop=True)

    # Add eval_id
    eval_df["eval_id"] = range(len(eval_df))

    # Keep only topic columns + id
    eval_df = eval_df[["eval_id"] + TOPIC_COLS]

    # Save final CSV
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(OUTPUT_PATH, index=False)

    print(f"[DONE] Wrote {len(eval_df)} eval prompts → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
    import pandas as pd

    df = pd.read_csv("../data/processed/eval_prompts.csv")

    # Check how many rows have duplicate (topic_1, topic_2, topic_3)
    dup_mask = df.duplicated(subset=["topic_1", "topic_2", "topic_3"], keep=False)
    dups = df[dup_mask].sort_values(["topic_1", "topic_2", "topic_3"])

    print(f"Total rows: {len(df)}")
    print(f"Rows that belong to a duplicated topic combo: {len(dups)}")

   