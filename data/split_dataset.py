# data/split_dataset.py

import os
import pandas as pd

# 1. Project root directory (parent of this file's directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. Data directory and source file path
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
SRC_PATH = os.path.join(DATA_DIR, "final_combined_jokes.csv")

print("Loading:", SRC_PATH)
if not os.path.exists(SRC_PATH):
    raise FileNotFoundError(f"❌ File not found: {SRC_PATH}")

# 3. Read the full original dataset
df = pd.read_csv(SRC_PATH)

# 4. Align column names: rename topic -> topics (to match JokeDataset)
if "topic" in df.columns and "topics" not in df.columns:
    df = df.rename(columns={"topic": "topics"})

# 5. Keep only the required columns
cols = [c for c in ["joke", "topics"] if c in df.columns]
df = df[cols]

# 6. Drop rows without valid jokes
df["joke"] = df["joke"].astype(str)
df = df[df["joke"].str.strip().astype(bool)]

# 7. If there's no topics column, create an empty one (unconditional generation)
if "topics" not in df.columns:
    df["topics"] = ""

# 8. Shuffle the dataset and select the first 100 rows
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

N_TOTAL = 100
if len(df) < N_TOTAL:
    raise ValueError(f"Not enough data rows ({len(df)} found, expected {N_TOTAL}).")

tiny = df.iloc[:N_TOTAL].copy().reset_index(drop=True)

# 9. Split into 80 / 10 / 10 for train / validation / test
train_df = tiny.iloc[:80].copy()
val_df = tiny.iloc[80:90].copy()
test_df = tiny.iloc[90:100].copy()

print("✅ Tiny split completed:")
print(f"  Train: {len(train_df)}")
print(f"  Val:   {len(val_df)}")
print(f"  Test:  {len(test_df)}")

# 10. Save the splits (overwrite current train/val/test for debugging)
os.makedirs(DATA_DIR, exist_ok=True)
train_df.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(DATA_DIR, "val.csv"), index=False)
test_df.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)

print(f"✅ Saved tiny train/val/test to {DATA_DIR}")
print("   (train=80, val=10, test=10)")