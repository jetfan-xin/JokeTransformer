# data/split_tiny_dataset.py

import os
import pandas as pd

# 1. Project root directory: one level above this script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. Data directory and source file path
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
SRC_PATH = os.path.join(DATA_DIR, "final_combined_jokes.csv")

print("Loading:", SRC_PATH)
if not os.path.exists(SRC_PATH):
    raise FileNotFoundError(f"❌ File not found: {SRC_PATH}")

# 3. Read the full original dataset
df = pd.read_csv(SRC_PATH)

# 4. Align column names: rename topic -> topics
if "topic" in df.columns and "topics" not in df.columns:
    df = df.rename(columns={"topic": "topics"})

# 5. Keep only the required columns
cols = []
for c in ["joke", "topics"]:
    if c in df.columns:
        cols.append(c)
df = df[cols]

# 6. Drop rows without valid jokes
df["joke"] = df["joke"].astype(str)
df = df[df["joke"].str.strip().astype(bool)]

# If there's no topics column, fill with empty strings
# (the model can still train, just unconditioned)
if "topics" not in df.columns:
    df["topics"] = ""

# 7. Shuffle the dataset and select the first 100 rows
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

N_TOTAL = 10000
if len(df) < N_TOTAL:
    raise ValueError(f"Not enough data rows ({len(df)} found, expected {N_TOTAL}).")

tiny = df.iloc[:N_TOTAL].copy().reset_index(drop=True)

# 8. Split into 80 / 10 / 10 for train / validation / test
train_df = tiny.iloc[:8000].copy()
val_df   = tiny.iloc[8000:9000].copy()
test_df  = tiny.iloc[9000:10000].copy()

print(f"✅ Tiny split completed:")
print(f"  Train: {len(train_df)}")
print(f"  Val:   {len(val_df)}")
print(f"  Test:  {len(test_df)}")

# 9. Save the splits (overwrite current train/val/test for debugging)
os.makedirs(DATA_DIR, exist_ok=True)
train_df.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(DATA_DIR, "val.csv"), index=False)
test_df.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)

print(f"✅ Saved tiny train/val/test to {DATA_DIR}")
print("   (train=80, val=10, test=10)")