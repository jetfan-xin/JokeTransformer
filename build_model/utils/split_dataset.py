import os
import pandas as pd
from sklearn.model_selection import train_test_split

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

# 8. Shuffle the dataset
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
print(f"✅ Total usable rows after cleaning: {len(df)}")

# 9. Split the dataset: 90% train, 5% val, 5% test
train_df, temp_df = train_test_split(df, test_size=0.10, random_state=42, shuffle=True)
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, shuffle=True)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# 10. Save the splits
os.makedirs(DATA_DIR, exist_ok=True)
train_df.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(DATA_DIR, "val.csv"), index=False)
test_df.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)

print(f"✅ Saved full shuffled dataset split to {DATA_DIR}")
print("   (train=90%, val=5%, test=5%)")