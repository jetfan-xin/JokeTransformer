import pandas as pd
from pathlib import Path

# -----------------------------
# Input files
# -----------------------------
base = Path("/ltstorage/home/4xin/uhh-ias-ml/data/ouputs")

files = [
    base / "combos_k1_stats.csv",
    base / "combos_k2_stats.csv",
    base / "combos_k3_stats.csv",
    base / "combos_k4_stats.csv",
    base / "combos_k5_stats.csv",
]

# -----------------------------
# Load + Combine frequencies
# -----------------------------
all_rows = []

for file in files:
    print(f"Loading {file} ...")
    df = pd.read_csv(file)

    df.columns = ["combo", "freq"]

    # Clean combo string: remove outer quotes, trim
    df["combo"] = df["combo"].astype(str).str.strip()
    df["combo"] = df["combo"].str.replace('^"|"$', '', regex=True)

    all_rows.append(df)

all_df = pd.concat(all_rows, ignore_index=True)

# -----------------------------
# Group by combo and sum freq
# -----------------------------
combo_freq = (
    all_df.groupby("combo")["freq"]
    .sum()
    .reset_index()
    .sort_values("freq", ascending=False)
)

# -----------------------------
# Apply required formatting:
# ALL combos must be double-quoted
# -----------------------------
def format_combo(c: str) -> str:
    """
    Add quotes around the entire combo string.
    e.g.:
    - dad → "dad"
    - bar, man → "bar, man"
    """
    c = c.strip()
    return f'"{c}"'

combo_freq["combo"] = combo_freq["combo"].apply(format_combo)

# -----------------------------
# Select top 5000
# -----------------------------
top_n = 5000
top_5000 = combo_freq.head(top_n)

# -----------------------------
# Save output
# -----------------------------
OUT_FILE = base / "combos_top5000_across_k1_to_k5.csv"
top_5000.to_csv(OUT_FILE, index=False)

print(f"Saved formatted top {top_n} combos to:")
print(OUT_FILE)