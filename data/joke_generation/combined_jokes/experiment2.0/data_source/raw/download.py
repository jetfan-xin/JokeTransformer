"""
Download Amirkid/jokes from Hugging Face and save it as CSV next to this script.
Usage (from repo root): python data/combined_jokes/experiment2.0/data_source/raw/download.py
"""
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def main():
    out_path = Path(__file__).resolve().parent / "amirkid_jokes.csv"
    if out_path.exists():
        print(f"[SKIP] {out_path} already exists ({out_path.stat().st_size} bytes)")
        return

    print("[LOAD] Fetching Amirkid/jokes (split=train) ...")
    ds = load_dataset("Amirkid/jokes", split="train")
    df = ds.to_pandas()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[DONE] Saved {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
