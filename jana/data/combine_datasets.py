#!/usr/bin/env python3
import gzip
from pathlib import Path
from typing import Iterable, Tuple, Optional
import pandas as pd

RAW_DIR = Path("raw")
OUT_DIR = Path("raw_combined")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def iter_two_col_tsv(path: Path) -> Iterable[Tuple[str, str]]:
    """Yield (col1, col2) by splitting each line at the FIRST tab."""
    open_fn = gzip.open if str(path).endswith(".gz") else open
    with open_fn(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            yield parts[0], parts[1]

def _best_col(df: pd.DataFrame, candidates: set[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    return None

def log(msg: str):
    print(f"[combine] {msg}")

# Sources 
def load_ysharma_short_jokes() -> list[str]:
    p = RAW_DIR / "ysharma_short_jokes.csv"
    jokes = []
    if p.exists():
        log(f"Loading {p} ...")
        df = pd.read_csv(p, dtype=str).fillna("")
        joke_col = _best_col(df, {"joke"})
        if not joke_col:
            log(f"WARN: no 'joke' column found. Columns: {list(df.columns)}")
            return jokes
        for j in df[joke_col]:
            j = str(j).strip()
            if j:
                jokes.append(j)
        log(f"Loaded {len(jokes)} jokes from ysharma.")
    else:
        log(f"WARN: {p} not found; skipping.")
    return jokes

def load_rjokes_train() -> list[str]:
    p_plain = RAW_DIR / "train.tsv"
    p_gzip  = RAW_DIR / "train.tsv.gz"
    p = p_plain if p_plain.exists() else (p_gzip if p_gzip.exists() else None)
    jokes = []
    if p and p.exists():
        log(f"Loading {p.name} ...")
        count = 0
        for score, text in iter_two_col_tsv(p):
            t = str(text).strip()
            if t:
                jokes.append(t)
                count += 1
        log(f"Loaded {count} from rJokesData.")
    else:
        log("WARN: train.tsv / train.tsv.gz not found; skipping.")
    return jokes

def load_kaggle_pos_jokes() -> list[str]:
    p = RAW_DIR / "kaggle_dataset.csv"
    jokes = []
    if p.exists():
        log(f"Loading {p} ...")
        df = pd.read_csv(p, dtype=str).fillna("")
        lower_map = {c.lower(): c for c in df.columns}
        text_col  = lower_map.get("text")
        label_col = lower_map.get("humor")
        if not text_col or not label_col:
            log(f"WARN: expected columns 'text' and 'humor'. Got {list(df.columns)}")
            return jokes

        def as_bool(x) -> bool:
            s = str(x).strip().lower()
            return s in {"1","true","yes","y","humor","humour","funny","positive"}

        df = df[df[label_col].map(as_bool)]
        for t in df[text_col].astype(str):
            t = t.strip()
            if t:
                jokes.append(t)
        log(f"Loaded {len(jokes)} positive jokes from Kaggle.")
    else:
        log(f"WARN: {p} not found; skipping.")
    return jokes

def load_dadjokes() -> list[str]:
    p = RAW_DIR / "shuttie_dadjokes.csv"
    jokes = []
    if p.exists():
        log(f"Loading {p} ...")
        df = pd.read_csv(p, dtype=str).fillna("")
        q_col = _best_col(df, {"question", "setup", "title"})
        r_col = _best_col(df, {"response", "answer", "punchline", "body"})
        if not q_col and not r_col:
            log(f"WARN: no question/response columns. Columns: {list(df.columns)}")
            return jokes
        for _, rr in df.iterrows():
            q = str(rr.get(q_col, "")).strip() if q_col else ""
            r = str(rr.get(r_col, "")).strip() if r_col else ""
            if q and r:
                joiner = "" if q.endswith(('.', '!', '?')) else " — "
                jokes.append(f"{q}{joiner}{r}")
            elif q or r:
                jokes.append(q or r)
        log(f"Loaded {len(jokes)} dadjokes.")
    else:
        log(f"WARN: {p} not found; skipping.")
    return jokes

def load_amirkid_jokes() -> list[str]:
    p = RAW_DIR / "amirkid_jokes.csv"
    jokes = []

    if p.exists():
        log(f"Loading local file {p} ...")
        df = pd.read_csv(p, dtype=str).fillna("")
    else:
        try:
            from datasets import load_dataset
            log("Loading Hugging Face dataset Amirkid/jokes (train split) ...")
            ds = load_dataset("Amirkid/jokes", split="train")
            df = ds.to_pandas()
        except Exception as e:
            log(f"WARN: Could not load Amirkid/jokes ({e}); skipping.")
            return jokes

    # Identify text column
    joke_col = _best_col(df, {"joke", "text", "content"})
    if not joke_col:
        log(f"WARN: No obvious joke column found. Columns: {list(df.columns)}")
        return jokes

    # Extract jokes
    for j in df[joke_col]:
        j = str(j).strip()
        if j:
            jokes.append(j)
    log(f"Loaded {len(jokes)} jokes from Amirkid/jokes.")
    return jokes

def main():
    all_rows: list[str] = []
    all_rows += load_ysharma_short_jokes()
    all_rows += load_rjokes_train()
    all_rows += load_kaggle_pos_jokes()
    all_rows += load_dadjokes()
    all_rows += load_amirkid_jokes()   

    df = pd.DataFrame({"raw_text": all_rows})
    out_csv = OUT_DIR / "combined_raw.csv"
    df.to_csv(out_csv, index=False)
    log(f"Wrote {out_csv} | rows={len(df)}")

if __name__ == "__main__":
    main()
