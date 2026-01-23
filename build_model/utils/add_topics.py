#!/usr/bin/env python3
from pathlib import Path
import multiprocessing
import pandas as pd
import spacy

IN_PATH  = Path("combined_data/clean_jokes.csv")
OUT_DIR  = Path("combined_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

STOP_ADD = set("""
joke jokes thing things one ones way time today tonight day guy guys people someone something anything everything nothing
""".split())

def log(msg: str):
    print(f"[topics] {msg}")

def topic_from_doc(doc, max_terms=3):
    nouns = [t.lemma_.lower().strip()
             for t in doc
             if t.pos_ in ("NOUN","PROPN") and not t.is_stop and len(t) >= 3]
    nouns = [w for w in nouns if w not in STOP_ADD]
    if not nouns:
        alts = [t.lemma_.lower().strip()
                for t in doc
                if t.pos_ in ("VERB","ADJ") and not t.is_stop and len(t) >= 3]
        nouns = alts or ["misc"]
    freq = {}
    for w in nouns: freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda kv: (kv[1], len(kv[0])), reverse=True)
    return ", ".join([w for w,_ in ranked[:max_terms]])

def main():
    if not IN_PATH.exists():
        raise SystemExit(f"Input not found: {IN_PATH} (run clean_and_dedup.py first)")

    # Load
    df = pd.read_csv(IN_PATH, dtype=str).fillna("")
    texts = df["joke"].tolist()

    # spaCy (tagger + lemmatizer only)
    log("Loading spaCy model: en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

    n_proc = 1  # spaCy recommends 1 for small model; adjust if needed
    batch_size = 1000

    topics = []
    log(f"Tagging topics for {len(texts):,} jokes ...")
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_proc):
        topics.append(topic_from_doc(doc))

    df["topic"] = topics
    out_csv = OUT_DIR / "final_combined_jokes.csv"
    out_parq = OUT_DIR / "final_combined_jokes.parquet"
    df.to_csv(out_csv, index=False)
    try:
        df.to_parquet(out_parq, index=False)
    except Exception as e:
        log(f"WARN: Parquet write failed: {e}")

    log(f"Wrote {out_csv} (+ parquet if supported)")

if __name__ == "__main__":
    main()
