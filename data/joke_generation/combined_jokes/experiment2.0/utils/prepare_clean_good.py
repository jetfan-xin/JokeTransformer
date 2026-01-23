"""
Format clean_good_jokes.csv into two outputs:
- clean_good_jokes_all_nouns.csv (adds topic_all_nouns)
- clean_good_jokes_single_topic.csv (adds topic_all_nouns + topic_single)

Default I/O:
  --input      outputs/final/clean_good_jokes.csv
  --out-all    outputs/final/clean_good_jokes_all_nouns.csv
  --out-single outputs/final/clean_good_jokes_single_topic.csv

Usage:
  python -m utils.prepare_clean_good
"""
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def load_spacy():
    import spacy

    return spacy.load("en_core_web_sm", disable=["ner", "parser"])


STOP_ADD = set(
    """
joke jokes thing things one ones way time today tonight day guy guys people someone something anything everything nothing
""".split()
)


def extract_all_nouns(texts, nlp, batch_size=1000, n_proc=1):
    def all_nouns_from_doc(doc):
        nouns = [
            t.lemma_.lower().strip()
            for t in doc
            if t.pos_ in ("NOUN", "PROPN") and not t.is_stop and len(t) >= 3
        ]
        nouns = [w for w in nouns if w not in STOP_ADD]
        if not nouns:
            return ""
        freq = {}
        for w in nouns:
            freq[w] = freq.get(w, 0) + 1
        ranked = sorted(freq.items(), key=lambda kv: (kv[1], len(kv[0])), reverse=True)
        return ", ".join([w for w, _ in ranked])

    out = []
    for doc in tqdm(
        nlp.pipe(texts, batch_size=batch_size, n_process=n_proc),
        total=len(texts),
        desc="all_nouns",
    ):
        out.append(all_nouns_from_doc(doc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/final/clean_good_jokes.csv"),
        help="Input CSV (clean_good_jokes)",
    )
    ap.add_argument(
        "--out-all",
        type=Path,
        default=Path("outputs/final/clean_good_jokes_all_nouns.csv"),
        help="Output CSV with topic_all_nouns added",
    )
    ap.add_argument(
        "--out-single",
        type=Path,
        default=Path("outputs/final/clean_good_jokes_single_topic.csv"),
        help="Output CSV with topic_all_nouns and topic_single",
    )
    ap.add_argument(
        "--text-col",
        default="joke_cleaned",
        help="Text column to use for noun extraction (default: joke_cleaned)",
    )
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    df = pd.read_csv(args.input, engine="python", on_bad_lines="skip")

    if args.text_col not in df.columns:
        raise SystemExit(f"Text column '{args.text_col}' not found in input")

    nlp = load_spacy()
    texts = df[args.text_col].astype(str).tolist()
    df["topic"] = extract_all_nouns(texts, nlp)

    # Write all_nouns output
    args.out_all.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_all, index=False)
    print(f"[WRITE] {args.out_all} rows={len(df)} cols={list(df.columns)}")

    # Create single-topic and write
    def first_topic(val: str) -> str:
        parts = [p.strip() for p in str(val).split(",") if p.strip()]
        return parts[0] if parts else ""

    df_single = df.copy()
    df_single["topic"] = df_single["topic"].apply(first_topic)
    args.out_single.parent.mkdir(parents=True, exist_ok=True)
    df_single.to_csv(args.out_single, index=False)
    print(f"[WRITE] {args.out_single} rows={len(df_single)} cols={list(df_single.columns)}")


if __name__ == "__main__":
    main()
