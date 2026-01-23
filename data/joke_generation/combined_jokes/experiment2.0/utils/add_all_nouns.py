
#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import spacy
from tqdm import tqdm

# Extra stop words to remove from noun candidates
STOP_ADD = set("""
joke jokes thing things one ones way time today tonight day guy guys people someone something anything everything nothing
""".split())


def log(msg: str):
    print(f"[all_nouns] {msg}")


def all_nouns_from_doc(doc):
    """
    Extract ALL noun-like tags from the doc:
    - NOUN or PROPN
    - not a stop word
    - length >= 3
    - filtered by STOP_ADD
    Returns a comma-separated string of unique lemmas,
    ranked by (frequency, length) in descending order.
    """
    nouns = [
        t.lemma_.lower().strip()
        for t in doc
        if t.pos_ in ("NOUN", "PROPN")
        and not t.is_stop
        and len(t) >= 3
    ]
    nouns = [w for w in nouns if w not in STOP_ADD]

    if not nouns:
        # No nouns found → empty string (we do NOT fall back to verbs or 'misc' here)
        return ""

    # Count frequencies to rank more important nouns first
    freq = {}
    for w in nouns:
        freq[w] = freq.get(w, 0) + 1

    # Sort by (frequency, length) descending
    ranked = sorted(freq.items(), key=lambda kv: (kv[1], len(kv[0])), reverse=True)

    # Keep all ranked nouns (no max_terms limit)
    return ", ".join([w for w, _ in ranked])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/final/clean_good_jokes.csv"),
        help="Input CSV (default: outputs/final/clean_good_jokes.csv)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/final/clean_good_jokes_all_nouns.csv"),
        help="Output CSV path",
    )
    ap.add_argument(
        "--text-cols",
        nargs="+",
        default=["joke_cleaned", "text"],
        help="Candidate text columns to use (first found is used)",
    )
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    log(f"Loading input CSV: {args.input}")
    df = pd.read_csv(args.input, dtype=str, engine="python", on_bad_lines="skip").fillna("")

    # pick text column
    text_col = None
    for c in args.text_cols:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        raise SystemExit(f"No text column found (tried {args.text_cols})")

    texts = df[text_col].astype(str).tolist()
    log(f"Found {len(texts):,} jokes.")

    # Load spaCy model (tagger + lemmatizer only)
    log("Loading spaCy model: en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

    n_proc = 1       # can increase if the server allows
    batch_size = 1000

    all_nouns_list = []
    log("Extracting all nouns (NOUN/PROPN) for each joke ...")

    for doc in tqdm(
        nlp.pipe(texts, batch_size=batch_size, n_process=n_proc),
        total=len(texts),
        desc="Extracting nouns"
    ):
        all_nouns_list.append(all_nouns_from_doc(doc))

    # Add new column with all noun tags
    df["topic_all_nouns"] = all_nouns_list

    log(f"Writing CSV to {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    log("Done.")


if __name__ == "__main__":
    main()
