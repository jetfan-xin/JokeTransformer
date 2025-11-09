#!/usr/bin/env python3
import re, hashlib, argparse, html, unicodedata
from pathlib import Path
import pandas as pd

IN_PATH  = Path("raw_combined/combined_raw.csv")
OUT_DIR  = Path("combined_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR = Path("combined_data/debug")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)
MAX_LEN = 1000        
MAX_LINES = 4       
SAVE_LONG_TO = "combined_data/debug/long_jokes.csv"    

def log(msg: str):
    print(f"[clean] {msg}")

RE_CONTROL = re.compile(r"[\u0000-\u001f\u007f-\u009f]")
RE_WS      = re.compile(r"\s+")
RE_URL     = re.compile(r"\b(?:https?://|http://|www\.)\S+\b", re.IGNORECASE)
RE_EMAIL   = re.compile(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", re.IGNORECASE)
RE_MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")        # [text](url) -> text
RE_HTMLTAG = re.compile(r"</?([a-z][a-z0-9]*)\b[^>]*>", re.IGNORECASE)
RE_CODEFENCE = re.compile(r"```.*?```", re.DOTALL)         # remove fenced code
RE_ANGLELINK = re.compile(r"<(https?://[^>]+)>", re.IGNORECASE)

# Broad emoji & pictograph ranges + variation selectors
RE_EMOJI = re.compile(
    "["                     
    "\U0001F1E6-\U0001F1FF" 
    "\U0001F300-\U0001F5FF" 
    "\U0001F600-\U0001F64F" 
    "\U0001F680-\U0001F6FF" 
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\u2600-\u26FF"         
    "\u2700-\u27BF"         
    "\uFE0E-\uFE0F"         
    "]", flags=re.UNICODE
)

# Keep letters, numbers, common punctuation and spacing
RE_KEEP_WHITELIST = re.compile(r"[^A-Za-z0-9\s.,!?;:'\"()\-–—_/&%$#*@+]")

def strip_markdown(text: str) -> str:
    text = RE_MD_LINK.sub(r"\1", text)   # keep link text
    # inline code `code`
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # headings / bullets / emphasis artifacts
    text = re.sub(r"(^|\s)[#>*\-]{1,3}\s*", r"\1", text)
    text = text.replace("**", "").replace("__", "").replace("*", "")
    return text

def strip_html(text: str) -> str:
    text = RE_HTMLTAG.sub(" ", text)
    text = html.unescape(text)
    return text

def normalize_text(s: str, remove_emojis=True, aggressive_symbol_strip=True) -> str:
    if not isinstance(s, str):
        return ""
    # Normalize unicode
    s = unicodedata.normalize("NFKC", s)

    # Remove code fences first (can contain urls etc.)
    s = RE_CODEFENCE.sub(" ", s)

    # Replace angle-bracketed links <http://...>
    s = RE_ANGLELINK.sub(" ", s)

    # Remove URLs & emails early
    s = RE_URL.sub(" ", s)
    s = RE_EMAIL.sub(" ", s)

    # Strip HTML and Markdown artifacts
    s = strip_html(s)
    s = strip_markdown(s)

    # Remove control chars
    s = RE_CONTROL.sub(" ", s)

    # Normalize quotes/dashes
    s = s.replace("’","'").replace("‘","'").replace("“", '"').replace("”", '"')
    s = s.replace("–","-").replace("—"," — ").replace("…","...")

    # Remove emojis/pictographs if requested
    if remove_emojis:
        s = RE_EMOJI.sub(" ", s)

    # Collapse leftover URLs that escaped earlier filters
    s = re.sub(r"\b[a-z0-9.-]+\.(?:com|net|org|io|co|ai|uk|de|fr|it|es|ru|cn|jp|ly)(?:/[^\s]*)?\b", " ", s, flags=re.IGNORECASE)

    # Aggressive symbol reduction (last resort)
    if aggressive_symbol_strip:
        s = RE_KEEP_WHITELIST.sub(" ", s)

    # Collapse whitespace + trim
    s = RE_WS.sub(" ", s).strip()

    # Remove trailing " - " or stray separators
    s = re.sub(r"\s*[-_/|]\s*$", "", s)

    return s

def stable_id(text: str) -> str:
    return hashlib.sha1(normalize_text(text, remove_emojis=False, aggressive_symbol_strip=False).lower().encode("utf-8")).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_len", type=int, default=10, help="Minimum length after cleaning.")
    ap.add_argument("--debug", action="store_true", help="Emit per-row drop reasons and samples.")
    args = ap.parse_args()

    if not IN_PATH.exists():
        raise SystemExit(f"Input not found: {IN_PATH} (run combine_datasets.py first)")

    df_in = pd.read_csv(IN_PATH, dtype=str).fillna("")
    n_total = len(df_in)
    log(f"Loaded {n_total} rows from {IN_PATH}")

    # Prepare debug slots
    dropped_records = []

    # Normalize
    df = df_in.copy()
    df["joke"] = df["raw_text"].map(lambda x: normalize_text(x))

    # Drop blanks
    before = len(df)
    mask_blank = df["joke"].str.len() == 0
    n_blank = int(mask_blank.sum())
    if n_blank:
        if args.debug:
            tmp = df.loc[mask_blank, ["raw_text"]].copy()
            tmp["reason"] = "blank_after_normalize"
            dropped_records.append(tmp)
        df = df.loc[~mask_blank]
    log(f"Dropped {n_blank} empty after normalize.")

    # Drop too short
    mask_short = df["joke"].str.len() < args.min_len
    n_short = int(mask_short.sum())
    if n_short:
        if args.debug:
            tmp = df.loc[mask_short, ["raw_text", "joke"]].copy()
            tmp["reason"] = f"too_short(<{args.min_len})"
            dropped_records.append(tmp)
        df = df.loc[~mask_short]
    log(f"Dropped {n_short} below min_len={args.min_len}.")

        # Drop too long (characters) 
    if MAX_LEN and MAX_LEN > 0:
        mask_long = df["joke"].str.len() > MAX_LEN
        n_long = int(mask_long.sum())
        if n_long:
            if args.debug:
                tmp = df.loc[mask_long, ["raw_text", "joke"]].copy()
                tmp["reason"] = f"too_long(>{MAX_LEN})"
                dropped_records.append(tmp)
            if SAVE_LONG_TO:
                Path(SAVE_LONG_TO).parent.mkdir(parents=True, exist_ok=True)
                df.loc[mask_long, ["joke"]].to_csv(SAVE_LONG_TO, mode="w", header=True, index=False)
                log(f"Saved {n_long} long jokes to {SAVE_LONG_TO}")
            df = df.loc[~mask_long]
        log(f"Dropped {n_long} above MAX_LEN={MAX_LEN}.")
    else:
        n_long = 0

    # Drop too many lines (multi-paragraph stories) 
    if MAX_LINES and MAX_LINES > 0:
        line_counts = df["joke"].str.count(r"\n") + 1
        mask_lines = line_counts > MAX_LINES
        n_lines = int(mask_lines.sum())
        if n_lines:
            if args.debug:
                tmp = df.loc[mask_lines, ["raw_text", "joke"]].copy()
                tmp["reason"] = f"too_many_lines(>{MAX_LINES})"
                dropped_records.append(tmp)
            if SAVE_LONG_TO:
                # append to same file if present
                Path(SAVE_LONG_TO).parent.mkdir(parents=True, exist_ok=True)
                df.loc[mask_lines, ["joke"]].to_csv(
                    SAVE_LONG_TO, mode="a", header=not Path(SAVE_LONG_TO).exists(), index=False
                )
            df = df.loc[~mask_lines]
        log(f"Dropped {n_lines} with > {MAX_LINES} lines.")
    else:
        n_lines = 0

    # Deduplicate (exact on lowercase)
    # dedup + log duplicates
    df["__norm"] = df["joke"].str.lower()
    before_dedup = len(df)

    # Find duplicates before dropping
    dupes = df[df.duplicated(subset="__norm", keep=False)].copy()

    if not dupes.empty:
        dupes = dupes.sort_values("__norm")
        dupes_path = DEBUG_DIR / "duplicates_report.csv"
        dupes.to_csv(dupes_path, index=False)
        log(f"Saved duplicate group report: {dupes_path}  ({len(dupes)} rows)")

    # Drop duplicates (keep first occurrence)
    df = df.drop_duplicates(subset="__norm").copy()
    n_dedup = before_dedup - len(df)
    log(f"Exact dedup removed {n_dedup} rows.")



    # Keep stable hash for reproducible joins
    df["stable_id"] = df["joke"].map(stable_id)

    #   Add simple row counter 
    df = df.sort_values("stable_id").reset_index(drop=True)  
    df.insert(0, "rid", range(1, len(df) + 1))               

    # Choose column order
    df = df[["rid", "stable_id", "joke"]]

    # Write outputs
    out_csv = OUT_DIR / "clean_jokes.csv"
    out_parq = OUT_DIR / "clean_jokes.parquet"
    df.to_csv(out_csv, index=False)
    try:
        df.to_parquet(out_parq, index=False)
    except Exception as e:
        log(f"WARN: Parquet write failed: {e}")

    # Emit debug artifacts
    if args.debug and dropped_records:
        dbg = pd.concat(dropped_records, ignore_index=True)
        dbg_path = DEBUG_DIR / "clean_drop_report.csv"
        dbg.to_csv(dbg_path, index=False)
        for reason, grp in dbg.groupby("reason"):
            sample_path = DEBUG_DIR / f"samples_{reason}.csv"
            grp.head(2000).to_csv(sample_path, index=False)
        log(f"Debug report written to {dbg_path}")

    # Summary
    n_final = len(df)
    log("----- SUMMARY -----")
    log(f"Total in           : {n_total:,}")
    log(f"Blank after norm   : {n_blank:,}")
    log(f"Too short          : {n_short:,}")
    log(f"Exact duplicates   : {n_dedup:,}")
    log(f"Too long (chars)   : {n_long:,}")
    log(f"Too many lines     : {n_lines:,}")
    log(f"Final out          : {n_final:,}")
    log(f"Wrote {out_csv} (+ parquet if supported)")
    log("-------------------")

if __name__ == "__main__":
    main()
