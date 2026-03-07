#!/usr/bin/env python3
import argparse
import csv
import gzip
import hashlib
import html
import json
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    import ftfy  # type: ignore

    def fix_text(value: str) -> str:
        return ftfy.fix_text(value)

except Exception:

    def fix_text(value: str) -> str:
        return value


@dataclass
class PipelineConfig:
    raw_dir: Path
    merged_out: Path
    clean_out: Path
    merge_stats_out: Path
    preprocess_stats_out: Path
    min_len: int
    max_len: int
    max_lines: int
    min_alpha_ratio: float
    run_download_script: bool


RE_CONTROL = re.compile(r"[\u0000-\u001f\u007f-\u009f]")
RE_WS = re.compile(r"\s+")
RE_URL = re.compile(r"\b(?:https?://|http://|www\.)\S+\b", re.IGNORECASE)
RE_EMAIL = re.compile(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", re.IGNORECASE)
RE_MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
RE_HTMLTAG = re.compile(r"</?([a-z][a-z0-9]*)\b[^>]*>", re.IGNORECASE)
RE_CODEFENCE = re.compile(r"```.*?```", re.DOTALL)
RE_ANGLELINK = re.compile(r"<(https?://[^>]+)>", re.IGNORECASE)
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
    "]",
    flags=re.UNICODE,
)
RE_KEEP_WHITELIST = re.compile(r"[^A-Za-z0-9\s.,!?;:'\"()\-_/&%$#*@+]")
RE_LEADING_BULLET = re.compile(r"^\s*(?:[:;,.!?]+|-{2,})\s+", re.VERBOSE)


def log(msg: str) -> None:
    print(f"[human-pipeline] {msg}")


def read_csv_flexible(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8", on_bad_lines="skip").fillna("")
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, encoding="latin1", on_bad_lines="skip").fillna("")


def iter_two_col_tsv(path: Path) -> Iterable[Tuple[str, str]]:
    open_fn = gzip.open if str(path).endswith(".gz") else open
    with open_fn(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            yield parts[0], parts[1]


def best_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower_map = {name.lower(): name for name in df.columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def as_bool(value: str) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "humor", "humour", "funny", "positive"}


def to_df(rows: List[Tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["raw_text", "source", "source_file"])


def load_short_jokes(raw_dir: Path) -> pd.DataFrame:
    candidates = [
        "short_jokes.csv",
        "shortjokes.csv",
        "ysharma_short_jokes.csv",
        "abhinavmoudgil_short_jokes.csv",
    ]
    for name in candidates:
        path = raw_dir / name
        if not path.exists():
            continue
        df = read_csv_flexible(path)
        joke_col = best_col(df, {"joke", "text", "content"})
        if not joke_col:
            continue
        rows = []
        for value in df[joke_col]:
            text = str(value).strip()
            if text:
                rows.append((text, "short_jokes_kaggle", path.name))
        return to_df(rows)
    return to_df([])


def load_rjokes(raw_dir: Path) -> pd.DataFrame:
    path_tsv = raw_dir / "train.tsv"
    path_gz = raw_dir / "train.tsv.gz"
    path = path_tsv if path_tsv.exists() else (path_gz if path_gz.exists() else None)
    if not path:
        return to_df([])
    rows = []
    for _score, text in iter_two_col_tsv(path):
        text = str(text).strip()
        if text:
            rows.append((text, "rjokesdata", path.name))
    return to_df(rows)


def ensure_amirkid_local(raw_dir: Path, run_download_script: bool) -> Optional[Path]:
    candidates = [
        raw_dir / "amirkid_jokes.csv",
        raw_dir / "amirkid.csv",
    ]
    for path in candidates:
        if path.exists():
            return path

    if not run_download_script:
        return None

    download_script = raw_dir / "download.py"
    if not download_script.exists():
        log("WARN: amirkid file missing and download.py not found")
        return None

    try:
        log("amirkid_jokes.csv missing, running data_source/raw/download.py")
        subprocess.run(
            [sys.executable, str(download_script)],
            cwd=raw_dir,
            check=True,
        )
    except Exception as exc:
        log(f"WARN: failed running download.py: {exc}")
        return None

    for path in candidates:
        if path.exists():
            return path
    return None


def load_amirkid(raw_dir: Path, run_download_script: bool) -> pd.DataFrame:
    path = ensure_amirkid_local(raw_dir, run_download_script)
    if not path:
        return to_df([])

    df = read_csv_flexible(path)
    joke_col = best_col(df, {"joke", "text", "content"})
    if not joke_col:
        return to_df([])
    rows = []
    for value in df[joke_col]:
        text = str(value).strip()
        if text:
            rows.append((text, "amirkid_jokes_hf", path.name))
    return to_df(rows)


def load_humor_200k(raw_dir: Path) -> pd.DataFrame:
    candidates = [
        "kaggle_dataset.csv",
        "200k_short_texts_for_humor_detection.csv",
        "deepcontractor_200k.csv",
    ]
    for name in candidates:
        path = raw_dir / name
        if not path.exists():
            continue
        df = read_csv_flexible(path)
        text_col = best_col(df, {"text", "joke", "content"})
        label_col = best_col(df, {"humor", "humour", "label", "is_humor"})
        if not text_col:
            continue
        if label_col:
            df = df[df[label_col].map(as_bool)]
        rows = []
        for value in df[text_col]:
            text = str(value).strip()
            if text:
                rows.append((text, "humor_detection_200k_kaggle", path.name))
        return to_df(rows)
    return to_df([])


def load_dad_jokes(raw_dir: Path) -> pd.DataFrame:
    candidates = [
        ("shuttie_dadjokes.csv", "shuttie_dadjokes_hf"),
        ("reddit_dad_jokes.csv", "reddit_dad_jokes_kaggle"),
        ("oktayozturk_reddit_dad_jokes.csv", "reddit_dad_jokes_kaggle"),
    ]
    all_rows: List[Tuple[str, str, str]] = []
    for file_name, source_name in candidates:
        path = raw_dir / file_name
        if not path.exists():
            continue
        df = read_csv_flexible(path)
        q_col = best_col(df, {"question", "setup", "title"})
        r_col = best_col(df, {"response", "answer", "punchline", "body"})
        t_col = best_col(df, {"joke", "text", "content"})
        if q_col or r_col:
            for _, row in df.iterrows():
                question = str(row.get(q_col, "")).strip() if q_col else ""
                response = str(row.get(r_col, "")).strip() if r_col else ""
                if question and response:
                    joiner = "" if question.endswith((".", "!", "?")) else " - "
                    text = f"{question}{joiner}{response}"
                else:
                    text = question or response
                if text:
                    all_rows.append((text, source_name, path.name))
        elif t_col:
            for value in df[t_col]:
                text = str(value).strip()
                if text:
                    all_rows.append((text, source_name, path.name))
    return to_df(all_rows)


def print_merge_source_table(source_stats: List[Dict[str, object]]) -> None:
    if not source_stats:
        log("no sources were loaded")
        return
    log("merge source stats:")
    header = "{:<34} {:<28} {:>10} {:>10}".format("source", "source_file", "loaded", "kept")
    print(header)
    print("-" * len(header))
    for row in source_stats:
        print(
            "{:<34} {:<28} {:>10} {:>10}".format(
                str(row["source"]),
                str(row["source_file"]),
                int(row["loaded_rows"]),
                int(row["kept_rows"]),
            )
        )


def strip_markdown(text: str) -> str:
    text = RE_MD_LINK.sub(r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"(^|\s)[#>*\-]{1,3}\s*", r"\1", text)
    text = text.replace("**", "").replace("__", "").replace("*", "")
    return text


def strip_html(text: str) -> str:
    text = RE_HTMLTAG.sub(" ", text)
    text = html.unescape(text)
    return text


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = fix_text(text)
    text = unicodedata.normalize("NFKC", text)
    text = RE_CODEFENCE.sub(" ", text)
    text = RE_ANGLELINK.sub(" ", text)
    text = RE_URL.sub(" ", text)
    text = RE_EMAIL.sub(" ", text)
    text = strip_html(text)
    text = strip_markdown(text)
    text = RE_CONTROL.sub(" ", text)
    text = (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", " - ")
        .replace("\u2026", "...")
    )
    text = RE_EMOJI.sub(" ", text)
    text = re.sub(
        r"\b[a-z0-9.-]+\.(?:com|net|org|io|co|ai|uk|de|fr|it|es|ru|cn|jp|ly)(?:/[^\s]*)?\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = RE_KEEP_WHITELIST.sub(" ", text)
    text = RE_WS.sub(" ", text).strip()
    text = re.sub(r"\s*[-_/|]\s*$", "", text)
    return text


def clean_joke_for_tokenization(text: str) -> str:
    text = text.lower()
    text = re.sub(r'"\s*n\s*"', '" ', text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"/r/\S+", "", text)
    text = re.sub(r"\.?@\w+|#\w+", "", text)
    text = re.sub(r"^\s*\.\s*", "", text)
    text = re.sub(r"[^a-z0-9\s.,!?;:'\"()\-\[\]...]", " ", text)
    text = re.sub(r"\.(?:\s*\.)+", "<ELLIPSIS>", text)
    text = re.sub(r"([!?])\1+", r"\1", text)
    text = " ".join(text.split())
    text = re.sub(r"(\w)'(\w)", r"\1<APOST>\2", text)
    text = re.sub(r"([.!?,;:(){}\[\]\"'])", r" \1 ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = text.replace("<APOST>", "'")
    text = text.replace("<ELLIPSIS>", " ... ")
    text = re.sub(r"\s+'\s+", "'", text)
    text = RE_LEADING_BULLET.sub("", text).strip()
    return text


def is_mostly_symbols(text: str, min_alpha_ratio: float) -> bool:
    if not isinstance(text, str) or not text:
        return True
    total_len = len(text)
    if total_len == 0:
        return True
    alnum_count = sum(ch.isalnum() for ch in text)
    return (alnum_count / total_len) < min_alpha_ratio


def stable_id(text: str) -> str:
    return hashlib.sha1(text.lower().encode("utf-8")).hexdigest()


def run_merge(config: PipelineConfig) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    loaders = [
        ("short_jokes_kaggle", load_short_jokes(config.raw_dir)),
        ("rjokesdata", load_rjokes(config.raw_dir)),
        ("amirkid_jokes_hf", load_amirkid(config.raw_dir, config.run_download_script)),
        ("humor_detection_200k_kaggle", load_humor_200k(config.raw_dir)),
        ("dad_jokes", load_dad_jokes(config.raw_dir)),
    ]

    frames: List[pd.DataFrame] = []
    source_stats: List[Dict[str, object]] = []
    for source_name, frame in loaders:
        if frame.empty:
            source_stats.append(
                {
                    "source": source_name,
                    "source_file": "missing_or_not_matched",
                    "loaded_rows": 0,
                    "kept_rows": 0,
                }
            )
            continue
        frame = frame.fillna("")
        loaded_rows = len(frame)
        frame["raw_text"] = frame["raw_text"].astype(str).str.strip()
        frame = frame[frame["raw_text"] != ""].copy()
        kept_rows = len(frame)
        frames.append(frame)

        for source_file, file_df in frame.groupby("source_file"):
            source_stats.append(
                {
                    "source": str(file_df["source"].iloc[0]),
                    "source_file": str(source_file),
                    "loaded_rows": loaded_rows if len(frame["source_file"].unique()) == 1 else len(file_df),
                    "kept_rows": len(file_df),
                }
            )

    merged = pd.concat(frames, ignore_index=True) if frames else to_df([])
    return merged, source_stats


def run_preprocess(config: PipelineConfig, merged_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    stats = {
        "input_rows": int(len(merged_df)),
        "dropped_empty_after_normalize": 0,
        "dropped_too_short": 0,
        "dropped_too_long": 0,
        "dropped_too_many_lines": 0,
        "dropped_dedup_normalized": 0,
        "dropped_empty_after_clean": 0,
        "dropped_symbolic": 0,
        "dropped_dedup_cleaned": 0,
        "final_rows": 0,
    }

    if merged_df.empty:
        return merged_df, stats

    df = merged_df.copy()
    # NOTE: drop_too_many_lines is intentionally disabled per project request.
    # Keep the stats key for backward compatibility with downstream readers.
    stats["dropped_too_many_lines"] = 0

    df["joke"] = df["raw_text"].map(normalize_text)

    mask_empty_norm = df["joke"].str.len() == 0
    stats["dropped_empty_after_normalize"] = int(mask_empty_norm.sum())
    df = df.loc[~mask_empty_norm].copy()

    mask_short = df["joke"].str.len() < config.min_len
    stats["dropped_too_short"] = int(mask_short.sum())
    df = df.loc[~mask_short].copy()

    mask_long = df["joke"].str.len() > config.max_len
    stats["dropped_too_long"] = int(mask_long.sum())
    df = df.loc[~mask_long].copy()

    before_norm_dedup = len(df)
    df["__norm_key"] = df["joke"].str.lower()
    df = df.drop_duplicates(subset="__norm_key", keep="first").copy()
    stats["dropped_dedup_normalized"] = before_norm_dedup - len(df)

    df["joke_cleaned"] = df["joke"].map(clean_joke_for_tokenization)

    mask_empty_clean = df["joke_cleaned"].str.len() == 0
    stats["dropped_empty_after_clean"] = int(mask_empty_clean.sum())
    df = df.loc[~mask_empty_clean].copy()

    mask_symbolic = df["joke_cleaned"].map(lambda s: is_mostly_symbols(s, config.min_alpha_ratio))
    stats["dropped_symbolic"] = int(mask_symbolic.sum())
    df = df.loc[~mask_symbolic].copy()

    before_clean_dedup = len(df)
    df["__clean_key"] = df["joke_cleaned"].str.lower()
    df = df.drop_duplicates(subset="__clean_key", keep="first").copy()
    stats["dropped_dedup_cleaned"] = before_clean_dedup - len(df)

    df["char_count"] = df["joke_cleaned"].str.len()
    df["word_count"] = df["joke_cleaned"].str.split().str.len()
    df["stable_id"] = df["joke_cleaned"].map(stable_id)
    df = df.sort_values("stable_id").reset_index(drop=True)
    df.insert(0, "rid", range(1, len(df) + 1))
    stats["final_rows"] = int(len(df))

    keep_cols = [
        "rid",
        "stable_id",
        "source",
        "source_file",
        "raw_text",
        "joke",
        "joke_cleaned",
        "char_count",
        "word_count",
    ]
    return df[keep_cols].copy(), stats


def print_preprocess_stats(stats: Dict[str, int]) -> None:
    log("preprocess stats:")
    ordered_keys = [
        "input_rows",
        "dropped_too_many_lines",
        "dropped_empty_after_normalize",
        "dropped_too_short",
        "dropped_too_long",
        "dropped_dedup_normalized",
        "dropped_empty_after_clean",
        "dropped_symbolic",
        "dropped_dedup_cleaned",
        "final_rows",
    ]
    for key in ordered_keys:
        print(f"  - {key}: {stats[key]}")


def print_source_distribution(clean_df: pd.DataFrame) -> None:
    if clean_df.empty:
        log("final source distribution: no rows")
        return
    log("final source distribution:")
    source_counts = (
        clean_df.groupby(["source", "source_file"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values("rows", ascending=False)
    )
    for _, row in source_counts.iterrows():
        print(f"  - source={row['source']}, source_file={row['source_file']}, rows={int(row['rows'])}")


def write_merge_stats(path: Path, source_stats: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source", "source_file", "loaded_rows", "kept_rows"],
        )
        writer.writeheader()
        for row in source_stats:
            writer.writerow(row)


def write_preprocess_stats(path: Path, preprocess_stats: Dict[str, int], clean_df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "preprocess_stats": preprocess_stats,
        "final_avg_char_count": float(clean_df["char_count"].mean()) if not clean_df.empty else 0.0,
        "final_avg_word_count": float(clean_df["word_count"].mean()) if not clean_df.empty else 0.0,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Human jokes data merging and preprocessing pipeline")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data_source/raw"),
        help="Directory containing raw source files",
    )
    parser.add_argument(
        "--merged-out",
        type=Path,
        default=Path("data_source/raw_combined/human_combined_raw.csv"),
        help="Output CSV for merged raw pool",
    )
    parser.add_argument(
        "--clean-out",
        type=Path,
        default=Path("outputs/preprocessed/human_clean_jokes.csv"),
        help="Output CSV for cleaned dataset",
    )
    parser.add_argument(
        "--merge-stats-out",
        type=Path,
        default=Path("outputs/preprocessed/human_merge_source_stats.csv"),
        help="Output CSV for per-source merge stats",
    )
    parser.add_argument(
        "--preprocess-stats-out",
        type=Path,
        default=Path("outputs/preprocessed/human_preprocess_stats.json"),
        help="Output JSON for preprocessing stats",
    )
    parser.add_argument("--min-len", type=int, default=10, help="Minimum text length after normalization")
    parser.add_argument("--max-len", type=int, default=1000, help="Maximum text length after normalization")
    parser.add_argument(
        "--max-lines",
        type=int,
        default=4,
        help="Deprecated: line-count filter is disabled and this value is ignored",
    )
    parser.add_argument(
        "--min-alpha-ratio",
        type=float,
        default=0.4,
        help="Minimum alphanumeric ratio in cleaned text",
    )
    parser.add_argument(
        "--run-download-script",
        action="store_true",
        help="Run data_source/raw/download.py if amirkid_jokes.csv is missing",
    )
    args = parser.parse_args()
    return PipelineConfig(
        raw_dir=args.raw_dir,
        merged_out=args.merged_out,
        clean_out=args.clean_out,
        merge_stats_out=args.merge_stats_out,
        preprocess_stats_out=args.preprocess_stats_out,
        min_len=args.min_len,
        max_len=args.max_len,
        max_lines=args.max_lines,
        min_alpha_ratio=args.min_alpha_ratio,
        run_download_script=args.run_download_script,
    )


def main() -> None:
    config = parse_args()
    log("stage 1/2: data merging")
    merged_df, source_stats = run_merge(config)
    print_merge_source_table(source_stats)

    config.merged_out.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(config.merged_out, index=False)
    write_merge_stats(config.merge_stats_out, source_stats)

    log(f"merged raw rows: {len(merged_df)}")
    log(f"merged raw output: {config.merged_out}")
    log(f"merge stats output: {config.merge_stats_out}")

    log("stage 2/2: preprocessing")
    clean_df, preprocess_stats = run_preprocess(config, merged_df)
    print_preprocess_stats(preprocess_stats)
    print_source_distribution(clean_df)

    config.clean_out.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(config.clean_out, index=False)
    write_preprocess_stats(config.preprocess_stats_out, preprocess_stats, clean_df)

    log(f"clean output: {config.clean_out}")
    log(f"preprocess stats output: {config.preprocess_stats_out}")
    log("pipeline complete")


if __name__ == "__main__":
    main()
