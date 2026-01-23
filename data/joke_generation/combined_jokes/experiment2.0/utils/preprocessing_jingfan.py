import argparse
import gzip
import hashlib
import html
import re
import unicodedata
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import ftfy
import pandas as pd
import spacy

from utils.logger import logger

class PreprocessConfig:

    # Directories
    RAW_DIR = Path("data_source/raw")
    COMBINED_DIR = Path("data_source/raw_combined")
    CLEANED_DIR = Path("outputs/preprocessed")
    OUTPUT_DIR = Path("outputs/final")
    DEBUG_DIR = Path("outputs/preprocessed/debug")

    # Cleaning parameters
    MAX_LEN = 1000  # Max characters after normalization
    MAX_LINES = 4  # Max newlines allowed
    MIN_LEN = 10  # Minimum length after cleaning

    # Topic extraction
    MAX_TOPICS = 3
    TOPIC_STOPWORDS = {
        "joke",
        "jokes",
        "thing",
        "things",
        "one",
        "ones",
        "way",
        "time",
        "today",
        "tonight",
        "day",
        "guy",
        "guys",
        "people",
        "someone",
        "something",
        "anything",
        "everything",
        "nothing",
    }


# COMBINE DATASETS


class DatasetCombiner:

    def __init__(self, config: PreprocessConfig = None):
        self.config = config or PreprocessConfig()
        self.config.COMBINED_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _iter_two_col_tsv(path: Path) -> Iterable[Tuple[str, str]]:
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

    @staticmethod
    def _best_col(df: pd.DataFrame, candidates: set) -> Optional[str]:
        lower_map = {c.lower(): c for c in df.columns}
        for c in candidates:
            if c in lower_map:
                return lower_map[c]
        return None

    def load_ysharma_short_jokes(self) -> List[str]:
        p = self.config.RAW_DIR / "ysharma_short_jokes.csv"
        jokes = []
        if p.exists():
            logger.info(f"[COMBINE] Loading {p}")
            df = pd.read_csv(p, dtype=str).fillna("")
            joke_col = self._best_col(df, {"joke"})
            if not joke_col:
                logger.warning(f"No 'joke' column found. Columns: {list(df.columns)}")
                return jokes
            for j in df[joke_col]:
                j = str(j).strip()
                if j:
                    jokes.append(j)
            logger.info(f"[COMBINE] Loaded {len(jokes)} jokes from ysharma")
        return jokes

    def load_rjokes_train(self) -> List[str]:
        p_plain = self.config.RAW_DIR / "train.tsv"
        p_gzip = self.config.RAW_DIR / "train.tsv.gz"
        p = p_plain if p_plain.exists() else (p_gzip if p_gzip.exists() else None)
        jokes = []
        if p and p.exists():
            logger.info(f"[COMBINE] Loading {p.name}")
            for score, text in self._iter_two_col_tsv(p):
                t = str(text).strip()
                if t:
                    jokes.append(t)
            logger.info(f"[COMBINE] Loaded {len(jokes)} from rJokesData")
        return jokes

    def load_kaggle_pos_jokes(self) -> List[str]:
        p = self.config.RAW_DIR / "kaggle_dataset.csv"
        jokes = []
        if p.exists():
            logger.info(f"[COMBINE] Loading {p}")
            df = pd.read_csv(p, dtype=str).fillna("")
            lower_map = {c.lower(): c for c in df.columns}
            text_col = lower_map.get("text")
            label_col = lower_map.get("humor")
            if not text_col or not label_col:
                logger.warning(
                    f"Expected 'text' and 'humor' columns. Got {list(df.columns)}"
                )
                return jokes

            def as_bool(x) -> bool:
                s = str(x).strip().lower()
                return s in {
                    "1",
                    "true",
                    "yes",
                    "y",
                    "humor",
                    "humour",
                    "funny",
                    "positive",
                }

            df = df[df[label_col].map(as_bool)]
            for t in df[text_col].astype(str):
                t = t.strip()
                if t:
                    jokes.append(t)
            logger.info(f"[COMBINE] Loaded {len(jokes)} positive jokes from Kaggle")
        return jokes

    def load_dadjokes(self) -> List[str]:
        p = self.config.RAW_DIR / "shuttie_dadjokes.csv"
        jokes = []
        if p.exists():
            logger.info(f"[COMBINE] Loading {p}")
            df = pd.read_csv(p, dtype=str).fillna("")
            q_col = self._best_col(df, {"question", "setup", "title"})
            r_col = self._best_col(df, {"response", "answer", "punchline", "body"})
            if not q_col and not r_col:
                logger.warning(
                    f"No question/response columns. Columns: {list(df.columns)}"
                )
                return jokes
            for _, rr in df.iterrows():
                q = str(rr.get(q_col, "")).strip() if q_col else ""
                r = str(rr.get(r_col, "")).strip() if r_col else ""
                if q and r:
                    joiner = "" if q.endswith((".", "!", "?")) else " — "
                    jokes.append(f"{q}{joiner}{r}")
                elif q or r:
                    jokes.append(q or r)
            logger.info(f"[COMBINE] Loaded {len(jokes)} dadjokes")
        return jokes

    def load_amirkid_jokes(self) -> List[str]:
        p = self.config.RAW_DIR / "amirkid_jokes.csv"
        jokes = []

        if p.exists():
            logger.info(f"[COMBINE] Loading local file {p}")
            df = pd.read_csv(p, dtype=str).fillna("")
        else:
            try:
                from datasets import load_dataset

                logger.info("[COMBINE] Loading HuggingFace dataset Amirkid/jokes")
                ds = load_dataset("Amirkid/jokes", split="train")
                df = ds.to_pandas()
            except Exception as e:
                logger.warning(f"Could not load Amirkid/jokes ({e}); skipping")
                return jokes

        joke_col = self._best_col(df, {"joke", "text", "content"})
        if not joke_col:
            logger.warning(f"No obvious joke column found. Columns: {list(df.columns)}")
            return jokes

        for j in df[joke_col]:
            j = str(j).strip()
            if j:
                jokes.append(j)
        logger.info(f"[COMBINE] Loaded {len(jokes)} jokes from Amirkid/jokes")
        return jokes

    def run(self) -> Path:
        rows_with_source = []

        def add_with_source(texts: List[str], source_name: str):
            for t in texts:
                rows_with_source.append((t, source_name))

        add_with_source(self.load_ysharma_short_jokes(), "ysharma_short_jokes.csv")
        add_with_source(self.load_rjokes_train(), "train.tsv|train.tsv.gz")
        # add_with_source(self.load_kaggle_pos_jokes(), "kaggle_dataset.csv")  # Skipped due to duplicated content
        add_with_source(self.load_dadjokes(), "shuttie_dadjokes.csv")
        add_with_source(self.load_amirkid_jokes(), "amirkid_jokes.csv")

        df = pd.DataFrame(rows_with_source, columns=["raw_text", "source_file"])
        out_csv = self.config.COMBINED_DIR / "combined_raw.csv"
        df.to_csv(out_csv, index=False)
        logger.info(f"[COMBINE] Wrote {out_csv} | rows={len(df)}")
        return out_csv


# CLEAN DATA


class DataCleaner:
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
        "\U0001f1e6-\U0001f1ff"
        "\U0001f300-\U0001f5ff"
        "\U0001f600-\U0001f64f"
        "\U0001f680-\U0001f6ff"
        "\U0001f700-\U0001f77f"
        "\U0001f780-\U0001f7ff"
        "\U0001f800-\U0001f8ff"
        "\U0001f900-\U0001f9ff"
        "\U0001fa00-\U0001fa6f"
        "\U0001fa70-\U0001faff"
        "\u2600-\u26ff"
        "\u2700-\u27bf"
        "\ufe0e-\ufe0f"
        "]",
        flags=re.UNICODE,
    )
    RE_KEEP_WHITELIST = re.compile(r"[^A-Za-z0-9\s.,!?;:'\"()\-–—_/&%$#*@+]")
    RE_LEADING_BULLET = re.compile(
        r"^\s*(?:[:;,.!?]+|-{2,})\s+",
        re.VERBOSE,
    )

    def __init__(self, config: PreprocessConfig = None):
        self.config = config or PreprocessConfig()
        self.config.CLEANED_DIR.mkdir(parents=True, exist_ok=True)
        self.config.DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    def strip_markdown(self, text: str) -> str:
        text = self.RE_MD_LINK.sub(r"\1", text)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        text = re.sub(r"(^|\s)[#>*\-]{1,3}\s*", r"\1", text)
        text = text.replace("**", "").replace("__", "").replace("*", "")
        return text

    def strip_html(self, text: str) -> str:
        text = self.RE_HTMLTAG.sub(" ", text)
        text = html.unescape(text)
        return text

    def normalize_text(
        self, s: str, remove_emojis: bool = True, aggressive_symbol_strip: bool = True
    ) -> str:
        if not isinstance(s, str):
            return ""

        s = unicodedata.normalize("NFKC", s)
        s = self.RE_CODEFENCE.sub(" ", s)
        s = self.RE_ANGLELINK.sub(" ", s)
        s = self.RE_URL.sub(" ", s)
        s = self.RE_EMAIL.sub(" ", s)
        s = self.strip_html(s)
        s = self.strip_markdown(s)
        s = self.RE_CONTROL.sub(" ", s)

        s = s.replace("'", "'").replace("'", "'").replace(""", '"').replace(""", '"')
        s = s.replace("–", "-").replace("—", " — ").replace("…", "...")

        if remove_emojis:
            s = self.RE_EMOJI.sub(" ", s)

        # Collapse leftover URLs
        s = re.sub(
            r"\b[a-z0-9.-]+\.(?:com|net|org|io|co|ai|uk|de|fr|it|es|ru|cn|jp|ly)(?:/[^\s]*)?\b",
            " ",
            s,
            flags=re.IGNORECASE,
        )

        if aggressive_symbol_strip:
            s = self.RE_KEEP_WHITELIST.sub(" ", s)

        s = self.RE_WS.sub(" ", s).strip()
        s = re.sub(r"\s*[-_/|]\s*$", "", s)

        return s

    def clean_jokes(self, joke: str) -> str:
        joke = ftfy.fix_text(str(joke))
        joke = unicodedata.normalize("NFKC", joke)
        joke = joke.lower()

        # Handle dialogue artifact
        joke = re.sub(r'"\s*n\s*"', '" ', joke)

        # Remove URLs, subreddit refs
        joke = re.sub(r"https?://\S+|www\.\S+", "", joke)
        joke = re.sub(r"/r/\S+", "", joke)

        # Remove mentions and hashtags
        joke = re.sub(r"\.?@\w+|#\w+", "", joke)
        joke = re.sub(r"^\s*\.\s*", "", joke)

        # Keep letters, numbers, and common punctuation
        joke = re.sub(r"[^a-z0-9\s.,!?;:'\"()\-\[\]…]", " ", joke)

        # Normalize ellipses
        joke = re.sub(r"\.(?:\s*\.)+", "<ELLIPSIS>", joke)
        joke = re.sub(r"([!?])\1+", r"\1", joke)
        joke = " ".join(joke.split())

        # Protect contractions
        joke = re.sub(r"(\w)'(\w)", r"\1<APOST>\2", joke)
        joke = re.sub(r"([.!?,;:(){}\[\]\"'])", r" \1 ", joke)
        joke = re.sub(r"\s{2,}", " ", joke).strip()

        # Restore markers
        joke = joke.replace("<APOST>", "'")
        joke = joke.replace("<ELLIPSIS>", " ... ")
        joke = re.sub(r"\s+'\s+", "'", joke)

        return joke.strip()

    def tidy_front(self, s: str) -> str:
        # Strip leading bullet punctuation.
        if not isinstance(s, str):
            return s
        if re.match(r"^\s*-\s*\d", s):
            return s.strip()
        s = s.strip()
        s = self.RE_LEADING_BULLET.sub("", s)
        return s.strip()

    @staticmethod
    def is_mostly_symbols(s: str, min_alpha_ratio: float = 0.4) -> bool:
        # Return True if string is mostly symbols (junk)
        if not isinstance(s, str) or not s:
            return True
        total = len(s)
        if total == 0:
            return True
        alnum = sum(ch.isalnum() for ch in s)
        return (alnum / total) < min_alpha_ratio

    @staticmethod
    def stable_id(text: str) -> str:
        return hashlib.sha1(text.lower().encode("utf-8")).hexdigest()

    def run(self, input_path: Path = None) -> Path:
        input_path = input_path or (self.config.COMBINED_DIR / "combined_raw.csv")

        if not input_path.exists():
            raise FileNotFoundError(
                f"Input not found: {input_path} (run combine first)"
            )

        df_in = pd.read_csv(input_path, dtype=str).fillna("")
        n_total = len(df_in)
        logger.info(f"[CLEAN] Loaded {n_total} rows from {input_path}")

        df = df_in.copy()
        if "source_file" not in df.columns:
            df["source_file"] = "unknown"

        # Normalize text
        df["joke"] = df["raw_text"].map(self.normalize_text)

        # Drop blanks
        mask_blank = df["joke"].str.len() == 0
        n_blank = int(mask_blank.sum())
        df = df.loc[~mask_blank]
        logger.info(f"[CLEAN] Dropped {n_blank} empty after normalize")

        # Drop too short
        mask_short = df["joke"].str.len() < self.config.MIN_LEN
        n_short = int(mask_short.sum())
        df = df.loc[~mask_short]
        logger.info(f"[CLEAN] Dropped {n_short} below min_len={self.config.MIN_LEN}")

        # Drop too long
        mask_long = df["joke"].str.len() > self.config.MAX_LEN
        n_long = int(mask_long.sum())
        df = df.loc[~mask_long]
        logger.info(f"[CLEAN] Dropped {n_long} above MAX_LEN={self.config.MAX_LEN}")

        # Deduplicate
        df["__norm"] = df["joke"].str.lower()
        before_dedup = len(df)
        df = df.drop_duplicates(subset="__norm").copy()
        n_dedup = before_dedup - len(df)
        logger.info(f"[CLEAN] Exact dedup removed {n_dedup} rows")

        # Tokenization-friendly cleaning
        df["joke_cleaned"] = df["joke"].map(self.clean_jokes)
        df["joke_cleaned"] = df["joke_cleaned"].map(self.tidy_front)

        # Drop blanks after cleaning
        mask_jc_blank = df["joke_cleaned"].str.len() == 0
        n_jc_blank = int(mask_jc_blank.sum())
        df = df.loc[~mask_jc_blank]
        logger.info(f"[CLEAN] Dropped {n_jc_blank} empty after clean_jokes")

        # Drop mostly-symbolic junk
        mask_symbolic = df["joke_cleaned"].map(self.is_mostly_symbols)
        n_symbolic = int(mask_symbolic.sum())
        df = df.loc[~mask_symbolic]
        logger.info(f"[CLEAN] Dropped {n_symbolic} mostly-symbolic junk")

        # Finalize
        df["word_count"] = df["joke_cleaned"].str.split().str.len()
        df["stable_id"] = df["joke_cleaned"].map(self.stable_id)
        df = df.sort_values("stable_id").reset_index(drop=True)
        df.insert(0, "rid", range(1, len(df) + 1))
        df = df[["rid", "stable_id", "joke", "joke_cleaned", "word_count", "source_file"]]

        # Write output
        out_csv = self.config.CLEANED_DIR / "clean_jokes.csv"
        df.to_csv(out_csv, index=False)

        n_final = len(df)
        logger.info(f"[CLEAN] Final: {n_final} jokes → {out_csv}")

        return out_csv


# ADD TOPICS
class TopicExtractor:
    WORD_RE = re.compile(r"^[a-z][a-z'_-]{1,}[a-z]$")
    RE_LAUGH = re.compile(r"^(ha)+h?$")
    RE_REPEATS = re.compile(r"(.)\1{2,}")
    VOWELS = set("aeiou")

    def __init__(self, config: PreprocessConfig = None):
        self.config = config or PreprocessConfig()
        self.config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.nlp = None

    def _load_spacy(self):
        if self.nlp is None:
            logger.info("[TOPICS] Loading spaCy model: en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        return self.nlp

    def is_good_topic_word(self, w: str) -> bool:
        if not w:
            return False
        if not self.WORD_RE.match(w):
            return False
        if w in self.config.TOPIC_STOPWORDS:
            return False
        if self.RE_LAUGH.match(w):
            return False
        if self.RE_REPEATS.search(w):
            return False
        if not any(v in w for v in self.VOWELS):
            return False
        return True

    def topic_from_doc(self, doc, max_terms: int = 3) -> str:
        nouns = []
        for t in doc:
            if t.pos_ not in ("NOUN", "PROPN"):
                continue
            if t.is_stop or t.like_num or not t.is_alpha:
                continue
            w = t.lemma_.lower().strip()
            if len(w) < 3:
                continue
            if not self.is_good_topic_word(w):
                continue
            nouns.append(w)

        # Fallback to verbs/adjectives
        if not nouns:
            alts = []
            for t in doc:
                if t.pos_ not in ("VERB", "ADJ"):
                    continue
                if t.is_stop or t.like_num or not t.is_alpha:
                    continue
                w = t.lemma_.lower().strip()
                if len(w) < 3:
                    continue
                if not self.is_good_topic_word(w):
                    continue
                alts.append(w)
            nouns = alts or ["misc"]

        # Rank by frequency
        freq = {}
        for w in nouns:
            freq[w] = freq.get(w, 0) + 1

        ranked = sorted(
            freq.items(),
            key=lambda kv: (kv[1], -len(kv[0])),
            reverse=True,
        )

        return ", ".join([w for w, _ in ranked[:max_terms]])

    def run(self, input_path: Path = None) -> Path:
        input_path = input_path or (self.config.CLEANED_DIR / "clean_jokes.csv")

        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path} (run clean first)")

        df = pd.read_csv(input_path, dtype=str).fillna("")

        # Use cleaned text
        if "joke_cleaned" in df.columns:
            texts = df["joke_cleaned"].tolist()
            logger.info("[TOPICS] Using 'joke_cleaned' column")
        else:
            texts = df["joke"].tolist()
            logger.info("[TOPICS] Using 'joke' column")

        nlp = self._load_spacy()

        topics = []
        logger.info(f"[TOPICS] Extracting topics for {len(texts):,} jokes...")
        for i, doc in enumerate(nlp.pipe(texts, batch_size=1000)):
            topics.append(self.topic_from_doc(doc, self.config.MAX_TOPICS))
            if (i + 1) % 5000 == 0:
                logger.info(f"[TOPICS] Processed {i + 1}/{len(texts)}")

        df["topic"] = topics

        out_csv = self.config.OUTPUT_DIR / "final_clean_jokes_single_topic.csv"
        df.to_csv(out_csv, index=False)
        logger.info(f"[TOPICS] Wrote {out_csv}")

        return out_csv


# MAIN PIPELINE


class PreprocessingPipeline:
    # Full preprocessing pipeline: combine → clean → topics.

    def __init__(self, config: PreprocessConfig = None):
        self.config = config or PreprocessConfig()
        self.combiner = DatasetCombiner(self.config)
        self.cleaner = DataCleaner(self.config)
        self.topic_extractor = TopicExtractor(self.config)

    def run_all(self) -> Path:
        logger.info("=" * 60)
        logger.info("STARTING FULL PREPROCESSING PIPELINE")
        logger.info("=" * 60)

        # Combine
        combined_path = self.combiner.run()

        # Clean
        cleaned_path = self.cleaner.run(combined_path)

        # Topics
        final_path = self.topic_extractor.run(cleaned_path)

        logger.info("=" * 60)
        logger.info(f"PREPROCESSING COMPLETE → {final_path}")
        logger.info("=" * 60)

        return final_path

    def run_combine(self) -> Path:
        return self.combiner.run()

    def run_clean(self, input_path: Path = None) -> Path:
        return self.cleaner.run(input_path)

    def run_topics(self, input_path: Path = None) -> Path:
        return self.topic_extractor.run(input_path)


def main():
    parser = argparse.ArgumentParser(description="Joke dataset preprocessing pipeline")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    parser.add_argument("--combine", action="store_true", help="Only combine datasets")
    parser.add_argument("--clean", action="store_true", help="Only clean data")
    parser.add_argument("--topics", action="store_true", help="Only extract topics")
    args = parser.parse_args()

    pipeline = PreprocessingPipeline()

    if args.all or not any([args.combine, args.clean, args.topics]):
        pipeline.run_all()
    else:
        if args.combine:
            pipeline.run_combine()
        if args.clean:
            pipeline.run_clean()
        if args.topics:
            pipeline.run_topics()


if __name__ == "__main__":
    main()
