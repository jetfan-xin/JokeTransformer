"""
Build and clean a combined LLM jokes dataset.

Steps:
1) Read Gemini and DeepSeek CSVs from data_source/raw/.
   - Gemini: keep rows where model_version == "gemini-2.5-pro" and column joke_text.
   - DeepSeek: take all rows from column joke_text.
2) Write combined raw CSV (raw_text + source_file) to data_source/raw_combined/combined_raw_llm.csv.
3) Run the same cleaning logic as preprocessing_jingfan.py --clean and write cleaned CSV to
   outputs/preprocessed/clean_jokes_llm.csv.

Usage (from repo root):
  python -m utils.llm_jokes_prepare
"""
import argparse
from pathlib import Path

import pandas as pd

from utils.logger import logger
from utils.preprocessing_jingfan import DataCleaner, PreprocessConfig


def build_combined(
    gemini_path: Path, deepseek_path: Path, out_path: Path
) -> Path:
    if not gemini_path.exists():
        raise FileNotFoundError(gemini_path)
    if not deepseek_path.exists():
        raise FileNotFoundError(deepseek_path)

    logger.info(f"[LOAD] Gemini CSV: {gemini_path}")
    df_g = pd.read_csv(gemini_path)
    if "model_version" not in df_g.columns:
        raise ValueError("Gemini CSV missing column 'model_version'")
    joke_col_g = "joke_text" if "joke_text" in df_g.columns else None
    if not joke_col_g:
        raise ValueError("Gemini CSV missing column 'joke_text'")
    df_g = df_g[df_g["model_version"] == "gemini-2.5-pro"]
    df_g = df_g[[joke_col_g]].rename(columns={joke_col_g: "raw_text"})
    df_g["source_file"] = gemini_path.name
    logger.info(f"[GEMINI] Kept {len(df_g)} rows with model_version == gemini-2.5-pro")

    logger.info(f"[LOAD] DeepSeek CSV: {deepseek_path}")
    df_d = pd.read_csv(deepseek_path)
    joke_col_d = "joke_text" if "joke_text" in df_d.columns else None
    if not joke_col_d:
        raise ValueError("DeepSeek CSV missing column 'joke_text'")
    df_d = df_d[[joke_col_d]].rename(columns={joke_col_d: "raw_text"})
    df_d["source_file"] = deepseek_path.name
    logger.info(f"[DEEPSEEK] Loaded {len(df_d)} rows")

    df = pd.concat([df_g, df_d], ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"[WRITE] Combined raw -> {out_path} (rows={len(df)})")
    return out_path


def clean_combined(combined_path: Path, cleaned_out: Path) -> Path:
    cfg = PreprocessConfig()
    # Override directories to avoid touching existing outputs
    cfg.CLEANED_DIR = cleaned_out.parent
    cfg.DEBUG_DIR = cleaned_out.parent / "debug"
    cleaner = DataCleaner(cfg)
    tmp_clean = cleaner.run(input_path=combined_path)

    cleaned_out.parent.mkdir(parents=True, exist_ok=True)
    Path(tmp_clean).rename(cleaned_out)
    logger.info(f"[WRITE] Cleaned -> {cleaned_out}")
    return cleaned_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gemini-csv",
        type=Path,
        default=Path("data_source/raw/gemini_jokes_dataset_500k.csv"),
        help="Path to Gemini jokes CSV",
    )
    parser.add_argument(
        "--deepseek-csv",
        type=Path,
        default=Path("data_source/raw/deepseek_jokes_dataset_500k.csv"),
        help="Path to DeepSeek jokes CSV",
    )
    parser.add_argument(
        "--combined-out",
        type=Path,
        default=Path("data_source/raw_combined/combined_raw_llm.csv"),
        help="Where to write combined raw CSV",
    )
    parser.add_argument(
        "--clean-out",
        type=Path,
        default=Path("outputs/preprocessed/clean_jokes_llm.csv"),
        help="Where to write cleaned CSV",
    )
    args = parser.parse_args()

    combined_path = build_combined(args.gemini_csv, args.deepseek_csv, args.combined_out)
    clean_combined(combined_path, args.clean_out)
    logger.info("[DONE]")


if __name__ == "__main__":
    main()
