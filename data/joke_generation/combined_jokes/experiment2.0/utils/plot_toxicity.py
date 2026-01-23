"""
Plot toxicity distributions per source using Plotly.

Typical inputs:
  - outputs/detox/clean_jokes_detox.csv
  - outputs/detox/clean_jokes_llm_detox.csv

Each source has its own color; all share the same x-axis (toxicity) and y-axis (density).

Usage (from repo root):
  python -m utils.plot_toxicity
  # or custom inputs/outputs
  python -m utils.plot_toxicity \
    --files outputs/detox/clean_jokes_detox.csv outputs/detox/clean_jokes_llm_detox.csv \
    --out stats_plots/detox/joke_toxicity/toxicity_distribution.html

    
python -m utils.plot_toxicity \
  --files outputs/detox/clean_jokes_detox.csv outputs/detox/clean_jokes_llm_detox.csv \
  --out stats_plots/detox/joke_toxicity/toxicity_distribution.html \
  --png stats_plots/detox/joke_toxicity/toxicity_distribution.png

"""
import argparse
from pathlib import Path
from typing import List

import pandas as pd
import plotly.express as px


def load_files(files: List[Path]) -> pd.DataFrame:
    dfs = []
    for p in files:
        if not p.exists():
            print(f"[WARN] skip missing {p}")
            continue
        df = pd.read_csv(p, engine="python", on_bad_lines="skip")
        if "toxicity" not in df.columns:
            print(f"[WARN] no 'toxicity' column in {p}, skipping")
            continue
        # source from column if present, else from filename
        if "source" not in df.columns:
            if "source_file" in df.columns:
                df["source"] = df["source_file"]
            else:
                df["source"] = p.name
        dfs.append(df)
    if not dfs:
        raise RuntimeError("No valid input files.")
    return pd.concat(dfs, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--files",
        nargs="+",
        type=Path,
        default=[
            Path("outputs/detox/clean_jokes_detox.csv"),
            Path("outputs/detox/clean_jokes_llm_detox.csv"),
        ],
        help="List of CSV files containing 'toxicity' and optional 'source' columns.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("stats_plots/detox/joke_toxicity/toxicity_distribution.html"),
        help="Output HTML path for the plot.",
    )
    ap.add_argument(
        "--png",
        type=Path,
        help="Optional PNG output path (requires kaleido installed).",
    )
    args = ap.parse_args()

    df = load_files(args.files)

    fig = px.histogram(
        df,
        x="toxicity",
        color="source",
        barmode="overlay",
        nbins=200,
        histnorm="probability density",
        opacity=0.6,
        title="Toxicity distribution by source",
    )
    fig.update_layout(xaxis_title="toxicity", yaxis_title="density")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.out)
    print(f"[WRITE] {args.out}")

    if args.png:
        try:
            args.png.parent.mkdir(parents=True, exist_ok=True)
            fig.write_image(args.png, width=1200, height=800, scale=2)
            print(f"[WRITE] {args.png}")
        except Exception as e:
            print(f"[WARN] Failed to write PNG ({e}). Ensure kaleido is installed: pip install -U kaleido")


if __name__ == "__main__":
    main()
