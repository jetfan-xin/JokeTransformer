"""
Plot PPL distributions per source (counts on y-axis).

Defaults:
  - Inputs: outputs/perplexity/train_quality_metrics.csv and outputs/perplexity/train_quality_metrics_llm.csv
  - Outputs: stats_plots/perplexity/joke_sources/ppl_joke_sources.html and stats_plots/perplexity/joke_sources/ppl_joke_sources.png

Usage (from repo root):
  python utils/plot_ppl_joke_sources.py \
    --files outputs/perplexity/train_quality_metrics.csv outputs/perplexity/train_quality_metrics_llm.csv \
    --out-html stats_plots/perplexity/joke_sources/ppl_joke_sources.html \
    --out-png stats_plots/perplexity/joke_sources/ppl_joke_sources.png \
    --nbins 200
"""
import argparse
from pathlib import Path
from typing import List

import pandas as pd
import plotly.express as px


def load_ppl_files(files: List[Path]) -> pd.DataFrame:
    """Load PPL CSVs and ensure 'perplexity' + 'source' columns exist."""
    dfs = []
    for p in files:
        if not p.exists():
            print(f"[WARN] skip missing {p}")
            continue
        df = pd.read_csv(p, engine="python", on_bad_lines="skip")
        if "perplexity" not in df.columns:
            print(f"[WARN] no 'perplexity' column in {p}, skipping")
            continue
        if "source" not in df.columns:
            if "source_file" in df.columns:
                df["source"] = df["source_file"]
            else:
                df["source"] = p.name
        dfs.append(df)
    if not dfs:
        raise RuntimeError("No valid input files for PPL plotting.")
    return pd.concat(dfs, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--files",
        nargs="+",
        type=Path,
        default=[
            Path("outputs/perplexity/train_quality_metrics.csv"),
            Path("outputs/perplexity/train_quality_metrics_llm.csv"),
        ],
        help="CSV files with 'perplexity' (and optional 'source').",
    )
    ap.add_argument(
        "--out-html",
        type=Path,
        default=Path("stats_plots/perplexity/joke_sources/ppl_joke_sources.html"),
        help="Output HTML path for PPL histogram overlay.",
    )
    ap.add_argument(
        "--out-png",
        type=Path,
        default=Path("stats_plots/perplexity/joke_sources/ppl_joke_sources.png"),
        help="Optional PNG output (requires kaleido). Set empty to skip.",
    )
    ap.add_argument(
        "--nbins",
        type=int,
        default=200,
        help="Number of bins for histograms.",
    )
    args = ap.parse_args()

    df = load_ppl_files(args.files)

    fig = px.histogram(
        df,
        x="perplexity",
        color="source",
        barmode="overlay",
        nbins=args.nbins,
        opacity=0.6,
        title="PPL distribution by source (counts)",
    )
    fig.update_layout(xaxis_title="perplexity", yaxis_title="count")

    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.out_html)
    print(f"[WRITE] {args.out_html}")

    if args.out_png:
        try:
            args.out_png.parent.mkdir(parents=True, exist_ok=True)
            fig.write_image(args.out_png, width=1200, height=800, scale=2)
            print(f"[WRITE] {args.out_png}")
        except Exception as e:
            print(f"[WARN] Failed to write PNG: {e} (install kaleido?)")


if __name__ == "__main__":
    main()
