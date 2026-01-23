"""
Plot PPL distributions before/after detox filtering per source.

Inputs:
  - Metrics (before): outputs/perplexity/train_quality_metrics.csv
  - Safe (after):    outputs/detox/clean_jokes_detox_safe.csv

For each source present in both files, generate a Plotly histogram overlay
of perplexity distributions (before vs after). One HTML (and optional PNG)
per source written to an output directory.

Usage (from repo root):
  python -m utils.plot_ppl_before_after

Custom paths/example:
  python -m utils.plot_ppl_before_after \
    --metrics outputs/perplexity/train_quality_metrics.csv \
    --safe outputs/detox/clean_jokes_detox_safe.csv \
    --out-dir stats_plots/perplexity/ppl_before_after \
    --png
"""
import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
import plotly.express as px


def load_with_source(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    if "perplexity" not in df.columns:
        raise ValueError(f"{path} missing 'perplexity' column")
    if "source" in df.columns:
        pass
    elif "source_file" in df.columns:
        df["source"] = df["source_file"]
    else:
        df["source"] = path.name
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metrics",
        type=Path,
        default=Path("outputs/perplexity/train_quality_metrics.csv"),
        help="CSV with perplexity before filtering",
    )
    ap.add_argument(
        "--safe",
        type=Path,
        default=Path("outputs/detox/clean_jokes_detox_safe.csv"),
        help="CSV with perplexity after filtering",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("stats_plots/perplexity/ppl_before_after"),
        help="Directory to write per-source plots",
    )
    ap.add_argument(
        "--png",
        action="store_true",
        help="Also write PNG (requires kaleido installed)",
    )
    ap.add_argument(
        "--nbins",
        type=int,
        default=200,
        help="Number of bins for histograms",
    )
    args = ap.parse_args()

    if not args.metrics.exists():
        raise FileNotFoundError(args.metrics)
    if not args.safe.exists():
        raise FileNotFoundError(args.safe)

    df_before = load_with_source(args.metrics)
    df_after = load_with_source(args.safe)

    common_sources = sorted(set(df_before["source"]).intersection(set(df_after["source"])))
    if not common_sources:
        raise RuntimeError("No common sources between metrics and safe files.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] sources: {common_sources}")

    for src in common_sources:
        b = df_before[df_before["source"] == src]
        a = df_after[df_after["source"] == src]
        combined = pd.concat(
            [
                pd.DataFrame({"perplexity": b["perplexity"], "stage": "before"}),
                pd.DataFrame({"perplexity": a["perplexity"], "stage": "after"}),
            ],
            ignore_index=True,
        )
        fig = px.histogram(
            combined,
            x="perplexity",
            color="stage",
            barmode="overlay",
            nbins=args.nbins,
            opacity=0.6,
            title=f"PPL distribution (before vs after) - {src}",
        )
        fig.update_layout(xaxis_title="perplexity", yaxis_title="count")

        html_path = args.out_dir / f"ppl_before_after_{src}.html"
        fig.write_html(html_path)
        print(f"[WRITE] {html_path}")

        if args.png:
            try:
                png_path = args.out_dir / f"ppl_before_after_{src}.png"
                fig.write_image(png_path, width=1200, height=800, scale=2)
                print(f"[WRITE] {png_path}")
            except Exception as e:
                print(f"[WARN] PNG failed for {src}: {e} (install kaleido?)")


if __name__ == "__main__":
    main()
