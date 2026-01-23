"""
Plot distributions of Detoxify dimensions on the whole dataset.

Dimensions (columns expected in input CSV):
  - toxicity
  - severe_toxicity
  - obscene
  - threat
  - insult
  - identity_attack

All dimensions share the same x-axis (score 0–1) and are distinguished by color.

Usage (from repo root):
  python -m utils.plot_toxicity_dims

or with custom inputs/outputs:
  python -m utils.plot_toxicity_dims \
    --files outputs/detox/clean_jokes_detox.csv outputs/detox/clean_jokes_llm_detox.csv \
    --out stats_plots/detox/joke_toxicity/toxicity_dims_distribution.html

python -m utils.plot_toxicity_dims \
  --out stats_plots/detox/joke_toxicity/toxicity_dims_distribution.html \
  --png stats_plots/detox/joke_toxicity/toxicity_dims_distribution.png

"""
import argparse
from pathlib import Path
from typing import List

import pandas as pd
import plotly.express as px


DEFAULT_DIMS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
]


def load_files(files: List[Path], dims: List[str]) -> pd.DataFrame:
    dfs = []
    for p in files:
        if not p.exists():
            print(f"[WARN] skip missing {p}")
            continue
        df = pd.read_csv(p, engine="python", on_bad_lines="skip")
        missing = [c for c in dims if c not in df.columns]
        if missing:
            print(f"[WARN] {p} missing columns {missing}, skipping")
            continue
        dfs.append(df[dims])
    if not dfs:
        raise RuntimeError("No valid input files with required columns.")
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
        help="CSV files with Detoxify columns.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("stats_plots/detox/joke_toxicity/toxicity_dims_distribution.html"),
        help="Output HTML path.",
    )
    ap.add_argument(
        "--png",
        type=Path,
        help="Optional PNG output (requires kaleido installed).",
    )
    args = ap.parse_args()

    df = load_files(args.files, DEFAULT_DIMS)

    # Melt into long format: columns -> dim, value
    long_df = df.melt(var_name="dimension", value_name="score")

    fig = px.histogram(
        long_df,
        x="score",
        color="dimension",
        barmode="overlay",
        nbins=200,
        histnorm="probability density",
        opacity=0.6,
        title="Detoxify dimension distributions",
    )
    fig.update_layout(xaxis_title="score", yaxis_title="density")

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
