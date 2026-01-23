import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px


DEFAULT_INPUT = Path(
    "/ltstorage/home/4xin/uhh-ias-ml/data/llm_jokes/experiment3.0/stats/llm_jokes_top_5000_topics_3_detox_safe_ppl.csv"
)
DEFAULT_OUT_HTML = Path(
    "/ltstorage/home/4xin/uhh-ias-ml/data/llm_jokes/experiment3.0/plots/ppl_hist.html"
)
DEFAULT_OUT_PNG = Path(
    "/ltstorage/home/4xin/uhh-ias-ml/data/llm_jokes/experiment3.0/plots/ppl_hist.png"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out-html", type=Path, default=DEFAULT_OUT_HTML)
    ap.add_argument("--out-png", type=Path, default=DEFAULT_OUT_PNG)
    ap.add_argument("--nbins", type=int, default=200)
    args = ap.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    df = pd.read_csv(args.input, engine="python", on_bad_lines="skip")
    if "perplexity" not in df.columns:
        raise ValueError("Input CSV missing 'perplexity' column.")

    fig = px.histogram(
        df,
        x="perplexity",
        nbins=args.nbins,
        opacity=0.8,
        title="PPL distribution (counts)",
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
