import argparse
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from detoxify import Detoxify
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm



DIMENSIONS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
]


def batch_iter(seq: List, batch_size: int):
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


def load_detox(model_name: str, device: str):
    return Detoxify(model_name, device=device)


def detox_scores(detox_model, texts: List[str], batch_size: int) -> List[Dict[str, float]]:
    scores: List[Dict[str, float]] = []
    total = (len(texts) + batch_size - 1) // batch_size
    for batch in tqdm(batch_iter(texts, batch_size), total=total, desc="Detoxify"):
        preds = detox_model.predict(batch)
        keys = list(preds.keys())
        for i in range(len(batch)):
            scores.append({k: float(preds[k][i]) for k in keys})
    return scores


def profanity_hits(text: str, base_list: List[str]) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in base_list)


def load_llamaguard(model_name: str, device: str, max_new_tokens: int):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.startswith("cuda") else None,
        device_map="auto",
        trust_remote_code=True,
    )
    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        device=0 if device.startswith("cuda") else -1,
        max_new_tokens=max_new_tokens,
    )
    return gen


def run_llamaguard(gen_pipe, texts: List[str]) -> List[str]:
    outputs = []
    for batch in tqdm(batch_iter(texts, 4), total=(len(texts) + 3) // 4, desc="LlamaGuard"):
        prompts = [f"[INST] {t} [/INST]" for t in batch]
        res = gen_pipe(prompts)
        for r in res:
            outputs.append(r[0]["generated_text"])
    return outputs


def default_output_paths(input_path: Path) -> tuple[Path, Path]:
    scored = input_path.with_name(input_path.stem + "_detox" + input_path.suffix)
    safe = scored.with_name(scored.stem + "_safe" + scored.suffix)
    return scored, safe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-csv",
        type=Path,
        default=Path(
            "/ltstorage/home/4xin/uhh-ias-ml/data/llm_jokes/experiment3.0/outputs/llm_jokes_top_5000_topics_3.csv"
        ),
        help="Input CSV to score and filter",
    )
    ap.add_argument("--text-col", default="joke_cleaned", help="Column with text to screen")
    ap.add_argument("--out-csv", type=Path, help="Output CSV with Detoxify scores")
    ap.add_argument("--safe-out-csv", type=Path, help="Output CSV with safe subset")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Drop if any Detoxify dimension >= threshold",
    )
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for Detoxify (and LlamaGuard if used)",
    )
    ap.add_argument("--detox-model", default="original", help="Detoxify model variant")
    ap.add_argument(
        "--detox-threshold",
        type=float,
        default=2.0,
        help="Flag if any Detoxify score >= threshold",
    )
    ap.add_argument("--detox-batch", type=int, default=64, help="Batch size for Detoxify")
    ap.add_argument("--profanity", nargs="*", default=[], help="Base profanity list (lowercase)")
    ap.add_argument("--extra-profanity", nargs="*", default=[], help="Additional profanity terms")
    ap.add_argument("--use-llamaguard", action="store_true", help="Enable LlamaGuard review")
    ap.add_argument("--llama-all", action="store_true", help="Run LlamaGuard on all rows")
    ap.add_argument("--llama-model", default="meta-llama/LlamaGuard-7b", help="LlamaGuard model name")
    ap.add_argument("--llama-max", type=int, default=128, help="max_new_tokens for LlamaGuard generation")
    ap.add_argument("--max-rows", type=int, help="Optional limit for debugging")
    args = ap.parse_args()

    if not args.data_csv.exists():
        raise FileNotFoundError(args.data_csv)

    out_csv, safe_out_csv = default_output_paths(args.data_csv)
    if args.out_csv:
        out_csv = args.out_csv
    if args.safe_out_csv:
        safe_out_csv = args.safe_out_csv

    print(f"[LOAD] data_csv={args.data_csv}")
    df = pd.read_csv(args.data_csv, engine="python", on_bad_lines="skip")
    if args.text_col not in df.columns:
        raise ValueError(f"text column '{args.text_col}' not found in CSV")
    if args.max_rows:
        df = df.head(args.max_rows)
        print(f"[INFO] limiting to first {len(df)} rows")

    texts = df[args.text_col].astype(str).tolist()
    profanity_list = [w.lower() for w in (args.profanity + args.extra_profanity)]

    print(f"[DETOX] model={args.detox_model} device={args.device} threshold={args.detox_threshold}")
    detox_model = load_detox(args.detox_model, args.device)
    detox_scores_list = detox_scores(detox_model, texts, args.detox_batch)

    detox_flags = []
    for t, sc in zip(texts, detox_scores_list):
        toxic = any(v >= args.detox_threshold for v in sc.values())
        prof = profanity_hits(t, profanity_list)
        detox_flags.append(bool(toxic or prof))

    df_scores = pd.DataFrame(detox_scores_list)
    df_flags = pd.DataFrame(
        {
            "detox_flag": detox_flags,
            "profanity_flag": [profanity_hits(t, profanity_list) for t in texts],
        }
    )

    llama_outputs = [""] * len(texts)
    llama_flags = [False] * len(texts)
    if args.use_llamaguard:
        target_indices = list(range(len(texts))) if args.llama_all else [i for i, f in enumerate(detox_flags) if f]
        if target_indices:
            print(f"[LLAMAGUARD] Loading {args.llama_model} for {len(target_indices)} rows")
            gen_pipe = load_llamaguard(args.llama_model, args.device, args.llama_max)
            batch_texts = [texts[i] for i in target_indices]
            outputs = run_llamaguard(gen_pipe, batch_texts)
            for idx, out in zip(target_indices, outputs):
                llama_outputs[idx] = out
                llama_flags[idx] = bool(re.search(r"unsafe|not safe|violation|reject", out, re.IGNORECASE))
        else:
            print("[LLAMAGUARD] No rows to review")

    df_out = pd.concat(
        [
            df.reset_index(drop=True),
            df_scores.reset_index(drop=True),
            df_flags.reset_index(drop=True),
            pd.DataFrame({"llamaguard_output": llama_outputs, "llamaguard_flag": llama_flags}),
        ],
        axis=1,
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print(f"[WRITE] Saved scored data -> {out_csv}")

    missing_dims = [c for c in DIMENSIONS if c not in df_out.columns]
    if missing_dims:
        raise ValueError(f"Missing Detoxify columns: {missing_dims}")

    toxic_mask = (df_out[DIMENSIONS] >= args.threshold).any(axis=1)
    removed_df = df_out[toxic_mask]
    safe_df = df_out[~toxic_mask]

    safe_out_csv.parent.mkdir(parents=True, exist_ok=True)
    safe_df.to_csv(safe_out_csv, index=False)
    print(f"[WRITE] Saved safe subset -> {safe_out_csv} (kept {len(safe_df)}/{len(df_out)})")

    print(f"Total rows: {len(df_out)}")
    print(f"Removed rows: {len(removed_df)}")
    print(f"Removed ratio: {len(removed_df) / len(df_out) if len(df_out) else 0.0:.4f}")
    for dim in DIMENSIONS:
        count = int((removed_df[dim] >= args.threshold).sum())
        print(f"Removed due to {dim}: {count}")


if __name__ == "__main__":
    main()

'''
[WRITE] Saved scored data -> /ltstorage/home/4xin/uhh-ias-ml/data/llm_jokes/experiment3.0/outputs/llm_jokes_top_5000_topics_3_detox.csv
[WRITE] Saved safe subset -> /ltstorage/home/4xin/uhh-ias-ml/data/llm_jokes/experiment3.0/outputs/llm_jokes_top_5000_topics_3_detox_safe.csv (kept 509087/522347)
Total rows: 522347
Removed rows: 13260
Removed ratio: 0.0254
Removed due to toxicity: 13260
Removed due to severe_toxicity: 0
Removed due to obscene: 1216
Removed due to threat: 68
Removed due to insult: 1388
Removed due to identity_attack: 66
'''