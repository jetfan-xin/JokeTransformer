"""
Two-stage safety filtering for jokes:
1) Fast pass: Detoxify + profanity list (low threshold) to flag likely toxic/offensive text.
2) Optional slow pass: LlamaGuard-7b on flagged rows for stricter review.

Usage (from repo root):
  python -m utils.safety_filter \
    --data-csv outputs/preprocessed/clean_jokes.csv \
    --text-col joke_cleaned \
    --out-csv outputs/detox/clean_jokes_detox.csv \
    --detox-threshold 0.5 \
    --use-llamaguard \
    --llama-max 128 \
    --max-rows 20000

Notes:
- Detoxify is fast and works on CPU/GPU (set --device).
- LlamaGuard-7b is large; requires GPU and the model weights (downloads on first run). Only applied to rows flagged by stage 1 unless you set --llama-all.
- Profanity list is a small built-in set; extend via --extra-profanity.
"""
import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import torch
from detoxify import Detoxify
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm

from utils.logger import logger


def batch_iter(seq: List, batch_size: int):
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


def load_detox(model_name: str, device: str):
    return Detoxify(model_name, device=device)


def detox_scores(detox_model, texts: List[str], batch_size: int) -> List[Dict[str, float]]:
    scores: List[Dict[str, float]] = []
    for batch in tqdm(batch_iter(texts, batch_size), total=(len(texts) + batch_size - 1) // batch_size, desc="Detoxify"):
        preds = detox_model.predict(batch)
        # preds is a dict of lists keyed by score name
        keys = list(preds.keys())
        for i in range(len(batch)):
            scores.append({k: float(preds[k][i]) for k in keys})
    return scores


def profanity_hits(text: str, base_list: List[str]) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in base_list)


def load_llamaguard(model_name: str, device: str, max_new_tokens: int):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device.startswith("cuda") else None, device_map="auto", trust_remote_code=True)
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
        # pipeline returns list of list[dict]; take first generated text
        for r in res:
            outputs.append(r[0]["generated_text"])
    return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-csv", type=Path, required=True, help="Input CSV to filter")
    ap.add_argument("--text-col", default="joke_cleaned", help="Column with text to screen")
    ap.add_argument("--out-csv", type=Path, required=True, help="Output CSV with scores/flags")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device for Detoxify (and LlamaGuard if used)")
    ap.add_argument("--detox-model", default="original", help="Detoxify model variant (original / multilingual)")
    ap.add_argument("--detox-threshold", type=float, default=2.0, help="Flag if any Detoxify score >= threshold")
    ap.add_argument("--detox-batch", type=int, default=64, help="Batch size for Detoxify")
    ap.add_argument("--profanity", nargs="*", default=[], help="Base profanity list (lowercase)") # ["fuck", "shit", "bitch", "cunt", "asshole", "nigger", "faggot", "slut", "whore"]
    ap.add_argument("--extra-profanity", nargs="*", default=[], help="Additional profanity terms")
    ap.add_argument("--use-llamaguard", action="store_true", help="Enable LlamaGuard second-stage review")
    ap.add_argument("--llama-all", action="store_true", help="If set, run LlamaGuard on all rows (otherwise only flagged ones)")
    ap.add_argument("--llama-model", default="meta-llama/LlamaGuard-7b", help="LlamaGuard model name")
    ap.add_argument("--llama-max", type=int, default=128, help="max_new_tokens for LlamaGuard generation")
    ap.add_argument("--max-rows", type=int, help="Optional limit for debugging")
    args = ap.parse_args()

    if not args.data_csv.exists():
        raise FileNotFoundError(args.data_csv)

    logger.info(f"[LOAD] data_csv={args.data_csv}")
    df = pd.read_csv(args.data_csv, engine="python", on_bad_lines="skip")
    if args.text_col not in df.columns:
        raise ValueError(f"text column '{args.text_col}' not found in CSV")
    if args.max_rows:
        df = df.head(args.max_rows)
        logger.info(f"[INFO] limiting to first {len(df)} rows")

    texts = df[args.text_col].astype(str).tolist()
    profanity_list = [w.lower() for w in (args.profanity + args.extra_profanity)]

    # Stage 1: Detoxify + profanity
    logger.info(f"[DETOX] model={args.detox_model} device={args.device} threshold={args.detox_threshold}")
    detox_model = load_detox(args.detox_model, args.device)
    detox_scores_list = detox_scores(detox_model, texts, args.detox_batch)

    detox_flags = []
    for t, sc in zip(texts, detox_scores_list):
        toxic = any(v >= args.detox_threshold for v in sc.values())
        prof = profanity_hits(t, profanity_list)
        detox_flags.append(bool(toxic or prof))

    df_scores = pd.DataFrame(detox_scores_list)
    df_out = pd.DataFrame(
        {
            "detox_flag": detox_flags,
            "profanity_flag": [profanity_hits(t, profanity_list) for t in texts],
        }
    )

    # Stage 2: LlamaGuard (optional)
    llama_outputs = [""] * len(texts)
    llama_flags = [False] * len(texts)
    if args.use_llamaguard:
        target_indices = list(range(len(texts))) if args.llama_all else [i for i, f in enumerate(detox_flags) if f]
        if target_indices:
            logger.info(f"[LLAMAGUARD] Loading {args.llama_model} for {len(target_indices)} rows")
            gen_pipe = load_llamaguard(args.llama_model, args.device, args.llama_max)
            batch_texts = [texts[i] for i in target_indices]
            outputs = run_llamaguard(gen_pipe, batch_texts)
            for idx, out in zip(target_indices, outputs):
                llama_outputs[idx] = out
                llama_flags[idx] = bool(re.search(r"unsafe|not safe|violation|reject", out, re.IGNORECASE))
        else:
            logger.info("[LLAMAGUARD] No rows to review")

    # Assemble final
    df_out = pd.concat(
        [
            df.reset_index(drop=True),
            df_scores.reset_index(drop=True),
            df_out.reset_index(drop=True),
            pd.DataFrame({"llamaguard_output": llama_outputs, "llamaguard_flag": llama_flags}),
        ],
        axis=1,
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.out_csv, index=False)
    logger.info(f"[WRITE] Saved -> {args.out_csv}")
    logger.info("[DONE]")


if __name__ == "__main__":
    main()
