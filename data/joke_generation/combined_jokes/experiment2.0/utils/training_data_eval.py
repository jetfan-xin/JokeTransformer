"""
Training-data quality metrics (no project checkpoints required).

Metrics per joke:
1) PPL via a small public LM (default: distilgpt2).
## 2) Max BLEU-4 to reference set (sample size configurable). # too slow for large datasets, removed
Corpus-level:
3) Distinct-1 / Distinct-2.

Example:
python -m utils.training_data_eval \
  --data-csv outputs/preprocessed/clean_jokes.csv \
  --text-col joke_cleaned \
  --out-csv outputs/perplexity/train_quality_metrics.csv \
  --device cuda

CUDA_VISIBLE_DEVICES=4 python -m utils.training_data_eval \
  --data-csv outputs/preprocessed/clean_jokes_llm.csv \
  --text-col joke_cleaned \
  --out-csv outputs/perplexity/train_quality_metrics_llm.csv \
  --device cuda
"""
import argparse
import math
import random
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.logger import logger

"""CUDA_VISIBLE_DEVICES=2 python -m utils.training_data_eval \
  --data-csv outputs/preprocessed/clean_jokes_llm.csv \
  --text-col joke_cleaned \
  --out-csv outputs/perplexity/train_quality_metrics_llm.csv \
  --device cuda \
  --batch-size 8 \
  --max-length 256
  
CUDA_VISIBLE_DEVICES=2 python -m utils.training_data_eval \
  --data-csv outputs/preprocessed/clean_jokes.csv \
  --text-col joke_cleaned \
  --out-csv outputs/perplexity/train_quality_metrics.csv \
  --device cuda \
  --batch-size 8 \
  --max-length 256
  """

def load_lm(model_name: str, device: str):
    """Load a small causal LM for scoring."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device.startswith("cuda") else None
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def calc_ppl_batch(texts: List[str], tokenizer, model, device: str, max_length: Optional[int]) -> List[float]:
    """Compute sentence-level perplexity for a batch."""
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=bool(max_length),
        max_length=max_length if max_length else None,
    )
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    # Shift labels
    labels = input_ids.clone()
    outputs = model(input_ids=input_ids, attention_mask=attn, labels=labels)
    # outputs.loss is mean over tokens with label != -100; here labels not masked, so equivalent to full seq
    # Recompute per-sample token-level NLL to get sentence PPL
    logits = outputs.logits  # (B, T, V)
    vocab_size = logits.size(-1)
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)
    log_probs = torch.nn.functional.log_softmax(logits_flat, dim=-1)
    nll_flat = -log_probs[torch.arange(labels_flat.numel(), device=device), labels_flat]
    nll = nll_flat.view(labels.shape)

    token_counts = attn.sum(dim=1)  # (B,)
    nll_sum = (nll * attn).sum(dim=1)  # (B,)
    ppl = torch.exp(nll_sum / token_counts.clamp(min=1)).tolist()
    return ppl


def max_bleu_to_refs(text: str, refs: List[str], smooth_fn, max_refs: int) -> float:
    if not text:
        return 0.0
    cand_tokens = text.split()
    if not cand_tokens:
        return 0.0
    refs_use = refs
    if max_refs and len(refs) > max_refs:
        refs_use = random.sample(refs, max_refs)
    max_bleu = 0.0
    for r in refs_use:
        ref_tokens = str(r).split()
        bleu = sentence_bleu(
            [ref_tokens],
            cand_tokens,
            smoothing_function=smooth_fn,
            weights=(0.25, 0.25, 0.25, 0.25),
        )
        if bleu > max_bleu:
            max_bleu = bleu
    return max_bleu


def distinct_n(texts: Iterable[str], n: int) -> float:
    ngrams = set()
    total = 0
    for t in texts:
        toks = t.split()
        if len(toks) < n:
            continue
        for i in range(len(toks) - n + 1):
            ngrams.add(tuple(toks[i : i + n]))
            total += 1
    if total == 0:
        return 0.0
    return len(ngrams) / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-csv", type=Path, required=True, help="CSV file to score")
    ap.add_argument("--text-col", default="joke_cleaned", help="Column containing joke text")
    ap.add_argument("--out-csv", type=Path, required=True, help="Where to write per-row metrics")
    ap.add_argument("--model-name", default="distilgpt2", help="HF causal LM for PPL (small & fast)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device for LM scoring")
    ap.add_argument("--batch-size", type=int, default=8, help="Batch size for PPL")
    ap.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Max tokens for LM scoring; leave empty to disable truncation",
    )
    ap.add_argument("--bleu-max-refs", type=int, default=1000, help="Max refs to sample per row for BLEU")
    ap.add_argument("--max-rows", type=int, help="Limit rows (debugging)")
    ap.add_argument(
        "--id-col",
        type=str,
        help="Optional ID column to carry over (e.g., rid or stable_id). If not set, tries rid then stable_id, otherwise uses row index.",
    )
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

    # pick id column
    id_col = args.id_col
    if id_col is None:
        for cand in ["rid", "stable_id"]:
            if cand in df.columns:
                id_col = cand
                break
    if id_col and id_col not in df.columns:
        logger.warn(f"[WARN] id_col '{id_col}' not found; falling back to row index")
        id_col = None
    if id_col is None:
        ids = list(range(len(df)))
    else:
        ids = df[id_col].tolist()
    logger.info(f"[ID] using id_col={id_col or 'row_index'}")

    # source handling: prefer source, then source_file, else filename
    if "source" in df.columns:
        source_values = df["source"].astype(str).tolist()
    elif "source_file" in df.columns:
        source_values = df["source_file"].astype(str).tolist()
    else:
        source_values = [args.data_csv.name] * len(df)

    texts = df[args.text_col].astype(str).tolist()

    # Load LM
    logger.info(f"[LM] loading {args.model_name} on {args.device}")
    tokenizer, model = load_lm(args.model_name, args.device)

    # PPL
    all_ppl: List[float] = []
    for i in range(0, len(texts), args.batch_size):
        batch = texts[i : i + args.batch_size]
        ppl_batch = calc_ppl_batch(batch, tokenizer, model, args.device, args.max_length)
        all_ppl.extend(ppl_batch)
        if (i // args.batch_size + 1) % 50 == 0:
            logger.info(f"[PPL] processed {min(len(texts), i + args.batch_size)}/{len(texts)}")

    # # BLEU
    # smooth_fn = SmoothingFunction().method1
    # bleu_scores = []
    # for idx, text in enumerate(texts):
    #     bleu = max_bleu_to_refs(text, texts, smooth_fn, args.bleu_max_refs)
    #     bleu_scores.append(bleu)
    #     if (idx + 1) % 500 == 0:
    #         logger.info(f"[BLEU] processed {idx + 1}/{len(texts)}")

    # Distinct
    dist1 = distinct_n(texts, 1)
    dist2 = distinct_n(texts, 2)
    logger.info(f"[DISTINCT] distinct_1={dist1:.4f}, distinct_2={dist2:.4f}")

    # Write out
    out_df = pd.DataFrame(
        {
            "source": source_values,
            "id": ids,
            "text": texts,
            "perplexity": all_ppl,
            # "max_bleu": bleu_scores,
        }
    )
    out_df.to_csv(args.out_csv, index=False)
    logger.info(f"[WRITE] per-row metrics -> {args.out_csv}")
    logger.info("[DONE]")


if __name__ == "__main__":
    main()
