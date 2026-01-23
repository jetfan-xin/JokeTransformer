import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_INPUT = Path(
    "/ltstorage/home/4xin/uhh-ias-ml/data/llm_jokes/experiment3.0/outputs/llm_jokes_top_5000_topics_3_detox_safe.csv"
)
DEFAULT_OUTPUT = Path(
    "/ltstorage/home/4xin/uhh-ias-ml/data/llm_jokes/experiment3.0/stats/llm_jokes_top_5000_topics_3_detox_safe_ppl.csv"
)


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return requested
        print("[WARN] cuda requested but not available; falling back to cpu")
        return "cpu"
    return "cpu"


def configure_cuda(device: str):
    if not device.startswith("cuda"):
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def load_lm(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device.startswith("cuda") else None
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    model.config.use_cache = False
    return tokenizer, model


@torch.no_grad()
def calc_ppl_batch(
    texts: List[str],
    tokenizer,
    model,
    device: str,
    max_length: Optional[int],
    use_amp: bool,
) -> List[float]:
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=bool(max_length),
        max_length=max_length if max_length else None,
    )
    input_ids = enc["input_ids"].to(device, non_blocking=True)
    attn = enc["attention_mask"].to(device, non_blocking=True)

    labels = input_ids.clone()
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if (use_amp and device.startswith("cuda"))
        else nullcontext()
    )
    with autocast_ctx:
        outputs = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        logits = outputs.logits
    vocab_size = logits.size(-1)
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)
    log_probs = torch.nn.functional.log_softmax(logits_flat, dim=-1)
    nll_flat = -log_probs[torch.arange(labels_flat.numel(), device=device), labels_flat]
    nll = nll_flat.view(labels.shape)

    token_counts = attn.sum(dim=1)
    nll_sum = (nll * attn).sum(dim=1)
    ppl = torch.exp(nll_sum / token_counts.clamp(min=1)).tolist()
    return ppl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-csv", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--text-col", default="joke_cleaned")
    ap.add_argument("--out-csv", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--model-name", default="distilgpt2")
    ap.add_argument(
        "--device",
        default="auto",
        help="Device for LM scoring: auto, cpu, cuda, cuda:0, ...",
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--max-rows", type=int, help="Limit rows for debugging")
    ap.add_argument(
        "--id-col",
        type=str,
        help="Optional ID column (e.g., rid or stable_id).",
    )
    args = ap.parse_args()

    if not args.data_csv.exists():
        raise FileNotFoundError(args.data_csv)

    device = resolve_device(args.device)
    configure_cuda(device)
    use_amp = device.startswith("cuda")

    print(f"[LOAD] data_csv={args.data_csv}")
    df = pd.read_csv(args.data_csv, engine="python", on_bad_lines="skip")
    if args.text_col not in df.columns:
        raise ValueError(f"text column '{args.text_col}' not found in CSV")
    if args.max_rows:
        df = df.head(args.max_rows)
        print(f"[INFO] limiting to first {len(df)} rows")

    id_col = args.id_col
    if id_col is None:
        for cand in ["rid", "stable_id"]:
            if cand in df.columns:
                id_col = cand
                break
    if id_col and id_col not in df.columns:
        print(f"[WARN] id_col '{id_col}' not found; using row index")
        id_col = None
    if id_col is None:
        ids = list(range(len(df)))
    else:
        ids = df[id_col].tolist()
    print(f"[ID] using id_col={id_col or 'row_index'}")

    texts = df[args.text_col].astype(str).tolist()

    print(f"[LM] loading {args.model_name} on {device}")
    tokenizer, model = load_lm(args.model_name, device)

    all_ppl: List[float] = []
    for i in range(0, len(texts), args.batch_size):
        batch = texts[i : i + args.batch_size]
        ppl_batch = calc_ppl_batch(
            batch, tokenizer, model, device, args.max_length, use_amp
        )
        all_ppl.extend(ppl_batch)
        if (i // args.batch_size + 1) % 50 == 0:
            print(f"[PPL] processed {min(len(texts), i + args.batch_size)}/{len(texts)}")

    out_df = pd.DataFrame(
        {
            "id": ids,
            "text": texts,
            "perplexity": all_ppl,
        }
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"[WRITE] per-row perplexity -> {args.out_csv}")
    print("[DONE]")


if __name__ == "__main__":
    main()
