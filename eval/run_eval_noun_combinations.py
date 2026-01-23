import json
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from metrics import (
    topic_recall,
    topic_soft_recall,
    gpt2_perplexity,
    max_bleu_to_training,
    is_copied_from_training,
    max_embedding_similarity_to_training,
    is_semantic_duplicate,
    diversity_metrics,
    encode_sentences,
)

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from tokenizers import Tokenizer
from decoder_only.model.decoder_only import TransformerDecoder


# -----------------------------
# Model loading
# -----------------------------
def load_model_and_tokenizer(model_ckpt: Path, device: str = "cuda"):
    device = device if torch.cuda.is_available() and device == "cuda" else "cpu"

    tokenizer_path = Path("../data/processed/tokenizer.json")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_padding()

    ckpt = torch.load(model_ckpt, map_location=device)

    if "config" in ckpt:
        cfg = ckpt["config"]
        model = TransformerDecoder(**cfg)
    else:
        model = TransformerDecoder(
            vocab_size=30000,
            emb_dim=384,
            context_size=256,
            num_att_heads=6,
            dropout=0.2,
        )

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    return {"model": model, "tokenizer": tokenizer, "device": device}


# -----------------------------
# Generation
# -----------------------------
@torch.no_grad()
def generate_joke(
    model_bundle,
    prompt: str,
    max_new_tokens: int = 40,
    temperature: float = 1.0,
    top_k: int = 0,
):
    model = model_bundle["model"]
    tokenizer = model_bundle["tokenizer"]
    device = model_bundle["device"]

    EOS_ID = tokenizer.token_to_id("[/S]")
    if EOS_ID is None:
        raise ValueError("Tokenizer has no token [/S].")

    tok = tokenizer.encode(prompt)
    ids = tok.ids

    if ids and ids[-1] == EOS_ID:
        ids = ids[:-1]

    idx = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.context_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        if temperature != 1.0:
            logits = logits / temperature

        if top_k > 0:
            values, indices = torch.topk(logits, top_k)
            probs = F.softmax(values, dim=-1)
            next_id = indices.gather(-1, torch.multinomial(probs, 1))
        else:
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_id), dim=1)

        if next_id.item() == EOS_ID:
            break

    out_ids = idx[0].tolist()
    decoded = tokenizer.decode(out_ids, skip_special_tokens=False)

    if "[/S]" in decoded:
        decoded = decoded.split("[/S]")[0] + "[/S]"

    return decoded


# -----------------------------
# New: load eval from combo CSV
# -----------------------------
def load_eval_combos(eval_csv: Path) -> pd.DataFrame:
    """
    Input CSV format, e.g.:

    rid,stable_id,joke,joke_cleaned,word_count,topic,topic_all_nouns,combo_key,combo_freq
    """
    df = pd.read_csv(eval_csv)
    required = ["combo_key"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Eval CSV is missing required columns: {missing}")
    return df


def extract_topics_from_combo_row(row: pd.Series) -> List[str]:
    """Use combo_key as the list of topics (1–5 nouns)."""
    combo = str(row.get("combo_key", "")).strip()
    if not combo:
        return []
    return [t.strip() for t in combo.split(",") if t.strip()]


def format_prompt_from_topics(topics: List[str]) -> str:
    topics_str = ", ".join(topics)
    return f"tell me a joke about {topics_str} [JOKE]"


# -----------------------------
# Training data + embeddings
# -----------------------------
def load_training_jokes(train_csv: Path, text_col="joke_cleaned") -> List[str]:
    df = pd.read_csv(train_csv)
    if text_col not in df.columns:
        raise ValueError(f"Training CSV has no column '{text_col}'.")
    return df[text_col].astype(str).tolist()


def load_or_build_train_embeddings(train_jokes, embeddings_path=None):
    if embeddings_path and embeddings_path.exists():
        arr = np.load(embeddings_path)["embeddings"]
        return arr

    print("Computing sentence embeddings for training jokes...")
    train_embs = encode_sentences(train_jokes)
    if embeddings_path:
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(embeddings_path, embeddings=train_embs)
        print(f"Saved train embeddings to {embeddings_path}")
    return train_embs


# -----------------------------
# Evaluation (adapted for combos CSV)
# -----------------------------
def evaluate_model_on_combos(
    model_bundle,
    eval_df,
    training_jokes,
    train_embeddings,
    max_new_tokens=64,
    temperature=1.0,
    top_k=0,
    max_bleu_refs=1000,
):
    rows = []
    generated_texts = []

    for idx, row in eval_df.iterrows():
        # You can treat stable_id or rid as eval_id; here we use stable_id if present:
        if "stable_id" in row:
            eval_id = str(row["stable_id"])
        elif "rid" in row:
            eval_id = str(row["rid"])
        else:
            eval_id = str(idx)

        topics = extract_topics_from_combo_row(row)
        num_topics = len(topics)

        if num_topics == 0:
            print(f"Warning: eval_id={eval_id} has no topics; skipping.")
            continue

        prompt = format_prompt_from_topics(topics)

        start = time.time()
        generated_joke = generate_joke(
            model_bundle=model_bundle,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        decode_time = time.time() - start

        generated_texts.append(generated_joke)

        hard_recall, hard_full_hit = topic_recall(generated_joke, topics)
        soft_recall, soft_full_hit = topic_soft_recall(generated_joke, topics)
        ppl = gpt2_perplexity(generated_joke)

        max_bleu = max_bleu_to_training(
            generated_joke, training_jokes, max_refs=max_bleu_refs
        )
        copy_flag = is_copied_from_training(max_bleu, 0.8)

        max_sim = max_embedding_similarity_to_training(
            generated_joke, train_embeddings
        )
        sem_dup_flag = is_semantic_duplicate(max_sim, 0.9)

        rows.append({
            "eval_id": eval_id,
            # keep original info if present
            "rid": row.get("rid", None),
            "stable_id": row.get("stable_id", None),
            "combo_key": row.get("combo_key", None),
            "combo_freq": row.get("combo_freq", None),
            "topic_all_nouns": row.get("topic_all_nouns", None),
            "num_topics": num_topics,
            "prompt": prompt,
            "generated_joke": generated_joke,
            "decode_time_sec": decode_time,
            "topic_recall_hard": hard_recall,
            "topic_full_hit_hard": hard_full_hit,
            "topic_recall_soft": soft_recall,
            "topic_full_hit_soft": soft_full_hit,
            "ppl_gpt2": ppl,
            "max_bleu_train": max_bleu,
            "copy_flag_bleu": copy_flag,
            "max_sim_train": max_sim,
            "sem_dup_flag": sem_dup_flag,
        })

    df_out = pd.DataFrame(rows)

    if generated_texts:
        df_out.attrs["diversity"] = diversity_metrics(generated_texts)

    return df_out


# -----------------------------
# Summary (unchanged core logic)
# -----------------------------
def build_summary(result_df, model_name, model_ckpt, max_new_tokens, temperature, top_k):
    summary = {
        "model_name": model_name,
        "model_checkpoint": str(model_ckpt),
        "decode_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
        },
        "num_examples": int(len(result_df)),
    }

    if len(result_df) == 0:
        return summary

    metric_cols = [
        "topic_recall_hard",
        "topic_full_hit_hard",
        "topic_recall_soft",
        "topic_full_hit_soft",
        "ppl_gpt2",
        "max_bleu_train",
        "copy_flag_bleu",
        "max_sim_train",
        "sem_dup_flag",
    ]

    for col in metric_cols:
        if col in result_df.columns:
            summary[f"mean_{col}"] = float(result_df[col].mean())
            summary[f"median_{col}"] = float(result_df[col].median())

    div = result_df.attrs.get("diversity")
    if div:
        summary["diversity"] = div

    per_ntopic = {}
    for n in sorted(result_df["num_topics"].unique()):
        df_n = result_df[result_df["num_topics"] == n]
        sub = {"num_examples": int(len(df_n))}
        for col in metric_cols:
            if col in df_n.columns:
                sub[f"mean_{col}"] = float(df_n[col].mean())
                sub[f"median_{col}"] = float(df_n[col].median())
        per_ntopic[str(n)] = sub

    summary["by_num_topics"] = per_ntopic
    return summary


# -----------------------------
# main() – configure your combo CSV & tag here
# -----------------------------
def main():
    model_ckpt = Path("/ltstorage/home/4xin/uhh-ias-ml/models/chekpoints/checkpoint_49500.pt")
    model_name = "decoder_50k"

    # eval_csv = Path("/ltstorage/home/4xin/uhh-ias-ml/data/ouputs/jokes_top5_combo_k3.csv")
    eval_csv = Path("/ltstorage/home/4xin/uhh-ias-ml/data/ouputs/jokes_p75_top5_combo_k3.csv")

    # embeddings
    train_csv = Path("../data/processed/final_clean_jokes.csv")
    train_text_col = "joke_cleaned"
    train_embeddings_path = Path("../data/processed/train_embeddings.npz")

    out_dir = Path("eval_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda"
    max_new_tokens = 64
    temperature = 1.0
    top_k = 0
    max_bleu_refs = 1000

    eval_tag = "null_combinations_k3_p75_top5"

    print("[LOAD] Loading model...")
    model_bundle = load_model_and_tokenizer(model_ckpt, device=device)

    print("[LOAD] Loading eval combos...")
    eval_df = load_eval_combos(eval_csv)

    print("[LOAD] Loading training jokes...")
    training_jokes = load_training_jokes(train_csv, text_col=train_text_col)

    print("[LOAD] Loading embeddings...")
    train_embeddings = load_or_build_train_embeddings(
        training_jokes, embeddings_path=train_embeddings_path
    )

    print("[EVAL] Running evaluation on noun-combination prompts...")
    result_df = evaluate_model_on_combos(
        model_bundle,
        eval_df,
        training_jokes,
        train_embeddings,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        max_bleu_refs=max_bleu_refs,
    )

    # ----- 文件名加上 eval_tag -----
    results_path = out_dir / f"results_{model_name}_{eval_tag}.csv"
    result_df.to_csv(results_path, index=False)
    print(f"[WRITE] Wrote per-example results → {results_path}")

    summary = build_summary(
        result_df,
        model_name,
        model_ckpt,
        max_new_tokens,
        temperature,
        top_k,
    )

    summary_path = out_dir / f"summary_{model_name}_{eval_tag}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[WRITE] Wrote summary → {summary_path}")
    print("[DONE]")


if __name__ == "__main__":
    main()