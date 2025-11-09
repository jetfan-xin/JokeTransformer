# utils/inference.py

import torch
from utils.config import Config  # 注意：从utils导入

cfg = Config()
tokenizer = cfg.tokenizer


@torch.no_grad()
def generate_joke(model, topics, max_new_tokens=40, temperature=0.8, top_k=50):
    """
    Generate a joke given one or more topics using the trained decoder-only model.

    Args:
        model: DecoderOnlyTransformer, already loaded & moved to device.
        topics: str or list/tuple of str, e.g. "cat and programming" or ["cat", "programming"]
        max_new_tokens: max number of tokens to generate for the joke.
        temperature: softmax temperature for sampling.
        top_k: if not None, apply top-k sampling.

    Returns:
        Generated joke string (without special tokens).
    """
    model.eval()
    device = next(model.parameters()).device

    # ---- format topics ----
    if isinstance(topics, (list, tuple, set)):
        topics_str = ", ".join(str(t) for t in topics)
    else:
        topics_str = str(topics)

    # ---- 1) encode prompt as single sequence -> [S] prompt [/S] ----
    prompt = f"tell me a joke about {topics_str}"
    enc = tokenizer.encode(prompt)
    ids = enc.ids  # e.g. [S] ... [/S]

    if len(ids) == 0:
        raise ValueError("Encoded prompt is empty; check tokenizer / input.")

    # ---- 2) replace closing [/S] with [JOKE] as generation start marker ----
    if ids[-1] == cfg.eos_token_id:
        ids[-1] = cfg.joke_token_id
    else:
        ids.append(cfg.joke_token_id)

    # ---- truncate if needed ----
    if len(ids) > cfg.max_seq_len:
        ids = ids[: cfg.max_seq_len]

    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, L]

    # ---- 3) autoregressive generation ----
    for _ in range(max_new_tokens):
        attn_mask = (input_ids != cfg.pad_token_id).long()  # [1, L]
        logits = model(input_ids, attn_mask=attn_mask)      # [1, L, V]

        next_logits = logits[:, -1, :] / max(temperature, 1e-6)

        # top-k filtering (optional)
        if top_k is not None:
            top_k_val = min(top_k, next_logits.size(-1))
            values, _ = torch.topk(next_logits, top_k_val)
            cutoff = values[:, -1].unsqueeze(-1)
            next_logits = torch.where(
                next_logits < cutoff,
                torch.full_like(next_logits, float("-inf")),
                next_logits,
            )

        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # [1, 1]
        token_id = next_id.item()

        # stop if model generates </S>
        if token_id == cfg.eos_token_id:
            input_ids = torch.cat([input_ids, next_id], dim=1)
            break

        # stop if reach max length
        if input_ids.size(1) >= cfg.max_seq_len:
            break

        input_ids = torch.cat([input_ids, next_id], dim=1)

    # ---- 4) extract tokens after [JOKE] ----
    full_ids = input_ids[0].tolist()

    if cfg.joke_token_id in full_ids:
        j_idx = full_ids.index(cfg.joke_token_id)
        joke_ids = full_ids[j_idx + 1:]
    else:
        joke_ids = full_ids

    # cut at </S> if exists
    if cfg.eos_token_id in joke_ids:
        eos_pos = joke_ids.index(cfg.eos_token_id)
        joke_ids = joke_ids[:eos_pos]

    # ---- 5) decode to text ----
    if len(joke_ids) == 0:
        return ""  # model didn't generate anything beyond [JOKE]

    joke = tokenizer.decode(joke_ids)
    return joke.strip()