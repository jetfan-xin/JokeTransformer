# utils/dataset.py

import torch
from torch.utils.data import Dataset
import pandas as pd
from utils.config import Config

cfg = Config()


class JokeDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = cfg.tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        topics = row["topics"]
        joke = str(row["joke"])

        # handle topics field
        if isinstance(topics, (list, tuple)):
            topics_str = ", ".join(topics)
        else:
            if pd.isna(topics):
                topics_str = ""
            else:
                topics_str = str(topics)

        # A = prompt, B = joke
        prompt = f"tell me a joke about {topics_str}"

        # uses pair template: [S] A [JOKE] B [/S]
        encoded = self.tokenizer.encode(prompt, joke)
        ids = encoded.ids

        # truncate
        if len(ids) > cfg.max_seq_len:
            ids = ids[: cfg.max_seq_len]

        # build loss_mask: only tokens AFTER [JOKE] up to [/S]
        loss_mask = [0] * len(ids)
        try:
            j_idx = ids.index(cfg.joke_token_id)
            for i in range(j_idx + 1, len(ids)):
                tok = ids[i]
                if tok in (cfg.pad_token_id, cfg.eos_token_id):
                    break
                loss_mask[i] = 1
        except ValueError:
            # no [JOKE] found -> this sample contributes no loss
            pass

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.float32),
        }


def collate_fn(batch):
    input_ids_list = [b["input_ids"] for b in batch]
    loss_mask_list = [b["loss_mask"] for b in batch]

    max_len = max(x.size(0) for x in input_ids_list)
    pad_id = cfg.pad_token_id

    padded_ids = []
    padded_masks = []

    for ids, m in zip(input_ids_list, loss_mask_list):
        pad_len = max_len - ids.size(0)
        if pad_len > 0:
            ids = torch.cat(
                [ids, torch.full((pad_len,), pad_id, dtype=torch.long)],
                dim=0,
            )
            m = torch.cat(
                [m, torch.zeros(pad_len, dtype=torch.float32)],
                dim=0,
            )
        padded_ids.append(ids)
        padded_masks.append(m)

    input_ids = torch.stack(padded_ids, dim=0)     # [B, L]
    loss_masks = torch.stack(padded_masks, dim=0)  # [B, L]

    # attention mask: 1 for real tokens, 0 for PAD
    attn_mask = (input_ids != pad_id).long()

    return {
        "input_ids": input_ids,
        "loss_mask": loss_masks,
        "attn_mask": attn_mask,
    }