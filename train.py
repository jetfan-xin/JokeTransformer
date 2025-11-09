import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from utils.config import Config
from utils.dataset import JokeDataset, collate_fn
from models.decoder_only import DecoderOnlyTransformer


def compute_loss(logits, input_ids, loss_mask, pad_token_id):
    """
    Compute autoregressive next-token prediction loss.

    Args:
        logits:    [B, L, V] model outputs.
        input_ids: [B, L] token ids.
        loss_mask:[B, L], 1 for joke tokens to train on, 0 otherwise.
        pad_token_id: id of the padding token.

    We only compute loss on joke tokens (after [JOKE]) and non-PAD positions.
    """
    B, L, V = logits.size()

    # Shift for next-token prediction
    logits = logits[:, :-1, :].contiguous()      # [B, L-1, V]
    targets = input_ids[:, 1:].contiguous()      # [B, L-1]
    mask = loss_mask[:, 1:].contiguous()         # [B, L-1]

    # Positions to ignore: non-joke regions or PAD tokens
    ignore_mask = (mask == 0) | (targets == pad_token_id)

    # Flatten
    logits = logits.view(-1, V)                  # [(B*(L-1)), V]
    targets = targets.view(-1)                   # [(B*(L-1))]
    ignore_mask = ignore_mask.view(-1)           # [(B*(L-1))]

    # Mark ignored positions with -100 to work with ignore_index
    targets[ignore_mask] = -100

    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(logits, targets)             # averaged over valid tokens
    return loss


def train_one_epoch(model, dataloader, optimizer, cfg):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(cfg.device)
        loss_mask = batch["loss_mask"].to(cfg.device)
        attn_mask = batch["attn_mask"].to(cfg.device)

        optimizer.zero_grad()
        logits = model(input_ids, attn_mask=attn_mask)

        loss = compute_loss(
            logits,
            input_ids,
            loss_mask,
            cfg.pad_token_id,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def eval_perplexity(model, dataloader, cfg):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(cfg.device)
        loss_mask = batch["loss_mask"].to(cfg.device)
        attn_mask = batch["attn_mask"].to(cfg.device)

        logits = model(input_ids, attn_mask=attn_mask)
        loss = compute_loss(
            logits,
            input_ids,
            loss_mask,
            cfg.pad_token_id,
        )

        total_loss += loss.item()
        num_batches += 1

    mean_loss = total_loss / max(num_batches, 1)
    ppl = torch.exp(torch.tensor(mean_loss))
    return mean_loss, ppl.item()


def main():
    cfg = Config()
    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    # Build dataset paths (assumes running from project root: python train.py)
    project_root = os.path.dirname(os.path.abspath(__file__))
    train_csv = os.path.join(project_root, "data", "processed", "train.csv")
    val_csv = os.path.join(project_root, "data", "processed", "val.csv")

    train_dataset = JokeDataset(train_csv)
    val_dataset = JokeDataset(val_csv)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = DecoderOnlyTransformer(cfg).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    best_val_loss = float("inf")

    for epoch in range(1, cfg.num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, cfg)
        val_loss, val_ppl = eval_perplexity(model, val_loader, cfg)

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_ppl={val_ppl:.2f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(project_root, "best_decoder_only.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> saved best model to {ckpt_path}")


if __name__ == "__main__":
    main()