import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm.auto import tqdm
import wandb

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


def train_one_epoch(model, dataloader, optimizer, cfg, epoch, run=None):
    """
    One training epoch with tqdm progress bar.
    Optionally logs batch & epoch loss to wandb.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch} [train]",
        leave=False
    )

    for batch in progress_bar:
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

        loss_val = loss.item()
        total_loss += loss_val
        num_batches += 1

        # Update tqdm postfix
        progress_bar.set_postfix(loss=f"{loss_val:.4f}")

        # Optional: log per-batch loss to wandb (not committing a new step every time)
        if run is not None:
            run.log({"train/batch_loss": loss_val}, commit=False)

    avg_loss = total_loss / max(num_batches, 1)

    # Log epoch-level train loss
    if run is not None:
        run.log({"train/epoch_loss": avg_loss}, step=epoch)

    return avg_loss


@torch.no_grad()
def eval_perplexity(model, dataloader, cfg, epoch, run=None):
    """
    Evaluate validation loss & perplexity with tqdm bar.
    Optionally logs to wandb.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch} [val]",
        leave=False
    )

    for batch in progress_bar:
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

        loss_val = loss.item()
        total_loss += loss_val
        num_batches += 1

        progress_bar.set_postfix(loss=f"{loss_val:.4f}")

    mean_loss = total_loss / max(num_batches, 1)
    ppl = torch.exp(torch.tensor(mean_loss)).item()

    # Log validation metrics
    if run is not None:
        run.log(
            {
                "val/loss": mean_loss,
                "val/perplexity": ppl,
            },
            step=epoch,
        )

    return mean_loss, ppl


# def main():
#     cfg = Config()
#     device = torch.device(cfg.device)
#     print(f"Using device: {device}")

#     # Build dataset paths (assumes running from project root: python train.py)
#     project_root = os.path.dirname(os.path.abspath(__file__))
#     train_csv = os.path.join(project_root, "data", "processed", "train.csv")
#     val_csv = os.path.join(project_root, "data", "processed", "val.csv")

#     train_dataset = JokeDataset(train_csv)
#     val_dataset = JokeDataset(val_csv)

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=cfg.batch_size,
#         shuffle=True,
#         collate_fn=collate_fn,
#     )
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=cfg.batch_size,
#         shuffle=False,
#         collate_fn=collate_fn,
#     )

#     model = DecoderOnlyTransformer(cfg).to(device)

#     optimizer = optim.AdamW(
#         model.parameters(),
#         lr=cfg.lr,
#         weight_decay=cfg.weight_decay,
#     )

#     # ---- wandb init ----
#     # Make sure: `pip install wandb` and `wandb login` before running.
#     try:
#         run = wandb.init(
#             project="efficient-joke-transformer",
#             dir="./wandb_logs",  # ⭐ 日志写在当前项目目录下
#             mode=os.getenv("WANDB_MODE", "online"),  # 可用环境变量控制
#             config={
#                 "architecture": "decoder-only",
#                 "d_model": cfg.d_model,
#                 "n_heads": cfg.n_heads,
#                 "d_ff": cfg.d_ff,
#                 "n_layers": cfg.n_layers,
#                 "dropout": cfg.dropout,
#                 "batch_size": cfg.batch_size,
#                 "lr": cfg.lr,
#                 "weight_decay": cfg.weight_decay,
#                 "num_epochs": cfg.num_epochs,
#                 "max_seq_len": cfg.max_seq_len,
#                 "device": str(device),
#             },
#         )
#         wandb.watch(model, log="all", log_freq=100)
#     except Exception as e:
#         print(f"[WARN] wandb init failed: {e}")
#         print("[WARN] Continuing without wandb logging.")
#         run = None

#     best_val_loss = float("inf")
#     best_val_ppl = None
#     ckpt_path = os.path.join(project_root, "best_decoder_only.pt")

#     for epoch in range(1, cfg.num_epochs + 1):
#         train_loss = train_one_epoch(model, train_loader, optimizer, cfg, epoch, run=run)
#         val_loss, val_ppl = eval_perplexity(model, val_loader, cfg, epoch, run=run)

#         print(
#             f"Epoch {epoch}: "
#             f"train_loss={train_loss:.4f}, "
#             f"val_loss={val_loss:.4f}, "
#             f"val_ppl={val_ppl:.2f}"
#         )

#         # Log summary-style epoch metrics as well
#         run.log(
#             {
#                 "epoch": epoch,
#                 "train/epoch_loss_logged": train_loss,
#                 "val/loss_logged": val_loss,
#                 "val/perplexity_logged": val_ppl,
#             },
#             step=epoch,
#         )

#         # Save best model
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_val_ppl = val_ppl
#             torch.save(model.state_dict(), ckpt_path)
#             print(f"  -> saved best model to {ckpt_path}")

#             run.summary["best_val_loss"] = best_val_loss
#             run.summary["best_val_perplexity"] = best_val_ppl

#     run.finish()

def main():
    cfg = Config()
    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    project_root = os.path.dirname(os.path.abspath(__file__))
    train_csv = os.path.join(project_root, "data", "processed", "train.csv")
    val_csv = os.path.join(project_root, "data", "processed", "val.csv")
    ckpt_path = os.path.join(project_root, "best_decoder_only.pt")

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

    # ----- build model -----
    model = DecoderOnlyTransformer(cfg).to(device)

    # ----- NEW: resume from existing checkpoint if present -----
    start_epoch = 6
    best_val_loss = float("inf")
    best_val_ppl = None

    if os.path.exists(ckpt_path):
        print(f"[INFO] Found existing checkpoint at {ckpt_path}, loading...")
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        print("[INFO] Loaded weights from best_decoder_only.pt")
        # 这里我们不知道之前训了多少 epoch，就当从 epoch 1 重新计数，
        # 但起点已经是之前的最优模型，相当于“warm restart”。

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # ---- wandb init ----
    try:
        run = wandb.init(
            project="efficient-joke-transformer",
            dir="./wandb_logs",
            mode=os.getenv("WANDB_MODE", "online"),
            config={
                "architecture": "decoder-only",
                "d_model": cfg.d_model,
                "n_heads": cfg.n_heads,
                "d_ff": cfg.d_ff,
                "n_layers": cfg.n_layers,
                "dropout": cfg.dropout,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "num_epochs": cfg.num_epochs,
                "max_seq_len": cfg.max_seq_len,
                "device": str(device),
                "resume_from_ckpt": os.path.exists(ckpt_path),
            },
        )
        wandb.watch(model, log="all", log_freq=100)
    except Exception as e:
        print(f"[WARN] wandb init failed: {e}")
        print("[WARN] Continuing without wandb logging.")
        run = None

    # 如果你想把“继续训练的 epoch 数”单独控制，比如再训 2 个：
    # extra_epochs = 2
    # epoch_range = range(start_epoch, start_epoch + extra_epochs)
    epoch_range = range(start_epoch, cfg.num_epochs + 1)

    for epoch in epoch_range:
        train_loss = train_one_epoch(model, train_loader, optimizer, cfg, epoch, run=run)
        val_loss, val_ppl = eval_perplexity(model, val_loader, cfg, epoch, run=run)

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_ppl={val_ppl:.2f}"
        )

        if run is not None:
            run.log(
                {
                    "epoch": epoch,
                    "train/epoch_loss_logged": train_loss,
                    "val/loss_logged": val_loss,
                    "val/perplexity_logged": val_ppl,
                },
                step=epoch,
            )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_ppl = val_ppl
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> saved best model to {ckpt_path}")

            if run is not None:
                run.summary["best_val_loss"] = best_val_loss
                run.summary["best_val_perplexity"] = best_val_ppl

    if run is not None:
        run.finish()

if __name__ == "__main__":
    main()