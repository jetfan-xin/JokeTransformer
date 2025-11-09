# models/decoder_only.py

import torch
import torch.nn as nn
import math
from utils.config import Config

cfg = Config()


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # x: [B, L, d_model]
        B, L, _ = x.size()

        q = self.W_q(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # [B,h,L,d]
        k = self.W_k(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B,h,L,L]

        # causal mask (LxL, upper triangle = -inf)
        causal_mask = torch.triu(
            torch.full((L, L), float("-inf"), device=x.device),
            diagonal=1,
        )
        scores = scores + causal_mask  # broadcast over B,h

        # attn_mask: [B,L], 1 for real, 0 for pad
        if attn_mask is not None:
            # expand to [B,1,1,L]
            pad_mask = (attn_mask == 0).unsqueeze(1).unsqueeze(2)  # True where pad
            scores = scores.masked_fill(pad_mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B,h,L,d_head]
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)  # [B,L,d_model]
        out = self.W_o(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, attn_mask=None):
        # Pre-LN + residual
        h = self.ln1(x)
        h = self.mha(h, attn_mask=attn_mask)
        x = x + h

        h2 = self.ln2(x)
        h2 = self.ffn(h2)
        x = x + h2
        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        self.blocks = nn.ModuleList([
            DecoderBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # tie weights
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, input_ids, attn_mask=None):
        # input_ids: [B, L]
        B, L = input_ids.size()
        device = input_ids.device

        pos = torch.arange(0, L, device=device).unsqueeze(0)  # [1, L]
        x = self.tok_emb(input_ids) + self.pos_emb(pos)       # [B, L, d_model]

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, L, vocab_size]
        return logits