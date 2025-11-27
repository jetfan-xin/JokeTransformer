
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, emb_dim: int, context_size: int, head_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(emb_dim, head_size, bias=False)
        self.query = nn.Linear(emb_dim, head_size, bias=False)
        self.value = nn.Linear(emb_dim, head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(context_size, context_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        
        B, T, E = x.shape
        k = self.key(x)      
        q = self.query(x)    

        weights = q @ k.transpose(-2, -1) * (E ** -0.5)  # (B, T, T)

        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        if attn_mask is not None:
            pad_mask = attn_mask[:, None, :].to(weights.device) 
            weights = weights.masked_fill(~pad_mask, float("-inf"))

        weights = F.softmax(weights, dim=-1)  
        weights = self.dropout(weights)

        v = self.value(x)  
        out = weights @ v  
        return out


class MultiHeadAttention(nn.Module):
    

    def __init__(
        self,
        num_heads: int,
        emb_dim: int,
        context_size: int,
        head_size: int,
        dropout: float,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(emb_dim, context_size, head_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(num_heads * head_size, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # concat heads along embedding dim
        out = torch.cat([h(x, attn_mask=attn_mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
 
    def __init__(self, emb_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.ReLU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    

    def __init__(self, emb_dim: int, context_size: int, num_heads: int, dropout: float):
        super().__init__()
        head_size = emb_dim // num_heads
        self.sa = MultiHeadAttention(num_heads, emb_dim, context_size, head_size, dropout)
        self.ffwd = FeedForward(emb_dim, dropout)
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.sa(self.ln1(x), attn_mask=attn_mask)
        x = x + self.ffwd(self.ln2(x))
        return x


class TransformerDecoder(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        context_size: int,
        num_att_heads: int,
        dropout: float,
        pad_token_id: int = -100,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.context_size = context_size
        self.pad_token_id = pad_token_id

        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(context_size, emb_dim)

        self.blocks = nn.ModuleList(
            [
                Block(emb_dim, context_size, num_att_heads, dropout)
                for _ in range(6)
            ]
        )
        self.ln_f = nn.LayerNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ):
        B, T = idx.shape

        token_emb = self.token_embedding(idx)  
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding(pos)  
        x = token_emb + pos_emb  

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)  

        loss = None
        if targets is not None:
            logits_flat = logits.view(B * T, self.vocab_size)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(
                logits_flat,
                targets_flat,
                ignore_index=self.pad_token_id,
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 256) -> torch.Tensor:
        """
        Autoregressively generate tokens, starting from idx.

        idx: (B, T_start)
        returns: (B, T_start + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_size :]  
            logits, _ = self(idx_cond)               
            logits_last = logits[:, -1, :]          
            probs = F.softmax(logits_last, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
