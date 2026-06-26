"""Minimal transformer trunk for the hybrid-ARC pipeline. The flat baseline, the latent-conditioned
filler, and (indirectly) the diffusion guidance all share this same bidirectional Trunk. The toy-era
nets (FSQ / Encoder / VAE prior / AR decoder / LatentDDPM) are intentionally dropped -- ARC uses only
Block + Trunk."""
import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, d, n_head, p=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_head, dropout=p, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(p))

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, key_padding_mask=key_padding_mask,
                         need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class Trunk(nn.Module):
    def __init__(self, d, n_head, n_layer, max_len, p=0.0):
        super().__init__()
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, n_head, p) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d)

    def forward(self, x, causal=False, key_padding_mask=None, attn_mask=None):
        T = x.shape[1]
        x = x + self.pos.weight[:T][None]
        m = attn_mask                                   # custom mask (e.g. prefix-bidir + suffix-causal) wins
        if m is None and causal:
            m = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        for b in self.blocks:
            x = b(x, attn_mask=m, key_padding_mask=key_padding_mask)
        return self.ln_f(x)
