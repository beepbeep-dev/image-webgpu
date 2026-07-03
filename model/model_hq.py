import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
    args = t[:, None].float() * freqs[None, :] * 1000.0
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FiLMResBlock(nn.Module):
    def __init__(self, cin, cout, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.film = nn.Linear(emb_dim, 2 * cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = F.silu(self.conv1(x))
        scale, shift = self.film(emb).chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """Plain multi-head self-attention over flattened spatial positions, with
    a GroupNorm + residual (standard ADM/DDPM-style attention block). Pure
    conv UNets only ever mix information within a 3x3 neighborhood per layer,
    so two distant regions of the image can only "see" each other after many
    stacked convs — attention lets every position attend to every other
    position in a single step, which is the standard fix for conv-only UNets
    producing locally-plausible but globally-incoherent output (color blobs
    with no real object structure). Only applied at the two lowest, cheapest
    resolutions (16x16 and 8x8) since cost grows with (H*W)^2."""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv.unbind(1)
        scale = (C // self.num_heads) ** -0.5
        attn = torch.einsum("bhdn,bhdm->bhnm", q * scale, k).softmax(dim=-1)
        out = torch.einsum("bhnm,bhdm->bhdn", attn, v).reshape(B, C, H, W)
        return x + self.proj(out)


class BigUNet(nn.Module):
    """256x256 -> 128 -> 64 -> 32 -> 16 -> 8 (mid) -> back up, FiLM-conditioned."""
    def __init__(self, channels=(20, 32, 48, 72, 104, 144), emb_dim=160, cdim=64):
        super().__init__()
        c0, c1, c2, c3, c4, c5 = channels
        self.emb_dim = emb_dim
        self.t_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.c_mlp = nn.Sequential(nn.Linear(cdim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))

        self.stem = nn.Conv2d(3, c0, 3, padding=1)
        self.down0 = FiLMResBlock(c0, c0, emb_dim)          # 256
        self.pool0 = nn.Conv2d(c0, c1, 3, stride=2, padding=1)   # ->128
        self.down1 = FiLMResBlock(c1, c1, emb_dim)          # 128
        self.pool1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)   # ->64
        self.down2 = FiLMResBlock(c2, c2, emb_dim)          # 64
        self.pool2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)   # ->32
        self.down3 = FiLMResBlock(c3, c3, emb_dim)          # 32
        self.pool3 = nn.Conv2d(c3, c4, 3, stride=2, padding=1)   # ->16
        self.down4 = FiLMResBlock(c4, c4, emb_dim)          # 16
        self.attn4 = SelfAttention2d(c4)
        self.pool4 = nn.Conv2d(c4, c5, 3, stride=2, padding=1)   # ->8

        self.mid1 = FiLMResBlock(c5, c5, emb_dim)
        self.mid_attn = SelfAttention2d(c5)
        self.mid2 = FiLMResBlock(c5, c5, emb_dim)

        self.up4_conv = nn.Conv2d(c5 + c4, c4, 3, padding=1)
        self.up4 = FiLMResBlock(c4, c4, emb_dim)
        self.up4_attn = SelfAttention2d(c4)
        self.up3_conv = nn.Conv2d(c4 + c3, c3, 3, padding=1)
        self.up3 = FiLMResBlock(c3, c3, emb_dim)
        self.up2_conv = nn.Conv2d(c3 + c2, c2, 3, padding=1)
        self.up2 = FiLMResBlock(c2, c2, emb_dim)
        self.up1_conv = nn.Conv2d(c2 + c1, c1, 3, padding=1)
        self.up1 = FiLMResBlock(c1, c1, emb_dim)
        self.up0_conv = nn.Conv2d(c1 + c0, c0, 3, padding=1)
        self.up0 = FiLMResBlock(c0, c0, emb_dim)

        self.head = nn.Conv2d(c0, 3, 3, padding=1)

    def forward(self, x, t01, cond):
        temb = self.t_mlp(sinusoidal_embedding(t01, self.emb_dim))
        cemb = self.c_mlp(cond)
        emb = temb + cemb

        h0 = F.silu(self.stem(x)); h0 = self.down0(h0, emb)           # 256
        h1 = F.silu(self.pool0(h0)); h1 = self.down1(h1, emb)         # 128
        h2 = F.silu(self.pool1(h1)); h2 = self.down2(h2, emb)         # 64
        h3 = F.silu(self.pool2(h2)); h3 = self.down3(h3, emb)         # 32
        h4 = F.silu(self.pool3(h3)); h4 = self.down4(h4, emb); h4 = self.attn4(h4)  # 16
        h5 = F.silu(self.pool4(h4))                                   # 8

        h5 = self.mid1(h5, emb); h5 = self.mid_attn(h5); h5 = self.mid2(h5, emb)

        u4 = F.interpolate(h5, scale_factor=2, mode="nearest")
        u4 = F.silu(self.up4_conv(torch.cat([u4, h4], dim=1))); u4 = self.up4(u4, emb); u4 = self.up4_attn(u4)
        u3 = F.interpolate(u4, scale_factor=2, mode="nearest")
        u3 = F.silu(self.up3_conv(torch.cat([u3, h3], dim=1))); u3 = self.up3(u3, emb)
        u2 = F.interpolate(u3, scale_factor=2, mode="nearest")
        u2 = F.silu(self.up2_conv(torch.cat([u2, h2], dim=1))); u2 = self.up2(u2, emb)
        u1 = F.interpolate(u2, scale_factor=2, mode="nearest")
        u1 = F.silu(self.up1_conv(torch.cat([u1, h1], dim=1))); u1 = self.up1(u1, emb)
        u0 = F.interpolate(u1, scale_factor=2, mode="nearest")
        u0 = F.silu(self.up0_conv(torch.cat([u0, h0], dim=1))); u0 = self.up0(u0, emb)

        return self.head(u0)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


class LatentUNet(nn.Module):
    """UNet for LATENT-space diffusion (the recipe that makes small-budget
    diffusion actually work): instead of denoising 256x256x3 pixels directly,
    it denoises the 32x32x4 latent produced by a small pretrained
    autoencoder (TAESD, MIT-licensed) — 48x fewer values, so the same
    training budget buys ~50x more optimization steps, and the decoder
    handles fine texture "for free". 3 levels (32 -> 16 -> 8), with
    self-attention at 16x16 and 8x8, FiLM-conditioned like BigUNet."""
    def __init__(self, channels=(192, 288, 384), emb_dim=256, cdim=64, in_ch=4):
        super().__init__()
        c0, c1, c2 = channels
        self.emb_dim = emb_dim
        self.t_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.c_mlp = nn.Sequential(nn.Linear(cdim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))

        self.stem = nn.Conv2d(in_ch, c0, 3, padding=1)
        self.down0 = FiLMResBlock(c0, c0, emb_dim)               # 32
        self.pool0 = nn.Conv2d(c0, c1, 3, stride=2, padding=1)   # ->16
        self.down1 = FiLMResBlock(c1, c1, emb_dim)               # 16
        self.attn1 = SelfAttention2d(c1)
        self.pool1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)   # ->8

        self.mid1 = FiLMResBlock(c2, c2, emb_dim)
        self.mid_attn = SelfAttention2d(c2)
        self.mid2 = FiLMResBlock(c2, c2, emb_dim)

        self.up1_conv = nn.Conv2d(c2 + c1, c1, 3, padding=1)
        self.up1 = FiLMResBlock(c1, c1, emb_dim)
        self.up1_attn = SelfAttention2d(c1)
        self.up0_conv = nn.Conv2d(c1 + c0, c0, 3, padding=1)
        self.up0 = FiLMResBlock(c0, c0, emb_dim)

        self.head = nn.Conv2d(c0, in_ch, 3, padding=1)

    def forward(self, x, t01, cond):
        temb = self.t_mlp(sinusoidal_embedding(t01, self.emb_dim))
        cemb = self.c_mlp(cond)
        emb = temb + cemb

        h0 = F.silu(self.stem(x)); h0 = self.down0(h0, emb)                     # 32
        h1 = F.silu(self.pool0(h0)); h1 = self.down1(h1, emb); h1 = self.attn1(h1)  # 16
        h2 = F.silu(self.pool1(h1))                                             # 8

        h2 = self.mid1(h2, emb); h2 = self.mid_attn(h2); h2 = self.mid2(h2, emb)

        u1 = F.interpolate(h2, scale_factor=2, mode="nearest")
        u1 = F.silu(self.up1_conv(torch.cat([u1, h1], dim=1))); u1 = self.up1(u1, emb); u1 = self.up1_attn(u1)
        u0 = F.interpolate(u1, scale_factor=2, mode="nearest")
        u0 = F.silu(self.up0_conv(torch.cat([u0, h0], dim=1))); u0 = self.up0(u0, emb)

        return self.head(u0)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


class CaptionEncoderV2(nn.Module):
    """Order-aware from-scratch text encoder (upgrade over the mean-pooled
    bag-of-words CaptionEncoder, which couldn't tell "dog chases cat" from
    "cat chases dog"): learned word embeddings + learned positional
    embeddings, one masked single-head self-attention layer, a small
    feed-forward layer, then masked mean-pooling. Still tiny (~150k params)
    and trained jointly with the UNet on the diffusion loss — small enough
    to port to plain JS loops exactly."""
    def __init__(self, vocab_size, dim=64, maxlen=14):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.randn(maxlen, dim) * 0.02)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.attn_out = nn.Linear(dim, dim)
        self.ff1 = nn.Linear(dim, 2 * dim)
        self.ff2 = nn.Linear(2 * dim, dim)
        self.dim = dim

    def forward(self, ids, mask):
        e = self.embed(ids) + self.pos[None, :, :]            # [B,L,D]
        q, k, v = self.q(e), self.k(e), self.v(e)
        scores = torch.einsum("bld,bmd->blm", q, k) / (self.dim ** 0.5)
        scores = scores.masked_fill(mask[:, None, :] == 0, -1e9)
        att = torch.einsum("blm,bmd->bld", scores.softmax(-1), v)
        e = e + self.attn_out(att)
        e = e + self.ff2(F.silu(self.ff1(e)))
        summed = (e * mask[:, :, None]).sum(1)
        count = mask.sum(1, keepdim=True).clamp(min=1.0)
        return summed / count
