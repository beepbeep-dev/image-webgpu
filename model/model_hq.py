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
        self.pool4 = nn.Conv2d(c4, c5, 3, stride=2, padding=1)   # ->8

        self.mid1 = FiLMResBlock(c5, c5, emb_dim)
        self.mid2 = FiLMResBlock(c5, c5, emb_dim)

        self.up4_conv = nn.Conv2d(c5 + c4, c4, 3, padding=1)
        self.up4 = FiLMResBlock(c4, c4, emb_dim)
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
        h4 = F.silu(self.pool3(h3)); h4 = self.down4(h4, emb)         # 16
        h5 = F.silu(self.pool4(h4))                                   # 8

        h5 = self.mid1(h5, emb); h5 = self.mid2(h5, emb)

        u4 = F.interpolate(h5, scale_factor=2, mode="nearest")
        u4 = F.silu(self.up4_conv(torch.cat([u4, h4], dim=1))); u4 = self.up4(u4, emb)
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
