import math
import torch
import torch.nn as nn
import torch.nn.functional as F

CDIM = 21


def sinusoidal_embedding(t, dim):
    # t: [B] float in [0,1] (we'll pass timestep/T)
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


class TinyUNet(nn.Module):
    """Small conditional noise-prediction UNet for 32x32 RGB.
    32 -> 16 -> 8 -> 16 -> 32, FiLM-conditioned on (timestep, semantic condition).
    """
    def __init__(self, c0=40, c1=64, c2=96, emb_dim=128):
        super().__init__()
        self.emb_dim = emb_dim
        self.t_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.c_mlp = nn.Sequential(nn.Linear(CDIM, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))

        self.stem = nn.Conv2d(3, c0, 3, padding=1)
        self.down1 = FiLMResBlock(c0, c0, emb_dim)
        self.pool1 = nn.Conv2d(c0, c1, 3, stride=2, padding=1)   # 32->16
        self.down2 = FiLMResBlock(c1, c1, emb_dim)
        self.pool2 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)   # 16->8

        self.mid1 = FiLMResBlock(c2, c2, emb_dim)
        self.mid2 = FiLMResBlock(c2, c2, emb_dim)

        self.up2_conv = nn.Conv2d(c2 + c1, c1, 3, padding=1)     # after upsample+concat skip(c1)
        self.up2 = FiLMResBlock(c1, c1, emb_dim)
        self.up1_conv = nn.Conv2d(c1 + c0, c0, 3, padding=1)     # after upsample+concat skip(c0)
        self.up1 = FiLMResBlock(c0, c0, emb_dim)

        self.head = nn.Conv2d(c0, 3, 3, padding=1)

    def forward(self, x, t01, cond):
        temb = self.t_mlp(sinusoidal_embedding(t01, self.emb_dim))
        cemb = self.c_mlp(cond)
        emb = temb + cemb

        h0 = F.silu(self.stem(x))            # [B,c0,32,32]
        h0 = self.down1(h0, emb)
        h1 = F.silu(self.pool1(h0))          # [B,c1,16,16]
        h1 = self.down2(h1, emb)
        h2 = F.silu(self.pool2(h1))          # [B,c2,8,8]

        h2 = self.mid1(h2, emb)
        h2 = self.mid2(h2, emb)

        u1 = F.interpolate(h2, scale_factor=2, mode="nearest")   # 8->16
        u1 = F.silu(self.up2_conv(torch.cat([u1, h1], dim=1)))
        u1 = self.up2(u1, emb)

        u0 = F.interpolate(u1, scale_factor=2, mode="nearest")   # 16->32
        u0 = F.silu(self.up1_conv(torch.cat([u0, h0], dim=1)))
        u0 = self.up1(u0, emb)

        return self.head(u0)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())
