"""
Train PocketPaint's own small DDPM-style diffusion model from scratch.

Unlike the existing "Neural model" engine (a conditional neural field /
CPPN that maps pixel coords directly to RGB), this is a genuine diffusion
model: a small conditional conv UNet trained to predict the noise added to
a 32x32 image at a random timestep, then sampled at inference time via an
iterative denoising loop (DDIM, ~20 steps). Training data comes from the
analytic scene+subject renderer in scene.py (infinite, generated on the
fly — same distillation strategy used for the original neural model).
"""
import json, math, time
import numpy as np
import torch
import torch.nn.functional as F
from model import TinyUNet
import scene_diffusion as scene

torch.manual_seed(0)
rng = np.random.default_rng(0)

S = 32  # training resolution
T = 1000
BETA_START, BETA_END = 1e-4, 0.02
betas = torch.linspace(BETA_START, BETA_END, T)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_acp = torch.sqrt(alphas_cumprod)
sqrt_1m_acp = torch.sqrt(1.0 - alphas_cumprod)

device = "cpu"
model = TinyUNet(c0=28, c1=44, c2=64, emb_dim=96).to(device)
print("params:", model.param_count())
LR = 1e-3
WARMUP = 200
opt = torch.optim.Adam(model.parameters(), lr=LR)

ys, xs = np.meshgrid(np.linspace(0, 1, S), np.linspace(0, 1, S), indexing="ij")


def make_batch(n):
    c = scene.sample_conditions(rng, n)                       # [n,21]
    cc = np.broadcast_to(c[:, None, None, :], (n, S, S, scene.CDIM))
    xx = np.broadcast_to(xs, (n, S, S))
    yy = np.broadcast_to(ys, (n, S, S))
    img = scene.render(xx, yy, cc)                             # [n,S,S,3] in [0,1]
    img = img * 2 - 1                                          # -> [-1,1]
    img = np.transpose(img, (0, 3, 1, 2)).astype(np.float32)   # [n,3,S,S]
    return torch.from_numpy(img), torch.from_numpy(c.astype(np.float32))


STEPS = 4000
BATCH = 48
t0 = time.time()
ema_loss = None
for step in range(1, STEPS + 1):
    x0, cond = make_batch(BATCH)
    t_idx = torch.randint(0, T, (BATCH,))
    noise = torch.randn_like(x0)
    xt = sqrt_acp[t_idx][:, None, None, None] * x0 + sqrt_1m_acp[t_idx][:, None, None, None] * noise

    for g in opt.param_groups:
        g["lr"] = LR * min(1.0, step / WARMUP)

    opt.zero_grad()
    pred = model(xt, t_idx.float() / T, cond)
    loss = F.mse_loss(pred, noise)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    ema_loss = loss.item() if ema_loss is None else 0.98 * ema_loss + 0.02 * loss.item()
    if step % 200 == 0 or step == 1:
        print(f"step {step:5d}  loss {loss.item():.4f}  ema {ema_loss:.4f}  ({time.time()-t0:6.1f}s)")

print("trained in %.1fs" % (time.time() - t0))

# ---------------------------------------------------------------------------
# Export: weights -> flat float32 array (base64 in the web app), plus the
# noise schedule constants needed to reproduce DDIM sampling in JS.
# ---------------------------------------------------------------------------
order = []
shapes = {}
flat = []
sd = model.state_dict()
for k, v in sd.items():
    order.append(k)
    shapes[k] = list(v.shape)
    flat.append(v.detach().cpu().numpy().astype(np.float32).ravel())
flat = np.concatenate(flat)
print("total params exported:", flat.size, " raw bytes:", flat.nbytes)

import base64
b64 = base64.b64encode(flat.tobytes()).decode("ascii")

meta = {
    "S": S, "T": T, "betaStart": BETA_START, "betaEnd": BETA_END,
    "c0": 28, "c1": 44, "c2": 64, "embDim": 96, "cdim": scene.CDIM,
}
out = {"meta": meta, "order": order, "shapes": shapes, "b64": b64}
with open("diffusion_model.json", "w") as f:
    json.dump(out, f)
print("saved diffusion_model.json, b64 len:", len(b64))
