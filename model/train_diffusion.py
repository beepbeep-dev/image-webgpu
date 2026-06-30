"""
Train PocketPaint's own small DDPM-style diffusion model from scratch.

Unlike the existing "Neural model" engine (a conditional neural field /
CPPN that maps pixel coords directly to RGB), this is a genuine diffusion
model: a small conditional conv UNet trained to predict the noise added to
a 32x32 image at a random timestep, then sampled at inference time via an
iterative denoising loop (DDIM, ~20 steps). Training data is drawn from
dataset.db (see build_dataset.py): a real SQLite database that exhaustively
covers the structural condition space (every time-of-day x biome x
mountain/neon/fire intensity x subject x placement combination, 13,500 rows)
rendered with the analytic scene+subject renderer in scene_diffusion.py —
a fixed, inspectable training set rather than only-ever-random sampling.
"""
import io, json, math, sqlite3, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
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

# ---------------------------------------------------------------------------
# Load the full training database into memory once (13,500 rows x 32x32x3 is
# only ~170MB as float32 — far cheaper than re-rendering or re-decoding PNGs
# every step).
# ---------------------------------------------------------------------------
con = sqlite3.connect("dataset.db")
db_rows = con.execute("SELECT condition, image FROM samples").fetchall()
con.close()
N = len(db_rows)
print("loaded", N, "rows from dataset.db")
DB_COND = np.zeros((N, scene.CDIM), dtype=np.float32)
DB_IMG = np.zeros((N, 3, S, S), dtype=np.float32)
for i, (cond_blob, png_blob) in enumerate(db_rows):
    DB_COND[i] = np.frombuffer(cond_blob, dtype=np.float32)
    img = np.asarray(Image.open(io.BytesIO(png_blob)).convert("RGB"), dtype=np.float32) / 255.0
    DB_IMG[i] = np.transpose(img * 2 - 1, (2, 0, 1))   # [3,S,S] in [-1,1]
del db_rows


def make_batch(n):
    idx = rng.integers(0, N, n)
    return torch.from_numpy(DB_IMG[idx]), torch.from_numpy(DB_COND[idx])


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
