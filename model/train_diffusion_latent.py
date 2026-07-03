"""
Train PocketPaint's LATENT-space "Diffusion HQ" model — the recipe that makes
small-budget diffusion actually work (and the same idea Stable Diffusion is
built on): a small pretrained autoencoder (TAESD, MIT license) compresses each
256x256x3 photo into a 32x32x4 latent, and OUR from-scratch model (LatentUNet,
model_hq.py) learns DDPM/DDIM denoising in that 48x-smaller space. Each
training step is ~50x cheaper than the old pixel-space 256x256 model, so the
same GPU budget buys ~50x more optimization — and the pretrained decoder
reconstructs crisp 256x256 texture from whatever latent we generate.

Inputs (fetched from the HF dataset repo by the vast.ai onstart script):
  latents_hq.npz            precomputed TAESD latents + captions (fp16)
  taesd_decoder.safetensors pretrained decoder weights, bundled into the
                            exported JSON so the browser can decode latents
Everything else (vocab, caption encoder, CFG, EMA, AdamW, export format)
matches train_diffusion_hq.py.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json, re, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_hq import LatentUNet

torch.manual_seed(0)
rng = np.random.default_rng(0)

LATENT_SIZE = 32
LATENT_CH = 4
T = 1000
BETA_START, BETA_END = 1e-4, 0.02
betas = torch.linspace(BETA_START, BETA_END, T)
alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
sqrt_acp = torch.sqrt(alphas_cumprod)
sqrt_1m_acp = torch.sqrt(1.0 - alphas_cumprod)

data = np.load("latents_hq.npz", allow_pickle=True)
LATENTS = data["latents"].astype(np.float32)      # [N,4,32,32]
captions = list(data["captions"])
N = len(captions)
LSCALE = 1.0 / float(LATENTS.std())               # normalize to unit variance for diffusion
LATENTS *= LSCALE
print(f"latents: {LATENTS.shape}  raw std -> unit (LSCALE={LSCALE:.4f})")

# Vocabulary from the real captions (same tokenizer rule as the JS side).
STOPWORDS = set("""a an the of in on at to from with and or for is are was
were be been being by as it its this that these those near over under
photo photograph picture image view taken file""".split())
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return [w for w in TOKEN_RE.findall(text.lower()) if w not in STOPWORDS and len(w) > 1]


token_lists = [tokenize(c) for c in captions]
freq = {}
for toks in token_lists:
    for w in set(toks):
        freq[w] = freq.get(w, 0) + 1
VOCAB_SIZE = 2000
vocab = [w for w, _ in sorted(freq.items(), key=lambda kv: -kv[1])[:VOCAB_SIZE]]
word2id = {w: i for i, w in enumerate(vocab)}
print("vocab size:", len(vocab), " e.g.:", vocab[:20])

MAXLEN = 14
EMBED_DIM = 64
ids_arr = np.zeros((N, MAXLEN), dtype=np.int64)
mask_arr = np.zeros((N, MAXLEN), dtype=np.float32)
for i, toks in enumerate(token_lists):
    known = [word2id[w] for w in toks if w in word2id][:MAXLEN]
    for j, tid in enumerate(known):
        ids_arr[i, j] = tid
        mask_arr[i, j] = 1.0


class CaptionEncoder(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)

    def forward(self, ids, mask):
        e = self.embed(ids)
        summed = (e * mask[:, :, None]).sum(1)
        count = mask.sum(1, keepdim=True).clamp(min=1.0)
        return summed / count


device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = device == "cuda"
print("device:", device, " mixed precision:", use_amp)

CHANNELS = (192, 288, 384)
EMB_DIM = 256
model = LatentUNet(channels=CHANNELS, emb_dim=EMB_DIM, cdim=EMBED_DIM).to(device)
cap_enc = CaptionEncoder(len(vocab), EMBED_DIM).to(device)
print("LatentUNet params:", f"{model.param_count():,}", " caption encoder params:",
      f"{sum(p.numel() for p in cap_enc.parameters()):,}")

LR = 2e-4
WARMUP = 1000
UNCOND_P = 0.15
GUIDANCE_SCALE = 3.0
EMA_DECAY = 0.9995
WEIGHT_DECAY = 0.01
opt = torch.optim.AdamW(list(model.parameters()) + list(cap_enc.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
print(f"optimizer: AdamW (lr={LR}, weight_decay={WEIGHT_DECAY})")

all_params = list(model.named_parameters()) + [("word_emb." + k, v) for k, v in cap_enc.embed.named_parameters()]
ema = {name: p.detach().clone() for name, p in all_params}

STEPS = 100000   # latent steps are ~50x cheaper than the pixel-space model's
BATCH = 128 if device == "cuda" else 16

LAT_T = torch.from_numpy(LATENTS)
IDS_T = torch.from_numpy(ids_arr)
MASK_T = torch.from_numpy(mask_arr)
if device == "cuda":
    LAT_T = LAT_T.to(device)      # ~125MB, trivially fits in VRAM
    IDS_T = IDS_T.to(device)
    MASK_T = MASK_T.to(device)

sqrt_acp_d = sqrt_acp.to(device)
sqrt_1m_acp_d = sqrt_1m_acp.to(device)

import base64


def export_model():
    order, shapes, flat = [], {}, []
    for k, v in ema.items():
        order.append(k)
        shapes[k] = list(v.shape)
        flat.append(v.detach().cpu().numpy().astype(np.float32).ravel())
    # Bundle the pretrained TAESD decoder so the browser has everything in
    # one file (prefix "taesd_dec." keeps it clearly separated from OUR
    # trained weights; MIT-licensed, credited in the README).
    from safetensors.torch import load_file
    dec = load_file("taesd_decoder.safetensors")
    for k, v in dec.items():
        key = "taesd_dec." + k
        order.append(key)
        shapes[key] = list(v.shape)
        flat.append(v.numpy().astype(np.float32).ravel())
    flat = np.concatenate(flat)
    b64 = base64.b64encode(flat.tobytes()).decode("ascii")
    meta = {
        "kind": "latent", "S": 256, "latentSize": LATENT_SIZE, "latentCh": LATENT_CH,
        "lscale": LSCALE, "T": T, "betaStart": BETA_START, "betaEnd": BETA_END,
        "channels": list(CHANNELS), "embDim": EMB_DIM, "cdim": EMBED_DIM,
        "vocab": vocab, "guidanceScale": GUIDANCE_SCALE,
    }
    out = {"meta": meta, "order": order, "shapes": shapes, "b64": b64}
    tmp = "diffusion_hq_model.json.tmp"
    with open(tmp, "w") as f:
        json.dump(out, f)
    os.replace(tmp, "diffusion_hq_model.json")


t0 = time.time()
ema_loss = None
for step in range(1, STEPS + 1):
    idx = torch.from_numpy(rng.integers(0, N, BATCH)).to(device)
    x0 = LAT_T[idx]
    # Horizontal-flip augmentation directly on latents: TAESD latents are
    # spatial feature maps, so flipping W approximates encoding the flipped
    # photo — cheap and effective at this dataset size.
    flip = torch.rand(BATCH, device=device) < 0.5
    x0 = torch.where(flip[:, None, None, None], x0.flip(-1), x0)
    ids_b = IDS_T[idx]
    mask_b = MASK_T[idx]

    t_idx = torch.randint(0, T, (BATCH,), device=device)
    noise = torch.randn_like(x0)
    xt = sqrt_acp_d[t_idx][:, None, None, None] * x0 + sqrt_1m_acp_d[t_idx][:, None, None, None] * noise

    for g in opt.param_groups:
        g["lr"] = LR * min(1.0, step / WARMUP)

    opt.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
        cond = cap_enc(ids_b, mask_b)
        uncond_mask = (torch.rand(BATCH, 1, device=device) < UNCOND_P).float()
        cond = cond * (1.0 - uncond_mask)
        pred = model(xt, t_idx.float() / T, cond)
        loss = F.mse_loss(pred, noise)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(cap_enc.parameters()), 1.0)
    opt.step()

    with torch.no_grad():
        for name, p in all_params:
            ema[name].mul_(EMA_DECAY).add_(p.detach(), alpha=1.0 - EMA_DECAY)

    ema_loss = loss.item() if ema_loss is None else 0.98 * ema_loss + 0.02 * loss.item()
    if step % 200 == 0 or step == 1:
        print(f"step {step:6d}  loss {loss.item():.4f}  ema {ema_loss:.4f}  ({time.time()-t0:8.1f}s)")
    if step % 10000 == 0:
        export_model()
        print(f"checkpoint saved at step {step}")

print("trained in %.1fs" % (time.time() - t0))
export_model()
print("saved diffusion_hq_model.json")
