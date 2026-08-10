"""
The 100%-from-scratch Diffusion HQ pipeline, in one run:

  STAGE 1  train OUR OWN autoencoder (own_ae.py) on our scraped photos —
           replaces the pretrained TAESD so every learned component is ours.
           Loss = L1 + MSE + an edge-gradient term (keeps reconstructions
           sharp without using anyone's pretrained perceptual network).
  STAGE 2  encode the whole dataset to 32x32x4 latents with OUR encoder.
  STAGE 3  train the LatentUNet + the order-aware CaptionEncoderV2
           (learned word + positional embeddings, one masked self-attention
           layer) jointly on the diffusion objective, exactly like
           train_diffusion_latent.py otherwise.

Export bundles OUR decoder under the same "taesd_dec." key prefix the
browser already runs (identical layer topology), with meta.decoder="own"
and meta.textEncoder="attn" so the JS side picks the new caption encoder.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import io, json, re, sqlite3, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from model_hq import LatentUNet, CaptionEncoderV2
from own_ae import OwnEncoder, OwnDecoder

torch.manual_seed(0)
rng = np.random.default_rng(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = device == "cuda"
print("device:", device, " mixed precision:", use_amp)

S = 256
LATENT_SIZE = 32
LATENT_CH = 4

# ---------------------------------------------------------------------------
# Load the dataset as COMPRESSED JPEG bytes and decode lazily.
#
# The old version decoded everything up front into one uint8 array, which is
# 256*256*3 = 192 KB of RAM per image. That was fine at 7.6k photos (1.5 GB)
# but the dataset is now ~200k, which would be ~40 GB and OOM any box we
# rent. Holding the JPEGs compressed is ~21 KB each (~4 GB at 200k) and
# batches are decoded on demand in a thread pool -- Pillow drops the GIL
# during decode, so threads genuinely overlap, and the AE batch (32 images)
# decodes well inside a GPU step.
# ---------------------------------------------------------------------------
DB_PATHS = [p for p in os.environ.get("DATASETS", "dataset_hq.db").split(",") if p.strip()]
BLOBS, captions, queries = [], [], []
for dbp in DB_PATHS:
    if not os.path.exists(dbp):
        print("  skip missing dataset:", dbp)
        continue
    con = sqlite3.connect(dbp)
    n0 = len(BLOBS)
    for q, cap, blob in con.execute("SELECT query, caption, image FROM samples"):
        BLOBS.append(blob)
        captions.append(cap or "")
        queries.append(q or "")
    con.close()
    print(f"  {dbp}: +{len(BLOBS)-n0} rows")
N = len(BLOBS)
print("dataset:", N, "photos  (~%.1f GB compressed in RAM)" % (sum(map(len, BLOBS)) / 1e9))

_decode_pool = ThreadPoolExecutor(max_workers=min(16, (os.cpu_count() or 4) * 2))


def _decode_one(i):
    im = Image.open(io.BytesIO(BLOBS[i]))
    im.draft("RGB", (S, S))
    im = im.convert("RGB")
    if im.size != (S, S):
        im = im.resize((S, S), Image.BILINEAR)
    return np.transpose(np.asarray(im, dtype=np.uint8), (2, 0, 1))


def decode_many(idx):
    return np.stack(list(_decode_pool.map(_decode_one, idx)))

# Oversample weight per row: aligned faces/bodies are a small slice of the
# dataset but are the only rows with consistent scale/position (see
# align_faces.py / align_people.py) — oversample them heavily so the model
# gets enough gradient signal to learn stable structure instead of
# averaging it away. A handful of scene topics that already render well
# (beach/desert/mountain/city) get a lighter boost just to make them more
# reliable, not to fix a detection gap.
TOPIC_PATTERNS = {
    "beach": (re.compile(r"\bbeach\b", re.I), 3),
    "desert": (re.compile(r"\bdesert\b|\bdune", re.I), 3),
    "mountain": (re.compile(r"\bmountain\b|\bpeak\b", re.I), 3),
    "city": (re.compile(r"\bcity\b|\bskyline\b", re.I), 3),
}
weight = np.ones(N, dtype=np.float64)
for i, (q, cap) in enumerate(zip(queries, captions)):
    if q.startswith("portrait_aligned:"):
        weight[i] = 16
    elif q.startswith("person_aligned:"):
        weight[i] = 10
    else:
        for name, (pat, w) in TOPIC_PATTERNS.items():
            if pat.search(cap or ""):
                weight[i] = max(weight[i], w)

# The two stages want different data, so they get different pools.
#
# The autoencoder is a compressor: it never sees a caption, and feeding it
# the widest variety of real photographs makes it generalise better. The
# diffusion model learns *what to generate*, so its training set decides the
# aesthetic of every image the app produces.
#
# The bulk CC12M rows are genuine photographs but visibly more mundane than
# the curated Pexels set (product shots, snapshots, posters), and they carry
# no captions, so in stage 3 they would only ever act as unconditional
# samples -- spending model capacity to make output *less* beautiful. So:
# everything trains the AE, only captioned photos train the diffusion model.
is_bulk = np.array([q.startswith("bulk:") for q in queries])
has_caption = np.array([bool((c or "").strip()) for c in captions])

AE_POOL = np.repeat(np.arange(N), weight.astype(np.int64))
diff_w = weight.copy()
diff_w[is_bulk | ~has_caption] = 0
if diff_w.sum() == 0:                      # no captioned rows at all -> fall back
    diff_w = weight.copy()

# Pin the aligned face/body rows to a TARGET SHARE of the diffusion pool
# rather than a fixed multiplier.
#
# A fixed multiplier silently weakens every time the dataset grows: at 34k
# rows a 16x weight put the 229 face crops at 7.4% of each batch and the
# model learned faces, but adding 62k Pexels photos pushed the same 16x down
# to 2.7% and portrait quality visibly regressed. The share is what the model
# actually experiences, so set that directly and let the multiplier fall out.
ALIGNED_TARGET_SHARE = {"portrait_aligned:": 0.075, "person_aligned:": 0.140}
_sel_cache = {p: (np.array([q.startswith(p) for q in queries]) & (diff_w > 0))
              for p in ALIGNED_TARGET_SHARE}
# Iterate: each prefix's weight changes the pool the *other* prefix is a
# share of, so a single pass lands short (portraits solve to 6.8% instead of
# 7.5% once the body rows are re-weighted afterwards). Three passes is well
# past the fixed point -- it converges by the second.
for _ in range(3):
    for prefix, target in ALIGNED_TARGET_SHARE.items():
        sel = _sel_cache[prefix]
        n_sel = int(sel.sum())
        if n_sel == 0:
            continue
        rest = float(diff_w[~sel].sum())
        # solve share = n*w / (rest + n*w)  ->  w = share*rest / (n*(1-share))
        diff_w[sel] = max(1.0, round(target * rest / (n_sel * (1.0 - target))))
for prefix, target in ALIGNED_TARGET_SHARE.items():
    sel = _sel_cache[prefix]
    if sel.any():
        print(f"  {prefix:18s} {int(sel.sum()):5d} rows -> weight {diff_w[sel][0]:.0f} "
              f"(target {target*100:.1f}% of diffusion pool)")

DIFF_POOL = np.repeat(np.arange(N), diff_w.astype(np.int64))
print(f"pools: AE={len(AE_POOL)} samples over {N} rows | "
      f"diffusion={len(DIFF_POOL)} samples over {int((diff_w>0).sum())} captioned rows")
for prefix in ALIGNED_TARGET_SHARE:
    sel = np.array([q.startswith(prefix) for q in queries])
    if sel.any():
        print(f"  actual {prefix:18s} share: "
              f"{100*diff_w[sel].sum()/max(1,diff_w.sum()):.2f}%")

for name in ["portrait_aligned", "person_aligned"] + list(TOPIC_PATTERNS):
    if name in ("portrait_aligned", "person_aligned"):
        n = sum(1 for q in queries if q.startswith(name + ":"))
    else:
        pat, _ = TOPIC_PATTERNS[name]
        n = sum(1 for cap in captions if pat.search(cap or ""))
    print(f"{name} rows: {n}")

def sample_indices(rng, batch, pool):
    return pool[rng.integers(0, len(pool), batch)]


def img_batch(idx):
    x = torch.from_numpy(decode_many(idx).astype(np.float32) / 255.0).to(device)
    flip = torch.rand(len(idx), device=device) < 0.5
    return torch.where(flip[:, None, None, None], x.flip(-1), x)


# ---------------------------------------------------------------------------
# STAGE 1: our own autoencoder.
# ---------------------------------------------------------------------------
enc = OwnEncoder(LATENT_CH).to(device)
dec = OwnDecoder(LATENT_CH).to(device)
print("AE params:", sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in dec.parameters()))
AE_STEPS = int(os.environ.get('AE_STEPS', 30000))
AE_BATCH = 32
ae_opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=3e-4, weight_decay=0.01)


def edge_loss(a, b):
    # L1 on horizontal+vertical image gradients: penalizes blur directly,
    # no pretrained perceptual net needed.
    dxa, dxb = a[..., :, 1:] - a[..., :, :-1], b[..., :, 1:] - b[..., :, :-1]
    dya, dyb = a[..., 1:, :] - a[..., :-1, :], b[..., 1:, :] - b[..., :-1, :]
    return (dxa - dxb).abs().mean() + (dya - dyb).abs().mean()


t0 = time.time()
ema_l = None
for step in range(1, AE_STEPS + 1):
    x = img_batch(sample_indices(rng, AE_BATCH, AE_POOL))
    ae_opt.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
        z = enc(x)
        y = dec(z)
        loss = (y - x).abs().mean() + F.mse_loss(y, x) + 0.5 * edge_loss(y, x)
    loss.backward()
    ae_opt.step()
    ema_l = loss.item() if ema_l is None else 0.98 * ema_l + 0.02 * loss.item()
    if step % 500 == 0 or step == 1:
        print(f"[AE] step {step:6d}  loss {loss.item():.4f}  ema {ema_l:.4f}  ({time.time()-t0:7.1f}s)")
enc.eval()
dec.eval()
print("[AE] done in %.1fs" % (time.time() - t0))

# ---------------------------------------------------------------------------
# STAGE 2: encode the dataset with OUR encoder.
# ---------------------------------------------------------------------------
lat_list = []
with torch.no_grad():
    for i in range(0, N, 64):
        chunk = np.arange(i, min(i + 64, N))
        x = torch.from_numpy(decode_many(chunk).astype(np.float32) / 255.0).to(device)
        lat_list.append(enc(x).float().cpu())
LATENTS = torch.cat(lat_list)
LSCALE = 1.0 / float(LATENTS.std())
LATENTS = LATENTS * LSCALE
print("latents:", tuple(LATENTS.shape), "LSCALE:", round(LSCALE, 4))

# ---------------------------------------------------------------------------
# STAGE 3: latent diffusion with the order-aware caption encoder.
# ---------------------------------------------------------------------------
T = 1000
BETA_START, BETA_END = 1e-4, 0.02
betas = torch.linspace(BETA_START, BETA_END, T)
acp = torch.cumprod(1.0 - betas, dim=0)
sqrt_acp_d = torch.sqrt(acp).to(device)
sqrt_1m_acp_d = torch.sqrt(1.0 - acp).to(device)

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
vocab = [w for w, _ in sorted(freq.items(), key=lambda kv: -kv[1])[:2000]]
word2id = {w: i for i, w in enumerate(vocab)}
print("vocab size:", len(vocab))

MAXLEN = 14
EMBED_DIM = 64
ids_arr = np.zeros((N, MAXLEN), dtype=np.int64)
mask_arr = np.zeros((N, MAXLEN), dtype=np.float32)
for i, toks in enumerate(token_lists):
    known = [word2id[w] for w in toks if w in word2id][:MAXLEN]
    for j, tid in enumerate(known):
        ids_arr[i, j] = tid
        mask_arr[i, j] = 1.0

# 43.2M params (was 16.1M at (192,288,384)). 2.7x the compute per step, so
# STEPS is raised with it -- this project's clearest failure was a 109M
# pixel-space model at 12k steps producing pure static, and 21M only worked
# once it got 60k. A bigger model on an unchanged step budget gets *worse*.
CHANNELS = (320, 480, 640)
EMB_DIM = 256
model = LatentUNet(channels=CHANNELS, emb_dim=EMB_DIM, cdim=EMBED_DIM).to(device)
cap_enc = CaptionEncoderV2(len(vocab), EMBED_DIM, MAXLEN).to(device)
print("LatentUNet params:", f"{model.param_count():,}",
      " caption encoder params:", f"{sum(p.numel() for p in cap_enc.parameters()):,}")

LR = 2e-4
UNCOND_P = 0.15
GUIDANCE_SCALE = 3.0
EMA_DECAY = 0.9995
opt = torch.optim.AdamW(list(model.parameters()) + list(cap_enc.parameters()), lr=LR, weight_decay=0.01)
print(f"optimizer: AdamW (lr={LR}, weight_decay=0.01)")

all_params = list(model.named_parameters()) + [("cap." + k, v) for k, v in cap_enc.named_parameters()]
ema = {name: p.detach().clone() for name, p in all_params}

STEPS = int(os.environ.get('STEPS', 90000))
BATCH = 128 if device == "cuda" else 16
LAT_T = LATENTS.to(device)
IDS_T = torch.from_numpy(ids_arr).to(device)
MASK_T = torch.from_numpy(mask_arr).to(device)

import base64


def export_model():
    order, shapes, flat = [], {}, []
    for k, v in ema.items():
        order.append(k)
        shapes[k] = list(v.shape)
        flat.append(v.detach().cpu().numpy().astype(np.float32).ravel())
    # OUR decoder, exported under the same key prefix/topology the browser
    # already runs (meta.decoder marks the provenance change).
    for k, v in enc.state_dict().items():
        key = "own_enc." + k
        order.append(key)
        shapes[key] = list(v.shape)
        flat.append(v.detach().cpu().numpy().astype(np.float32).ravel())
    for k, v in dec.state_dict().items():
        key = "taesd_dec." + k
        order.append(key)
        shapes[key] = list(v.shape)
        flat.append(v.detach().cpu().numpy().astype(np.float32).ravel())
    flat = np.concatenate(flat)
    meta = {
        "kind": "latent", "decoder": "own", "textEncoder": "attn",
        "S": S, "latentSize": LATENT_SIZE, "latentCh": LATENT_CH,
        "lscale": LSCALE, "T": T, "betaStart": BETA_START, "betaEnd": BETA_END,
        "channels": list(CHANNELS), "embDim": EMB_DIM, "cdim": EMBED_DIM,
        "maxlen": MAXLEN, "vocab": vocab, "guidanceScale": GUIDANCE_SCALE,
    }
    out = {"meta": meta, "order": order, "shapes": shapes,
           "b64": base64.b64encode(flat.tobytes()).decode("ascii")}
    tmp = "diffusion_hq_model.json.tmp"
    with open(tmp, "w") as f:
        json.dump(out, f)
    os.replace(tmp, "diffusion_hq_model.json")


t0 = time.time()
ema_loss = None
for step in range(1, STEPS + 1):
    idx = torch.from_numpy(sample_indices(rng, BATCH, DIFF_POOL)).to(device)
    x0 = LAT_T[idx]
    flip = torch.rand(BATCH, device=device) < 0.5
    x0 = torch.where(flip[:, None, None, None], x0.flip(-1), x0)
    t_idx = torch.randint(0, T, (BATCH,), device=device)
    noise = torch.randn_like(x0)
    xt = sqrt_acp_d[t_idx][:, None, None, None] * x0 + sqrt_1m_acp_d[t_idx][:, None, None, None] * noise

    for g in opt.param_groups:
        g["lr"] = LR * min(1.0, step / 1000)

    opt.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
        cond = cap_enc(IDS_T[idx], MASK_T[idx])
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
