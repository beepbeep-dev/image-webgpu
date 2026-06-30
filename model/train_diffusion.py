"""
Train PocketPaint's own small DDPM-style diffusion model from scratch, on a
REAL captioned photo dataset (model/dataset_real.db — see scrape_dataset.py),
not synthetic renders.

This is a genuine diffusion model: a small conditional convolutional UNet
trained to predict the noise added to a 32x32 image at a random timestep,
then sampled at inference time via iterative denoising (DDIM, ~20 steps).
Conditioning comes from real captions, not a hand-designed category vector:
a tiny trainable word-embedding table (built from the dataset's own
vocabulary) maps each caption to a fixed-size vector by mean-pooling the
embeddings of its known words — essentially a minimal from-scratch text
encoder, trained jointly with the UNet on the actual diffusion objective.
"""
import io, json, re, sqlite3, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from model import TinyUNet

torch.manual_seed(0)
rng = np.random.default_rng(0)

S = 32        # training resolution
T = 1000
BETA_START, BETA_END = 1e-4, 0.02
betas = torch.linspace(BETA_START, BETA_END, T)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_acp = torch.sqrt(alphas_cumprod)
sqrt_1m_acp = torch.sqrt(1.0 - alphas_cumprod)

# ---------------------------------------------------------------------------
# Load the real captioned database and build a small vocabulary + tokenized
# captions. Tokenization (lowercase, split on non [a-z0-9]) is intentionally
# trivial so the exact same rule can be reimplemented in plain JS at
# inference time with no external NLP dependency.
# ---------------------------------------------------------------------------
STOPWORDS = set("""a an the of in on at to from with and or for is are was
were be been being by as it its this that these those near over under
photo photograph picture image view taken file""".split())

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return [w for w in TOKEN_RE.findall(text.lower()) if w not in STOPWORDS and len(w) > 1]


con = sqlite3.connect("dataset_real.db")
db_rows = con.execute("SELECT caption, image FROM samples").fetchall()
con.close()
N = len(db_rows)
print("loaded", N, "rows from dataset_real.db")

token_lists = [tokenize(cap) for cap, _ in db_rows]
freq = {}
for toks in token_lists:
    for w in set(toks):
        freq[w] = freq.get(w, 0) + 1
VOCAB_SIZE = 800
vocab = [w for w, _ in sorted(freq.items(), key=lambda kv: -kv[1])[:VOCAB_SIZE]]
word2id = {w: i for i, w in enumerate(vocab)}
print("vocab size:", len(vocab), " e.g.:", vocab[:20])

MAXLEN = 12
EMBED_DIM = 48
ids = np.zeros((N, MAXLEN), dtype=np.int64)
mask = np.zeros((N, MAXLEN), dtype=np.float32)
for i, toks in enumerate(token_lists):
    known = [word2id[w] for w in toks if w in word2id][:MAXLEN]
    for j, tid in enumerate(known):
        ids[i, j] = tid
        mask[i, j] = 1.0

DB_IMG = np.zeros((N, 3, S, S), dtype=np.float32)
for i, (_, png_blob) in enumerate(db_rows):
    img = np.asarray(Image.open(io.BytesIO(png_blob)).convert("RGB"), dtype=np.float32) / 255.0
    DB_IMG[i] = np.transpose(img * 2 - 1, (2, 0, 1))   # [3,S,S] in [-1,1]
del db_rows

IDS = torch.from_numpy(ids)
MASK = torch.from_numpy(mask)


def make_batch(n):
    idx = rng.integers(0, N, n)
    return torch.from_numpy(DB_IMG[idx]), IDS[idx], MASK[idx]


# ---------------------------------------------------------------------------
# Tiny from-scratch text encoder: word embeddings, mean-pooled over the
# tokens actually present in a caption (padding positions are masked out of
# both the sum and the gradient).
# ---------------------------------------------------------------------------
class CaptionEncoder(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)

    def forward(self, ids, mask):
        e = self.embed(ids)                                    # [B,MAXLEN,D]
        summed = (e * mask[:, :, None]).sum(1)
        count = mask.sum(1, keepdim=True).clamp(min=1.0)
        return summed / count


device = "cpu"
model = TinyUNet(c0=28, c1=44, c2=64, emb_dim=96, cdim=EMBED_DIM).to(device)
cap_enc = CaptionEncoder(len(vocab), EMBED_DIM).to(device)
print("UNet params:", model.param_count(), " caption encoder params:",
      sum(p.numel() for p in cap_enc.parameters()))

LR = 1e-3
WARMUP = 200
opt = torch.optim.Adam(list(model.parameters()) + list(cap_enc.parameters()), lr=LR)

STEPS = 6000
BATCH = 48
t0 = time.time()
ema_loss = None
for step in range(1, STEPS + 1):
    x0, ids_b, mask_b = make_batch(BATCH)
    t_idx = torch.randint(0, T, (BATCH,))
    noise = torch.randn_like(x0)
    xt = sqrt_acp[t_idx][:, None, None, None] * x0 + sqrt_1m_acp[t_idx][:, None, None, None] * noise

    for g in opt.param_groups:
        g["lr"] = LR * min(1.0, step / WARMUP)

    opt.zero_grad()
    cond = cap_enc(ids_b, mask_b)
    pred = model(xt, t_idx.float() / T, cond)
    loss = F.mse_loss(pred, noise)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(cap_enc.parameters()), 1.0)
    opt.step()

    ema_loss = loss.item() if ema_loss is None else 0.98 * ema_loss + 0.02 * loss.item()
    if step % 200 == 0 or step == 1:
        print(f"step {step:5d}  loss {loss.item():.4f}  ema {ema_loss:.4f}  ({time.time()-t0:6.1f}s)")

print("trained in %.1fs" % (time.time() - t0))

# ---------------------------------------------------------------------------
# Export: UNet weights + the caption word-embedding table, as one flat
# float32 array (base64 in the web app), plus the noise schedule constants
# and vocabulary needed to reproduce DDIM sampling + tokenization in JS.
# ---------------------------------------------------------------------------
order = []
shapes = {}
flat = []
sd = dict(model.state_dict())
sd["word_emb.weight"] = cap_enc.embed.weight.detach()
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
    "c0": 28, "c1": 44, "c2": 64, "embDim": 96, "cdim": EMBED_DIM,
    "vocab": vocab,
}
out = {"meta": meta, "order": order, "shapes": shapes, "b64": b64}
with open("diffusion_model.json", "w") as f:
    json.dump(out, f)
print("saved diffusion_model.json, b64 len:", len(b64))
