"""
Train PocketPaint's own "Diffusion HQ" model: a SEPARATE, deeper diffusion
model generating natively at 256x256 (the smaller 32x32 "Diffusion (ours)"
model in train_diffusion.py is untouched), trained on ~100k real captioned
photos scraped from Wikimedia Commons (model/dataset_hq.db — see
scrape_dataset_hq.py).

Same recipe as the 32x32 model — DDPM/DDIM, a from-scratch word-embedding
text encoder trained jointly via classifier-free-guidance dropout, and EMA
weight averaging for the exported weights — just a bigger UNet (BigUNet,
model_hq.py) and images decoded lazily per batch from SQLite instead of
preloaded into RAM: at 256x256, the full ~100k-image dataset would need
~80GB as float32, far more than this machine has, so each step fetches and
JPEG-decodes only the BATCH images it actually needs.
"""
import io, json, re, sqlite3, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from model_hq import BigUNet

torch.manual_seed(0)
rng = np.random.default_rng(0)

S = 256
T = 1000
BETA_START, BETA_END = 1e-4, 0.02
betas = torch.linspace(BETA_START, BETA_END, T)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_acp = torch.sqrt(alphas_cumprod)
sqrt_1m_acp = torch.sqrt(1.0 - alphas_cumprod)

# ---------------------------------------------------------------------------
# Vocabulary (cheap: just reads captions, not images).
# ---------------------------------------------------------------------------
STOPWORDS = set("""a an the of in on at to from with and or for is are was
were be been being by as it its this that these those near over under
photo photograph picture image view taken file""".split())
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return [w for w in TOKEN_RE.findall(text.lower()) if w not in STOPWORDS and len(w) > 1]


con = sqlite3.connect("dataset_hq.db", check_same_thread=False)
id_caption_rows = con.execute("SELECT id, caption FROM samples").fetchall()
N = len(id_caption_rows)
print("dataset has", N, "rows in dataset_hq.db")

ROW_IDS = np.array([r[0] for r in id_caption_rows], dtype=np.int64)
token_lists = [tokenize(cap) for _, cap in id_caption_rows]
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
del id_caption_rows, token_lists

# ---------------------------------------------------------------------------
# Lazy per-batch image loading: fetch + JPEG-decode only the rows a given
# step actually needs, straight from SQLite (id is the indexed primary key).
# ---------------------------------------------------------------------------
_cur = con.cursor()


def load_images(row_indices):
    db_ids = ROW_IDS[row_indices]
    placeholders = ",".join("?" * len(db_ids))
    rows = _cur.execute(f"SELECT id, image FROM samples WHERE id IN ({placeholders})", db_ids.tolist()).fetchall()
    by_id = {rid: blob for rid, blob in rows}
    out = np.empty((len(row_indices), 3, S, S), dtype=np.float32)
    for k, idx in enumerate(row_indices):
        img = np.asarray(Image.open(io.BytesIO(by_id[int(ROW_IDS[idx])])).convert("RGB"), dtype=np.float32) / 255.0
        out[k] = np.transpose(img * 2 - 1, (2, 0, 1))
    return out


def make_batch(n):
    idx = rng.integers(0, N, n)
    imgs = load_images(idx)
    return torch.from_numpy(imgs), torch.from_numpy(ids_arr[idx]), torch.from_numpy(mask_arr[idx])


# ---------------------------------------------------------------------------
# Tiny from-scratch text encoder (identical design to the 32x32 model's).
# ---------------------------------------------------------------------------
class CaptionEncoder(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)

    def forward(self, ids, mask):
        e = self.embed(ids)
        summed = (e * mask[:, :, None]).sum(1)
        count = mask.sum(1, keepdim=True).clamp(min=1.0)
        return summed / count


device = "cpu"
CHANNELS = (16, 24, 36, 54, 80, 112)
EMB_DIM = 160
model = BigUNet(channels=CHANNELS, emb_dim=EMB_DIM, cdim=EMBED_DIM).to(device)
cap_enc = CaptionEncoder(len(vocab), EMBED_DIM).to(device)
print("UNet params:", model.param_count(), " caption encoder params:",
      sum(p.numel() for p in cap_enc.parameters()))

LR = 1e-3
WARMUP = 300
UNCOND_P = 0.15
GUIDANCE_SCALE = 3.0
EMA_DECAY = 0.999
opt = torch.optim.Adam(list(model.parameters()) + list(cap_enc.parameters()), lr=LR)

all_params = list(model.named_parameters()) + [("word_emb." + k, v) for k, v in cap_enc.embed.named_parameters()]
ema = {name: p.detach().clone() for name, p in all_params}

STEPS = 8000
BATCH = 16
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
    uncond_mask = (torch.rand(BATCH, 1) < UNCOND_P).float()
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
    if step % 50 == 0 or step == 1:
        print(f"step {step:5d}  loss {loss.item():.4f}  ema {ema_loss:.4f}  ({time.time()-t0:7.1f}s)")

print("trained in %.1fs" % (time.time() - t0))

# ---------------------------------------------------------------------------
# Export (same flat float32 + base64 format as the 32x32 model).
# ---------------------------------------------------------------------------
order = []
shapes = {}
flat = []
sd = dict(ema)
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
    "channels": list(CHANNELS), "embDim": EMB_DIM, "cdim": EMBED_DIM,
    "vocab": vocab, "guidanceScale": GUIDANCE_SCALE,
}
out = {"meta": meta, "order": order, "shapes": shapes, "b64": b64}
with open("diffusion_hq_model.json", "w") as f:
    json.dump(out, f)
print("saved diffusion_hq_model.json, b64 len:", len(b64))
