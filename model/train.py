"""
Train PocketPaint's own tiny generative model.

The model is a CONDITIONAL NEURAL FIELD (a small MLP / CPPN):
    input  = [ Fourier(x,y) , x, y, r , condition_vector(12) ]   (39 dims)
    output = RGB at that pixel                                    (3 dims)

It is trained by DISTILLING a hand-written analytic "scene renderer" R(x,y,c):
the renderer composes recognizable landscapes (sky by time-of-day, ground by
biome, optional mountains / neon / fire) from a 12-D semantic condition vector.
Because R is smooth and deterministic, a small MLP can learn to reproduce it and
then generalize/interpolate across conditions — giving us a real, trained neural
generator that is only ~20k params, renders at any resolution, and runs in the
browser (WebGPU shader or JS) with zero downloads.

Outputs: model.json (weights, float16-ish) consumed by the web app, plus a few
preview PNGs so we can eyeball quality before shipping.
"""
import numpy as np, json, struct, zlib, math, time

rng = np.random.default_rng(0)

# ----------------------------------------------------------------------------
# Condition vector layout (must match the JS prompt->condition mapping exactly).
# All values in [0,1].
#  0 night     1 sunset    2 day       3 overcast      (sky / time-of-day)
#  4 ocean     5 forest    6 desert    7 snow      8 plain   (ground biome)
#  9 mountains 10 neon     11 fire
# ----------------------------------------------------------------------------
CDIM = 12
HORIZON = 0.58

def _mix(a, b, t):
    return a + (b - a) * t

def render(x, y, c):
    """Analytic ground-truth scene renderer, fully vectorized.
    x,y: arrays in [0,1] (y=0 top). c: array [...,12]. Returns [...,3] in [0,1]."""
    x = np.asarray(x); y = np.asarray(y)
    c = np.asarray(c)
    night = c[..., 0]; sunset = c[..., 1]; day = c[..., 2]; over = c[..., 3]
    ocean = c[..., 4]; forest = c[..., 5]; desert = c[..., 6]; snow = c[..., 7]; plain = c[..., 8]
    mount = c[..., 9]; neon = c[..., 10]; fire = c[..., 11]

    sky_w = night + sunset + day + over + 1e-3
    # vertical param within sky (0 top .. 1 horizon)
    ts = np.clip(y / HORIZON, 0, 1)

    def grad(top, hor):
        return np.stack([_mix(top[i], hor[i], ts) for i in range(3)], axis=-1)

    night_c  = grad((0.03, 0.04, 0.12), (0.12, 0.10, 0.22))
    sunset_c = grad((0.17, 0.15, 0.42), (1.00, 0.50, 0.22))
    day_c    = grad((0.20, 0.45, 0.88), (0.70, 0.86, 0.98))
    over_c   = grad((0.55, 0.57, 0.61), (0.78, 0.79, 0.80))

    sky = (night_c * night[..., None] + sunset_c * sunset[..., None] +
           day_c * day[..., None] + over_c * over[..., None]) / sky_w[..., None]

    # Sun / moon glow (smooth radial), blended by time weights.
    def glow(cx, cy, rad, col, strength):
        d2 = (x - cx) ** 2 + (y - cy) ** 2
        g = np.exp(-d2 / (rad * rad))
        return np.stack([col[i] * g * strength for i in range(3)], axis=-1)

    sun = glow(0.50, 0.46, 0.16, (1.0, 0.80, 0.45), sunset)        # low warm sun
    sun += glow(0.32, 0.22, 0.13, (1.0, 0.95, 0.80), day)          # high bright sun
    moon = glow(0.70, 0.22, 0.09, (0.92, 0.94, 1.0), night)        # moon
    sky = np.clip(sky + sun + moon, 0, 1)

    # Mountains: dark silhouette just above the horizon (only where mount>0).
    ridge = (0.5 + 0.30 * np.sin(x * 7.0 + 1.3) + 0.18 * np.sin(x * 17.0 + 4.0))
    mh = HORIZON - 0.20 * ridge                                    # ridge top y
    m_mask = np.clip((HORIZON - y) / 0.02, 0, 1) * np.clip((y - mh) / 0.02 + 0.5, 0, 1)
    m_mask = np.clip(m_mask, 0, 1) * (y < HORIZON)
    mountain_c = np.array([0.15, 0.14, 0.20])
    sky = _mix(sky, np.broadcast_to(mountain_c, sky.shape), (m_mask * mount)[..., None])

    # Ground (y >= horizon)
    tg = np.clip((y - HORIZON) / (1 - HORIZON), 0, 1)
    def gground(top, bot):
        return np.stack([_mix(top[i], bot[i], tg) for i in range(3)], axis=-1)
    # gentle horizontal water shimmer
    wave = 0.05 * np.sin(y * 60.0) * np.clip(1 - tg, 0, 1)
    ocean_c  = gground((0.06, 0.28, 0.52), (0.02, 0.10, 0.28)) + wave[..., None]
    forest_c = gground((0.12, 0.36, 0.14), (0.04, 0.16, 0.06))
    desert_c = gground((0.86, 0.72, 0.46), (0.60, 0.44, 0.25))
    snow_c   = gground((0.90, 0.93, 0.98), (0.68, 0.76, 0.90))
    plain_c  = gground((0.36, 0.56, 0.26), (0.17, 0.30, 0.12))
    gnd_w = ocean + forest + desert + snow + plain + 1e-3
    gnd = (ocean_c * ocean[..., None] + forest_c * forest[..., None] +
           desert_c * desert[..., None] + snow_c * snow[..., None] +
           plain_c * plain[..., None]) / gnd_w[..., None]
    gnd = np.clip(gnd, 0, 1)

    is_ground = (y >= HORIZON)[..., None].astype(np.float64)
    col = sky * (1 - is_ground) + gnd * is_ground

    # Fire: warm the lower sky / horizon strongly.
    fire_glow = np.clip(1 - np.abs(y - HORIZON) / 0.35, 0, 1)
    col = col + (np.array([0.9, 0.35, 0.08]) * (fire_glow * fire)[..., None]) * 0.6
    col = np.clip(col, 0, 1)

    # Neon: push toward a magenta-cyan synthwave grade + horizon glow.
    lum = (col[..., 0] * 0.3 + col[..., 1] * 0.59 + col[..., 2] * 0.11)
    neon_grade = np.stack([0.6 + 0.4 * np.sin(lum * 6.0 + 0.0),
                           0.2 + 0.3 * np.sin(lum * 6.0 + 2.0),
                           0.7 + 0.3 * np.sin(lum * 6.0 + 4.0)], axis=-1)
    horizon_glow = np.clip(1 - np.abs(y - HORIZON) / 0.12, 0, 1)
    neon_grade = np.clip(neon_grade + np.array([0.7, 0.1, 0.9]) * horizon_glow[..., None] * 0.5, 0, 1)
    col = _mix(col, neon_grade, (neon * 0.85)[..., None])

    return np.clip(col, 0, 1)

# ----------------------------------------------------------------------------
# Condition sampler for training: cover the space with realistic archetypes
# plus blends and noise so the network learns a smooth, generalizing manifold.
# ----------------------------------------------------------------------------
def sample_conditions(n):
    c = np.zeros((n, CDIM))
    # time-of-day: pick one (sometimes blend two for smoothness)
    t = rng.integers(0, 4, n)
    for i, ti in enumerate(t):
        c[i, ti] = 1.0
        if rng.random() < 0.25:                      # blend a neighbor a bit
            tj = rng.integers(0, 4)
            c[i, tj] = max(c[i, tj], rng.uniform(0.2, 0.6))
    # biome: pick one
    b = rng.integers(4, 9, n)
    c[np.arange(n), b] = 1.0
    if_blend = rng.random(n) < 0.2
    bb = rng.integers(4, 9, n)
    c[np.arange(n)[if_blend], bb[if_blend]] = np.maximum(
        c[np.arange(n)[if_blend], bb[if_blend]], rng.uniform(0.2, 0.6, if_blend.sum()))
    # flags
    c[:, 9] = (rng.random(n) < 0.35) * rng.uniform(0.4, 1.0, n)   # mountains
    c[:, 10] = (rng.random(n) < 0.18) * rng.uniform(0.5, 1.0, n)  # neon
    c[:, 11] = (rng.random(n) < 0.18) * rng.uniform(0.4, 1.0, n)  # fire
    # small noise + clamp
    c = np.clip(c + rng.normal(0, 0.03, c.shape), 0, 1)
    return c

# ----------------------------------------------------------------------------
# Feature encoding (must match the web app exactly).
# ----------------------------------------------------------------------------
K = 6  # Fourier octaves
def encode(x, y, c):
    xp = x * 2 - 1; yp = y * 2 - 1
    r = np.sqrt(xp * xp + yp * yp)
    feats = [xp, yp, r]
    for k in range(K):
        f = (2.0 ** k) * math.pi
        feats += [np.sin(f * xp), np.cos(f * xp), np.sin(f * yp), np.cos(f * yp)]
    base = np.stack(feats, axis=-1)               # [...,3+4K]
    return np.concatenate([base, c], axis=-1)     # [...,3+4K+12]

DIM = 3 + 4 * K + CDIM
print("input dim DIM =", DIM)

# ----------------------------------------------------------------------------
# Tiny MLP: DIM -> H -> H -> H -> 3 with tanh hidden, sigmoid output.
# ----------------------------------------------------------------------------
H = 96
def glorot(a, b):
    return rng.normal(0, math.sqrt(2.0 / (a + b)), (a, b))
P = {
    "W0": glorot(DIM, H), "b0": np.zeros(H),
    "W1": glorot(H, H),   "b1": np.zeros(H),
    "W2": glorot(H, H),   "b2": np.zeros(H),
    "W3": glorot(H, 3),   "b3": np.zeros(3),
}
def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))

def forward(X, cache=None):
    z0 = X @ P["W0"] + P["b0"]; a0 = np.tanh(z0)
    z1 = a0 @ P["W1"] + P["b1"]; a1 = np.tanh(z1)
    z2 = a1 @ P["W2"] + P["b2"]; a2 = np.tanh(z2)
    z3 = a2 @ P["W3"] + P["b3"]; out = sigmoid(z3)
    if cache is not None:
        cache.update(dict(X=X, a0=a0, a1=a1, a2=a2, out=out))
    return out

# Adam state
m = {k: np.zeros_like(v) for k, v in P.items()}
v = {k: np.zeros_like(v) for k, v in P.items()}
def adam(grads, t, lr=2e-3, b1=0.9, b2=0.999, eps=1e-8):
    for k in P:
        m[k] = b1 * m[k] + (1 - b1) * grads[k]
        v[k] = b2 * v[k] + (1 - b2) * grads[k] ** 2
        mh = m[k] / (1 - b1 ** t); vh = v[k] / (1 - b2 ** t)
        P[k] -= lr * mh / (np.sqrt(vh) + eps)

# ----------------------------------------------------------------------------
# Training loop: fresh analytic data each step (infinite dataset).
# ----------------------------------------------------------------------------
STEPS = 6000
NC = 96            # conditions per batch
NP = 160           # pixels per condition  -> batch = 15360
t0 = time.time()
for step in range(1, STEPS + 1):
    c = sample_conditions(NC)                              # [NC,12]
    px = rng.random((NC, NP)); py = rng.random((NC, NP))   # pixel coords
    cc = np.repeat(c[:, None, :], NP, axis=1)              # [NC,NP,12]
    X = encode(px, py, cc).reshape(-1, DIM)
    Y = render(px, py, cc).reshape(-1, 3)

    cache = {}
    out = forward(X, cache)
    diff = out - Y                                         # MSE on sigmoid out
    B = X.shape[0]
    # backprop
    dz3 = (2.0 / B) * diff * out * (1 - out)              # through sigmoid
    gW3 = cache["a2"].T @ dz3; gb3 = dz3.sum(0)
    da2 = dz3 @ P["W3"].T; dz2 = da2 * (1 - cache["a2"] ** 2)
    gW2 = cache["a1"].T @ dz2; gb2 = dz2.sum(0)
    da1 = dz2 @ P["W2"].T; dz1 = da1 * (1 - cache["a1"] ** 2)
    gW1 = cache["a0"].T @ dz1; gb1 = dz1.sum(0)
    da0 = dz1 @ P["W1"].T; dz0 = da0 * (1 - cache["a0"] ** 2)
    gW0 = cache["X"].T @ dz0; gb0 = dz0.sum(0)
    grads = {"W0": gW0, "b0": gb0, "W1": gW1, "b1": gb1,
             "W2": gW2, "b2": gb2, "W3": gW3, "b3": gb3}
    adam(grads, step)

    if step % 500 == 0 or step == 1:
        mse = (diff ** 2).mean()
        print(f"step {step:5d}  mse {mse:.5f}  ({time.time()-t0:5.1f}s)")

print("trained in %.1fs" % (time.time() - t0))

# ----------------------------------------------------------------------------
# Export weights to JSON (float values; the web app reads them directly).
# ----------------------------------------------------------------------------
meta = {"DIM": DIM, "H": H, "K": K, "CDIM": CDIM, "HORIZON": HORIZON}
weights = {k: P[k].astype(np.float32).ravel().tolist() for k in P}
with open("model.json", "w") as f:
    json.dump({"meta": meta, "weights": weights}, f)
print("saved model.json  (params:", sum(P[k].size for k in P), ")")

# ----------------------------------------------------------------------------
# Minimal PNG writer (no PIL) + render previews from the TRAINED net.
# ----------------------------------------------------------------------------
def write_png(path, img):  # img: HxWx3 uint8
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))
    def chunk(typ, data):
        c = typ + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    idat = zlib.compress(raw, 9)
    with open(path, "wb") as f:
        f.write(sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b""))

def net_render(c, S=128):
    ys, xs = np.meshgrid(np.linspace(0, 1, S), np.linspace(0, 1, S), indexing="ij")
    cc = np.broadcast_to(np.asarray(c), (S, S, CDIM))
    X = encode(xs, ys, cc).reshape(-1, DIM)
    out = forward(X).reshape(S, S, 3)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)

def C(**kw):
    c = np.zeros(CDIM)
    idx = {"night":0,"sunset":1,"day":2,"overcast":3,"ocean":4,"forest":5,
           "desert":6,"snow":7,"plain":8,"mountains":9,"neon":10,"fire":11}
    for k, val in kw.items():
        c[idx[k]] = val
    return c

previews = {
    "sunset_ocean":  C(sunset=1, ocean=1),
    "night_mountain":C(night=1, mountains=1, plain=1),
    "day_forest":    C(day=1, forest=1),
    "desert_day":    C(day=1, desert=1),
    "neon_city":     C(night=1, neon=1, plain=1),
    "snow_mountain": C(day=1, snow=1, mountains=1),
}
for name, c in previews.items():
    write_png(f"preview_{name}.png", net_render(c, 160))
print("wrote previews:", list(previews))
