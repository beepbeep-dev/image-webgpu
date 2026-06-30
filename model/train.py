"""
Train PocketPaint's own tiny generative model.

The model is a CONDITIONAL NEURAL FIELD (a small MLP / CPPN):
    input  = [ Fourier(x,y) , x, y, r , condition_vector(21) ]   (48 dims)
    output = RGB at that pixel                                    (3 dims)

It is trained by DISTILLING a hand-written analytic "scene renderer" R(x,y,c):
the renderer composes recognizable landscapes (sky by time-of-day, ground by
biome, optional mountains / neon / fire) AND simple silhouette SUBJECTS
(person / horse / dog / car / bird / boat / tree / building, placed on the
ground or sky) from a 21-D semantic condition vector. Because R is smooth and
deterministic, a small MLP can learn to reproduce it and then
generalize/interpolate across conditions — giving us a real, trained neural
generator that is only ~32k params, renders at any resolution, and runs in
the browser (plain JS) with zero downloads.

Outputs: model.json (weights) consumed by the web app, plus a few preview
PNGs so we can eyeball quality before shipping.
"""
import numpy as np, json, struct, zlib, math, time

rng = np.random.default_rng(0)

# ----------------------------------------------------------------------------
# Condition vector layout (must match the JS prompt->condition mapping exactly).
# All values in [0,1].
#  0 night     1 sunset    2 day       3 overcast      (sky / time-of-day)
#  4 ocean     5 forest    6 desert    7 snow      8 plain   (ground biome)
#  9 mountains 10 neon     11 fire
#  12 person   13 horse    14 tree     15 building        (subjects, on ground)
#  17 dog      18 car      19 bird     20 boat             (more subjects)
#  16 subj_x   (0..1, horizontal placement of the subject group, default 0.5)
# ----------------------------------------------------------------------------
CDIM = 21
HORIZON = 0.58

def _mix(a, b, t):
    return a + (b - a) * t

# ----------------------------------------------------------------------------
# Soft shape primitives for subject silhouettes (smooth, learnable edges).
# ----------------------------------------------------------------------------
def soft_rect(u, v, hw, hh, edge=0.045):
    du = np.clip((np.abs(u) - hw) / edge, 0, 1)
    dv = np.clip((np.abs(v) - hh) / edge, 0, 1)
    return np.clip((1 - du) * (1 - dv), 0, 1)

def soft_circle(u, v, r, edge=0.045):
    d = np.sqrt(u * u + v * v) - r
    return np.clip(1 - d / edge, 0, 1)

def soft_ellipse(u, v, ru, rv, edge=0.10):
    d = (u / ru) ** 2 + (v / rv) ** 2 - 1
    return np.clip(1 - d / edge, 0, 1)

def person_mask(u, v):
    """u,v local coords centered at feet (v=0 feet .. v=1 head), unit height."""
    leg1 = soft_rect(u + 0.09, v - 0.20, 0.045, 0.20)
    leg2 = soft_rect(u - 0.09, v - 0.20, 0.045, 0.20)
    body = soft_rect(u, v - 0.58, 0.16, 0.18)
    arm1 = soft_rect(u + 0.22, v - 0.55, 0.045, 0.16)
    arm2 = soft_rect(u - 0.22, v - 0.55, 0.045, 0.16)
    head = soft_circle(u, v - 0.88, 0.13)
    return np.maximum.reduce([leg1, leg2, body, arm1, arm2, head])

def horse_mask(u, v):
    legs = np.maximum.reduce([
        soft_rect(u + 0.30, v - 0.16, 0.035, 0.16),
        soft_rect(u + 0.14, v - 0.16, 0.035, 0.16),
        soft_rect(u - 0.14, v - 0.16, 0.035, 0.16),
        soft_rect(u - 0.30, v - 0.16, 0.035, 0.16),
    ])
    body = soft_ellipse(u, v - 0.48, 0.36, 0.18)
    neck = soft_ellipse(u - 0.40, v - 0.66, 0.13, 0.20)
    head = soft_ellipse(u - 0.52, v - 0.84, 0.10, 0.13)
    return np.maximum.reduce([legs, body, neck, head])

def tree_mask(u, v):
    trunk = soft_rect(u, v - 0.18, 0.045, 0.18)
    canopy = soft_circle(u, v - 0.62, 0.30)
    canopy2 = soft_circle(u - 0.16, v - 0.50, 0.20)
    canopy3 = soft_circle(u + 0.16, v - 0.50, 0.20)
    return np.maximum.reduce([trunk, canopy, canopy2, canopy3]), trunk, np.maximum(canopy, np.maximum(canopy2, canopy3))

def building_mask(u, v):
    body = soft_rect(u, v - 0.40, 0.30, 0.40)
    roof = soft_rect(u, v - 0.82, 0.34, 0.06)
    return np.maximum(body, roof)

def dog_mask(u, v):
    legs = np.maximum.reduce([
        soft_rect(u + 0.20, v - 0.10, 0.03, 0.10),
        soft_rect(u + 0.08, v - 0.10, 0.03, 0.10),
        soft_rect(u - 0.08, v - 0.10, 0.03, 0.10),
        soft_rect(u - 0.20, v - 0.10, 0.03, 0.10),
    ])
    body = soft_ellipse(u, v - 0.30, 0.24, 0.13)
    head = soft_circle(u - 0.26, v - 0.40, 0.11)
    tail = soft_circle(u + 0.30, v - 0.36, 0.06)
    return np.maximum.reduce([legs, body, head, tail])

def car_mask(u, v):
    body = soft_rect(u, v - 0.20, 0.32, 0.12)
    cabin = soft_rect(u - 0.04, v - 0.32, 0.18, 0.10)
    wheel1 = soft_circle(u - 0.20, v - 0.08, 0.08)
    wheel2 = soft_circle(u + 0.20, v - 0.08, 0.08)
    return np.maximum.reduce([body, cabin, wheel1, wheel2])

def boat_mask(u, v):
    hull = soft_rect(u, v - 0.06, 0.28, 0.07)
    mast = soft_rect(u, v - 0.28, 0.018, 0.22)
    sail = soft_rect(u + 0.08, v - 0.32, 0.10, 0.14)
    return np.maximum.reduce([hull, mast, sail])

def bird_mask(u, v):
    body = soft_ellipse(u, v, 0.08, 0.05)
    wing1 = soft_ellipse(u - 0.16, v + 0.03, 0.15, 0.045)
    wing2 = soft_ellipse(u + 0.16, v + 0.03, 0.15, 0.045)
    return np.maximum.reduce([body, wing1, wing2])

def place(x, y, cx, scale):
    """Local coords for a subject standing at (cx, ground) with given scale."""
    baseline = HORIZON + 0.018
    u = (x - cx) / scale
    v = (baseline - y) / scale
    return u, v

def place_float(x, y, cx, scale, sink=0.0):
    """Like place() but the baseline can sit above/below the horizon (boats)."""
    baseline = HORIZON + sink
    u = (x - cx) / scale
    v = (baseline - y) / scale
    return u, v

def place_sky(x, y, cx, cy, scale):
    """Local coords for something floating in the sky at (cx, cy)."""
    u = (x - cx) / scale
    v = (cy - y) / scale
    return u, v

def render(x, y, c):
    """Analytic ground-truth scene renderer, fully vectorized.
    x,y: arrays in [0,1] (y=0 top). c: array [...,17]. Returns [...,3] in [0,1]."""
    x = np.asarray(x); y = np.asarray(y)
    c = np.asarray(c)
    night = c[..., 0]; sunset = c[..., 1]; day = c[..., 2]; over = c[..., 3]
    ocean = c[..., 4]; forest = c[..., 5]; desert = c[..., 6]; snow = c[..., 7]; plain = c[..., 8]
    mount = c[..., 9]; neon = c[..., 10]; fire = c[..., 11]
    person = c[..., 12]; horse = c[..., 13]; tree = c[..., 14]; building = c[..., 15]
    dog = c[..., 17]; car = c[..., 18]; bird = c[..., 19]; boat = c[..., 20]
    subj_x = c[..., 16] * 0.5 + 0.25  # keep group roughly within frame [0.25,0.75]

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

    # ---- Subjects: simple silhouettes standing on the ground, drawn back to
    # front (tree, building, horse, person) so they layer sensibly when more
    # than one is requested at once. Each gets a small fixed offset from
    # subj_x so a "person and a horse" don't perfectly overlap.
    def paint(mask, color):
        nonlocal col
        col = col * (1 - mask[..., None]) + np.asarray(color, dtype=np.float64) * mask[..., None]

    ux, vt = place(x, y, np.clip(subj_x - 0.20, 0.08, 0.92), 0.40)
    tm, trunk_m, canopy_m = tree_mask(ux, vt)
    tree_paint = (np.clip(trunk_m, 0, 1) * tree)[..., None] * np.array([0.30, 0.20, 0.10]) + \
                 (np.clip(canopy_m, 0, 1) * tree)[..., None] * np.array([0.10, 0.30, 0.08])
    tree_alpha = np.clip(tm * tree, 0, 1)
    col = col * (1 - tree_alpha[..., None]) + tree_paint

    ub, vb = place(x, y, np.clip(subj_x + 0.32, 0.08, 0.92), 0.50)
    bm = building_mask(ub, vb)
    paint(np.clip(bm * building, 0, 1), (0.32, 0.30, 0.34))

    ubo, vbo = place_float(x, y, np.clip(subj_x - 0.30, 0.08, 0.92), 0.30, sink=-0.01)
    bom = boat_mask(ubo, vbo)
    paint(np.clip(bom * boat, 0, 1), (0.35, 0.22, 0.10))

    uc, vc = place(x, y, np.clip(subj_x + 0.18, 0.08, 0.92), 0.26)
    cm = car_mask(uc, vc)
    paint(np.clip(cm * car, 0, 1), (0.65, 0.10, 0.10))

    uh, vh = place(x, y, np.clip(subj_x + 0.05, 0.08, 0.92), 0.22)
    hm = horse_mask(uh, vh)
    paint(np.clip(hm * horse, 0, 1), (0.32, 0.20, 0.12))

    ud, vd = place(x, y, np.clip(subj_x - 0.10, 0.08, 0.92), 0.14)
    dm = dog_mask(ud, vd)
    paint(np.clip(dm * dog, 0, 1), (0.55, 0.42, 0.25))

    ubr, vbr = place_sky(x, y, np.clip(subj_x + 0.15, 0.1, 0.9), HORIZON * 0.35, 0.10)
    brm = bird_mask(ubr, vbr)
    paint(np.clip(brm * bird, 0, 1), (0.08, 0.08, 0.10))

    up, vp = place(x, y, subj_x, 0.30)
    pm = person_mask(up, vp)
    paint(np.clip(pm * person, 0, 1), (0.10, 0.10, 0.14))

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
    # subjects: each independently has a chance to appear; bias toward at
    # least one subject often so the net sees plenty of foreground examples.
    c[:, 12] = (rng.random(n) < 0.25) * rng.uniform(0.7, 1.0, n)  # person
    c[:, 13] = (rng.random(n) < 0.20) * rng.uniform(0.7, 1.0, n)  # horse
    c[:, 14] = (rng.random(n) < 0.25) * rng.uniform(0.7, 1.0, n)  # tree
    c[:, 15] = (rng.random(n) < 0.16) * rng.uniform(0.7, 1.0, n)  # building
    c[:, 16] = rng.uniform(0.0, 1.0, n)                            # subj_x
    c[:, 17] = (rng.random(n) < 0.18) * rng.uniform(0.7, 1.0, n)  # dog
    c[:, 18] = (rng.random(n) < 0.16) * rng.uniform(0.7, 1.0, n)  # car
    c[:, 19] = (rng.random(n) < 0.18) * rng.uniform(0.7, 1.0, n)  # bird
    c[:, 20] = (rng.random(n) < 0.14) * rng.uniform(0.7, 1.0, n)  # boat
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
    return np.concatenate([base, c], axis=-1)     # [...,3+4K+17]

DIM = 3 + 4 * K + CDIM
print("input dim DIM =", DIM)

# ----------------------------------------------------------------------------
# Tiny MLP: DIM -> H -> H -> H -> 3 with tanh hidden, sigmoid output.
# ----------------------------------------------------------------------------
H = 112
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
STEPS = 8000
NC = 96            # conditions per batch
NP = 192           # pixels per condition  -> batch = 18432
t0 = time.time()
for step in range(1, STEPS + 1):
    c = sample_conditions(NC)                              # [NC,17]
    px = rng.random((NC, NP)); py = rng.random((NC, NP))   # pixel coords
    cc = np.repeat(c[:, None, :], NP, axis=1)              # [NC,NP,17]
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
# Export weights, packed exactly the way index.html's #pp-model expects:
# float32 tensors concatenated in `order` and base64-encoded.
# ----------------------------------------------------------------------------
import base64
meta = {"DIM": DIM, "H": H, "K": K, "CDIM": CDIM, "HORIZON": HORIZON}
order = ["W0", "b0", "W1", "b1", "W2", "b2", "W3", "b3"]
shapes = {k: list(P[k].shape) for k in order}
buf = b"".join(P[k].astype(np.float32).ravel().tobytes() for k in order)
b64 = base64.b64encode(buf).decode("ascii")
with open("model_web.json", "w") as f:
    json.dump({"meta": meta, "order": order, "shapes": shapes, "b64": b64}, f)
print("saved model_web.json  (params:", sum(P[k].size for k in P), ")")

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
           "desert":6,"snow":7,"plain":8,"mountains":9,"neon":10,"fire":11,
           "person":12,"horse":13,"tree":14,"building":15,"subj_x":16,
           "dog":17,"car":18,"bird":19,"boat":20}
    for k, val in kw.items():
        c[idx[k]] = val
    if "subj_x" not in kw:
        c[16] = 0.5
    return c

previews = {
    "sunset_ocean":  C(sunset=1, ocean=1),
    "night_mountain":C(night=1, mountains=1, plain=1),
    "day_forest":    C(day=1, forest=1, tree=1),
    "desert_day":    C(day=1, desert=1),
    "person_plain":  C(day=1, plain=1, person=1),
    "horse_pasture": C(day=1, plain=1, horse=1),
    "village_day":   C(day=1, plain=1, building=1, tree=0.6),
    "person_horse_sunset": C(sunset=1, plain=1, person=1, horse=1, subj_x=0.4),
    "dog_park":      C(day=1, plain=1, dog=1, tree=0.6),
    "road_car":      C(day=1, desert=1, car=1),
    "sailing_boat":  C(day=1, ocean=1, boat=1, bird=0.8),
    "birds_at_sunset": C(sunset=1, ocean=1, bird=1),
}
for name, c in previews.items():
    write_png(f"preview_{name}.png", net_render(c, 160))
print("wrote previews:", list(previews))
