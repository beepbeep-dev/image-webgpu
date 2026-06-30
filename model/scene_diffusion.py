"""
Analytic scene+subject renderer used as the DIFFUSION model's training target.
Produces SxSx3 images in [0,1] from a 21-D condition vector (same semantic
layout as index.html's promptToCondition(), so the same prompt mapping can
drive this engine too):
  0 night 1 sunset 2 day 3 overcast | 4 ocean 5 forest 6 desert 7 snow 8 plain
  | 9 mountains 10 neon 11 fire
  | 12 person 13 horse 14 tree 15 building 16 subj_x (placement, 0..1)
  | 17 dog 18 car 19 bird 20 boat

This is intentionally simple/blocky (silhouettes built from circles & rects) —
the point is to give a small conv UNet enough local structure to learn real
denoising, not photorealism.
"""
import numpy as np

CDIM = 21
HORIZON = 0.58


def _mix(a, b, t):
    return a + (b - a) * t


def render(x, y, c):
    """x,y: [...,] in [0,1] (y=0 top). c: [...,21]. Returns [...,3] in [0,1]."""
    x = np.asarray(x); y = np.asarray(y); c = np.asarray(c)
    night, sunset, day, over = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    ocean, forest, desert, snow, plain = c[..., 4], c[..., 5], c[..., 6], c[..., 7], c[..., 8]
    mount, neon, fire = c[..., 9], c[..., 10], c[..., 11]
    person, horse, tree, building, subj_x = c[..., 12], c[..., 13], c[..., 14], c[..., 15], c[..., 16]
    dog, car, bird, boat = c[..., 17], c[..., 18], c[..., 19], c[..., 20]

    sky_w = night + sunset + day + over + 1e-3
    ts = np.clip(y / HORIZON, 0, 1)

    def grad(top, hor):
        return np.stack([_mix(top[i], hor[i], ts) for i in range(3)], axis=-1)

    night_c = grad((0.03, 0.04, 0.12), (0.12, 0.10, 0.22))
    sunset_c = grad((0.17, 0.15, 0.42), (1.00, 0.50, 0.22))
    day_c = grad((0.20, 0.45, 0.88), (0.70, 0.86, 0.98))
    over_c = grad((0.55, 0.57, 0.61), (0.78, 0.79, 0.80))
    sky = (night_c * night[..., None] + sunset_c * sunset[..., None] +
           day_c * day[..., None] + over_c * over[..., None]) / sky_w[..., None]

    def glow(cx, cy, rad, col, strength):
        d2 = (x - cx) ** 2 + (y - cy) ** 2
        g = np.exp(-d2 / (rad * rad))
        return np.stack([col[i] * g * strength for i in range(3)], axis=-1)

    sun = glow(0.50, 0.46, 0.16, (1.0, 0.80, 0.45), sunset)
    sun = sun + glow(0.32, 0.22, 0.13, (1.0, 0.95, 0.80), day)
    moon = glow(0.70, 0.22, 0.09, (0.92, 0.94, 1.0), night)
    sky = np.clip(sky + sun + moon, 0, 1)

    ridge = (0.5 + 0.30 * np.sin(x * 7.0 + 1.3) + 0.18 * np.sin(x * 17.0 + 4.0))
    mh = HORIZON - 0.20 * ridge
    m_mask = np.clip((HORIZON - y) / 0.02, 0, 1) * np.clip((y - mh) / 0.02 + 0.5, 0, 1)
    m_mask = np.clip(m_mask, 0, 1) * (y < HORIZON)
    sky = _mix(sky, np.broadcast_to(np.array([0.15, 0.14, 0.20]), sky.shape), (m_mask * mount)[..., None])

    tg = np.clip((y - HORIZON) / (1 - HORIZON), 0, 1)

    def gground(top, bot):
        return np.stack([_mix(top[i], bot[i], tg) for i in range(3)], axis=-1)

    wave = 0.05 * np.sin(y * 60.0) * np.clip(1 - tg, 0, 1)
    ocean_c = gground((0.06, 0.28, 0.52), (0.02, 0.10, 0.28)) + wave[..., None]
    forest_c = gground((0.12, 0.36, 0.14), (0.04, 0.16, 0.06))
    desert_c = gground((0.86, 0.72, 0.46), (0.60, 0.44, 0.25))
    snow_c = gground((0.90, 0.93, 0.98), (0.68, 0.76, 0.90))
    plain_c = gground((0.36, 0.56, 0.26), (0.17, 0.30, 0.12))
    gnd_w = ocean + forest + desert + snow + plain + 1e-3
    gnd = (ocean_c * ocean[..., None] + forest_c * forest[..., None] +
           desert_c * desert[..., None] + snow_c * snow[..., None] +
           plain_c * plain[..., None]) / gnd_w[..., None]
    gnd = np.clip(gnd, 0, 1)

    is_ground = (y >= HORIZON)[..., None].astype(np.float64)
    col = sky * (1 - is_ground) + gnd * is_ground

    fire_glow = np.clip(1 - np.abs(y - HORIZON) / 0.35, 0, 1)
    col = col + (np.array([0.9, 0.35, 0.08]) * (fire_glow * fire)[..., None]) * 0.6
    col = np.clip(col, 0, 1)

    lum = (col[..., 0] * 0.3 + col[..., 1] * 0.59 + col[..., 2] * 0.11)
    neon_grade = np.stack([0.6 + 0.4 * np.sin(lum * 6.0), 0.2 + 0.3 * np.sin(lum * 6.0 + 2.0),
                            0.7 + 0.3 * np.sin(lum * 6.0 + 4.0)], axis=-1)
    horizon_glow = np.clip(1 - np.abs(y - HORIZON) / 0.12, 0, 1)
    neon_grade = np.clip(neon_grade + np.array([0.7, 0.1, 0.9]) * horizon_glow[..., None] * 0.5, 0, 1)
    col = _mix(col, neon_grade, (neon * 0.85)[..., None])

    # ---- subjects: simple silhouettes placed on/above the horizon ----
    def ellipse(cx, cy, rx, ry):
        return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2

    def blend_shape(col, mask, color, soft=0.12):
        a = np.clip(1 - (mask - 1) / soft, 0, 1) * (mask < 1 + soft)
        return _mix(col, np.broadcast_to(np.array(color), col.shape), a[..., None])

    sx = 0.2 + subj_x * 0.6  # placement x in [0.2,0.8]

    # tree: trunk + canopy, base sits on horizon
    trunk = ellipse(sx, HORIZON - 0.02, 0.012, 0.07)
    canopy = ellipse(sx, HORIZON - 0.13, 0.075, 0.075)
    col = blend_shape(col, trunk / np.maximum(tree, 1e-3) + (1 - tree) * 9, (0.30, 0.18, 0.10))
    col = blend_shape(col, canopy / np.maximum(tree, 1e-3) + (1 - tree) * 9, (0.10, 0.32, 0.10))

    # building: rectangle block with a couple of window dots
    bw, bh = 0.16, 0.20
    bx0, bx1 = sx - bw / 2, sx + bw / 2
    by0, by1 = HORIZON - bh, HORIZON
    in_box = np.maximum(np.maximum(bx0 - x, x - bx1), np.maximum(by0 - y, y - by1)) / 0.01
    win = (np.minimum(((x - sx + 0.04) % 0.06 - 0.03) ** 2, ((x - sx - 0.04) % 0.06 - 0.03) ** 2) +
           ((y - (HORIZON - bh * 0.6)) % 0.05 - 0.025) ** 2)
    bcol = np.where(win < 0.0006, 0.95, 0.42)
    bldg_col = np.stack([bcol * 0.55, bcol * 0.55, bcol * 0.6], axis=-1)
    col = _mix(col, bldg_col, (np.clip(1 - in_box, 0, 1) * building)[..., None])

    # person: head circle + body capsule, standing on horizon
    head = ellipse(sx, HORIZON - 0.20, 0.022, 0.022)
    body = ellipse(sx, HORIZON - 0.09, 0.028, 0.10)
    pmask = np.minimum(head, body)
    col = blend_shape(col, pmask / np.maximum(person, 1e-3) + (1 - person) * 9, (0.85, 0.55, 0.35), 0.10)

    # dog: small low body + head + legs (approx as a squat ellipse)
    dbody = ellipse(sx, HORIZON - 0.035, 0.06, 0.028)
    dhead = ellipse(sx + 0.05, HORIZON - 0.05, 0.022, 0.022)
    dmask = np.minimum(dbody, dhead)
    col = blend_shape(col, dmask / np.maximum(dog, 1e-3) + (1 - dog) * 9, (0.55, 0.38, 0.20), 0.10)

    # horse: bigger body + neck/head + legs (approx ellipse body)
    hbody = ellipse(sx, HORIZON - 0.06, 0.11, 0.045)
    hhead = ellipse(sx + 0.11, HORIZON - 0.10, 0.03, 0.05)
    hmask = np.minimum(hbody, hhead)
    col = blend_shape(col, hmask / np.maximum(horse, 1e-3) + (1 - horse) * 9, (0.40, 0.25, 0.14), 0.10)

    # car: body rectangle + 2 wheel circles, sits on horizon
    cw, ch = 0.16, 0.045
    cx0, cx1 = sx - cw / 2, sx + cw / 2
    cy0, cy1 = HORIZON - ch, HORIZON
    car_box = np.maximum(np.maximum(cx0 - x, x - cx1), np.maximum(cy0 - y, y - cy1)) / 0.01
    wheelL = ellipse(sx - cw * 0.28, HORIZON, 0.018, 0.018)
    wheelR = ellipse(sx + cw * 0.28, HORIZON, 0.018, 0.018)
    car_mask = np.minimum(np.clip(1 - car_box, 0, 1) * 9, np.minimum(wheelL, wheelR) * 0 + 9)
    car_mask = np.maximum(np.clip(1 - car_box, 0, 1), np.clip(1 - wheelL, 0, 1))
    car_mask = np.maximum(car_mask, np.clip(1 - wheelR, 0, 1))
    car_col = np.array([0.75, 0.15, 0.15])
    col = _mix(col, np.broadcast_to(car_col, col.shape), (car_mask * car)[..., None])

    # bird: small V-shape in the sky (two short angled marks) -> approximate as two thin ellipses
    by = HORIZON - 0.30 - 0.08 * np.sin(sx * 31.0)
    bird1 = ellipse(sx - 0.02, by, 0.022, 0.006)
    bird2 = ellipse(sx + 0.02, by, 0.022, 0.006)
    bmask = np.minimum(bird1, bird2)
    col = blend_shape(col, bmask / np.maximum(bird, 1e-3) + (1 - bird) * 9, (0.08, 0.08, 0.10), 0.08)

    # boat: triangular hull sitting on the horizon + mast
    hull = ellipse(sx, HORIZON + 0.02, 0.09, 0.022)
    mast = ellipse(sx, HORIZON - 0.05, 0.006, 0.05)
    boatmask = np.minimum(hull, mast)
    col = blend_shape(col, boatmask / np.maximum(boat, 1e-3) + (1 - boat) * 9, (0.45, 0.30, 0.15), 0.10)

    return np.clip(col, 0, 1)


def sample_conditions(rng, n):
    c = np.zeros((n, CDIM))
    t = rng.integers(0, 4, n)
    c[np.arange(n), t] = 1.0
    b = rng.integers(4, 9, n)
    c[np.arange(n), b] = 1.0
    c[:, 9] = (rng.random(n) < 0.30) * rng.uniform(0.4, 1.0, n)
    c[:, 10] = (rng.random(n) < 0.15) * rng.uniform(0.5, 1.0, n)
    c[:, 11] = (rng.random(n) < 0.15) * rng.uniform(0.4, 1.0, n)
    # exactly one (or zero, 25% of the time) subject per scene, for clean supervision
    subj_idx = [12, 13, 17, 18, 19, 20, 14, 15]  # person horse dog car bird boat tree building
    has_subj = rng.random(n) > 0.20
    pick = rng.integers(0, len(subj_idx), n)
    for i in range(n):
        if has_subj[i]:
            c[i, subj_idx[pick[i]]] = 1.0
    c[:, 16] = rng.uniform(0.15, 0.85, n)  # subj_x placement
    if not None:
        pass
    return c
