"""
Build a real, structured SQLite training dataset for PocketPaint's own
diffusion model, covering the FULL combinatorial space of the 21-D scene
condition vector (see scene_diffusion.py): every time-of-day x biome x
mountain/neon/fire intensity x subject x placement combination.

This replaces "infinite random on-the-fly sampling" with a fixed, inspectable,
queryable dataset: model/dataset.db. Each row is one rendered scene plus the
exact condition vector and structured metadata columns (time_of_day, biome,
subject, ...) so you can browse/query it (e.g. "every night desert scene with
a horse") rather than only ever seeing it as opaque training tensors.

Usage:  python3 build_dataset.py [--size 32] [--out dataset.db]
"""
import argparse, io, sqlite3, time
import numpy as np
from PIL import Image
import scene_diffusion as scene

TIME_NAMES = ["night", "sunset", "day", "overcast"]      # condition idx 0-3
BIOME_NAMES = ["ocean", "forest", "desert", "snow", "plain"]  # condition idx 4-8
SUBJECT_NAMES = ["none", "person", "horse", "tree", "building", "dog", "car", "bird", "boat"]
SUBJECT_IDX = {"person": 12, "horse": 13, "tree": 14, "building": 15,
               "dog": 17, "car": 18, "bird": 19, "boat": 20}
INTENSITIES = [0.0, 0.5, 0.9]   # off / medium / strong, for mountain & neon & fire
PLACEMENTS = [0.2, 0.5, 0.8]    # horizontal subject position when a subject is present


def build_conditions():
    """Enumerate every structural combination -> list of (meta_dict, condition[21])."""
    rows = []
    for ti, tname in enumerate(TIME_NAMES):
        for bi, bname in enumerate(BIOME_NAMES):
            for mountain in INTENSITIES:
                for neon in INTENSITIES:
                    for fire in INTENSITIES:
                        for subject in SUBJECT_NAMES:
                            placements = PLACEMENTS if subject != "none" else [0.5]
                            for subj_x in placements:
                                c = np.zeros(scene.CDIM, dtype=np.float32)
                                c[ti] = 1.0
                                c[4 + bi] = 1.0
                                c[9] = mountain
                                c[10] = neon
                                c[11] = fire
                                c[16] = subj_x
                                if subject != "none":
                                    c[SUBJECT_IDX[subject]] = 1.0
                                meta = dict(time_of_day=tname, biome=bname, mountain=mountain,
                                            neon=neon, fire=fire, subject=subject, subj_x=subj_x)
                                rows.append((meta, c))
    return rows


def render_image(c, S):
    ys, xs = np.meshgrid(np.linspace(0, 1, S), np.linspace(0, 1, S), indexing="ij")
    img = scene.render(xs, ys, np.broadcast_to(c, (S, S, scene.CDIM)))
    return np.clip(img * 255, 0, 255).astype(np.uint8)   # [S,S,3] uint8


def png_bytes(img_u8):
    buf = io.BytesIO()
    Image.fromarray(img_u8, mode="RGB").save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--out", default="dataset.db")
    args = ap.parse_args()

    rows = build_conditions()
    print(f"enumerated {len(rows)} structural scene combinations "
          f"({len(TIME_NAMES)} time-of-day x {len(BIOME_NAMES)} biomes x "
          f"{len(INTENSITIES)}^3 mountain/neon/fire x {len(SUBJECT_NAMES)} subjects x placement)")

    con = sqlite3.connect(args.out)
    cur = con.cursor()
    cur.executescript("""
        DROP TABLE IF EXISTS samples;
        CREATE TABLE samples (
            id          INTEGER PRIMARY KEY,
            time_of_day TEXT NOT NULL,
            biome       TEXT NOT NULL,
            mountain    REAL NOT NULL,
            neon        REAL NOT NULL,
            fire        REAL NOT NULL,
            subject     TEXT NOT NULL,
            subj_x      REAL NOT NULL,
            width       INTEGER NOT NULL,
            height      INTEGER NOT NULL,
            condition   BLOB NOT NULL,   -- 21 x float32, little-endian
            image       BLOB NOT NULL    -- PNG-encoded RGB
        );
        CREATE INDEX idx_time    ON samples(time_of_day);
        CREATE INDEX idx_biome   ON samples(biome);
        CREATE INDEX idx_subject ON samples(subject);
    """)

    t0 = time.time()
    batch = []
    for i, (meta, c) in enumerate(rows):
        img = render_image(c, args.size)
        batch.append((meta["time_of_day"], meta["biome"], meta["mountain"], meta["neon"],
                       meta["fire"], meta["subject"], meta["subj_x"], args.size, args.size,
                       c.tobytes(), png_bytes(img)))
        if len(batch) >= 500:
            cur.executemany(
                "INSERT INTO samples (time_of_day,biome,mountain,neon,fire,subject,subj_x,"
                "width,height,condition,image) VALUES (?,?,?,?,?,?,?,?,?,?,?)", batch)
            con.commit()
            batch.clear()
            print(f"  {i+1}/{len(rows)}  ({time.time()-t0:.1f}s)")
    if batch:
        cur.executemany(
            "INSERT INTO samples (time_of_day,biome,mountain,neon,fire,subject,subj_x,"
            "width,height,condition,image) VALUES (?,?,?,?,?,?,?,?,?,?,?)", batch)
        con.commit()

    n = cur.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    size_mb = cur.execute("SELECT SUM(LENGTH(image)) FROM samples").fetchone()[0] / 1e6
    print(f"done: {n} rows, {size_mb:.1f}MB of PNGs, in {time.time()-t0:.1f}s -> {args.out}")

    print("\nrows per time_of_day:")
    for r in cur.execute("SELECT time_of_day, COUNT(*) FROM samples GROUP BY time_of_day"):
        print(" ", r)
    print("rows per subject:")
    for r in cur.execute("SELECT subject, COUNT(*) FROM samples GROUP BY subject"):
        print(" ", r)
    con.close()


if __name__ == "__main__":
    main()
