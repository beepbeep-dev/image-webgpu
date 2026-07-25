"""
Full-body companion to align_faces.py: detects whole standing/walking
people (not just faces) with OpenCV's classical HOG + default people SVM
detector (the standard non-deep-learning pedestrian detector), crops a
square region around the detected person with enough margin to keep scene
context (beach, street, etc.) visible, and inserts each as a new row tagged
"person_aligned:". Unlike align_faces.py, the ORIGINAL cleaned caption is
kept (not templated) since the scene context in the caption ("on the
beach", "in the city") is exactly the signal we want the model to learn
alongside a consistently-scaled human figure.
"""
import hashlib
import io
import sqlite3
import time

import cv2
import numpy as np
from PIL import Image

from align_faces import PERSON_WORDS, crop_around_face

HOG = cv2.HOGDescriptor()
HOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
MARGIN = 1.6
MIN_PERSON = 60


def biggest_person(img_bgr):
    rects, weights = HOG.detectMultiScale(img_bgr, winStride=(4, 4), padding=(8, 8), scale=1.05)
    if len(rects) == 0:
        return None
    best = None
    best_w = -1e9
    for (x, y, w, h), wt in zip(rects, weights):
        if min(w, h) < MIN_PERSON:
            continue
        if wt > best_w:
            best_w, best = wt, (x, y, w, h)
    return best


def stable_id(native_id):
    h = hashlib.md5(f"person_aligned:{native_id}".encode()).hexdigest()
    return int(h[:15], 16)


def main():
    con = sqlite3.connect("dataset_hq.db", timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")
    cur = con.cursor()
    all_rows = cur.execute(
        "SELECT page_id, query, caption, image FROM samples "
        "WHERE query NOT LIKE 'portrait_aligned:%' AND query NOT LIKE 'person_aligned:%'"
    ).fetchall()
    rows = [r for r in all_rows if PERSON_WORDS.search(r[2] or "") or PERSON_WORDS.search(r[1] or "")]
    print(f"scanning {len(rows)} person-likely images (of {len(all_rows)} total) for full bodies", flush=True)

    saved = 0
    t0 = time.time()
    for i, (pid, q, cap, blob) in enumerate(rows):
        try:
            img = Image.open(io.BytesIO(blob)).convert("RGB")
        except Exception:
            continue
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        person = biggest_person(bgr)
        if person is None:
            continue
        crop = crop_around_face(img, person, margin=MARGIN).resize((256, 256), Image.LANCZOS)
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=92)
        new_pid = stable_id(pid)
        try:
            cur.execute(
                "INSERT OR IGNORE INTO samples (page_id,query,title,caption,artist,license,"
                "source_url,width,height,image) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (new_pid, f"person_aligned:{q}", "", cap, "", "", "", 256, 256, buf.getvalue()))
            if cur.rowcount:
                saved += 1
        except sqlite3.IntegrityError:
            pass
        if (i + 1) % 500 == 0:
            con.commit()
            print(f"[{i+1}/{len(rows)}] aligned bodies found so far: {saved}  ({time.time()-t0:.0f}s)", flush=True)
    con.commit()
    print("done:", saved, "aligned bodies in", round(time.time() - t0, 1), "s", flush=True)
    con.close()


if __name__ == "__main__":
    main()
