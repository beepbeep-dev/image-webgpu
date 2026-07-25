"""
Face-alignment pass over the existing HQ dataset.

The core problem with "make it draw a person": faces in the raw dataset
appear at wildly different scale/position/framing, so a 16M-param model
never sees enough *consistent* facial structure to converge on what a face
looks like (this is the same reason aligned datasets like CelebA-HQ/FFHQ
were such a big deal for small GANs — alignment removes nuisance variance
so the network can spend its capacity on the actual signal).

This scans every photo already in dataset_hq.db with a classical (not
learned-embedding) OpenCV Haar cascade face detector, crops tightly around
the largest detected face with head-and-shoulders margin, resizes to
256x256, and inserts each as a NEW row tagged with a "portrait_aligned:"
query prefix and a caption built to always contain "portrait"/"face" so the
text encoder's vocab picks it up strongly. These aligned rows get
oversampled heavily during training (see train_diffusion_own.py).
"""
import hashlib
import io
import re
import sqlite3
import sys
import time

import cv2
import numpy as np
from PIL import Image

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
MARGIN = 2.2
MIN_FACE = 40

# Haar face cascades throw plenty of false positives on non-face textures
# (arches, cloud formations, flower petals all matched in an earlier run) —
# restricting the scan to captions that plausibly contain a person, and then
# requiring a detected eye pair *inside* the candidate face box, cuts that
# false-positive rate down to something usable.
PERSON_WORDS = re.compile(
    r"\b(person|people|man|men|woman|women|boy|girl|child|children|kid|kids|"
    r"portrait|face|smil\w*|musician|dancer|athlete|player|runner|cyclist|"
    r"hiker|skier|swimmer|surfer|climber|farmer|chef|artist|teacher|worker|"
    r"vendor|photographer|crowd|festival|couple|family|tourist|selfie|actor|"
    r"actress|model|bride|groom|baby|elderly|senior|student|soldier|nurse|"
    r"doctor|cook|driver|pilot|singer|guitarist|drummer)\b", re.I)


def biggest_face(gray):
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=7, minSize=(MIN_FACE, MIN_FACE))
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    for f in faces:
        x, y, w, h = f
        face_roi = gray[y:y + h, x:x + w]
        eyes = EYE_CASCADE.detectMultiScale(face_roi, scaleFactor=1.05, minNeighbors=4, minSize=(w // 10, h // 10))
        if len(eyes) >= 1:
            return f
    return None


def crop_around_face(img, face, margin=MARGIN):
    x, y, w, h = face
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * margin
    W, H = img.size
    half = side / 2.0
    left, top, right, bottom = cx - half, cy - half, cx + half, cy + half
    # shift the box back inside bounds rather than shrinking it, so the
    # face stays centered-ish even near an image edge
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > W:
        left -= (right - W)
        right = W
    if bottom > H:
        top -= (bottom - H)
        bottom = H
    left, top = max(0, left), max(0, top)
    right, bottom = min(W, right), min(H, bottom)
    return img.crop((int(left), int(top), int(right), int(bottom)))


GENDER_WOMAN = re.compile(r"\b(woman|women|female|girl|lady|actress|mother|wife|daughter)\b", re.I)
GENDER_MAN = re.compile(r"\b(man|men|male|boy|guy|gentleman|actor|father|husband|son)\b", re.I)


def gender_hint(caption):
    if GENDER_WOMAN.search(caption or ""):
        return "woman"
    if GENDER_MAN.search(caption or ""):
        return "man"
    return "person"


def stable_id(native_id):
    h = hashlib.md5(f"portrait_aligned:{native_id}".encode()).hexdigest()
    return int(h[:15], 16)


def main():
    con = sqlite3.connect("dataset_hq.db", timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")
    cur = con.cursor()
    all_rows = cur.execute(
        "SELECT page_id, query, caption, image FROM samples WHERE query NOT LIKE 'portrait_aligned:%'"
    ).fetchall()
    rows = [r for r in all_rows if PERSON_WORDS.search(r[2] or "") or PERSON_WORDS.search(r[1] or "")]
    print(f"scanning {len(rows)} person-likely images (of {len(all_rows)} total) for faces", flush=True)
    saved = 0
    t0 = time.time()
    for i, (pid, q, cap, blob) in enumerate(rows):
        try:
            img = Image.open(io.BytesIO(blob)).convert("RGB")
        except Exception:
            continue
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        face = biggest_face(gray)
        if face is None:
            continue
        crop = crop_around_face(img, face).resize((256, 256), Image.LANCZOS)
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=92)
        gh = gender_hint(cap)
        new_caption = f"portrait of a {gh} face closeup"
        new_pid = stable_id(pid)
        try:
            cur.execute(
                "INSERT OR IGNORE INTO samples (page_id,query,title,caption,artist,license,"
                "source_url,width,height,image) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (new_pid, f"portrait_aligned:{q}", "", new_caption, "", "", "", 256, 256, buf.getvalue()))
            if cur.rowcount:
                saved += 1
        except sqlite3.IntegrityError:
            pass
        if (i + 1) % 1000 == 0:
            con.commit()
            print(f"[{i+1}/{len(rows)}] aligned faces found so far: {saved}  ({time.time()-t0:.0f}s)", flush=True)
    con.commit()
    print("done:", saved, "aligned faces in", round(time.time() - t0, 1), "s", flush=True)
    con.close()


if __name__ == "__main__":
    main()
