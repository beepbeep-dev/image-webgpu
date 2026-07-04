"""
Add a SECOND, THIRD, FOURTH, and FIFTH source to the training database,
on top of Wikimedia Commons (scrape_dataset_hq.py) — depending on exactly
one API source meant every image, license, and caption style came from the
same place. All four new sources are queryable with NO signup/API key and
are legally clean for training a derivative model:

  - Openverse       api.openverse.org — aggregates Flickr/museums/etc,
                    filtered here to permissive licenses only (cc0, pdm,
                    by, by-sa — explicitly excluding nc/nd variants)
  - NASA Images     images-api.nasa.gov — all US-government-work public domain
  - Library of Congress  loc.gov/photos — public-domain/no-known-restrictions
                    items only
  - Met Museum      collectionapi.metmuseum.org — Open Access, filtered to
                    isPublicDomain=true AND medium=Photographs (excludes
                    paintings/sculpture — we want photos, same reasoning as
                    scrape_dataset.py's looks_like_photo filter)

Writes into the SAME dataset_hq.db / samples table scrape_dataset_hq.py
uses. page_id is a stable hash of (source, native_id) so it can't collide
with Wikimedia Commons pageids or across the four new sources.
"""
import hashlib
import html
import io
import re
import sqlite3
import sys
import time
import urllib.parse
import urllib.request

from PIL import Image
from scrape_dataset import UA, clean_caption, looks_like_photo, center_crop_square

OUT_SIZE = 256
TIMEOUT = 20


def _get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        import json
        return json.loads(r.read().decode("utf-8", errors="replace"))


def _get_bytes(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.read()
    except Exception:
        return None


def _stable_id(source, native_id):
    h = hashlib.md5(f"{source}:{native_id}".encode()).hexdigest()
    return int(h[:15], 16)


def _process_and_check(raw_bytes, min_size=OUT_SIZE, max_ar=3.2):
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img.draft("RGB", (OUT_SIZE, OUT_SIZE))
        img = img.convert("RGB")
    except Exception:
        return None
    w, h = img.size
    if min(w, h) < min_size:
        return None
    ar = w / max(1, h)
    if ar > max_ar or ar < 1 / max_ar:
        return None
    img = center_crop_square(img).resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Openverse
# ---------------------------------------------------------------------------
def fetch_openverse(query, pages=3, page_size=20):
    # Anonymous (no API key) requests are capped at page_size=20 — anything
    # larger gets a 401, not a clean error, so this isn't a soft limit.
    out = []
    for page in range(1, pages + 1):
        q = urllib.parse.quote(query)
        url = (f"https://api.openverse.org/v1/images/?q={q}&license=cc0,pdm,by,by-sa"
               f"&page_size={page_size}&page={page}&mature=false")
        try:
            data = _get_json(url)
        except Exception:
            break
        results = data.get("results", [])
        if not results:
            break
        for r in results:
            title = r.get("title", "") or ""
            caption = clean_caption(title, "")
            if len(caption) < 3 or not looks_like_photo(caption, ""):
                continue
            img_url = r.get("url")
            if not img_url:
                continue
            raw = _get_bytes(img_url)
            if raw is None:
                continue
            jpg = _process_and_check(raw)
            if jpg is None:
                continue
            out.append({
                "native_id": r.get("id"), "title": title, "caption": caption,
                "artist": (r.get("creator") or "")[:120], "license": r.get("license", "unknown"),
                "source_url": r.get("foreign_landing_url", ""), "image": jpg,
            })
        time.sleep(0.2)
    return out


# ---------------------------------------------------------------------------
# NASA Images API
# ---------------------------------------------------------------------------
def fetch_nasa(query, pages=1, page_size=50):
    out = []
    for page in range(1, pages + 1):
        q = urllib.parse.quote(query)
        url = f"https://images-api.nasa.gov/search?q={q}&media_type=image&page={page}"
        try:
            data = _get_json(url)
        except Exception:
            break
        items = data.get("collection", {}).get("items", [])
        if not items:
            break
        for it in items[:page_size]:
            meta = it["data"][0]
            title = meta.get("title", "") or ""
            desc = meta.get("description", "") or ""
            caption = clean_caption(title, desc)
            if len(caption) < 3 or not looks_like_photo(caption, ""):
                continue
            links = it.get("links", [])
            if not links:
                continue
            raw = _get_bytes(links[0]["href"])
            if raw is None:
                continue
            jpg = _process_and_check(raw)
            if jpg is None:
                continue
            out.append({
                "native_id": meta.get("nasa_id"), "title": title, "caption": caption,
                "artist": "NASA", "license": "public domain (NASA)",
                "source_url": f"https://images.nasa.gov/details-{meta.get('nasa_id')}", "image": jpg,
            })
    return out


# ---------------------------------------------------------------------------
# Library of Congress
# ---------------------------------------------------------------------------
def fetch_loc(query, pages=1):
    out = []
    for page in range(1, pages + 1):
        q = urllib.parse.quote(query)
        url = f"https://www.loc.gov/photos/?q={q}&fo=json&c=40&sp={page}"
        try:
            data = _get_json(url)
        except Exception:
            break
        results = data.get("results", [])
        if not results:
            break
        for r in results:
            if r.get("access_restricted"):
                continue
            rights = (r.get("rights_advisory") or "")
            if rights:   # any advisory text present = not clean public domain
                continue
            title = r.get("title", "") or ""
            caption = clean_caption(title, "")
            if len(caption) < 3 or not looks_like_photo(caption, ""):
                continue
            urls = r.get("image_url") or []
            # prefer the largest-looking variant (LoC lists small->large or mixed order)
            best = max(urls, key=lambda u: len(u), default=None) if urls else None
            if not best:
                continue
            best = best.split("#")[0]
            raw = _get_bytes(best)
            if raw is None:
                continue
            jpg = _process_and_check(raw, min_size=200)   # LoC thumbs run smaller
            if jpg is None:
                continue
            out.append({
                "native_id": r.get("id") or best, "title": title, "caption": caption,
                "artist": "", "license": "public domain / no known restrictions (LoC)",
                "source_url": r.get("id", ""), "image": jpg,
            })
        time.sleep(0.2)
    return out


# ---------------------------------------------------------------------------
# Met Museum Open Access
# ---------------------------------------------------------------------------
def _met_one(oid):
    try:
        obj = _get_json(f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{oid}")
    except Exception:
        return None
    if not obj.get("isPublicDomain") or not obj.get("primaryImage"):
        return None
    title = obj.get("title", "") or ""
    desc = " ".join(filter(None, [obj.get("culture", ""), obj.get("period", "")]))
    caption = clean_caption(title, desc)
    if len(caption) < 3:
        return None
    raw = _get_bytes(obj["primaryImage"])
    if raw is None:
        return None
    jpg = _process_and_check(raw)
    if jpg is None:
        return None
    return {
        "native_id": oid, "title": title, "caption": caption,
        "artist": (obj.get("artistDisplayName") or "")[:120],
        "license": "public domain (Met Open Access)",
        "source_url": obj.get("objectURL", ""), "image": jpg,
    }


def fetch_met(query, max_items=30):
    from concurrent.futures import ThreadPoolExecutor
    q = urllib.parse.quote(query)
    url = f"https://collectionapi.metmuseum.org/public/collection/v1/search?q={q}&hasImages=true&medium=Photographs"
    try:
        data = _get_json(url)
    except Exception:
        return []
    ids = (data.get("objectIDs") or [])[:max_items]
    # Each object needs its own detail request (no batch endpoint) — this is
    # network-latency-bound, not CPU-bound, so a thread pool overlaps the
    # round-trips instead of paying for them serially.
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_met_one, ids))
    return [r for r in results if r is not None]


SOURCES = {
    "openverse": fetch_openverse,
    "nasa": fetch_nasa,
    "loc": fetch_loc,
    "met": fetch_met,
}


def insert_results(con, source, query, results):
    cur = con.cursor()
    saved = 0
    for r in results:
        pid = _stable_id(source, r["native_id"])
        try:
            cur.execute(
                "INSERT OR IGNORE INTO samples (page_id,query,title,caption,artist,license,"
                "source_url,width,height,image) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (pid, f"{source}:{query}", r["title"], r["caption"], r["artist"], r["license"],
                 r["source_url"], OUT_SIZE, OUT_SIZE, r["image"]))
            if cur.rowcount:
                saved += 1
        except sqlite3.IntegrityError:
            pass
    con.commit()
    return saved


def main():
    queries = sys.argv[2:] if len(sys.argv) > 2 else None
    source_name = sys.argv[1] if len(sys.argv) > 1 else None
    if source_name not in SOURCES:
        print("usage: scrape_multi_source.py <openverse|nasa|loc|met> [query ...]")
        sys.exit(1)
    if not queries:
        # scrape_dataset_hq's QUERIES includes the 43 person-focused queries
        # added on top of the base 485 -- use that (528 total), not the base
        # list, so the new sources cover people too.
        from scrape_dataset_hq import QUERIES
        queries = QUERIES
    # busy_timeout + WAL: multiple source scripts run concurrently against
    # the same dataset_hq.db, and SQLite's default rollback-journal mode
    # errors immediately ("database is locked") under concurrent writers
    # instead of waiting.
    con = sqlite3.connect("dataset_hq.db", timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")
    fetch = SOURCES[source_name]
    total = 0
    t0 = time.time()
    for i, q in enumerate(queries):
        try:
            results = fetch(q)
        except Exception as e:
            print(f"[{i+1}/{len(queries)}] {q!r}: ERROR {e}")
            continue
        saved = insert_results(con, source_name, q, results)
        total += saved
        print(f"[{i+1}/{len(queries)}] {q!r}: +{saved} (total {total})  ({time.time()-t0:.0f}s)")
    con.close()
    print(f"done: +{total} rows from {source_name} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
