"""
Build the larger, higher-resolution training database for the "Diffusion HQ
(ours)" model: 256x256 real photos + captions from Wikimedia Commons, using
the SAME 485 subject queries as scrape_dataset.py but paginating deeper into
each query's search results (via gsroffset) to reach ~100k images instead of
~90 per query. Reuses all the filtering/cleaning/licensing logic from
scrape_dataset.py so both datasets are curated the same way.
"""
import io, sys, threading, time, sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from scrape_dataset import (
    QUERIES, UA, api_get, fetch_bytes, clean_caption, looks_like_photo,
    center_crop_square,
)
import html, re

MAX_AR = 3.2
THUMB_W = 850   # Wikimedia scales thumbs to this WIDTH; for a landscape photo at
                # the widest aspect ratio we allow (MAX_AR), the resulting HEIGHT
                # must still be >= OUT_SIZE, so this needs to be >= OUT_SIZE*MAX_AR
OUT_SIZE = 256
PAGES_PER_QUERY = 5     # up to 5 * 90 = 450 raw results considered per query
PAGE_SIZE = 90


def process_page(p, seen_pageids):
    pid = p.get("pageid")
    if pid in seen_pageids:
        return None
    info = p.get("imageinfo")
    if not info:
        return None
    info = info[0]
    thumb_url = info.get("thumburl")
    if not thumb_url or not re.search(r"\.(jpe?g|png)$", thumb_url, re.I):
        return None
    em = info.get("extmetadata", {})
    license_short = em.get("LicenseShortName", {}).get("value", "unknown")
    artist_raw = em.get("Artist", {}).get("value", "")
    artist = re.sub(r"<[^>]+>", "", html.unescape(artist_raw)).strip()[:120]
    desc = em.get("ImageDescription", {}).get("value", "") or em.get("ObjectName", {}).get("value", "")
    caption = clean_caption(p.get("title", ""), desc)
    categories = em.get("Categories", {}).get("value", "")
    if len(caption) < 3 or not looks_like_photo(caption, categories):
        return None
    if info.get("width", 0) and info.get("height", 0):
        ar = info["width"] / max(1, info["height"])
        if ar > MAX_AR or ar < 1 / MAX_AR:
            return None
        if min(info["width"], info["height"]) < OUT_SIZE:
            return None   # source too small to give a real 256px crop

    raw = fetch_bytes(thumb_url)
    if raw is None:
        return None
    try:
        img = Image.open(io.BytesIO(raw))
        # JPEG "draft mode": decode at a reduced internal DCT scale when we're
        # about to downsize a lot anyway (850px source -> 256px output) —
        # much cheaper than a full-resolution decode followed by a resize.
        img.draft("RGB", (OUT_SIZE, OUT_SIZE))
        img = img.convert("RGB")
    except Exception:
        return None
    if min(img.size) < OUT_SIZE:
        return None
    img = center_crop_square(img).resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)   # JPEG: photos compress far smaller than PNG at this res
    return (pid, p.get("title", ""), artist, license_short, caption,
            info.get("descriptionurl", ""), buf.getvalue())


def fetch_query_pages(q, img_pool, seen_pageids):
    """Runs the full sequential pagination chain for one query (search ->
    download candidate images -> next page's search -> ...) and returns the
    list of successfully processed (not-yet-inserted) results. This is the
    unit of work run CONCURRENTLY across many queries at once — the search+
    continuation round-trips are the real bottleneck (network latency, not
    CPU), so overlapping many queries' round-trips is what actually speeds
    this up, more than parallelizing downloads within a single query does."""
    results = []
    continue_params = {}
    for page_num in range(PAGES_PER_QUERY):
        params = {
            "action": "query", "generator": "search", "gsrsearch": q,
            "gsrlimit": PAGE_SIZE, "gsrnamespace": 6,
            "prop": "imageinfo", "iiprop": "url|extmetadata|size",
            "iiurlwidth": THUMB_W,
        }
        params.update(continue_params)
        data = api_get(params)
        if not data or "query" not in data:
            break
        pages = list(data["query"]["pages"].values())
        for result in img_pool.map(lambda p: process_page(p, seen_pageids), pages):
            if result is not None:
                results.append(result)
        cont = data.get("continue")
        if not cont:
            break
        continue_params = cont
    return results


def main():
    resume = "--resume" in sys.argv
    con = sqlite3.connect("dataset_hq.db")
    cur = con.cursor()
    if not resume:
        cur.executescript("DROP TABLE IF EXISTS samples;")
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS samples (
            id            INTEGER PRIMARY KEY,
            page_id       INTEGER UNIQUE,
            query         TEXT,
            title         TEXT,
            caption       TEXT,
            artist        TEXT,
            license       TEXT,
            source_url    TEXT,
            width         INTEGER,
            height        INTEGER,
            image         BLOB
        );
        CREATE INDEX IF NOT EXISTS idx_query ON samples(query);
    """)

    seen_pageids = set()
    if resume:
        seen_pageids = {r[0] for r in cur.execute("SELECT page_id FROM samples")}
        print(f"resuming: {len(seen_pageids)} rows already saved")
    total_saved = len(seen_pageids)
    t0 = time.time()

    img_pool = ThreadPoolExecutor(max_workers=48)     # per-page image download/decode parallelism
    query_pool = ThreadPoolExecutor(max_workers=12)   # concurrent queries, to overlap search-latency
    db_lock = threading.Lock()
    done_count = 0

    def insert_results(q, results):
        nonlocal total_saved
        saved_here = 0
        with db_lock:
            for result in results:
                pid, title, artist, license_short, caption, source_url, jpg_bytes = result
                if pid in seen_pageids:
                    continue
                seen_pageids.add(pid)
                cur.execute(
                    "INSERT OR IGNORE INTO samples (page_id,query,title,caption,artist,license,"
                    "source_url,width,height,image) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (pid, q, title, caption, artist, license_short,
                     source_url, OUT_SIZE, OUT_SIZE, jpg_bytes))
                saved_here += 1
                total_saved += 1
            con.commit()
        return saved_here

    futures = {query_pool.submit(fetch_query_pages, q, img_pool, seen_pageids): (qi, q)
               for qi, q in enumerate(QUERIES)}
    for future in as_completed(futures):
        qi, q = futures[future]
        results = future.result()
        saved_here = insert_results(q, results)
        done_count += 1
        print(f"[{done_count}/{len(QUERIES)}] {q!r}: +{saved_here} (total {total_saved})  ({time.time()-t0:.0f}s)")

    print("done:", total_saved, "rows in", time.time() - t0, "s")
    con.close()


if __name__ == "__main__":
    main()
