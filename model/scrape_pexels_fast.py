"""
High-throughput photo scraper for beautiful photography.

WHY THIS IS FAST (the three levers, in order of impact):

  1. ASK THE CDN FOR THE SIZE WE NEED. Every previous scraper in this repo
     downloaded a full-resolution original and threw 95% of the pixels away.
     Pexels' CDN resizes server-side via a query param, so requesting
     `?auto=compress&cs=tinysrgb&w=512` returns ~42 KB instead of a 3-10 MB
     original. That is a ~100x bandwidth reduction and it is the single
     biggest win -- no amount of concurrency makes up for downloading
     pixels you are about to discard.

  2. HIGH-CONCURRENCY ASYNC I/O WITH CONNECTION REUSE. Image fetches are
     latency-bound (~500 ms round trip each), not bandwidth-bound per
     connection, so throughput is concurrency / latency. One aiohttp event
     loop with a large keep-alive pool and cached DNS saturates the link
     with a single thread.

  3. DECODE IN A PROCESS POOL, WITH JPEG DCT SCALING. Pillow's `draft()`
     decodes a JPEG directly at 1/2 or 1/4 scale inside the DCT step, which
     is ~1.6x faster than decoding full-size and resizing after. Decode is
     CPU-bound and the GIL makes threads useless here, so it runs in a
     process pool sized to the core count.

MEASURED ON A 4-CORE BOX WITH A ~110 MB/s LINK:
  --raw    (store CDN bytes as-is)  ~2,000-2,600 img/s  -> network-bound
  --decode (crop/resize to 256px)   ~1,500 img/s        -> CPU-bound

  Decode is the ceiling on a 4-core machine. Use --raw to hit peak ingest
  rate and defer resizing, or run --decode on a box with more cores.

SOURCES (both real photography, not AI-generated, not artwork):
  janpf    opendiffusionai/pexels-janpf-sharp -- 64,925 sharpness-filtered
           Pexels photos, every one carrying a LLaVA-38B caption (~875
           chars). This is the quality source: the captions are far richer
           than the filename-derived captions the older scrapers produced.
  meta     terminusresearch/pexels-metadata-1.71M -- 1.71M Pexels IDs with
           Google SafeSearch flags. Captions are only ~0.4% populated, so
           this is the volume/benchmark source. URLs are reconstructed from
           the bare photo ID, which is verified to work for arbitrary IDs.

Be a good citizen: --concurrency is capped and defaults to a level that is
brisk but not abusive. Cranking it to thousands to win a benchmark is a
denial-of-service against someone else's CDN.
"""
import argparse
import asyncio
import io
import os
import re
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor

DEFAULT_UA = "PocketPaintDatasetBuilder/2.0 (research/education)"
# fm=jpg forces JPEG output regardless of the stored format. That matters:
# ~3% of Pexels photos exist only as .png, and a resized PNG comes back at
# ~474 KB where the same image as JPEG is ~44 KB -- a 10x difference. The
# path extension still has to match the stored asset or the CDN 404s, so we
# preserve it and let fm= handle the encoding.
PEXELS_TMPL = ("https://images.pexels.com/photos/{id}/pexels-photo-{id}.{ext}"
               "?auto=compress&cs=tinysrgb&fm=jpg&w={w}")
_ID_RE = re.compile(r"/photos/(\d+)/")


def rewrite_url(url, width, ext=None):
    """Force any Pexels URL to a CDN-resized JPEG of the requested width.

    The source manifests carry inconsistent URLs (some .png, some
    full-resolution, some already parameterised). Normalising them all to
    one compressed-JPEG form is what keeps the payload at ~40 KB.
    """
    url = (url or "").strip()
    m = _ID_RE.search(url)
    if not m:
        return None
    if ext is None:
        ext = url.split("?")[0].rsplit(".", 1)[-1].lower()
        if ext not in ("jpeg", "jpg", "png"):
            ext = "jpeg"
    return PEXELS_TMPL.format(id=m.group(1), ext=ext, w=width)


# ---------------------------------------------------------------------------
# Decode workers (separate processes -- decode is CPU-bound and holds the GIL)
# ---------------------------------------------------------------------------
def _decode_batch(args):
    """Crop to square + resize, returning re-encoded JPEG bytes.

    Batched because sending 42 KB blobs through IPC one at a time costs more
    than the decode itself.
    """
    from PIL import Image
    batch, out_size, quality = args
    out = []
    for key, caption, blob in batch:
        try:
            im = Image.open(io.BytesIO(blob))
            # DCT-domain downscale: decodes at 1/2 or 1/4 size directly.
            im.draft("RGB", (out_size, out_size))
            im = im.convert("RGB")
            w, h = im.size
            if min(w, h) < out_size // 2:
                continue
            s = min(w, h)
            im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
            im = im.resize((out_size, out_size), Image.BILINEAR)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=quality)
            out.append((key, caption, buf.getvalue(), out_size, out_size))
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------
def load_manifest(source, limit, width, caption_field, cache_dir):
    import pyarrow.parquet as pq
    import urllib.request

    os.makedirs(cache_dir, exist_ok=True)

    def fetch(url, path):
        if os.path.exists(path) and os.path.getsize(path) > 1024:
            return path
        print(f"  downloading manifest {os.path.basename(path)} ...", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": DEFAULT_UA})
        with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
            while True:
                c = r.read(1 << 20)
                if not c:
                    break
                f.write(c)
        return path

    items = []
    if source == "janpf":
        p = fetch(
            "https://huggingface.co/datasets/opendiffusionai/pexels-janpf-sharp/resolve/main/data.parquet",
            os.path.join(cache_dir, "janpf.parquet"))
        t = pq.read_table(p, columns=["url", caption_field]).to_pydict()
        for url, cap in zip(t["url"], t[caption_field]):
            u = rewrite_url(url, width)
            if not u:
                continue
            items.append((u, (cap or "").strip()))
            if limit and len(items) >= limit:
                break
    elif source == "meta":
        p = fetch(
            "https://huggingface.co/datasets/terminusresearch/pexels-metadata-1.71M/resolve/main/photos_sequential.parquet",
            os.path.join(cache_dir, "pexels_meta.parquet"))
        pf = pq.ParquetFile(p)
        SAFE = {"very_unlikely", "unlikely", None, ""}
        cols = ["id", "adult", "racy", "violence", "alt_text", "title", "description"]
        for rg in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(rg, columns=cols).to_pydict()
            for i in range(len(t["id"])):
                if t["adult"][i] not in SAFE or t["violence"][i] not in SAFE:
                    continue
                if t["racy"][i] in ("likely", "very_likely"):
                    continue
                pid = t["id"][i]
                if pid is None:
                    continue
                cap = (t["alt_text"][i] or t["description"][i] or t["title"][i] or "").strip()
                items.append((PEXELS_TMPL.format(id=int(pid), ext="jpeg", w=width), cap))
                if limit and len(items) >= limit:
                    return items
    else:
        raise SystemExit(f"unknown source {source!r}")
    return items


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------
def open_db(path):
    con = sqlite3.connect(path, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.executescript("""
        CREATE TABLE IF NOT EXISTS samples (
            id INTEGER PRIMARY KEY, page_id INTEGER UNIQUE, query TEXT, title TEXT,
            caption TEXT, artist TEXT, license TEXT, source_url TEXT,
            width INTEGER, height INTEGER, image BLOB);
        CREATE INDEX IF NOT EXISTS idx_query ON samples(query);
    """)
    return con


# ---------------------------------------------------------------------------
# Async fetch pipeline
# ---------------------------------------------------------------------------
async def run(items, args):
    import aiohttp

    con = open_db(args.db)
    seen = {r[0] for r in con.execute("SELECT page_id FROM samples")}
    todo = []
    for u, cap in items:
        m = _ID_RE.search(u)
        pid = int(m.group(1))
        if pid not in seen:
            todo.append((pid, u, cap))
    print(f"manifest: {len(items)}  new: {len(todo)}  already in db: {len(items)-len(todo)}", flush=True)
    if not todo:
        con.close()
        return

    q_decode = asyncio.Queue(maxsize=args.concurrency * 4)
    stats = {"fetched": 0, "bytes": 0, "saved": 0, "failed": 0}
    t0 = time.time()

    pool = None if args.raw else ProcessPoolExecutor(max_workers=args.workers)
    loop = asyncio.get_running_loop()
    pending_writes = []

    def flush(rows):
        if not rows:
            return
        con.executemany(
            "INSERT OR IGNORE INTO samples (page_id,query,title,caption,artist,license,"
            "source_url,width,height,image) VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        con.commit()

    async def consumer():
        """Pulls fetched bytes, decodes (or not), writes in batches."""
        batch = []
        inflight = []
        while True:
            item = await q_decode.get()
            if item is None:
                break
            batch.append(item)
            if len(batch) >= args.batch:
                if args.raw:
                    rows = [(k, f"pexels_fast:{args.source}", "", c, "", "Pexels", "", 0, 0, b)
                            for k, c, b in batch]
                    flush(rows)
                    stats["saved"] += len(rows)
                else:
                    fut = loop.run_in_executor(pool, _decode_batch, (batch, args.out_size, args.quality))
                    inflight.append(fut)
                    if len(inflight) >= args.workers * 2:
                        done = inflight.pop(0)
                        res = await done
                        rows = [(k, f"pexels_fast:{args.source}", "", c, "", "Pexels", "", w, h, b)
                                for k, c, b, w, h in res]
                        flush(rows)
                        stats["saved"] += len(rows)
                batch = []
        # drain
        if batch:
            if args.raw:
                rows = [(k, f"pexels_fast:{args.source}", "", c, "", "Pexels", "", 0, 0, b) for k, c, b in batch]
                flush(rows)
                stats["saved"] += len(rows)
            else:
                inflight.append(loop.run_in_executor(pool, _decode_batch, (batch, args.out_size, args.quality)))
        for fut in inflight:
            res = await fut
            rows = [(k, f"pexels_fast:{args.source}", "", c, "", "Pexels", "", w, h, b) for k, c, b, w, h in res]
            flush(rows)
            stats["saved"] += len(rows)

    idx = 0
    lock = asyncio.Lock()

    async def worker(sess):
        nonlocal idx
        while True:
            async with lock:
                if idx >= len(todo):
                    return
                pid, url, cap = todo[idx]
                idx += 1
            try:
                async with sess.get(url) as resp:
                    if resp.status != 200:
                        stats["failed"] += 1
                        continue
                    blob = await resp.read()
            except Exception:
                stats["failed"] += 1
                continue
            if len(blob) < 1024:
                stats["failed"] += 1
                continue
            stats["fetched"] += 1
            stats["bytes"] += len(blob)
            await q_decode.put((pid, cap[:1500], blob))

    async def reporter():
        last = 0
        while True:
            await asyncio.sleep(2)
            el = time.time() - t0
            f = stats["fetched"]
            rate = (f - last) / 2.0
            last = f
            print(f"  [{el:6.1f}s] fetched={f:7d} saved={stats['saved']:7d} "
                  f"fail={stats['failed']:5d}  {rate:7.1f} img/s  "
                  f"{stats['bytes']/el/1048576:6.1f} MB/s", flush=True)

    conn = aiohttp.TCPConnector(limit=args.concurrency, ttl_dns_cache=600,
                                use_dns_cache=True, force_close=False)
    timeout = aiohttp.ClientTimeout(total=args.timeout, connect=10)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout,
                                     headers={"User-Agent": DEFAULT_UA}) as sess:
        cons = asyncio.create_task(consumer())
        rep = asyncio.create_task(reporter())
        await asyncio.gather(*[worker(sess) for _ in range(args.concurrency)])
        await q_decode.put(None)
        await cons
        rep.cancel()

    if pool:
        pool.shutdown()
    el = time.time() - t0
    n = con.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    con.close()
    print("-" * 72)
    print(f"done in {el:.1f}s   fetched={stats['fetched']}  saved={stats['saved']}  failed={stats['failed']}")
    print(f"rate: {stats['fetched']/el:.0f} img/s   {stats['bytes']/el/1048576:.1f} MB/s   "
          f"avg {stats['bytes']/max(1,stats['fetched'])/1024:.1f} KB/img")
    print(f"projected for 100,000 images: {100000*el/max(1,stats['fetched']):.0f}s")
    print(f"db now holds {n} rows -> {args.db}")


def main():
    ap = argparse.ArgumentParser(description="Fast Pexels photography scraper")
    ap.add_argument("--source", default="janpf", choices=["janpf", "meta"])
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--width", type=int, default=512, help="CDN-side resize width")
    ap.add_argument("--out-size", type=int, default=256)
    ap.add_argument("--concurrency", type=int, default=192)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4)))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--raw", action="store_true", help="store CDN bytes as-is (no decode; fastest)")
    ap.add_argument("--caption-field", default="llava38b", choices=["llava38b", "interlm7b", "wd14"])
    ap.add_argument("--db", default="dataset_pexels.db")
    ap.add_argument("--cache-dir", default=".manifest_cache")
    args = ap.parse_args()
    if args.concurrency > 512:
        sys.exit("refusing --concurrency > 512: that is abusive to the CDN")

    print(f"source={args.source} limit={args.limit} width={args.width} "
          f"concurrency={args.concurrency} workers={args.workers} mode={'raw' if args.raw else 'decode'}")
    items = load_manifest(args.source, args.limit, args.width, args.caption_field, args.cache_dir)
    asyncio.run(run(items, args))


if __name__ == "__main__":
    main()
