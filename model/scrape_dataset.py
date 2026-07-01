"""
Build a REAL, captioned training database by querying the Wikimedia Commons
API (commons.wikimedia.org) — every file hosted there is required by Commons
policy to be public domain or under a free license that permits commercial
use and derivative works (no NC/ND content is allowed on Commons at all), so
this is a legally clean source of real photographs with real captions for
training a derivative model. We store full attribution (artist, license,
source URL) per row for transparency.
"""
import html, io, json, re, sqlite3, sys, time
import urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

API = "https://commons.wikimedia.org/w/api.php"
UA = "PocketPaintDatasetBuilder/1.0 (research/education; contact: jasoncomerfordny@gmail.com)"
THUMB_W = 96
OUT_SIZE = 32

QUERIES = [
    # --- nature / landscape (original set) ---
    "sunset sky", "sunrise sky", "starry night sky", "full moon night",
    "desert dunes", "snowy mountain", "mountain range", "green meadow field",
    "forest path", "pine forest", "autumn forest", "tropical beach",
    "ocean waves", "calm lake", "waterfall", "river valley", "rolling hills",
    "rainbow sky", "foggy morning", "storm clouds", "northern lights",
    "city skyline night", "city skyline day", "neon street night",
    "skyscraper building", "old town street", "village houses",
    "countryside farm", "wheat field", "vineyard hills",
    "horse running field", "horse grazing", "dog running", "dog playing park",
    "cat sitting", "wild bird flying", "eagle flying", "owl perched",
    "deer in forest", "fox in snow", "rabbit grass", "sheep grazing",
    "cow pasture", "elephant savanna", "lion grass", "tiger jungle",
    "fish underwater", "dolphin ocean", "sailboat lake", "sailing ship sea",
    "fishing boat harbor", "rowboat river", "red car road", "vintage car",
    "sports car street", "bicycle path", "train tracks", "airplane sky",
    "hot air balloon", "lighthouse coast", "windmill field", "bridge river",
    "castle hill", "church building", "barn farmhouse", "cabin in woods",
    "tent campsite", "campfire night", "bonfire beach", "fireworks night",
    "person walking beach", "person hiking mountain", "child playing park",
    "people market street", "farmer field", "fisherman boat",
    "cyclist road", "runner trail", "skier mountain", "surfer wave",
    "tree alone field", "oak tree", "palm tree beach", "cherry blossom tree",
    "flower garden", "sunflower field", "rose garden", "cactus desert",
    "icebergs arctic", "glacier mountain", "volcano eruption", "canyon rock",
    "cave entrance", "waterfall jungle", "lake reflection mountain",
    "thunderstorm lightning", "rain street", "snow covered street",
    "autumn leaves road", "spring blossoms park", "summer beach sunset",
    "winter forest snow", "harbor boats sunset", "market stalls street",
    "street food vendor", "night market lights", "desert camel",
    "savanna sunset", "jungle river", "coral reef fish", "penguin ice",
    "polar bear snow", "wolf forest", "owl night", "butterfly flower",

    # --- more animals ---
    "elephant herd", "giraffe savanna", "zebra grassland", "kangaroo outback",
    "panda eating bamboo", "monkey tree", "gorilla forest", "chimpanzee",
    "shark underwater", "whale ocean", "octopus reef", "jellyfish sea",
    "snake forest floor", "lizard rock", "frog pond", "turtle beach",
    "hedgehog grass", "squirrel tree", "raccoon forest", "bear river",
    "moose forest", "bison plains", "llama mountain", "alpaca farm",
    "peacock feathers", "parrot tropical", "flamingo lake", "swan lake",
    "duck pond", "goose flying", "chicken farm", "pig farm", "goat hill",
    "donkey field", "camel desert caravan", "hippopotamus river",
    "rhinoceros savanna", "crocodile river", "koala tree", "sloth tree",
    "hawk flying", "falcon hunting", "crow perched", "robin bird garden",
    "hummingbird flower", "seal beach rocks", "otter river", "beaver dam",
    "ant macro", "bee flower", "ladybug leaf", "spider web", "dragonfly pond",
    "snail leaf", "starfish beach", "crab beach", "lobster ocean",

    # --- vehicles / transport ---
    "truck highway", "motorcycle road", "helicopter sky", "rocket launch",
    "submarine ocean", "tractor field", "scooter street", "city bus street",
    "train station platform", "subway train", "ferry boat", "cruise ship sea",
    "tram city street", "ambulance street", "fire truck", "police car",
    "race car track", "monster truck", "go kart track", "snowmobile snow",
    "kayak river", "canoe lake", "jet ski water", "cable car mountain",
    "hot air balloons festival", "glider sky", "biplane vintage",
    "freight train", "yacht harbor", "tugboat harbor",

    # --- food / drink ---
    "pizza table", "burger plate", "cake birthday", "coffee cup",
    "fresh bread bakery", "fruit basket", "vegetables market",
    "ice cream cone", "sushi plate", "pasta dish", "salad bowl",
    "pancakes breakfast", "tacos plate", "soup bowl", "chocolate bar",
    "wine glass", "tea cup", "fresh vegetables farm", "apple orchard",
    "orange grove", "grapes vineyard", "strawberries basket", "watermelon slice",
    "barbecue grill", "street food stall", "farmers market produce",

    # --- architecture / places ---
    "pyramid desert", "ancient temple", "mosque architecture",
    "cathedral interior", "stadium crowd", "public library building",
    "museum building", "stone tower", "dam river", "skyscraper street",
    "amphitheater ancient", "palace garden", "windmill countryside",
    "monastery mountain", "fortress wall", "harbor town", "fishing village",
    "ski resort mountain", "vineyard estate", "rice terraces farm",
    "greenhouse plants", "subway station", "rooftop city view",
    "street market night", "alleyway old town", "town square fountain",

    # --- sports / activities ---
    "soccer match field", "basketball court game", "tennis court match",
    "swimming pool race", "rock climbing cliff", "yoga outdoors",
    "dancing stage performance", "painting artist studio", "cooking kitchen",
    "reading book park", "camping tent mountain", "fishing lake shore",
    "golf course green", "baseball field game", "volleyball beach",
    "skateboarding park", "snowboarding mountain", "surfing big wave",
    "rowing team river", "marathon runners street", "gymnastics performance",
    "boxing match ring", "archery target", "horseback riding trail",

    # --- everyday scenes / interiors ---
    "kitchen interior modern", "bedroom interior cozy", "office workspace desk",
    "classroom students", "science laboratory", "factory machinery",
    "library bookshelves", "art gallery paintings", "concert stage crowd",
    "bakery shop interior", "flower shop interior", "bookstore shelves",
    "workshop tools", "garden greenhouse", "balcony plants city",

    # --- textures / phenomena ---
    "fire flames closeup", "water splash macro", "smoke abstract",
    "ice crystals macro", "lava flow volcano", "crystal mineral",
    "tornado storm", "hailstorm clouds", "sandstorm desert",
    "lightning storm night", "aurora borealis sky", "fog mountain valley",
    "frost window pattern", "bubbles water macro", "steam rising",

    # --- space ---
    "galaxy stars space", "planet space telescope", "astronaut spacewalk",
    "satellite orbit earth", "space shuttle launch", "milky way night sky",
    "solar eclipse sky", "comet night sky",

    # --- musical instruments / tech / misc objects ---
    "acoustic guitar closeup", "grand piano concert", "violin closeup",
    "drum set stage", "trumpet musician", "robot machine",
    "vintage computer", "smartphone closeup", "drone flying sky",
    "camera photography closeup", "telescope observatory", "windsurfing sea",
    "umbrella rain street", "lantern night street", "candle flame closeup",

    # --- plants ---
    "mushroom forest floor", "fern forest green", "bamboo forest path",
    "lotus flower pond", "tulip field colorful", "orchid flower closeup",
    "cactus garden desert", "moss covered rock", "ivy covered wall",
    "wildflower meadow", "pine cone closeup", "autumn pumpkin patch",

    # --- landmarks / famous places ---
    "eiffel tower paris", "great wall china", "taj mahal india",
    "golden gate bridge", "grand canyon", "niagara falls", "mount everest",
    "colosseum rome", "statue of liberty", "big ben london",
    "sydney opera house", "machu picchu", "stonehenge", "petra jordan",
    "santorini greece", "venice canal", "amsterdam canal houses",
    "kyoto temple garden", "great barrier reef", "yellowstone geyser",
    "mount fuji", "victoria falls", "angkor wat temple", "acropolis athens",
    "burj khalifa dubai", "times square new york", "red square moscow",

    # --- more everyday objects ---
    "wall clock closeup", "wooden chair", "dining table set",
    "table lamp closeup", "stack of books", "old key closeup",
    "eyeglasses closeup", "wrist watch closeup", "leather backpack",
    "pair of shoes", "straw hat", "bicycle helmet", "wooden door closeup",
    "window with curtains", "vase of flowers", "wine bottle closeup",
    "typewriter vintage", "old telephone", "sewing machine vintage",
    "toolbox workshop", "paintbrush palette", "chess board pieces",
    "playing cards table", "board game pieces", "puzzle pieces",

    # --- more food ---
    "sandwich plate", "donut closeup", "cupcake closeup", "cheese platter",
    "grilled steak plate", "seafood platter", "noodles bowl asian",
    "curry dish", "dumplings plate", "croissant bakery", "waffles breakfast",
    "smoothie glass", "cocktail drink", "beer glass pub", "cheese wheel",
    "honey jar closeup", "spices market closeup", "herbs garden closeup",

    # --- more human activities / portraits ---
    "chef cooking kitchen", "doctor hospital", "teacher classroom",
    "firefighter action", "construction worker site", "artist painting canvas",
    "musician playing stage", "photographer camera", "scientist lab coat",
    "farmer harvesting field", "baker bakery kitchen", "carpenter workshop",
    "student studying library", "business meeting office",
    "family picnic park", "friends laughing outdoors", "couple walking beach",
    "grandmother knitting", "baby sleeping crib", "children playing playground",

    # --- celebrations / holidays ---
    "christmas tree lights", "halloween pumpkin carving", "birthday party cake",
    "wedding ceremony flowers", "new year fireworks", "easter eggs basket",
    "thanksgiving dinner table", "carnival parade costumes", "diwali lights festival",
    "graduation ceremony cap",

    # --- more nature macro / wildlife ---
    "leaf closeup veins", "tree bark texture", "rock texture closeup",
    "sand dune texture closeup", "water droplets leaf", "spider macro closeup",
    "beetle macro closeup", "grasshopper macro closeup", "moth closeup wings",
    "coral reef closeup", "seashell beach closeup", "pebbles beach closeup",
    "bird nest closeup", "bird eggs closeup", "feather closeup macro",
    "owl closeup portrait", "wolf closeup portrait", "fox closeup portrait",
    "eagle closeup portrait", "tiger closeup portrait", "lion closeup portrait",

    # --- more transportation / tech ---
    "container ship port", "cargo plane airport", "high speed train",
    "electric car charging", "solar panels field", "wind turbines field",
    "oil rig ocean", "power plant industrial", "bridge suspension night",
    "highway traffic night", "airport terminal interior", "subway tunnel",

    # --- more weather / sky ---
    "cloudy sky dramatic", "clear blue sky", "sunset clouds orange",
    "hazy sky city", "double rainbow field", "meteor shower night sky",
    "sunbeams forest", "golden hour field", "blue hour city",
]


def api_get(params, retries=4):
    params = dict(params)
    params["format"] = "json"
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read())
        except Exception as e:
            if attempt == retries - 1:
                print("  api_get failed:", e, file=sys.stderr)
                return None
            time.sleep(1.5 * (attempt + 1))


def fetch_bytes(url, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=20) as resp:
                return resp.read()
        except Exception as e:
            if attempt == retries - 1:
                return None
            time.sleep(1.0 * (attempt + 1))


PLACEHOLDER_DESCS = {"see title", "see filename", "no description", "untitled", ""}


def clean_caption(title, desc):
    t = title.replace("File:", "")
    t = re.sub(r"\.(jpe?g|png|gif|tiff?|webp)$", "", t, flags=re.I)
    t = t.replace("_", " ")
    cap = desc.strip() if desc and len(desc.strip()) > 0 else ""
    cap = re.sub(r"<[^>]+>", " ", html.unescape(cap))   # strip any HTML
    # Wikidata structured-data leakage, e.g. `label QS:Len,"Sunset"` or
    # `title QS:P1476,en:"..."` — cut everything from the first QS: marker on.
    cap = re.sub(r"\b(label|title|description)\s+QS:.*$", "", cap, flags=re.I)
    cap = re.sub(r"\s+", " ", cap).strip().strip('"').strip()
    if cap.lower() in PLACEHOLDER_DESCS or len(cap) < 3:
        cap = t   # fall back to the (cleaned) filename, which is usually descriptive
    return cap[:200]


BLOCK_WORDS = re.compile(
    r"\b(montage|collage|composite|panorama|map|chart|diagram|painting|drawing|"
    r"sketch|illustration|logo|flag|screenshot|graph|stamp|postcard|poster|"
    r"by vincent van gogh|emblem|coat of arms|locator|infographic|panel of|"
    r"set of \d|tiled|grid of|comparison)\b", re.I)


def looks_like_photo(caption, categories):
    text = (caption or "") + " " + (categories or "")
    return not BLOCK_WORDS.search(text)


def center_crop_square(img):
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def main():
    resume = "--resume" in sys.argv
    con = sqlite3.connect("dataset_real.db")
    cur = con.cursor()
    if resume:
        cur.execute("PRAGMA journal_mode=DELETE")  # clear any stale -journal cleanly
    else:
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
        print(f"resuming: {len(seen_pageids)} rows already saved (all {len(QUERIES)} queries will be "
              f"re-run with a higher per-query limit so already-scraped topics get topped up with new results too)")
    total_saved = len(seen_pageids)
    t0 = time.time()
    pool = ThreadPoolExecutor(max_workers=16)

    def process_page(p):
        """Runs in a worker thread: filter + download + decode + resize only
        (no SQLite access here — all DB writes happen back on the main thread)."""
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
            if ar > 3.2 or ar < 1 / 3.2:   # skip extreme panoramas/strips
                return None

        raw = fetch_bytes(thumb_url)
        if raw is None:
            return None
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            return None
        if min(img.size) < 16:
            return None
        img = center_crop_square(img).resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return (pid, p.get("title", ""), artist, license_short, caption,
                info.get("descriptionurl", ""), buf.getvalue())

    for qi, q in enumerate(QUERIES):
        data = api_get({
            "action": "query", "generator": "search", "gsrsearch": q,
            "gsrlimit": 90, "gsrnamespace": 6,
            "prop": "imageinfo", "iiprop": "url|extmetadata|size",
            "iiurlwidth": THUMB_W,
        })
        if not data or "query" not in data:
            print(f"[{qi+1}/{len(QUERIES)}] {q!r}: no results")
            continue
        pages = list(data["query"]["pages"].values())
        saved_here = 0
        for result in pool.map(process_page, pages):
            if result is None:
                continue
            pid, title, artist, license_short, caption, source_url, png_bytes = result
            if pid in seen_pageids:   # a concurrent duplicate across queries
                continue
            seen_pageids.add(pid)
            cur.execute(
                "INSERT OR IGNORE INTO samples (page_id,query,title,caption,artist,license,"
                "source_url,width,height,image) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (pid, q, title, caption, artist, license_short,
                 source_url, OUT_SIZE, OUT_SIZE, png_bytes))
            saved_here += 1
            total_saved += 1
        con.commit()
        print(f"[{qi+1}/{len(QUERIES)}] {q!r}: +{saved_here} (total {total_saved})  ({time.time()-t0:.0f}s)")

    print("done:", total_saved, "rows in", time.time() - t0, "s")
    con.close()


if __name__ == "__main__":
    main()
