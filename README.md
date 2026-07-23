# PocketPaint — a local, in-browser image generator

A complete **single-file** image generation app (`index.html`) that runs entirely
in the browser. No backend, no cloud API, no login, no paid service, and **no
Node.js needed to run the finished app** — just open the file.

It is built to start instantly and stay within a **~3 GB RAM** budget on low-end
devices — explicitly including the **iPad 9 (3 GB, iPadOS Safari)** — while still
using modern in-browser inference tech (WebGPU / WebAssembly / ONNX Runtime Web).
Output is **at least 256 × 256 px** (default size on low-memory devices) and up to
640 × 640 on capable hardware.

---

## 1. What it does

- **Text prompt box**, **Generate** button, and a live **preview** canvas.
- **Progress bar + status messages** while loading/generating.
- **Settings:** image size (256–640), **seed** (reproducible), and **steps**.
- **Capability detection** (WebGPU / WebGL2 / WebAssembly / device RAM / cores /
  iOS) with **graceful error handling and fallbacks**.
- **Per-tab memory budget guard** so the heavy diffusion path never crashes a
  low-RAM tab (e.g. iPad 9). Oversized models are refused with a clear message
  and the app falls back to the built-in neural model.
- **Download PNG** and **copy prompt**.
- Clean, modern dark UI. All code + comments are in the one HTML file.

---

## 2. Model / runtime chosen, and why

The app ships **six engines**. The default is **my own neural network, trained
from scratch** specifically to fit this 3 GB / iPad-9 target; there is also a
**second, self-trained model** — a genuine small diffusion model (see section A2) —
and a **third, bigger self-trained diffusion model** generating natively at
256×256 (section A3, served in two flavors: WebGPU 🎨 and CPU 🐌).

### A. 🧬 Neural model — DEFAULT (I trained this myself)
- **What it is:** a **~31,000-parameter conditional neural field** (a small MLP /
  CPPN). It takes `(x, y)` pixel coordinates — Fourier-encoded — plus a **21-D
  semantic condition vector** and outputs the RGB colour at that pixel:

  ```
  input  = [ Fourier(x,y, 6 octaves), x, y, r, condition(21) ]   (48 dims)
  hidden = 48 → 112 → 112 → 112   (tanh)
  output = 3 (sigmoid → RGB)
  ```

- **How it was trained** (`model/train.py`, pure NumPy, no GPU, ~19 min on CPU,
  8000 steps): I wrote an analytic **scene renderer** that composes recognizable
  scenes — sky by **time of day** (night / sunset / day / overcast), ground by
  **biome** (ocean / forest / desert / snow / plain), optional **mountains / neon
  / fire**, plus simple **silhouette subjects** standing/floating in the scene:
  **person, horse, dog, car, bird, boat, tree, building** (procedurally drawn with
  soft signed-distance shape primitives — legs, torsos, wheels, canopies, hulls,
  etc. — and placed via a `subj_x` placement slot in the condition vector) — from
  the 21-D condition. The network is trained by **distilling** that renderer: each
  step samples fresh random conditions + pixels and regresses the network's output
  to the renderer's (Adam, MSE). Final MSE ≈ `1.1e-3` (RMSE ≈ 0.034) — a bit higher
  than the landscape-only version because the condition space is much larger now.
- **How it runs in the browser:** the trained weights (`float32`, base64, ~165 KB)
  are **embedded directly in `index.html`** and decoded at startup. Your prompt is
  mapped to the same 21-D condition the model learned, and the network is evaluated
  **per pixel** in optimized JavaScript (typed arrays), rendered at a capped
  internal resolution and upscaled (the field is smooth, so this looks clean). The
  JS forward pass is **bit-for-bit identical** to the Python training code (verified
  against reference pixels, max diff ~1e-15).
- **Why this design:** a real Stable-Diffusion model is **gigabytes** and cannot
  load in a ~1–1.5 GB iOS Safari tab. So instead of depending on a giant model, I
  **made my own tiny one** that is genuine neural-network inference, needs **zero
  downloads**, starts **instantly**, renders at **any resolution (≥256 px)**, and
  uses only a few MB of RAM — so it actually runs on an **iPad 9**.
- **Honest limitation:** because it's tiny and trained to reproduce a procedural
  scene generator, it produces **stylized scenes with simple silhouette subjects**,
  not arbitrary photoreal images. It genuinely understands the prompt *vocabulary*
  it was trained on (times of day, biomes, mountains/neon/fire, and the 8 subject
  types above) and interpolates smoothly between them. Bigger/foreground subjects
  (person, horse, tree, building) render clearly and recognizably; smaller ones
  (dog, car, bird, boat) are lower-fidelity, soft colour blobs rather than crisp
  shapes — a result of their tiny footprint in the training images. See sample
  outputs in `model/previews.png`.

### A2. 🌀 Diffusion (ours) — a SECOND model I trained myself, on REAL scraped photos
- **What it is:** unlike the neural field above (which maps coordinates straight
  to RGB and is distilled from a hand-written analytic renderer), this is a
  genuine **DDPM/DDIM diffusion model** — a small conditional **convolutional
  UNet** trained to predict the noise added to a 32×32 image at a random
  timestep, then sampled by iterative denoising starting from pure Gaussian
  noise. It is a real instance of the same family of model SD-Turbo belongs
  to, just tiny: **~831,000 parameters total** (741k UNet + 90k text encoder),
  working resolution 32×32 (upscaled to the requested output size).

  ```
  UNet: 32×32 → (down, stride-2 conv) 16×16 → (down) 8×8 → mid (2 FiLM ResBlocks)
        → (up, nearest+conv, skip-concat) 16×16 → (up) 32×32 → 3ch noise pred
  Each level: FiLM residual block — conv3x3 → SiLU → FiLM(scale,shift from
  timestep+condition embedding) → SiLU → conv3x3 → + residual
  Conditioning: sinusoidal timestep embedding + a 64-D CAPTION embedding
  (see below), summed and fed to every FiLM block
  channel count: 36 → 58 → 84 (down/up), UNet embedding dim 128
  ```

- **Trained on a real, captioned photo database I scraped myself — not
  synthetic renders.** `model/scrape_dataset.py` queries the **Wikimedia
  Commons API** (**485 search topics** — landscapes/weather, ~70 animal
  species, vehicles/transport, food & drink, architecture & famous
  landmarks, sports & activities, everyday objects & interiors, human
  activities/portraits, celebrations, textures/phenomena, space,
  instruments/tech, plants) and downloads real photographs with their real
  captions/descriptions into `model/dataset_real.db` (SQLite, **27,487
  rows**: page id, query, title, caption, artist, license, source URL, and
  a center-cropped 32×32 PNG). Downloads run through a 16-worker thread
  pool (network fetch + decode/resize per candidate image in parallel;
  all SQLite writes stay on the main thread), which cut scraping time by
  roughly **13×** versus the original one-image-at-a-time version. Every
  file on Wikimedia Commons is required by Commons policy to be public
  domain or under a free license permitting reuse and derivative works (no
  NC/ND content is hosted there at all), so this is a legally clean source
  for training a derivative model — full attribution (artist, license,
  source URL) is kept per-row for transparency. A keyword filter drops
  obvious non-photos (paintings, collages, maps, diagrams, screenshots),
  and captions are cleaned of Wikidata template leakage (e.g.
  `label QS:Len,"..."`) and placeholder text (`"See title"`) before being
  stored. (The database itself isn't committed to the repo — at this size
  it's a regenerable build artifact, not something that belongs in source
  control; run the script to rebuild it.)
- **A tiny from-scratch TEXT ENCODER, also trained by me, replaces the
  neural model's fixed category vector.** Instead of mapping prompts to a
  hand-designed 21-D vector, this model learns its own **1,400-word
  vocabulary** straight from the scraped captions and a **trainable word
  embedding table** (64-D). A caption (or, at inference time, your prompt)
  is tokenized with a trivial rule (lowercase, split on non-alphanumerics,
  drop stopwords) and turned into one vector by **mean-pooling the
  embeddings of its known words** — a minimal bag-of-words text encoder,
  trained jointly with the UNet on the actual denoising loss, so the
  vocabulary it ends up caring about is whatever the photos' real captions
  actually used (`park`, `street`, `forest`, `sunset`, `mountain`, `horse`,
  `elephant`, `desert`, `guitar`, `galaxy`, ...).
- **Two standard quality techniques, added after the first version looked
  noisy/incoherent:**
  - **EMA (exponential moving average) of the weights.** Instead of
    shipping the raw end-of-training weights (noisy, since Adam keeps
    perturbing them step to step), a shadow-averaged copy (decay 0.999) is
    tracked throughout training and *that's* what gets exported — visibly
    smoother, less grainy samples for the same architecture and data.
  - **Classifier-free guidance (CFG).** During training, captions are
    randomly zeroed out 15% of the time so the model also learns the
    *unconditional* denoising distribution. At generation time, the JS
    side runs the UNet twice per step (once with your prompt's embedding,
    once with a zero vector) and extrapolates *away* from the
    unconditional prediction toward the conditional one
    (`guidance_scale=3.0`, baked into the model's metadata) — this is the
    single biggest lever for prompt-adherence/coherence in small diffusion
    models and produces noticeably cleaner, less speckled output than
    without it.
- **How it was trained** (`model/train_diffusion.py`, PyTorch, CPU-only,
  16,000 steps, batch 64, ~3 hours): the whole database is decoded into
  memory once; each step samples a random batch of real (image, caption)
  rows, embeds the captions through the word-embedding table (with CFG
  dropout applied), adds noise at a random timestep, and trains the UNet
  (+ text encoder, same optimizer, same Adam/MSE/gradient-clipping/
  linear-warmup setup as before) to predict that noise. The loss had a
  couple of brief spikes mid-run (a known small-model diffusion-training
  quirk) but recovered within ~200 steps each time and finished stable.
- **How it runs in the browser:** the trained (EMA) weights — UNet + the
  word embedding table + the learned vocabulary + the guidance scale — are
  embedded in this file (`#pp-diffusion-model`, float32, base64, ~4.3 MB)
  and decoded on first use. Your prompt is tokenized and embedded by a JS
  port of the exact same rule (`promptToCaptionEmbedding`), then **DDIM
  sampling with classifier-free guidance** (Steps slider, 1–20 denoising
  steps, two UNet forward passes per step) runs fully client-side in plain
  JavaScript (conv2d, FiLM, nearest-upsample, embedding lookup all
  hand-implemented as typed-array loops; no WebGPU/WASM/ONNX dependency,
  so it runs on literally any device this app supports, including the
  iPad 9, just a bit slower per image with the extra forward pass). Both
  the UNet forward pass and the caption-embedding lookup were numerically
  cross-checked against the PyTorch model (max abs diff ~2e-6 / exact
  match respectively) before shipping.
- **Why this is a genuinely different thing from the 🧬 Neural model:** different
  training data (real scraped photographs vs. an analytic renderer), different
  conditioning (a learned text encoder over real captions vs. a hand-designed
  semantic vector), different *class* of generative model (score/noise-prediction
  + iterative refinement, the same paradigm as Stable Diffusion, vs. a direct
  coordinate→RGB field), and a real multi-step sampling loop instead of one
  forward pass.
- **Honest limitation:** at 32×32 working resolution, ~831k parameters, and
  ~27.5k real training photos (vs. billions of images / billions of parameters
  for an actual Stable Diffusion), output is still abstract/impressionistic —
  EMA + CFG made it noticeably cleaner and more coherent (larger solid color
  regions, less speckle noise) but it picks up rough color palette and mood
  from the prompt (e.g. warm tones for "sunset", blue-black for "night sky",
  green for "forest") rather than sharp recognizable objects. Real-world
  photos are a much harder training target than the neural model's clean
  analytic renders, and this is an honest, tiny, from-scratch model, not a
  scaled one. It's offered alongside the neural model precisely so you can
  see the difference between the two model families and the two training
  data sources, side by side, both made from scratch, both fully local.

### A3. 🎨/🐌 Diffusion HQ (ours) — a THIRD self-trained model, native 256×256
- **What it is:** the same genuine DDPM/DDIM recipe as A2, scaled up: a
  6-level FiLM-conditioned convolutional UNet (256→128→64→32→16→8 and back
  up with skip connections) with **multi-head self-attention** at the 16×16
  and 8×8 levels, **~23M parameters** (22.97M UNet + 128k word-embedding
  text encoder), generating **natively at 256×256** — no upscaling from a
  tiny working resolution like A2's 32×32.
- **Data:** `model/scrape_dataset_hq.py` paginates much deeper into the same
  Wikimedia Commons queries as A2, at 256×256 (JPEG, quality 90), plus **43
  additional person-focused queries** (portraits, musicians, cyclists,
  hikers, etc. — added specifically because early versions of this model
  had never seen enough photos of people) for **528 topics total**, into
  `model/dataset_hq.db`. On top of Wikimedia, `model/scrape_multi_source.py`
  adds four more no-signup, legally-clean sources — **Openverse**
  (permissive-license aggregator), **NASA Images**, **Library of Congress**,
  and **Met Museum Open Access** (filtered to public-domain photographs) —
  each with its own license/rights filtering, run to completion twice —
  **33,251 photos combined**. Same caption cleaning as A2. Training adds
  random horizontal flips and mild brightness/contrast jitter.
- **Training** (`model/train_diffusion_hq.py`): unlike the CPU-trained small
  models, this one needs a real GPU — it was trained on a rented RTX 4090
  (bf16 autocast, batch 32, EMA, classifier-free guidance, checkpoint every
  5k steps). At 256×256 the dataset can't be preloaded into RAM, so batches
  are decoded lazily from SQLite with a background prefetch thread.
- **An honest journey, documented because the failures taught the real
  lessons:**
  1. **109M params / 12k steps** → pure colored static. Too much model for
     too little data and far too few steps.
  2. **21M params / 12k steps** → still static. The diagnostic that cracked
     it: sampled output's pixel std was ~1.0 (noise-scale) vs ~0.48 for real
     training photos — the model simply hadn't trained long enough, at any
     size. 256×256 has 64× the pixels of the 32×32 model and needs
     proportionally more optimization.
  3. **21M params / 60k steps** → real structure at last, but blown out —
     fixed at sampling time (no retraining) by **rescaling the model's
     clean-image estimate toward the real data's variance** each DDIM step
     (same idea as dynamic thresholding). Output became smooth abstract
     color compositions.
  4. **+self-attention, +more data, AdamW** — pure-conv UNets only mix
     information within 3×3 neighborhoods per layer, which showed up as
     locally-plausible but globally-incoherent output, so attention blocks
     were added at the two cheapest resolutions. Output became
     prompt-differentiated (city verticals vs. dune sweeps) but still
     impressionistic.
  5. **Current: LATENT diffusion — the change that actually cracked it.**
     Pixel-space 256×256 from scratch is the hardest version of this
     problem; every successful small-budget diffusion project (and Stable
     Diffusion itself) denoises a compressed latent instead. **TAESD** (a
     tiny pretrained autoencoder, MIT license) compresses each photo to a
     32×32×4 latent — 48× fewer values — and our from-scratch **LatentUNet**
     (16M params, attention at 16×16/8×8) learns denoising there, making
     each training step ~50× cheaper. A 40k-step run took 36 minutes
     (~$0.25 of GPU) and produces **recognizable photographic scenes**:
     real-looking dunes at sunset, city lights at night, streams through
     misty vegetation. Initially the autoencoder was the pretrained TAESD
     (credited, MIT); see step 6.
  6. **Current: 100% from scratch.** The pretrained TAESD was replaced by
     **our own autoencoder** (`model/own_ae.py`, same generic conv topology,
     weights trained from zero on our photos with an L1+MSE+edge-gradient
     loss — no borrowed perceptual network), and the bag-of-words text
     encoder was upgraded to an **order-aware** one (`CaptionEncoderV2`:
     learned word + positional embeddings and one masked self-attention
     layer, so "dog chases cat" ≠ "cat chases dog"). One combined GPU run
     (`model/train_diffusion_own.py`) trains AE → encodes the dataset →
     trains UNet + text encoder. Every learned weight in the shipped
     pipeline — generator, text encoder, autoencoder — is now trained by
     us on our own scraped data. Quality matched or improved on the TAESD
     version.
  7. **More data, same limitation.** After the dataset grew from 9,590 to
     15,922 photos (the multi-source expansion above) the model was
     retrained with a proportionally larger budget (AE 30k steps, diffusion
     90k steps, ~90 min total). Scene/landscape prompts (city streets at
     night, desert dunes, snowy peaks) are consistently solid. Person and
     portrait prompts, despite 43 dedicated queries and several thousand
     more person-containing photos, still come out as abstract color/shape
     compositions with no recognizable face or figure — a real result, not
     a bug: at 16M parameters and low-thousands of person photos (spread
     across many distinct poses/scenes/lighting conditions), there just
     isn't enough repeated structure for the model to learn what a face or
     body reliably looks like. Human anatomy needs far more image density
     than scenery to converge, and that's a compute/data scale problem, not
     something the architecture can fix.
  8. **Doubling the data again, on a shorter run.** The Openverse/NASA/LoC
     scrapes were run to completion, growing the dataset from 15,922 to
     **33,251 photos**. Retrained with a shorter 40k-step diffusion run
     (down from 90k, ~37 min total) to see how much of the improvement
     comes from data volume versus step count. Result: scene prompts stayed
     solid, and person/animal prompts started showing faint emergent
     *shape* for the first time — a human silhouette on a beach, a rough
     four-legged form on a horse-riding prompt — where earlier runs gave
     pure abstract color fields. Still nowhere near a recognizable face or
     portrait, but the direction is real: more data density is doing more
     for this model than more optimizer steps at this point.
- **How it runs in the browser:** the weights (~93 MB: our UNet + the
  bundled TAESD decoder under a `taesd_dec.` prefix) are **fetched once on
  demand** from a public Hugging Face model repo (cached by the browser).
  At latent size, plain JavaScript is genuinely fast (~0.3s per UNet pass),
  so both 🎨 and 🐌 share one JS path and generate in roughly half a minute
  on any device — **no WebGPU required**. The JS UNet forward, the TAESD
  decoder port, and the full sampling loop were all numerically verified
  against PyTorch (max abs diff ~3e-7 / ~8e-6).
- **Honest limitation:** ~33k photos and 16M params is still ~5 orders of
  magnitude less data/compute than a real Stable Diffusion. Landscapes and
  scenes (where our training data is dense) come out genuinely
  photographic-looking; specific objects and creatures, and especially
  people/portraits, are still soft, abstract, or implied rather than
  crisply drawn — more data helped narrow this gap but did not close it.

### B. ⚡ Procedural — WebGPU shader (broadest compatibility)
- A hand-written **WebGPU** fragment shader (WGSL) with a **Canvas2D CPU fallback**:
  domain-warped fractal (fBm) fields colored by a keyword palette, reproducible by
  seed. A few MB of RAM, instant, runs essentially anywhere.

### C. 🧠 Diffusion (ONNX Runtime Web) — optional, a real pretrained model
- **ONNX Runtime Web** (WebGPU EP, WASM fallback), lazy-loaded only when selected.
- **"Load recommended model"** pre-wires a genuine, publicly hosted pretrained
  text-to-image model — **[SD-Turbo](https://huggingface.co/schmuell/sd-turbo-ort-web)**
  (Stability AI's distilled, 1–4-step Stable Diffusion, ONNX export) — so you
  don't have to hunt down and manually wire model files yourself. One button
  click downloads it (~2.5 GB, cached by the browser after) and runs a **real
  pipeline I implemented**: a CLIP BPE tokenizer (verified token-for-token
  against the actual Python `transformers.CLIPTokenizer` on this model's
  vocab/merges), the real `text_encoder` ONNX session, a real Euler-discrete
  scheduler loop (matching `scheduler_config.json`: scaled-linear betas,
  epsilon prediction, trailing timestep spacing) driving the real `unet`
  session, and the real `vae_decoder` session to produce final pixels. No
  classifier-free guidance is needed (SD-Turbo uses `guidance_scale=0`), so
  each step is a single UNet forward pass.
- **License:** SD-Turbo is distributed by Stability AI under a
  [non-commercial research license](https://huggingface.co/stabilityai/sd-turbo/blob/main/LICENSE.TXT) —
  personal/research use only. This is disclosed in the app's UI next to the
  load button.
- You can still **bring your own ONNX model file(s)** instead via "Advanced"
  in the Diffusion setup panel; that path runs a best-effort generic forward
  pass (not the full SD-Turbo pipeline above, since tokenizer/scheduler
  specifics are model-dependent).
- A **hard memory-budget guard refuses models too big for the device** —
  checked against the *known* file sizes before a single byte is downloaded —
  and falls back to the neural model, so it never crashes a low-RAM tab.
- **Requires real WebGPU**, not just WASM. I downloaded and statically
  inspected the actual `.onnx` graphs while building this (to get exact
  input/output names, shapes, and dtypes right — e.g. `text_encoder`'s
  `input_ids` is `int32` not the more common `int64`, and the UNet's
  `timestep` input is rank-1 `[steps]`, not a scalar) and confirmed this
  export's graph contains fused/precision-cast ops (`SimplifiedLayerNormFusion`
  + float16 casts) that only ONNX Runtime Web's WebGPU kernels implement —
  it fails to even *initialize* on a plain CPU/WASM execution provider. So
  the app gates this engine on `caps.webgpu` specifically and refuses (with a
  clear message, falling back to the neural model) rather than attempting a
  WASM run that would fail with a cryptic type-mismatch error.
- **Honest limitation on testing:** I verified, against the real model files
  (not assumptions): the tokenizer is byte-for-byte identical to Python's
  `transformers.CLIPTokenizer` on this model's actual vocab/merges (including
  the quirk that `"!"` is a directly-mapped added token sharing id 0 with the
  pad token); the scheduler's `timesteps`/`sigmas` match the published
  `scheduler_config.json` math (e.g. the 1-step schedule resolves to `t=999`
  and the 4-step schedule to `[999, 749, 499, 249]`, matching SD-Turbo's known
  published schedule); and every input/output name, shape and dtype my code
  uses was read directly from the downloaded graphs, not guessed. This
  sandboxed environment has no GPU/WebGPU hardware, so I could not execute the
  actual UNet/text-encoder forward passes here (they require WebGPU, see
  above) — final image *quality* is unverified end-to-end. If you hit an
  issue, please file it with the prompt/device used.

### Will it run on an iPad 9?
**Yes — the default Neural model and the Procedural engine both do, reliably, at
256–512 px.** They use only a few MB of RAM and need no downloads, so they start
instantly and won't be killed by Safari's tab-memory limits. On low-memory/iOS
devices the app auto-selects **256 × 256** and caps the neural render resolution.

The **Diffusion engine is gated by a memory budget**: iPadOS Safari terminates tabs
well below the 3 GB physical limit (often ~1–1.5 GB), so the app computes a
conservative budget (~880 MB of weights for a 3 GB iPad) and **refuses any model
larger than that**. A full Stable-Diffusion model is refused on an iPad 9 — but you
don't need it, because the built-in neural model already runs there.

**Diffusion HQ** (🎨) needs real WebGPU; without it, it falls back to the neural
model with a clear message. The 🐌 CPU variant runs anywhere (its ~125 MB of
float32 weights fit the iPad budget) but takes minutes per image — it's an
explicit opt-in, never an automatic fallback.

---

## 3. How to run it

**Easiest:** double-click `index.html` (or drag it into Chrome/Edge/Firefox/Safari).
The default **Neural model** works immediately, offline — the weights are baked
into the file.

**Recommended (enables WebGPU for the Procedural engine):** serve it over `http://`
with any static server — no build step, no Node required for the app itself:

```bash
# Python (already on most machines)
python3 -m http.server 8000
# then open http://localhost:8000/index.html

# …or any other static server, e.g.:
npx serve .        # (uses Node only as a convenience, not required)
```

> Tip: WebGPU needs a recent Chrome/Edge (or Firefox with WebGPU enabled). If it's
> unavailable, the app automatically uses the Canvas2D CPU fallback — generation
> just runs a bit slower.

### Using the optional Diffusion engine (model file setup)
1. Click the **🧠 Diffusion (ONNX)** engine, then open **Diffusion mode setup**.
2. Download a **small / quantized** text-to-image ONNX model **locally** (e.g. a
   quantized SD-Turbo / Tiny-SD export). On a 3 GB device, prefer the smallest
   int8 variants.
3. Use the file picker to select the `.onnx` (and any `.onnx_data`) files. They
   stay on your machine — nothing is uploaded.
4. Press **Generate**. If the model is too big for the device budget, or needs a
   multi-stage pipeline this minimal bridge doesn't implement, the app explains and
   falls back to the built-in neural model.

### Re-training / improving the neural model
The model is fully reproducible:

```bash
pip install numpy
python3 model/train.py          # ~19 min on CPU (8000 steps) → writes model_web.json + previews
# then paste model_web.json's contents into the <script id="pp-model"> tag in index.html
```

---

## 4. Limitations & how to improve quality later

- **Neural model is tiny and stylized.** With ~23k params it generates smooth
  landscape scenes, not arbitrary photoreal images, and only understands the
  vocabulary it was trained on (times of day, biomes, mountains/neon/fire).
  *Improve by:* (1) growing the network (wider/deeper) and the condition
  vocabulary; (2) training on **real photos** instead of a procedural renderer —
  e.g. encode each image's caption to the condition and regress, or train a small
  conditional GAN/VAE; (3) adding a learned text encoder so free-form prompts map
  to conditions; (4) running the forward pass in a **WebGPU compute shader** for
  speed at higher resolutions.
- **Procedural engine is not semantic** — it maps keywords→palette and
  noise→structure. *Improve by:* a richer keyword/style dictionary and
  compositional templates.
- **Diffusion engine depends on your model.** A full pipeline (CLIP tokenizer +
  text encoder + scheduler UNet loop + VAE decoder) is model-specific; the
  single-file bridge runs a best-effort forward pass. *Improve by:* wiring a proper
  tokenizer (e.g. via `transformers.js`), an Euler/DDIM scheduler loop, and VAE
  decode for a specific export; use int8/4-bit weights and SD-Turbo-style 1–4 step
  models to cut memory and time.
- **Network use:** the Neural and Procedural engines need **no network at all**.
  Only the optional Diffusion engine fetches ONNX Runtime Web once (then cached).

---

## Files
- `index.html` — the entire app (HTML + CSS + JS, fully commented), with the
  trained model weights embedded.
- `model/train.py` — trains the neural model from scratch (pure NumPy).
- `model/model_web.json` — exported weights (also embedded in `index.html`).
- `model/previews.png` — sample outputs from the trained model.
- `model/model.py` — the small conditional UNet architecture (PyTorch).
- `model/scrape_dataset.py` — scrapes `model/dataset_real.db` (run it to
  regenerate), a real SQLite database of photos + captions from the
  Wikimedia Commons API (485 search topics, public-domain/freely-licensed
  only). **Not committed to the repo** — it's a large (~100+ MB), fully
  regenerable build artifact, not something that belongs in source control;
  only the script and the final trained weights are checked in.
- `model/train_diffusion.py` — trains the diffusion model + its own small
  word-embedding text encoder from `model/dataset_real.db` (PyTorch, CPU)
  and exports `model/diffusion_model.json`.
- `model/diffusion_model.json` — exported diffusion model weights, learned
  vocabulary, and word embeddings (also embedded in `index.html`).
- `model/model_hq.py` — the bigger 6-level attention UNet used by
  Diffusion HQ (PyTorch).
- `model/scrape_dataset_hq.py` — scrapes the 256×256 `model/dataset_hq.db`
  (resumable via `--resume`; not committed, same reasoning as above).
- `model/train_diffusion_hq.py` — trains Diffusion HQ (PyTorch, needs a real
  GPU) and exports `diffusion_hq_model.json`, which is published to a public
  Hugging Face model repo and **fetched on demand** by the app rather than
  embedded (~125 MB).
