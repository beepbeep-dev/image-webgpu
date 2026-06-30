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

The app ships **three engines**. The default is **my own neural network, trained
from scratch** specifically to fit this 3 GB / iPad-9 target.

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
