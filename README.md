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
- **What it is:** a **~23,000-parameter conditional neural field** (a small MLP /
  CPPN). It takes `(x, y)` pixel coordinates — Fourier-encoded — plus a **12-D
  semantic condition vector** and outputs the RGB colour at that pixel:

  ```
  input  = [ Fourier(x,y, 6 octaves), x, y, r, condition(12) ]   (39 dims)
  hidden = 39 → 96 → 96 → 96   (tanh)
  output = 3 (sigmoid → RGB)
  ```

- **How it was trained** (`model/train.py`, pure NumPy, no GPU, ~10 min on CPU):
  I wrote an analytic **scene renderer** that composes recognizable landscapes —
  sky by **time of day** (night / sunset / day / overcast), ground by **biome**
  (ocean / forest / desert / snow / plain), plus optional **mountains / neon /
  fire** — from the 12-D condition. The network is trained by **distilling** that
  renderer: each step samples fresh random conditions + pixels and regresses the
  network's output to the renderer's (Adam, MSE). Final MSE ≈ `3e-4` (RMSE ≈ 0.017).
- **How it runs in the browser:** the trained weights (`float32`, base64, ~120 KB)
  are **embedded directly in `index.html`** and decoded at startup. Your prompt is
  mapped to the same 12-D condition the model learned, and the network is evaluated
  **per pixel** in optimized JavaScript (typed arrays), rendered at a capped
  internal resolution and upscaled (the field is smooth, so this looks clean). The
  JS forward pass is **bit-for-bit identical** to the Python training code (verified
  against reference pixels).
- **Why this design:** a real Stable-Diffusion model is **gigabytes** and cannot
  load in a ~1–1.5 GB iOS Safari tab. So instead of depending on a giant model, I
  **made my own tiny one** that is genuine neural-network inference, needs **zero
  downloads**, starts **instantly**, renders at **any resolution (≥256 px)**, and
  uses only a few MB of RAM — so it actually runs on an **iPad 9**.
- **Honest limitation:** because it's tiny and trained to reproduce a procedural
  scene generator, it produces **stylized landscapes**, not arbitrary photoreal
  scenes. It genuinely understands the prompt *vocabulary* it was trained on
  (times of day, biomes, mountains/neon/fire) and interpolates smoothly between
  them. See sample outputs in `model/previews.png`.

### B. ⚡ Procedural — WebGPU shader (broadest compatibility)
- A hand-written **WebGPU** fragment shader (WGSL) with a **Canvas2D CPU fallback**:
  domain-warped fractal (fBm) fields colored by a keyword palette, reproducible by
  seed. A few MB of RAM, instant, runs essentially anywhere.

### C. 🧠 Diffusion (ONNX Runtime Web) — optional, experimental
- **ONNX Runtime Web** (WebGPU EP, WASM fallback), lazy-loaded only when selected.
  Runs a **real text-to-image ONNX model that you supply locally** (files stay in
  your browser). A **hard memory-budget guard refuses models too big for the
  device** and falls back to the neural model, so it never crashes a low-RAM tab.

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
python3 model/train.py          # ~10 min on CPU → writes model_web.json + previews
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
