# PocketPaint — a local, in-browser image generator

A complete **single-file** image generation app (`index.html`) that runs entirely
in the browser. No backend, no cloud API, no login, no paid service, and **no
Node.js needed to run the finished app** — just open the file.

It is built to start instantly and stay within a **~3 GB RAM** budget on low-end
devices, while still using modern in-browser inference tech (WebGPU / WebAssembly
/ ONNX Runtime Web).

---

## 1. What it does

- **Text prompt box**, **Generate** button, and a live **preview** canvas.
- **Progress bar + status messages** while loading/generating.
- **Settings:** image size (256–640), **seed** (reproducible), and **steps**.
- **Capability detection** (WebGPU / WebGL2 / WebAssembly / device RAM / cores)
  with **graceful error handling and fallbacks**.
- **Download PNG** and **copy prompt**.
- Clean, modern dark UI. All code + comments are in the one HTML file.

---

## 2. Model / runtime chosen, and why

The app ships **two engines**:

### A. Local Generative Engine — default, always works
- **Runtime:** a hand-written **WebGPU** fragment shader (WGSL), with an automatic
  **Canvas2D CPU fallback** when WebGPU is missing or flaky.
- **What it is:** real *procedural* image synthesis — domain-warped fractal
  (fBm) fields colored by a **keyword-driven palette** and made reproducible by the
  seed. Your prompt's words (e.g. *ocean, sunset, fire, neon, forest, night*) pick
  the colors and mood; the seed fixes the composition; steps add detail.
- **Why:** it uses only a few **megabytes** of memory, starts **instantly**, and
  runs on essentially any device — exactly the 3 GB / fast-startup / compatibility
  target. It is honest about what it is: **not** a semantic neural net, so it won't
  "understand" complex scenes, but it reliably turns prompts into distinct,
  shareable images on hardware where a real diffusion model simply won't load.

### B. Diffusion (ONNX Runtime Web) — optional, experimental
- **Runtime:** **ONNX Runtime Web** (`onnxruntime-web`) with the **WebGPU** execution
  provider (WASM fallback), lazy-loaded from a CDN only when you select this engine.
- **What it is:** a bridge to run a **real neural text-to-image model** that *you*
  supply as local `.onnx` files (they never leave your browser).
- **Why optional:** full **Stable-Diffusion-class** models need far more than 3 GB
  in a browser tab. Only the smallest **quantized (int8)** exports have any chance
  of fitting, so making this the only path would mean the app fails to start on the
  target hardware. It's provided for capable devices, and the app **falls back to
  the Local engine and tells you why** if the model is missing or too heavy.

**Bottom line / honest limitation:** true general text-to-image is too heavy for a
3 GB device, so the **default working version** is the lightweight procedural
engine, with an honest, optional path to real diffusion for users who have both a
capable device and a small model.

---

## 3. How to run it

**Easiest:** double-click `index.html` (or drag it into Chrome/Edge/Firefox).
The Local engine works immediately, offline.

**Recommended (enables WebGPU reliably):** serve it over `http://` with any static
server — no build step, no Node required for the app itself:

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
4. Press **Generate**. If the model can't load or needs a multi-stage pipeline this
   minimal bridge doesn't implement, the app explains and falls back to the Local
   engine.

---

## 4. Limitations & how to improve quality later

- **Local engine is not semantic.** It maps keywords→palette and noise→structure,
  so it produces beautiful abstract/landscape-like fields, not literal objects.
  *Improve by:* expanding the keyword→style dictionary, adding compositional
  templates (horizon, radial, etc.), or layering simple shape primitives.
- **Diffusion engine depends on your model.** A full pipeline (CLIP tokenizer +
  text encoder + scheduler UNet loop + VAE decoder) is model-specific; the
  single-file bridge runs a best-effort forward pass. *Improve by:* wiring a proper
  tokenizer (e.g. via `transformers.js`), an Euler/DDIM scheduler loop, and the VAE
  decode step for a specific model export.
- **Memory ceiling.** Real diffusion may exceed 3 GB. *Improve by:* using int8/4-bit
  quantized weights, smaller latent sizes, tiled VAE decoding, and SD-Turbo-style
  1–4 step models to cut both memory and time.
- **First diffusion load needs network once** (to fetch ONNX Runtime Web from CDN);
  it's cached afterward. The Local engine needs no network at all.

---

## Files
- `index.html` — the entire app (HTML + CSS + JS, fully commented).
