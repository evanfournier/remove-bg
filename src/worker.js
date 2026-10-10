// Runs the whole background-removal pipeline off the main thread.
//
// Quality guarantees:
//  - the output has exactly the input's width and height;
//  - only the alpha channel is computed: the RGB of every fully opaque source
//    pixel is the decoded original, bit for bit;
//  - the output is a lossless PNG; the source's colour profile is kept when the
//    browser can decode without colour conversion (Chrome, Edge, Firefox).
// The AI model only sees a 1024x1024 copy of the image to compute the mask;
// the mask is then cleaned, snapped to the photo's own edges at full
// resolution (refine.js) and applied to the original pixels.

import { extractColorChunks } from './color.js';
import { applyMask, downscaleAlpha, prepareMask } from './mask.js';
import { encodePng } from './png.js';
import { refineAlpha } from './refine.js';

const MODEL_CACHE = 'remove-bg-models';
const PREVIEW_MAX = 2048; // longest side of the on-screen preview

const post = (msg, transfer) => self.postMessage(msg, transfer || []);
const url = (path) => new URL(path, self.location.href).href;

let ort = null;
let manifest = null;
let engine = null; // { id, session, model }

async function getManifest() {
  if (!manifest) {
    const res = await fetch(url('models/manifest.json'), { cache: 'no-cache' });
    if (!res.ok) throw new Error(`Could not load the model list (HTTP ${res.status})`);
    manifest = await res.json();
  }
  return manifest;
}

/** Downloads a model (or reads it from Cache Storage), reporting progress. */
async function fetchModel(model, onProgress) {
  const modelUrl = url(model.file);
  let cache = null;
  try {
    cache = await caches.open(MODEL_CACHE);
    const hit = await cache.match(modelUrl);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  } catch {
    // Cache Storage can be unavailable (private mode); just download.
  }

  const res = await fetch(modelUrl);
  if (!res.ok) throw new Error(`Could not download the AI model (HTTP ${res.status})`);
  const total = model.bytes;
  const bytes = new Uint8Array(total);
  const reader = res.body.getReader();
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (loaded + value.length > total) throw new Error('The AI model download is larger than expected');
    bytes.set(value, loaded);
    loaded += value.length;
    onProgress(loaded, total);
  }
  if (loaded !== total) throw new Error('The AI model download was interrupted; please retry');

  try {
    await cache?.put(modelUrl, new Response(bytes));
    // Model files are content-hashed: drop any that this site no longer uses.
    const current = new Set((await getManifest()).models.map((m) => url(m.file)));
    for (const request of await cache.keys()) {
      if (!current.has(request.url)) await cache.delete(request);
    }
  } catch {
    // Not fatal: the model will simply be downloaded again next time.
  }
  return bytes;
}

async function getEngine(modelId, onProgress) {
  const { models } = await getManifest();
  const model = models.find((m) => m.id === modelId) || models.find((m) => m.default) || models[0];
  if (engine?.id === model.id) return engine;

  if (!ort) {
    ort = await import(url('ort/ort.wasm.min.js'));
    ort.env.wasm.wasmPaths = {
      mjs: url('ort/ort-wasm-simd-threaded.js'),
      wasm: url('ort/ort-wasm-simd-threaded.wasm'),
    };
    // Multi-threading needs cross-origin isolation (see coi-serviceworker).
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1;
  }
  // Only one model in memory at a time: they share the 4 GB WebAssembly heap.
  if (engine) {
    await engine.session.release().catch(() => {});
    engine = null;
  }

  const bytes = await fetchModel(model, (loaded, total) => {
    post({ type: 'model-progress', label: model.label, loaded, total, done: false });
    onProgress(loaded / total);
  });
  post({ type: 'model-progress', done: true });
  // The model runs on the CPU (WebAssembly): ONNX Runtime's WebGPU backend lacks
  // operators these models need (ceil-mode MaxPool, DeformConv).
  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  const threads = ort.env.wasm.numThreads;
  post({ type: 'engine', text: `CPU · ${threads} thread${threads > 1 ? 's' : ''}` });
  engine = { id: model.id, session, model };
  return engine;
}

function decodeOptions(keepColorSpace) {
  return {
    imageOrientation: 'from-image', // apply EXIF rotation like every viewer does
    premultiplyAlpha: 'none',
    colorSpaceConversion: keepColorSpace ? 'none' : 'default',
  };
}

function canvas2d(width, height) {
  return new OffscreenCanvas(width, height).getContext('2d', { willReadFrequently: true });
}

/**
 * We want the raw pixel values (no colour conversion) whenever we can attach the
 * source's colour profile to the output. Check on a small copy that the browser
 * really honours colorSpaceConversion: 'none' (Safari ignores it) and that it
 * changes anything; if not, decode normally (to sRGB) and write no profile.
 */
async function shouldKeepColorSpace(file) {
  const read = async (keep) => {
    const bitmap = await createImageBitmap(file, { ...decodeOptions(keep), resizeWidth: 64, resizeHeight: 64 });
    const ctx = canvas2d(64, 64);
    ctx.drawImage(bitmap, 0, 0, 64, 64);
    bitmap.close();
    return ctx.getImageData(0, 0, 64, 64).data;
  };
  const [raw, converted] = await Promise.all([read(true), read(false)]);
  return raw.some((v, i) => v !== converted[i]);
}

/**
 * Copies the full-resolution pixels out of the bitmap in horizontal bands. A
 * single huge canvas is not an option: Safari cannot allocate one above 16.7 MP
 * (iOS 17) and then silently reads back transparent black.
 */
function readPixels(bitmap, width, height) {
  const pixels = new Uint8ClampedArray(width * height * 4);
  const bandHeight = Math.max(1, Math.min(height, Math.floor((1 << 22) / width)));
  const ctx = canvas2d(width, bandHeight);
  ctx.fillRect(0, 0, 1, 1);
  if (ctx.getImageData(0, 0, 1, 1).data[3] !== 255) {
    throw new Error('This image is too large for this browser');
  }
  for (let y = 0; y < height; y += bandHeight) {
    const h = Math.min(bandHeight, height - y);
    ctx.clearRect(0, 0, width, bandHeight);
    ctx.drawImage(bitmap, 0, y, width, h, 0, 0, width, h); // 1:1 copy, no resampling
    pixels.set(ctx.getImageData(0, 0, width, h).data, y * width * 4);
  }
  return pixels;
}

/**
 * The model's input: the image squeezed to size x size, normalised, planar RGB.
 * The squeezed RGBA is returned too: it guides the edge refinement.
 */
function modelInput(bitmap, { size, mean, std }) {
  const ctx = canvas2d(size, size);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bitmap, 0, 0, size, size);
  const rgba = ctx.getImageData(0, 0, size, size).data;
  const plane = size * size;
  const input = new Float32Array(3 * plane);
  for (let c = 0; c < 3; c++) {
    const scale = 1 / (255 * std[c]), offset = mean[c] / std[c];
    for (let i = c, j = c * plane, end = j + plane; j < end; i += 4, j++) input[j] = rgba[i] * scale - offset;
  }
  return { input, guide: rgba };
}

/** A small RGBA preview of the result for the page (the PNG is full size). */
async function makePreview(bitmap, pixels, width, height) {
  const scale = Math.min(1, PREVIEW_MAX / Math.max(width, height));
  const pw = Math.max(1, Math.round(width * scale)), ph = Math.max(1, Math.round(height * scale));
  const ctx = canvas2d(pw, ph);
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bitmap, 0, 0, pw, ph);
  const image = ctx.getImageData(0, 0, pw, ph);
  const alpha = downscaleAlpha(pixels, width, height, pw, ph);
  for (let i = 0; i < alpha.length; i++) image.data[4 * i + 3] = alpha[i];
  ctx.putImageData(image, 0, 0);
  return ctx.canvas.convertToBlob({ type: 'image/webp', quality: 0.9 });
}

async function removeBackground(id, file, modelId, sharpen) {
  const started = performance.now();
  const stage = (text, progress = null) => post({ type: 'stage', id, text, progress });

  stage('Reading image…', 0.02);
  const bytes = new Uint8Array(await file.arrayBuffer());
  const colorChunks = await extractColorChunks(bytes).catch(() => []);
  const keepColorSpace = colorChunks.length > 0 && (await shouldKeepColorSpace(file).catch(() => false));

  let bitmap;
  try {
    bitmap = await createImageBitmap(file, decodeOptions(keepColorSpace));
  } catch {
    throw new Error('this browser cannot open this file type (try JPEG, PNG or WebP)');
  }
  const { width, height } = bitmap;

  try {
    const { session, model } = await getEngine(modelId, (p) => stage('Downloading AI model (first time only)…', p));
    const { input, guide } = modelInput(bitmap, model);

    stage('Detecting the subject…', null);
    const tensor = new ort.Tensor('float32', input, [1, 3, model.size, model.size]);
    const out = await session.run({ [session.inputNames[0]]: tensor }, [session.outputNames[0]]);
    const mask = prepareMask(out[session.outputNames[0]].data, model);
    out[session.outputNames[0]].dispose?.();

    stage(sharpen ? 'Refining the edges…' : 'Building the full-resolution cut-out…', 0.75);
    const pixels = readPixels(bitmap, width, height);
    if (sharpen) refineAlpha(pixels, width, height, mask, model.size, guide);
    else applyMask(pixels, width, height, mask, model.size);
    const preview = await makePreview(bitmap, pixels, width, height);

    stage('Saving lossless PNG…', 0.9);
    const blob = await encodePng(pixels, width, height, keepColorSpace ? colorChunks : []);
    post({
      type: 'done',
      id,
      blob,
      preview,
      width,
      height,
      colorProfile: keepColorSpace,
      modelLabel: model.label,
      sharpen,
      seconds: (performance.now() - started) / 1000,
    });
  } finally {
    bitmap.close();
  }
}

// Process images one at a time: large photos need a lot of memory.
const cancelled = new Set();
let queue = Promise.resolve();
self.onmessage = ({ data }) => {
  if (data.type === 'cancel') {
    for (const id of data.ids) cancelled.add(id);
    return;
  }
  if (data.type !== 'process') return;
  queue = queue.then(async () => {
    if (cancelled.has(data.id)) return;
    try {
      await removeBackground(data.id, data.file, data.model, data.sharpen !== false);
    } catch (err) {
      console.error(err);
      post({ type: 'error', id: data.id, message: err?.message || String(err) });
    }
  });
};
