// Runs the whole background-removal pipeline off the main thread.
//
// Quality guarantees:
//  - the output has exactly the input's width and height;
//  - every RGB value is the decoded original, only the alpha channel changes;
//  - the output is a lossless PNG that keeps the original colour profile.
// The AI model only sees a 1024x1024 copy of the image to compute the mask;
// the mask is then scaled up and applied to the full-resolution pixels.

import { extractColorChunks } from './color.js';
import { encodePng } from './png.js';

const MODEL_URL = new URL('models/isnet.onnx', self.location.href).href;
const MODEL_CACHE = 'remove-bg-model-v1';
const SIZE = 1024; // model input resolution
// The raw model output leaves solid subjects slightly see-through (alpha
// 230-254) and leaves faint haze in the background. Map confident values to
// fully opaque / fully transparent and keep only the real edges soft.
const MASK_LOW = 0.2;
const MASK_HIGH = 0.9;

const post = (msg) => self.postMessage(msg);

let ort = null;
let session = null;

async function fetchModel() {
  let cache = null;
  try {
    cache = await caches.open(MODEL_CACHE);
    const hit = await cache.match(MODEL_URL);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  } catch {
    // Cache Storage can be unavailable (private mode); just download.
  }

  const res = await fetch(MODEL_URL);
  if (!res.ok) throw new Error(`Could not download the model (HTTP ${res.status})`);
  const total = Number(res.headers.get('content-length')) || 0;
  const reader = res.body.getReader();
  const parts = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    loaded += value.length;
    post({ type: 'model-progress', loaded, total });
  }
  const model = new Uint8Array(loaded);
  let o = 0;
  for (const p of parts) { model.set(p, o); o += p.length; }

  try {
    await cache?.put(MODEL_URL, new Response(model));
  } catch {
    // Not fatal: the model will simply be downloaded again next time.
  }
  return model;
}

// The model runs on the CPU (WebAssembly). ONNX Runtime's WebGPU backend cannot
// run this model (MaxPool with ceil_mode is not implemented there).
async function createSession() {
  ort = await import(new URL('ort/ort.wasm.min.js', self.location.href).href);
  ort.env.wasm.wasmPaths = {
    mjs: new URL('ort/ort-wasm-simd-threaded.js', self.location.href).href,
    wasm: new URL('ort/ort-wasm-simd-threaded.wasm', self.location.href).href,
  };
  // Multi-threading needs cross-origin isolation (see coi-serviceworker).
  ort.env.wasm.numThreads = self.crossOriginIsolated
    ? Math.min(8, navigator.hardwareConcurrency || 4)
    : 1;

  post({ type: 'engine', text: 'Downloading AI model (first time only)…' });
  const model = await fetchModel();
  post({ type: 'engine', text: 'Starting AI model…' });
  const created = await ort.InferenceSession.create(model, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  const threads = ort.env.wasm.numThreads;
  post({ type: 'engine', text: `Engine: CPU, ${threads} thread${threads > 1 ? 's' : ''}` });
  return created;
}

/** Returns the 1024x1024 foreground mask (values 0..1). */
async function predictMask(input) {
  session ??= await createSession();
  const tensor = new ort.Tensor('float32', input, [1, 3, SIZE, SIZE]);
  const out = await session.run({ [session.inputNames[0]]: tensor });
  const mask = out[session.outputNames[0]].data;

  const levels = new Float32Array(mask.length);
  for (let i = 0; i < mask.length; i++) {
    const v = (mask[i] - MASK_LOW) / (MASK_HIGH - MASK_LOW);
    levels[i] = v < 0 ? 0 : v > 1 ? 1 : v;
  }
  return levels;
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
 * really honours colorSpaceConversion: 'none' and that it changes anything; if
 * not, decode normally (to sRGB) and write no profile.
 */
async function shouldKeepColorSpace(file) {
  const read = async (keep) => {
    const bitmap = await createImageBitmap(file, {
      ...decodeOptions(keep),
      resizeWidth: 64,
      resizeHeight: 64,
    });
    const ctx = canvas2d(64, 64);
    ctx.drawImage(bitmap, 0, 0, 64, 64);
    bitmap.close();
    return ctx.getImageData(0, 0, 64, 64).data;
  };
  const [raw, converted] = await Promise.all([read(true), read(false)]);
  return raw.some((v, i) => v !== converted[i]);
}

function modelInput(bitmap) {
  const ctx = canvas2d(SIZE, SIZE);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bitmap, 0, 0, SIZE, SIZE);
  const rgba = ctx.getImageData(0, 0, SIZE, SIZE).data;
  const plane = SIZE * SIZE;
  const input = new Float32Array(3 * plane);
  for (let i = 0, j = 0; j < plane; i += 4, j++) {
    input[j] = (rgba[i] - 128) / 256;
    input[j + plane] = (rgba[i + 1] - 128) / 256;
    input[j + 2 * plane] = (rgba[i + 2] - 128) / 256;
  }
  return input;
}

/** Scales the mask to full size (bilinear) and multiplies it into the alpha channel. */
function applyMask(pixels, width, height, mask) {
  const x0 = new Int32Array(width), x1 = new Int32Array(width), fx = new Float32Array(width);
  for (let x = 0; x < width; x++) {
    const p = Math.min(Math.max((x + 0.5) * (SIZE / width) - 0.5, 0), SIZE - 1);
    x0[x] = Math.floor(p);
    x1[x] = Math.min(x0[x] + 1, SIZE - 1);
    fx[x] = p - x0[x];
  }
  for (let y = 0; y < height; y++) {
    const q = Math.min(Math.max((y + 0.5) * (SIZE / height) - 0.5, 0), SIZE - 1);
    const r0 = Math.floor(q) * SIZE;
    const r1 = Math.min(Math.floor(q) + 1, SIZE - 1) * SIZE;
    const fy = q - Math.floor(q);
    for (let x = 0, o = y * width * 4 + 3; x < width; x++, o += 4) {
      const top = mask[r0 + x0[x]] + (mask[r0 + x1[x]] - mask[r0 + x0[x]]) * fx[x];
      const bottom = mask[r1 + x0[x]] + (mask[r1 + x1[x]] - mask[r1 + x0[x]]) * fx[x];
      pixels[o] = pixels[o] * (top + (bottom - top) * fy); // keeps existing transparency
    }
  }
}

async function removeBackground(id, file) {
  const started = performance.now();
  const status = (text) => post({ type: 'status', id, text });

  status('Reading image…');
  const bytes = new Uint8Array(await file.arrayBuffer());
  const colorChunks = await extractColorChunks(bytes).catch(() => []);
  const keepColorSpace =
    colorChunks.length > 0 && (await shouldKeepColorSpace(file).catch(() => false));

  const bitmap = await createImageBitmap(file, decodeOptions(keepColorSpace));
  const { width, height } = bitmap;
  const pixels = (() => {
    const ctx = canvas2d(width, height);
    ctx.drawImage(bitmap, 0, 0);
    return ctx.getImageData(0, 0, width, height).data;
  })();
  const input = modelInput(bitmap);
  bitmap.close();

  status('Detecting the subject…');
  const mask = await predictMask(input);

  status('Building full-resolution cut-out…');
  applyMask(pixels, width, height, mask);

  status('Saving lossless PNG…');
  const blob = await encodePng(pixels, width, height, keepColorSpace ? colorChunks : []);

  post({
    type: 'done',
    id,
    blob,
    width,
    height,
    colorProfile: keepColorSpace,
    seconds: (performance.now() - started) / 1000,
  });
}

// Process images one at a time: large photos need a lot of memory.
let queue = Promise.resolve();
self.onmessage = ({ data }) => {
  queue = queue.then(() =>
    removeBackground(data.id, data.file).catch((err) => {
      console.error(err);
      post({ type: 'error', id: data.id, message: err?.message || String(err) });
    }),
  );
};
