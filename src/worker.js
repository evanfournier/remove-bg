// Runs the background-removal pipeline off the main thread, one image at a time
// (the page queues the jobs and restarts this worker to switch model or cancel).
//
// Quality guarantees:
//  - the output has exactly the input's width and height;
//  - only the alpha channel is computed: the RGB of every fully opaque source
//    pixel is the decoded original, bit for bit;
//  - the output is a lossless PNG; the source's colour profile is kept when the
//    browser can decode without colour conversion (Chrome, Edge, Firefox).
// The AI model only sees a 1024x1024 copy of the image to compute the mask;
// the mask is then cleaned, snapped to the photo's own edges at full
// resolution (refine.js) and applied to the original pixels. The mask is also
// sent to the page: objects added or removed there (segment-worker.js) come
// back as an edited mask, which is rendered the same way.

import { extractColorChunks } from './color.js';
import { isOutOfMemoryMessage } from './errors.js';
import { applyMask, prepareMask, resizeRGBA } from './mask.js';
import { fetchModel, getManifest, loadOrt, sessionOptions } from './models.js';
import { encodePng } from './png.js';
import { refineAlpha } from './refine.js';

const PREVIEW_MAX = 2048; // longest side of the on-screen preview

const post = (msg, transfer) => self.postMessage(msg, transfer);

let ort = null;
let engine = null; // { id, session, model }

/** The manifest entry of a model (the default one if the id is unknown). */
async function getModel(modelId) {
  const manifest = await getManifest();
  const { models } = manifest;
  const model = models.find((m) => m.id === modelId) || models.find((m) => m.default) || models[0];
  return { model, manifest };
}

async function getEngine(modelId, onProgress) {
  const { model, manifest } = await getModel(modelId);
  if (engine?.id === model.id) return engine;
  if (engine) throw new Error('internal error: one worker per model'); // the page restarts us

  ort = await loadOrt();
  let bytes;
  try {
    bytes = await fetchModel(model, manifest, (loaded, total) => {
      post({ type: 'model-progress', label: model.label, loaded, total, done: false });
      onProgress(loaded / total);
    });
  } finally {
    post({ type: 'model-progress', done: true });
  }
  const session = await ort.InferenceSession.create(bytes, sessionOptions);
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
    throw new Error('this image is too large for this browser');
  }
  for (let y = 0; y < height; y += bandHeight) {
    const h = Math.min(bandHeight, height - y);
    ctx.clearRect(0, 0, width, bandHeight);
    ctx.drawImage(bitmap, 0, y, width, h, 0, 0, width, h); // 1:1 copy, no resampling
    pixels.set(ctx.getImageData(0, 0, width, h).data, y * width * 4);
  }
  return pixels;
}

/** Normalised planar RGB from the size x size RGBA copy of the image. */
function modelInput(rgba, { size, mean, std }) {
  const plane = size * size;
  const input = new Float32Array(3 * plane);
  for (let c = 0; c < 3; c++) {
    const scale = 1 / (255 * std[c]), offset = mean[c] / std[c];
    for (let i = c, j = c * plane, end = j + plane; j < end; i += 4, j++) input[j] = rgba[i] * scale - offset;
  }
  return input;
}

/** Out of memory while running the model (a lighter model may then fit). */
const isOutOfMemory = (err) =>
  !err?.imageStage && (err instanceof RangeError || isOutOfMemoryMessage(err?.message || err));

/**
 * @param {Uint8Array} [edited] the mask on the model's grid, as edited on the
 *   page (objects added or removed): the model is then not run again.
 */
async function removeBackground(id, file, modelId, sharpen, edited) {
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
    throw Object.assign(new Error('this browser cannot open this file type (try JPEG, PNG or WebP)'), { imageStage: true });
  }
  const { width, height } = bitmap;
  let pixels;
  try {
    pixels = readPixels(bitmap, width, height);
  } catch (err) {
    // Memory spent on the image itself: a lighter model would not help.
    const memory = err instanceof RangeError || isOutOfMemoryMessage(err?.message);
    throw Object.assign(new Error(memory ? 'not enough memory to open this image' : err.message), { imageStage: true });
  } finally {
    bitmap.close();
  }

  let model, session;
  if (edited) ({ model } = await getModel(modelId));
  else ({ session, model } = await getEngine(modelId, (p) => stage('Downloading AI model (first time only)…', p)));
  // The same squeezed copy feeds the model and guides the edge refinement.
  const guide = !edited || sharpen ? resizeRGBA(pixels, width, height, model.size, model.size) : null;

  let mask, modelMask = null;
  if (edited) {
    if (edited.length !== model.size * model.size) throw new Error('internal error: edited mask size');
    mask = new Float32Array(edited.length);
    for (let i = 0; i < edited.length; i++) mask[i] = edited[i] / 255;
  } else {
    stage('Detecting the subject…', null);
    const tensor = new ort.Tensor('float32', modelInput(guide, model), [1, 3, model.size, model.size]);
    const out = await session.run({ [session.inputNames[0]]: tensor }, [session.outputNames[0]]);
    mask = prepareMask(out[session.outputNames[0]].data, model);
    out[session.outputNames[0]].dispose?.();
    // Sent to the page, which edits it when objects are added or removed.
    modelMask = new Uint8Array(mask.length);
    for (let i = 0; i < mask.length; i++) modelMask[i] = Math.round(mask[i] * 255);
  }

  stage(sharpen ? 'Refining the edges…' : 'Building the full-resolution cut-out…', 0.75);
  if (sharpen) refineAlpha(pixels, width, height, mask, model.size, guide, model.refine);
  else applyMask(pixels, width, height, mask, model.size);

  stage('Saving lossless PNG…', 0.85);
  const chunks = keepColorSpace ? colorChunks : [];
  const blob = await encodePng(pixels, width, height, chunks);
  // Preview: same pixels and colour profile as the PNG, so it looks identical.
  const scale = Math.min(1, PREVIEW_MAX / Math.max(width, height));
  const pw = Math.max(1, Math.round(width * scale)), ph = Math.max(1, Math.round(height * scale));
  const preview = await encodePng(resizeRGBA(pixels, width, height, pw, ph, true), pw, ph, chunks);

  post({
    type: 'done',
    id,
    blob,
    preview,
    width,
    height,
    colorProfile: keepColorSpace,
    modelLabel: model.label,
    mask: modelMask,
    sharpen,
    seconds: (performance.now() - started) / 1000,
  }, modelMask ? [modelMask.buffer] : []);
}

let busy = false;
self.onmessage = async ({ data }) => {
  if (data.type !== 'process' || busy) return;
  busy = true;
  try {
    await removeBackground(data.id, data.file, data.model, data.sharpen !== false, data.mask);
  } catch (err) {
    console.error(err);
    post({ type: 'error', id: data.id, message: err?.message || String(err), outOfMemory: isOutOfMemory(err) });
  } finally {
    busy = false;
  }
};
