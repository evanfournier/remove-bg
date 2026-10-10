// Selects whole objects in a photo from a click or a box, with Segment Anything
// (SAM ViT-B, quantised). The page uses it to add objects to a cut-out (the
// plate under the food, the hands holding the subject) or to remove them; the
// AI agent drives it the same way.
//
// Requests (each answered with the same `reqId`):
//   open    {key, file}         encode the image (the slow step, once per image)
//   segment {key, prompt, size} one object as a mask on the size x size model grid
//   close   {key}               forget the image
// Prompt coordinates are fractions of the image's width and height (0..1).

import { coverage, fillSmallHoles } from './compose.js';
import { fetchModel, getManifest, loadOrt, sessionOptions } from './models.js';

// The encoder's input: the image fitted into this canvas, portrait images
// rotated to landscape first (as rembg does: it gives aligned masks).
const W = 1024, H = 684;
const KEEP = 6; // encoded images kept in memory (4 MB each)
const VIEW_MAX = 1024; // longest side of the copy given to the page for the AI agent

const post = (msg, transfer) => self.postMessage(msg, transfer);

let sam = null; // { ort, encoder, decoder }
const images = new Map(); // key -> { embedding, rot, sw, sh }

async function getSam(progress) {
  if (sam) return sam;
  const ort = await loadOrt();
  const manifest = await getManifest();
  if (!manifest.sam) throw new Error('the selection model is not available on this site');
  const { encoder, decoder } = manifest.sam;
  const total = encoder.bytes + decoder.bytes;
  const buffers = [];
  let done = 0;
  for (const part of [encoder, decoder]) {
    buffers.push(await fetchModel(part, manifest, (loaded) =>
      progress(`Downloading the selection model (first time only)… ${Math.round(((done + loaded) / total) * 100)}%`)));
    done += part.bytes;
  }
  progress('Loading the selection model…');
  sam = {
    ort,
    encoder: await ort.InferenceSession.create(buffers[0], sessionOptions),
    decoder: await ort.InferenceSession.create(buffers[1], sessionOptions),
  };
  return sam;
}

async function open(key, file, progress) {
  const { ort, encoder } = await getSam(progress);
  progress('Analysing the photo…');
  let bitmap;
  try {
    bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
  } catch {
    throw new Error('this browser cannot open this file type');
  }
  const rot = bitmap.height > bitmap.width;
  const iw = rot ? bitmap.height : bitmap.width, ih = rot ? bitmap.width : bitmap.height;
  const s = Math.min(W / iw, H / ih);
  const sw = Math.max(1, Math.round(iw * s)), sh = Math.max(1, Math.round(ih * s));
  const ctx = new OffscreenCanvas(W, H).getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, W, H);
  ctx.imageSmoothingQuality = 'high';
  if (rot) {
    ctx.translate(sw, 0);
    ctx.rotate(Math.PI / 2);
    ctx.drawImage(bitmap, 0, 0, sh, sw);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  } else {
    ctx.drawImage(bitmap, 0, 0, sw, sh);
  }

  // A small upright copy of the photo for the AI agent's eyes.
  const vs = Math.min(1, VIEW_MAX / Math.max(bitmap.width, bitmap.height));
  const vw = Math.max(1, Math.round(bitmap.width * vs)), vh = Math.max(1, Math.round(bitmap.height * vs));
  const viewCtx = new OffscreenCanvas(vw, vh).getContext('2d', { willReadFrequently: true });
  viewCtx.imageSmoothingQuality = 'high';
  viewCtx.drawImage(bitmap, 0, 0, vw, vh);
  const view = viewCtx.getImageData(0, 0, vw, vh);
  bitmap.close();

  const rgba = ctx.getImageData(0, 0, W, H).data;
  const hwc = new Float32Array(W * H * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
    hwc[j] = rgba[i];
    hwc[j + 1] = rgba[i + 1];
    hwc[j + 2] = rgba[i + 2];
  }
  const out = await encoder.run({ [encoder.inputNames[0]]: new ort.Tensor('float32', hwc, [H, W, 3]) });
  const embedding = out[encoder.outputNames[0]];

  images.delete(key);
  images.set(key, { embedding, rot, sw, sh });
  while (images.size > KEEP) {
    const [oldest] = images.keys();
    images.get(oldest).embedding.dispose?.();
    images.delete(oldest);
  }
  return { view: { width: vw, height: vh, data: view.data } };
}

/** Image fractions (x, y) -> pixels of the encoder's canvas. */
function toCanvas(x, y, { rot, sw, sh }) {
  const cx = rot ? 1 - y : x, cy = rot ? x : y;
  return [Math.min(1, Math.max(0, cx)) * sw, Math.min(1, Math.max(0, cy)) * sh];
}

async function segment(key, prompt, size) {
  const image = images.get(key);
  if (!image) throw Object.assign(new Error('internal error: image not opened'), { code: 'not-opened' });
  const { ort, decoder } = sam;
  const coords = [], labels = [];
  for (const [x, y, label] of prompt.points || []) {
    coords.push(...toCanvas(x, y, image));
    labels.push(label ? 1 : 0);
  }
  if (prompt.box) {
    const [x0, y0, x1, y1] = prompt.box;
    const [ax, ay] = toCanvas(x0, y0, image), [bx, by] = toCanvas(x1, y1, image);
    coords.push(Math.min(ax, bx), Math.min(ay, by), Math.max(ax, bx), Math.max(ay, by));
    labels.push(2, 3);
  } else {
    coords.push(0, 0); // padding point, as SAM expects without a box
    labels.push(-1);
  }
  const n = labels.length;
  const out = await decoder.run({
    image_embeddings: image.embedding,
    point_coords: new ort.Tensor('float32', Float32Array.from(coords), [1, n, 2]),
    point_labels: new ort.Tensor('float32', Float32Array.from(labels), [1, n]),
    mask_input: new ort.Tensor('float32', new Float32Array(256 * 256), [1, 1, 256, 256]),
    has_mask_input: new ort.Tensor('float32', new Float32Array([0]), [1]),
    orig_im_size: new ort.Tensor('float32', Float32Array.from([H, W]), [2]),
  }, ['masks', 'iou_predictions']);
  const logits = out.masks.data; // H x W, > 0 inside the object
  const score = out.iou_predictions.data[0];
  out.masks.dispose?.();
  const mask = toGrid(logits, image, size);
  // Specks left inside the object: up to 1% of its area (0.2% of the image).
  const area = coverage(mask) * size * size;
  fillSmallHoles(mask, size, Math.min(0.01 * area, 0.002 * size * size));
  return { mask, score };
}

/**
 * Resamples the mask logits onto the model's size x size grid (the image
 * squeezed to a square), rotating back portrait images. SAM's mask is
 * logit > 0; a narrow ramp around 0 (instead of a sigmoid, which leaves
 * objects partly transparent: logits are often only 1-2 inside) keeps the
 * edge anti-aliased.
 */
function toGrid(logits, { rot, sw, sh }, size) {
  const grid = new Uint8Array(size * size);
  for (let gy = 0; gy < size; gy++) {
    const yn = (gy + 0.5) / size;
    for (let gx = 0; gx < size; gx++) {
      const xn = (gx + 0.5) / size;
      const cx = (rot ? 1 - yn : xn) * sw - 0.5, cy = (rot ? xn : yn) * sh - 0.5;
      const x0 = Math.min(sw - 1, Math.max(0, Math.floor(cx))), y0 = Math.min(sh - 1, Math.max(0, Math.floor(cy)));
      const x1 = Math.min(sw - 1, x0 + 1), y1 = Math.min(sh - 1, y0 + 1);
      const fx = Math.min(1, Math.max(0, cx - x0)), fy = Math.min(1, Math.max(0, cy - y0));
      const top = logits[y0 * W + x0] * (1 - fx) + logits[y0 * W + x1] * fx;
      const bottom = logits[y1 * W + x0] * (1 - fx) + logits[y1 * W + x1] * fx;
      const v = top * (1 - fy) + bottom * fy;
      grid[gy * size + gx] = Math.round(255 * Math.min(1, Math.max(0, 0.5 + v)));
    }
  }
  return grid;
}

// One request at a time, in order.
let chain = Promise.resolve();
self.onmessage = ({ data }) => {
  chain = chain.then(async () => {
    const { reqId, type, key } = data;
    const progress = (text) => post({ type: 'progress', reqId, text });
    try {
      if (type === 'open') {
        const result = await open(key, data.file, progress);
        post({ type: 'result', reqId, result }, [result.view.data.buffer]);
      } else if (type === 'segment') {
        const result = await segment(key, data.prompt, data.size);
        post({ type: 'result', reqId, result }, [result.mask.buffer]);
      } else if (type === 'close') {
        images.get(key)?.embedding.dispose?.();
        images.delete(key);
        post({ type: 'result', reqId, result: null });
      }
    } catch (err) {
      console.error(err);
      post({ type: 'error', reqId, message: err?.message || String(err), code: err?.code });
    }
  });
};
