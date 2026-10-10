// Page side of segment-worker.js: select whole objects in an image.

const KEEP = 6; // same as the worker: images kept ready

let worker = null;
let nextReq = 0;
const pending = new Map(); // reqId -> { resolve, reject, onProgress }
const opened = new Map(); // key -> Promise<view>

function failAll(message) {
  for (const { reject } of pending.values()) reject(new Error(message));
  pending.clear();
  opened.clear();
}

function getWorker() {
  if (worker) return worker;
  worker = new Worker('segment-worker.js', { type: 'module' });
  worker.onmessage = ({ data }) => {
    const call = pending.get(data.reqId);
    if (!call) return;
    if (data.type === 'progress') return call.onProgress?.(data.text);
    pending.delete(data.reqId);
    if (data.type === 'result') call.resolve(data.result);
    else call.reject(Object.assign(new Error(data.message), { code: data.code }));
  };
  worker.onerror = (event) => {
    event.preventDefault();
    worker?.terminate();
    worker = null;
    failAll(event.message || 'the selection tool stopped unexpectedly; please retry');
  };
  return worker;
}

function call(message, onProgress) {
  const reqId = nextReq++;
  return new Promise((resolve, reject) => {
    pending.set(reqId, { resolve, reject, onProgress });
    getWorker().postMessage({ ...message, reqId });
  });
}

/**
 * Prepares an image for selection (downloads the model the first time, then
 * encodes the image: a few seconds). Resolves to a small upright RGBA copy of
 * the photo ({width, height, data}, longest side 1024).
 */
export function openImage(key, file, onProgress) {
  let view = opened.get(key);
  if (!view) {
    view = call({ type: 'open', key, file }, onProgress).then((result) => result.view);
    view.catch(() => opened.delete(key));
    opened.delete(key);
    opened.set(key, view);
    while (opened.size > KEEP) opened.delete(opened.keys().next().value);
  }
  return view;
}

/**
 * Selects one object: prompt = { points: [[x, y, 1 inside | 0 outside]], box: [x0, y0, x1, y1] | null }
 * in fractions of the image. Resolves to { mask: Uint8Array (size x size), score }.
 */
export async function selectObject(key, file, prompt, size, onProgress) {
  await openImage(key, file, onProgress);
  try {
    return await call({ type: 'segment', key, prompt, size });
  } catch (err) {
    if (err.code !== 'not-opened') throw err;
    opened.delete(key); // the worker made room for other images: encode it again
    await openImage(key, file, onProgress);
    return call({ type: 'segment', key, prompt, size });
  }
}

export function closeImage(key) {
  if (!opened.delete(key) || !worker) return;
  call({ type: 'close', key }).catch(() => {});
}

/** Stops everything (the page was cleared). */
export function resetSelection() {
  worker?.terminate();
  worker = null;
  failAll('cancelled');
}
