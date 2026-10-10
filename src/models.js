// Model files and the ONNX Runtime, shared by the workers. Models are downloaded
// once and kept in Cache Storage; their file names are content-hashed.

const MODEL_CACHE = 'remove-bg-models';

export const url = (path) => new URL(path, self.location.href).href;

export async function getManifest() {
  const res = await fetch(url('models/manifest.json'), { cache: 'no-cache' });
  if (!res.ok) throw new Error(`could not load the model list (HTTP ${res.status})`);
  return res.json();
}

/** Every model file the site currently uses. */
const manifestFiles = (manifest) => [
  ...manifest.models.map((m) => m.file),
  ...(manifest.sam ? [manifest.sam.encoder.file, manifest.sam.decoder.file] : []),
];

/**
 * Downloads a model file (or reads it from Cache Storage), reporting progress.
 * @param {{file: string, bytes: number}} model
 */
export async function fetchModel(model, manifest, onProgress) {
  const modelUrl = url(model.file);
  let cache = null;
  try {
    cache = await caches.open(MODEL_CACHE);
    await caches.delete('remove-bg-model-v1'); // first release's cache (exact name: shared github.io origin)
    const hit = await cache.match(modelUrl);
    if (hit) {
      const bytes = new Uint8Array(await hit.arrayBuffer());
      if (bytes.length === model.bytes) return bytes;
      await cache.delete(modelUrl); // damaged entry
    }
  } catch {
    // Cache Storage can be unavailable (private mode); just download.
  }

  const res = await fetch(modelUrl);
  if (!res.ok) throw new Error(`could not download the AI model (HTTP ${res.status})`);
  const total = model.bytes;
  const bytes = new Uint8Array(total);
  const reader = res.body.getReader();
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (loaded + value.length > total) throw new Error('the AI model download is larger than expected');
    bytes.set(value, loaded);
    loaded += value.length;
    onProgress(loaded, total);
  }
  if (loaded !== total) throw new Error('the AI model download was interrupted; please retry');

  try {
    await cache?.put(modelUrl, new Response(bytes));
    // Drop the files that this site no longer uses.
    const current = new Set(manifestFiles(manifest).map(url));
    for (const request of await cache.keys()) {
      if (!current.has(request.url)) await cache.delete(request);
    }
  } catch {
    // Not fatal: the model will simply be downloaded again next time.
  }
  return bytes;
}

let ort = null;

/** ONNX Runtime (WebAssembly build), multi-threaded when the page is cross-origin isolated. */
export async function loadOrt() {
  if (ort) return ort;
  ort = await import(url('ort/ort.wasm.min.js'));
  ort.env.wasm.wasmPaths = {
    mjs: url('ort/ort-wasm-simd-threaded.js'),
    wasm: url('ort/ort-wasm-simd-threaded.wasm'),
  };
  // Multi-threading needs cross-origin isolation (see coi-serviceworker).
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1;
  return ort;
}

/**
 * Models run on the CPU (WebAssembly): ONNX Runtime's WebGPU backend lacks
 * operators they need (ceil-mode MaxPool, DeformConv).
 */
export const sessionOptions = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
