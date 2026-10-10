// Assembles the static website in dist/:
//   - the app (src/)
//   - ONNX Runtime Web (runs the AI models with WebAssembly)
//   - the AI models + models/manifest.json describing them
//   - coi-serviceworker (enables multi-threading on GitHub Pages)
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { copyFile, cp, mkdir, readdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const root = new URL('../', import.meta.url);
const dist = new URL('dist/', root);
const modules = new URL('node_modules/', root);
const cache = new URL('.cache/', root);
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

await rm(dist, { recursive: true, force: true });
await cp(new URL('src/', root), dist, { recursive: true });

// ONNX Runtime: the JS API plus its WebAssembly build. Files are renamed
// .mjs -> .js so any static host serves them with a JavaScript MIME type.
const ort = new URL('onnxruntime-web/dist/', modules);
await mkdir(new URL('ort/', dist));
await copyFile(new URL('ort.wasm.min.mjs', ort), new URL('ort/ort.wasm.min.js', dist));
await copyFile(new URL('ort-wasm-simd-threaded.mjs', ort), new URL('ort/ort-wasm-simd-threaded.js', dist));
await copyFile(new URL('ort-wasm-simd-threaded.wasm', ort), new URL('ort/ort-wasm-simd-threaded.wasm', dist));

await copyFile(
  new URL('coi-serviceworker/coi-serviceworker.min.js', modules),
  new URL('coi-serviceworker.min.js', dist),
);

// ---- Models -------------------------------------------------------------------

/** ISNet (general use, fp16) as packaged by IMG.LY: 4 MB chunks named by SHA-256. */
async function isnet() {
  const data = new URL('@imgly/background-removal-data/dist/', modules);
  const resources = JSON.parse(await readFile(new URL('resources.json', data), 'utf8'));
  const entry = resources['/models/medium'];
  const chunks = await Promise.all(
    entry.chunks.map(async ({ hash }) => {
      const bytes = await readFile(new URL(hash, data));
      if (sha256(bytes) !== hash) throw new Error(`ISNet chunk ${hash} is corrupted`);
      return bytes;
    }),
  );
  const model = Buffer.concat(chunks);
  if (model.length !== entry.size) throw new Error('ISNet size mismatch');
  return model;
}

/**
 * BiRefNet-lite (MIT), published as ONNX by the rembg project. It is converted by
 * scripts/convert_birefnet.py so it fits in the browser's memory (see that file).
 * Needs Python 3 with onnx and numpy (a .venv in the repo is used automatically);
 * the result is cached in .cache/.
 */
const BIREFNET = {
  url: 'https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx',
  sha256: '5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333',
};

/** Python to run the conversion: $PYTHON, else a local .venv, else the system one. */
function python() {
  if (process.env.PYTHON) return process.env.PYTHON;
  const venv = new URL(process.platform === 'win32' ? '.venv/Scripts/python.exe' : '.venv/bin/python', root);
  if (existsSync(venv)) return fileURLToPath(venv);
  return process.platform === 'win32' ? 'python' : 'python3';
}

async function birefnet() {
  const script = await readFile(new URL('scripts/convert_birefnet.py', root));
  const key = sha256(BIREFNET.sha256 + sha256(script)).slice(0, 16);
  const converted = new URL(`birefnet-${key}.onnx`, cache);
  await mkdir(cache, { recursive: true });
  for (const name of await readdir(cache)) {
    if (/^birefnet-[0-9a-f]{16}\.(onnx|tmp)$/.test(name) && !name.startsWith(`birefnet-${key}.onnx`)) {
      await rm(new URL(name, cache)); // conversions from older script versions
    }
  }
  if (existsSync(converted)) return readFile(converted);

  const source = new URL('birefnet-source.onnx', cache);
  if (!existsSync(source) || sha256(await readFile(source)) !== BIREFNET.sha256) {
    console.log('Downloading BiRefNet-lite (224 MB)…');
    const res = await fetch(BIREFNET.url);
    if (!res.ok) throw new Error(`BiRefNet download failed: HTTP ${res.status}`);
    const bytes = Buffer.from(await res.arrayBuffer());
    if (sha256(bytes) !== BIREFNET.sha256) throw new Error('BiRefNet download has the wrong checksum');
    await writeFile(source, bytes);
  }

  console.log('Converting BiRefNet-lite for the browser…');
  const tmp = new URL(`birefnet-${key}.tmp`, cache);
  const result = spawnSync(
    python(),
    [fileURLToPath(new URL('scripts/convert_birefnet.py', root)), fileURLToPath(source), fileURLToPath(tmp)],
    { stdio: 'inherit' },
  );
  if (result.status !== 0) {
    throw new Error(`conversion failed (${result.error?.message || `exit ${result.status}`}); ` +
      'set up Python with: python3 -m venv .venv && .venv/bin/pip install onnx numpy');
  }
  await rename(tmp, converted);
  return readFile(converted);
}

const MODELS = [
  {
    id: 'birefnet',
    label: 'Best quality (BiRefNet)',
    load: birefnet,
    // ImageNet normalisation; the output is a logit.
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
    sigmoid: true,
    // Confident model: only trim faint haze and near-opaque noise.
    levels: [0.03, 0.97],
    // Its edges are already well placed: 2 sharpening passes keep fur natural.
    refine: { iters: 2 },
    // Needs ~2.8 GB of WebAssembly memory; phones may not have it.
    fallback: 'isnet',
  },
  {
    id: 'isnet',
    label: 'Faster (ISNet)',
    load: isnet,
    // (x - 128) / 256 on 0..255 values, as in IMG.LY's pipeline; output in 0..1.
    mean: [128 / 255, 128 / 255, 128 / 255],
    std: [256 / 255, 256 / 255, 256 / 255],
    sigmoid: false,
    // Raw ISNet leaves solid subjects slightly see-through and faint haze around.
    levels: [0.2, 0.9],
    refine: { iters: 3 },
  },
];

await mkdir(new URL('models/', dist));
const manifest = { models: [] };
for (const { load, ...model } of MODELS) {
  let bytes;
  try {
    bytes = await load();
  } catch (err) {
    // Deploys must ship every model; local builds can go on without one.
    if (process.env.CI) throw err;
    console.warn(`Skipping model "${model.id}": ${err.message}`);
    continue;
  }
  // Content-hashed name: browsers fetch and cache a new file whenever it changes.
  const file = `models/${model.id}-${sha256(bytes).slice(0, 12)}.onnx`;
  await writeFile(new URL(file, dist), bytes);
  manifest.models.push({ ...model, file, bytes: bytes.length, size: 1024 });
  console.log(`  ${file} (${(bytes.length / 1024 / 1024).toFixed(1)} MB)`);
}
if (!manifest.models.length) throw new Error('No model could be built');
manifest.models[0].default = true;
await writeFile(new URL('models/manifest.json', dist), JSON.stringify(manifest, null, 2));

console.log('Built dist/');
