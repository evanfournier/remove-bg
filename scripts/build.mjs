// Assembles the static website in dist/:
//   - the app (src/)
//   - ONNX Runtime Web (runs the AI model with WebAssembly)
//   - the ISNet background-removal model
//   - coi-serviceworker (enables multi-threading on GitHub Pages)
import { createHash } from 'node:crypto';
import { copyFile, cp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';

const root = new URL('../', import.meta.url);
const dist = new URL('dist/', root);
const modules = new URL('node_modules/', root);

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

// Model: ISNet (general use, fp16) as packaged by IMG.LY. The npm package ships
// it in 4 MB chunks named by their SHA-256; verify and join them.
const data = new URL('@imgly/background-removal-data/dist/', modules);
const resources = JSON.parse(await readFile(new URL('resources.json', data), 'utf8'));
const entry = resources['/models/medium'];
const chunks = await Promise.all(
  entry.chunks.map(async ({ hash }) => {
    const bytes = await readFile(new URL(hash, data));
    if (createHash('sha256').update(bytes).digest('hex') !== hash) {
      throw new Error(`Model chunk ${hash} is corrupted`);
    }
    return bytes;
  }),
);
const model = Buffer.concat(chunks);
if (model.length !== entry.size) throw new Error('Model size mismatch');
await mkdir(new URL('models/', dist));
await writeFile(new URL('models/isnet.onnx', dist), model);

console.log(`Built dist/ (model ${(model.length / 1024 / 1024).toFixed(1)} MB)`);
