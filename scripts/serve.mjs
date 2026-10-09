// Tiny static server for dist/. It sends the COOP/COEP headers that enable
// multi-threaded WebAssembly, so the CPU fallback runs at full speed locally.
import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import { createServer } from 'node:http';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const dist = fileURLToPath(new URL('../dist/', import.meta.url));
const port = Number(process.env.PORT) || 8080;

const types = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.wasm': 'application/wasm',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
};

createServer(async (req, res) => {
  try {
    let file = join(dist, normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname)));
    if ((await stat(file)).isDirectory()) file = join(file, 'index.html');
    const { size } = await stat(file);
    res.writeHead(200, {
      'Content-Type': types[extname(file)] || 'application/octet-stream',
      'Content-Length': size,
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cache-Control': 'no-cache',
    });
    createReadStream(file).pipe(res);
  } catch {
    res.writeHead(404).end('Not found');
  }
}).listen(port, () => {
  console.log(`Remove Background is running at http://localhost:${port}`);
});
