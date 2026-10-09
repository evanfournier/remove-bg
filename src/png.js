// Lossless PNG encoder for 8-bit RGBA pixels.
//
// We encode the PNG ourselves instead of using canvas.toBlob(): canvases store
// premultiplied alpha, which rounds the colours of semi-transparent pixels.
// Here every byte we are given ends up in the file unchanged.

const CRC_TABLE = new Uint32Array(256);
for (let n = 0; n < 256; n++) {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  CRC_TABLE[n] = c >>> 0;
}

function crc32(bytes) {
  let crc = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) {
    crc = CRC_TABLE[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
  const out = new Uint8Array(12 + data.length);
  const view = new DataView(out.buffer);
  view.setUint32(0, data.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  view.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
}

/** zlib-compresses bytes with the browser's native CompressionStream. */
export async function zlib(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream('deflate'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/**
 * Writes one filtered scanline (filter byte + row) into `out`, choosing the
 * PNG filter with the smallest sum of absolute values (the libpng heuristic).
 */
function filterRow(cur, prev, out, scratch) {
  const n = cur.length;
  const [sub, up, avg, paeth] = scratch;
  let sNone = 0, sSub = 0, sUp = 0, sAvg = 0, sPaeth = 0;
  for (let i = 0; i < n; i++) {
    const x = cur[i];
    const a = i >= 4 ? cur[i - 4] : 0;
    const b = prev ? prev[i] : 0;
    const c = i >= 4 && prev ? prev[i - 4] : 0;
    const p = a + b - c;
    const pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
    const pred = pa <= pb && pa <= pc ? a : pb <= pc ? b : c;

    const vSub = (x - a) & 255, vUp = (x - b) & 255;
    const vAvg = (x - ((a + b) >> 1)) & 255, vPaeth = (x - pred) & 255;
    sub[i] = vSub; up[i] = vUp; avg[i] = vAvg; paeth[i] = vPaeth;

    sNone += x < 128 ? x : 256 - x;
    sSub += vSub < 128 ? vSub : 256 - vSub;
    sUp += vUp < 128 ? vUp : 256 - vUp;
    sAvg += vAvg < 128 ? vAvg : 256 - vAvg;
    sPaeth += vPaeth < 128 ? vPaeth : 256 - vPaeth;
  }

  let type = 0, best = sNone, row = cur;
  if (sSub < best) { type = 1; best = sSub; row = sub; }
  if (sUp < best) { type = 2; best = sUp; row = up; }
  if (sAvg < best) { type = 3; best = sAvg; row = avg; }
  if (sPaeth < best) { type = 4; row = paeth; }
  out[0] = type;
  out.set(row, 1);
}

/**
 * Encodes RGBA pixels as a PNG Blob.
 * @param {Uint8Array|Uint8ClampedArray} rgba  width*height*4 bytes
 * @param {{type: string, data: Uint8Array}[]} extraChunks  e.g. colour profile
 *   chunks (iCCP, sRGB, ...) to place before the image data.
 */
export async function encodePng(rgba, width, height, extraChunks = []) {
  const rowBytes = width * 4;
  const compressor = new CompressionStream('deflate');
  const writer = compressor.writable.getWriter();

  // Drain the compressed output while we write, re-chunking it into IDATs.
  const idats = [];
  const drained = (async () => {
    const reader = compressor.readable.getReader();
    let pending = [], pendingSize = 0;
    const flush = () => {
      if (!pendingSize) return;
      const data = new Uint8Array(pendingSize);
      let o = 0;
      for (const p of pending) { data.set(p, o); o += p.length; }
      idats.push(chunk('IDAT', data));
      pending = []; pendingSize = 0;
    };
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      pending.push(value);
      pendingSize += value.length;
      if (pendingSize >= 1 << 20) flush();
    }
    flush();
  })();

  const scratch = [0, 1, 2, 3].map(() => new Uint8Array(rowBytes));
  const rowsPerBatch = Math.max(1, Math.floor((4 << 20) / (rowBytes + 1)));
  for (let y = 0; y < height; y += rowsPerBatch) {
    const rows = Math.min(rowsPerBatch, height - y);
    const batch = new Uint8Array(rows * (rowBytes + 1));
    for (let r = 0; r < rows; r++) {
      const start = (y + r) * rowBytes;
      const cur = rgba.subarray(start, start + rowBytes);
      const prev = y + r > 0 ? rgba.subarray(start - rowBytes, start) : null;
      filterRow(cur, prev, batch.subarray(r * (rowBytes + 1), (r + 1) * (rowBytes + 1)), scratch);
    }
    await writer.write(batch);
  }
  await writer.close();
  await drained;

  const ihdr = new Uint8Array(13);
  const view = new DataView(ihdr.buffer);
  view.setUint32(0, width);
  view.setUint32(4, height);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 6; // colour type: RGBA
  // compression, filter and interlace methods: 0

  const signature = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);
  return new Blob(
    [
      signature,
      chunk('IHDR', ihdr),
      ...extraChunks.map((c) => chunk(c.type, c.data)),
      ...idats,
      chunk('IEND', new Uint8Array(0)),
    ],
    { type: 'image/png' },
  );
}
