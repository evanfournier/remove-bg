import assert from 'node:assert/strict';
import { test } from 'node:test';
import { inflateSync } from 'node:zlib';
import { crc32, encodePng } from '../src/png.js';

/** Minimal decoder for our own output: 8-bit RGBA, non-interlaced. */
function decodePng(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  assert.deepEqual([...bytes.subarray(0, 8)], [137, 80, 78, 71, 13, 10, 26, 10]);
  const chunks = [];
  let width, height;
  const idat = [];
  for (let o = 8; o < bytes.length; ) {
    const len = view.getUint32(o);
    const type = String.fromCharCode(...bytes.subarray(o + 4, o + 8));
    const data = bytes.subarray(o + 8, o + 8 + len);
    assert.equal(view.getUint32(o + 8 + len), crc32(bytes.subarray(o + 4, o + 8 + len)), `CRC of ${type}`);
    chunks.push({ type, data });
    if (type === 'IHDR') {
      width = view.getUint32(o + 8);
      height = view.getUint32(o + 12);
      assert.deepEqual([...data.subarray(8)], [8, 6, 0, 0, 0]);
    }
    if (type === 'IDAT') idat.push(data);
    o += 12 + len;
  }
  const raw = inflateSync(Buffer.concat(idat));
  const stride = width * 4, out = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    const filter = raw[y * (stride + 1)];
    const line = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    for (let i = 0; i < stride; i++) {
      const a = i >= 4 ? out[y * stride + i - 4] : 0;
      const b = y ? out[(y - 1) * stride + i] : 0;
      const c = i >= 4 && y ? out[(y - 1) * stride + i - 4] : 0;
      const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
      const pred = [0, a, b, (a + b) >> 1, pa <= pb && pa <= pc ? a : pb <= pc ? b : c][filter];
      out[y * stride + i] = (line[i] + pred) & 255;
    }
  }
  return { width, height, pixels: out, chunks };
}

test('encodePng is lossless and keeps extra chunks', async () => {
  const width = 173, height = 61, pixels = new Uint8ClampedArray(width * height * 4);
  let seed = 7;
  for (let i = 0; i < pixels.length; i++) {
    seed = (seed * 1103515245 + 12345) >>> 0;
    // mix of gradients (favour Sub/Up/Paeth) and noise (favours None)
    pixels[i] = i % 3 === 0 ? seed >>> 24 : (i * 7 + (i >> 9)) & 255;
  }
  const iccp = { type: 'iCCP', data: new Uint8Array([65, 0, 0, 120, 156, 3, 0, 0, 0, 0, 1]) };
  const blob = await encodePng(pixels, width, height, [iccp]);
  assert.equal(blob.type, 'image/png');
  const png = decodePng(new Uint8Array(await blob.arrayBuffer()));
  assert.equal(png.width, width);
  assert.equal(png.height, height);
  assert.deepEqual(png.pixels, new Uint8Array(pixels.buffer));
  assert.deepEqual(png.chunks.map((c) => c.type).filter((t) => t !== 'IDAT'), ['IHDR', 'iCCP', 'IEND']);
  assert.deepEqual(png.chunks[1].data, iccp.data);
});

test('crc32 matches the reference value', () => {
  assert.equal(crc32(new TextEncoder().encode('123456789')), 0xcbf43926);
});
