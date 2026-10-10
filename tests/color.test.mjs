import assert from 'node:assert/strict';
import { test } from 'node:test';
import { inflateSync } from 'node:zlib';
import { extractColorChunks } from '../src/color.js';

const ascii = (s) => [...s].map((c) => c.charCodeAt(0));
const icc = (space) => {
  const p = new Uint8Array(300).map((_, i) => (i * 13) & 255);
  p.set(ascii(space), 16);
  return p;
};

test('JPEG: multi-segment ICC profile becomes an iCCP chunk', async () => {
  const profile = icc('RGB ');
  const seg = (seq, part) => {
    const body = [...ascii('ICC_PROFILE\0'), seq, 2, ...part];
    return [0xff, 0xe2, (body.length + 2) >> 8, (body.length + 2) & 255, ...body];
  };
  const jpeg = new Uint8Array([0xff, 0xd8, ...seg(2, profile.subarray(150)), ...seg(1, profile.subarray(0, 150)), 0xff, 0xda, 0, 2]);
  const [chunk] = await extractColorChunks(jpeg);
  assert.equal(chunk.type, 'iCCP');
  const nul = chunk.data.indexOf(0);
  assert.equal(chunk.data[nul + 1], 0); // compression method
  assert.deepEqual(new Uint8Array(inflateSync(chunk.data.subarray(nul + 2))), profile);
});

test('JPEG: CMYK profile is ignored', async () => {
  const body = [...ascii('ICC_PROFILE\0'), 1, 1, ...icc('CMYK')];
  const jpeg = new Uint8Array([0xff, 0xd8, 0xff, 0xe2, (body.length + 2) >> 8, (body.length + 2) & 255, ...body, 0xff, 0xda]);
  assert.deepEqual(await extractColorChunks(jpeg), []);
});

test('PNG: colour chunks are copied, grayscale skipped', async () => {
  const chunk = (type, data) => {
    const out = new Uint8Array(12 + data.length);
    new DataView(out.buffer).setUint32(0, data.length);
    out.set(ascii(type), 4);
    out.set(data, 8);
    return [...out];
  };
  const png = (colorType) => new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10,
    ...chunk('IHDR', [0, 0, 0, 1, 0, 0, 0, 1, 8, colorType, 0, 0, 0]),
    ...chunk('gAMA', [0, 0, 177, 143]), ...chunk('sRGB', [0]), ...chunk('IDAT', [1, 2]), ...chunk('cHRM', [9])]);
  assert.deepEqual((await extractColorChunks(png(6))).map((c) => c.type), ['gAMA', 'sRGB']);
  assert.deepEqual(await extractColorChunks(png(0)), []);
});
