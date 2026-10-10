import assert from 'node:assert/strict';
import { test } from 'node:test';
import { applyMask, prepareMask, resizeRGBA } from '../src/mask.js';
import { refineAlpha } from '../src/refine.js';

const S = 1024;

test('prepareMask applies sigmoid and levels', () => {
  const m = prepareMask(Float32Array.of(-20, 0, 20), { sigmoid: true, levels: [0.03, 0.97] });
  assert.deepEqual([...m].map((v) => Math.round(v * 1000) / 1000), [0, 0.5, 1]);
  const l = prepareMask(Float32Array.of(0.1, 0.55, 0.95), { sigmoid: false, levels: [0.2, 0.9] });
  assert.deepEqual([...l].map((v) => Math.round(v * 1000) / 1000), [0, 0.5, 1]);
});

/** Red left part, blue right part, boundary at x = edge; RGB is noisy. */
function scene(width, height, edge) {
  const px = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const o = (y * width + x) * 4, n = (x * 31 + y * 17) % 7;
    px.set(x < edge ? [200 + n, 40, 30 + n, 255] : [30, 60 + n, 190, 255], o);
  }
  return px;
}

function squeeze(px, width, height) {
  const g = new Uint8ClampedArray(S * S * 4);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const sx = Math.min(width - 1, Math.floor(((x + 0.5) * width) / S)), sy = Math.min(height - 1, Math.floor(((y + 0.5) * height) / S));
    g.set(px.subarray((sy * width + sx) * 4, (sy * width + sx) * 4 + 4), (y * S + x) * 4);
  }
  return g;
}

test('applyMask and refineAlpha never touch RGB', () => {
  const width = 300, height = 200, px = scene(width, height, 150);
  const mask = new Float32Array(S * S).map((_, i) => ((i % S) < 512 ? 1 : 0));
  for (const run of [
    (p) => applyMask(p, width, height, mask, S),
    (p) => refineAlpha(p, width, height, mask, S, squeeze(px, width, height)),
  ]) {
    const copy = px.slice();
    run(copy);
    for (let i = 0; i < copy.length; i += 4) {
      assert.equal(copy[i], px[i]);
      assert.equal(copy[i + 1], px[i + 1]);
      assert.equal(copy[i + 2], px[i + 2]);
    }
  }
});

test('refineAlpha snaps a misaligned mask edge to the image edge', () => {
  const width = 3000, height = 2000, edge = 1500, px = scene(width, height, edge);
  // Model mask edge 2 grid px (~6 image px) too far right, and soft (a typical
  // model error; the filter only reaches edges within its window radius).
  const mask = new Float32Array(S * S);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) mask[y * S + x] = Math.min(1, Math.max(0, (514 + 2 - x) / 4));
  refineAlpha(px, width, height, mask, S, squeeze(px, width, height));
  const row = 1000 * width * 4;
  assert.equal(px[row + (edge - 40) * 4 + 3], 255, 'inside stays opaque');
  assert.equal(px[row + (edge + 40) * 4 + 3], 0, 'outside is transparent');
  let transition = 0;
  for (let x = 0; x < width; x++) if (px[row + x * 4 + 3] >= 128) transition = x;
  assert.ok(Math.abs(transition - (edge - 1)) <= 3, `edge at ${transition}, expected ${edge - 1}`);
});

test('refineAlpha removes a faint stray blob but keeps the subject', () => {
  const width = 800, height = 800, px = scene(width, height, 400);
  const mask = new Float32Array(S * S);
  for (let y = 100; y < 900; y++) for (let x = 0; x < 512; x++) mask[y * S + x] = 1; // subject
  for (let y = 950; y < 960; y++) for (let x = 900; x < 910; x++) mask[y * S + x] = 0.5; // faint blob
  refineAlpha(px, width, height, mask, S, squeeze(px, width, height));
  const at = (x, y) => px[(y * width + x) * 4 + 3];
  assert.equal(at(100, 400), 255);
  assert.equal(at(Math.round((905 * width) / S), Math.round((955 * height) / S)), 0);
});

test('resizeRGBA averages areas when shrinking and interpolates when enlarging', () => {
  // 4x2 image: left half (200,100,0) opaque, right half transparent.
  const px = new Uint8ClampedArray(4 * 2 * 4);
  for (let i = 0; i < 8; i++) if (i % 4 < 2) px.set([200, 100, 0, 255], i * 4);
  // Model input: composite on black, opaque.
  assert.deepEqual([...resizeRGBA(px, 4, 2, 2, 1)], [200, 100, 0, 255, 0, 0, 0, 255]);
  assert.deepEqual([...resizeRGBA(px, 4, 2, 1, 1)], [100, 50, 0, 255]);
  // Preview: alpha averaged, colour un-premultiplied (no darkening at edges).
  assert.deepEqual([...resizeRGBA(px, 4, 2, 1, 1, true)], [200, 100, 0, 128]);
  // Enlarging a 2x1 gradient is linear between pixel centres.
  const g = Uint8ClampedArray.of(0, 0, 0, 255, 200, 200, 200, 255);
  const up = resizeRGBA(g, 2, 1, 4, 1);
  assert.deepEqual([up[0], up[4], up[8], up[12]], [0, 50, 150, 200]);
});

test('resizeRGBA handles non-integer shrink factors exactly', () => {
  const w = 5, px = new Uint8ClampedArray(w * 4);
  for (let x = 0; x < w; x++) px.set([x * 50, 0, 0, 255], x * 4);
  // 5 -> 2: [0, 50, half of 100] and [half of 100, 150, 200], each 2.5 px wide.
  const out = resizeRGBA(px, w, 1, 2, 1);
  assert.equal(out[0], Math.round((0 + 50 + 0.5 * 100) / 2.5));
  assert.equal(out[4], Math.round((0.5 * 100 + 150 + 200) / 2.5));
});
