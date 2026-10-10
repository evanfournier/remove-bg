import assert from 'node:assert/strict';
import test from 'node:test';
import { composeMask, coverage, cutoutPreview, fillSmallHoles, overlayPixels, partsInside } from '../src/compose.js';

const grid = (size, fn) => {
  const m = new Uint8Array(size * size);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) m[y * size + x] = fn(x, y);
  return m;
};

test('composeMask: keep adds, remove cuts, in order', () => {
  const size = 8;
  const base = grid(size, (x) => (x < 4 ? 255 : 0)); // left half
  const plate = grid(size, (x, y) => (y >= 4 ? 255 : 0)); // bottom half
  const spot = grid(size, (x, y) => (x === 1 && y === 1 ? 255 : 0));

  const kept = composeMask(base, [{ mode: 'keep', mask: plate }]);
  assert.equal(kept[0 * size + 6], 0); // top right: still background
  assert.equal(kept[6 * size + 6], 255); // bottom right: added
  assert.equal(kept[1 * size + 1], 255);

  const cut = composeMask(base, [{ mode: 'keep', mask: plate }, { mode: 'remove', mask: spot }]);
  assert.equal(cut[1 * size + 1], 0);
  assert.equal(cut[6 * size + 6], 255);

  // A later keep wins over an earlier remove.
  const back = composeMask(base, [{ mode: 'remove', mask: spot }, { mode: 'keep', mask: spot }]);
  assert.equal(back[1 * size + 1], 255);

  // The base is not modified, soft values combine with max/min.
  assert.equal(base[6 * size + 6], 0);
  const soft = composeMask(Uint8Array.of(100, 200), [{ mode: 'keep', mask: Uint8Array.of(150, 50) }]);
  assert.deepEqual([...soft], [150, 200]);
  const softCut = composeMask(Uint8Array.of(200, 200), [{ mode: 'remove', mask: Uint8Array.of(150, 50) }]);
  assert.deepEqual([...softCut], [105, 200]);
});

test('coverage counts the selected fraction', () => {
  assert.equal(coverage(Uint8Array.of(0, 127, 128, 255)), 0.5);
});

test('cutoutPreview paints removed areas magenta, keeps the photo elsewhere', () => {
  const size = 4;
  const mask = grid(size, (x) => (x < 2 ? 255 : 0));
  const view = { width: 8, height: 2, data: new Uint8ClampedArray(8 * 2 * 4).fill(100) };
  const out = cutoutPreview(view, mask, size);
  assert.deepEqual([...out.subarray(0, 4)], [100, 100, 100, 255]); // left: photo
  assert.deepEqual([...out.subarray(7 * 4, 8 * 4)], [255, 0, 255, 255]); // right: magenta
});

test('overlayPixels tints only what is removed', () => {
  const out = overlayPixels(Uint8Array.of(255, 0));
  assert.equal(out[3], 0);
  assert.ok(out[7] > 100);
});

test('fillSmallHoles fills enclosed specks, not openings or big holes', () => {
  const size = 12;
  // A ring object: border background outside, a 2x2 speck and a 4x4 hole inside.
  const mask = grid(size, (x, y) => (x >= 1 && x <= 10 && y >= 1 && y <= 10 ? 255 : 0));
  for (const [x, y] of [[2, 2], [3, 2], [2, 3], [3, 3]]) mask[y * size + x] = 0;
  for (let y = 5; y < 9; y++) for (let x = 5; x < 9; x++) mask[y * size + x] = 0;
  fillSmallHoles(mask, size, 8);
  assert.equal(mask[2 * size + 2], 255); // speck filled
  assert.equal(mask[6 * size + 6], 0); // 16-pixel hole kept
  assert.equal(mask[0], 0); // outside untouched
});

test('partsInside takes the separate parts lying entirely inside a box', () => {
  const size = 20;
  // A big subject on the left crossing the box edge, two small parts inside the box.
  const mask = grid(size, (x, y) => (x < 12 ? 255 : (x === 15 || x === 17) && y >= 4 && y <= 6 ? 255 : 0));
  const out = partsInside(mask, size, [0.5, 0.1, 1, 0.5]); // x 10..19, y 2..9
  assert.equal(out[5 * size + 15], 255);
  assert.equal(out[5 * size + 17], 255);
  assert.equal(out[5 * size + 16], 255); // soft edge between them
  assert.equal(out[5 * size + 11], 0); // the big subject crosses the box edge: untouched
  assert.equal(out[12 * size + 15], 0); // outside the box
});
