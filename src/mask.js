// Mask post-processing: model output -> clean 0..1 mask -> full-resolution alpha.

/**
 * Turns raw model output into a 0..1 mask: optional sigmoid, then "levels" that
 * map values below `low` to fully transparent and above `high` to fully opaque.
 */
export function prepareMask(raw, { sigmoid, levels: [low, high] }) {
  const mask = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) {
    let v = raw[i];
    if (sigmoid) v = 1 / (1 + Math.exp(-v));
    v = (v - low) / (high - low);
    mask[i] = v < 0 ? 0 : v > 1 ? 1 : v;
  }
  return mask;
}

/** Bilinear sampling positions (pixel-centre aligned) from `src` samples to `dst`. */
function axis(dst, src) {
  const i0 = new Int32Array(dst), i1 = new Int32Array(dst), f = new Float32Array(dst);
  for (let x = 0; x < dst; x++) {
    const p = Math.min(Math.max((x + 0.5) * (src / dst) - 0.5, 0), src - 1);
    i0[x] = Math.floor(p);
    i1[x] = Math.min(i0[x] + 1, src - 1);
    f[x] = p - i0[x];
  }
  return { i0, i1, f };
}

/**
 * Scales the size x size mask to width x height (bilinear) and multiplies it into
 * the alpha channel of `pixels` (RGBA), which keeps any existing transparency.
 * RGB values are never modified.
 */
export function applyMask(pixels, width, height, mask, size) {
  const xs = axis(width, size), ys = axis(height, size);
  for (let y = 0; y < height; y++) {
    const r0 = ys.i0[y] * size, r1 = ys.i1[y] * size, fy = ys.f[y];
    for (let x = 0, o = y * width * 4 + 3; x < width; x++, o += 4) {
      const a = xs.i0[x], b = xs.i1[x], fx = xs.f[x];
      const top = mask[r0 + a] + (mask[r0 + b] - mask[r0 + a]) * fx;
      const bottom = mask[r1 + a] + (mask[r1 + b] - mask[r1 + a]) * fx;
      pixels[o] = pixels[o] * (top + (bottom - top) * fy);
    }
  }
}

/**
 * Box-averages the full-resolution alpha of `pixels` down to pw x ph (for the
 * on-screen preview). Returns a Uint8ClampedArray of pw * ph alpha values.
 */
export function downscaleAlpha(pixels, width, height, pw, ph) {
  const sum = new Float64Array(pw * ph), count = new Uint32Array(pw * ph);
  const col = new Int32Array(width);
  for (let x = 0; x < width; x++) col[x] = Math.min(pw - 1, Math.floor((x * pw) / width));
  for (let y = 0; y < height; y++) {
    const row = Math.min(ph - 1, Math.floor((y * ph) / height)) * pw;
    for (let x = 0, o = y * width * 4 + 3; x < width; x++, o += 4) {
      sum[row + col[x]] += pixels[o];
      count[row + col[x]]++;
    }
  }
  const out = new Uint8ClampedArray(pw * ph);
  for (let i = 0; i < out.length; i++) out[i] = sum[i] / count[i];
  return out;
}
