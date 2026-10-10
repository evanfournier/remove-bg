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
 * 1-D resampling weights from `src` samples to `dst`: exact area averaging when
 * shrinking, linear interpolation when enlarging. Returns per-output lists.
 */
function weights(dst, src) {
  const start = new Int32Array(dst), count = new Int32Array(dst), w = [];
  const scale = src / dst;
  for (let i = 0; i < dst; i++) {
    const taps = [];
    if (scale >= 1) {
      const a = i * scale, b = a + scale;
      for (let s = Math.floor(a); s < Math.min(src, Math.ceil(b)); s++) {
        taps.push(Math.min(b, s + 1) - Math.max(a, s));
      }
      start[i] = Math.floor(a);
    } else {
      const p = Math.min(Math.max((i + 0.5) * scale - 0.5, 0), src - 1);
      const s0 = Math.floor(p), f = p - s0;
      start[i] = s0;
      taps.push(1 - f);
      if (s0 + 1 < src) taps.push(f);
    }
    const total = taps.reduce((x, y) => x + y, 0);
    count[i] = taps.length;
    w.push(Float32Array.from(taps, (t) => t / total));
  }
  return { start, count, w };
}

/**
 * Resizes RGBA pixels to outW x outH, averaging every covered source pixel
 * (linear interpolation when enlarging). Done in JS rather than with a canvas:
 * browsers differ in how well they downscale (Firefox ignores
 * imageSmoothingQuality), and this works for images larger than a canvas may be.
 * keepAlpha = false: composite on black, opaque output (model input).
 * keepAlpha = true: average alpha, colours un-premultiplied (preview).
 */
export function resizeRGBA(pixels, width, height, outW, outH, keepAlpha = false) {
  const xs = weights(outW, width), ys = weights(outH, height);
  const out = new Uint8ClampedArray(outW * outH * 4);
  const row = new Float32Array(width * 4); // vertically combined, premultiplied
  for (let y = 0; y < outH; y++) {
    row.fill(0);
    for (let k = 0; k < ys.count[y]; k++) {
      const wy = ys.w[y][k];
      for (let x = 0, o = (ys.start[y] + k) * width * 4, r = 0; x < width; x++, o += 4, r += 4) {
        const a = pixels[o + 3] * wy;
        row[r] += pixels[o] * a;
        row[r + 1] += pixels[o + 1] * a;
        row[r + 2] += pixels[o + 2] * a;
        row[r + 3] += a;
      }
    }
    for (let x = 0, o = y * outW * 4; x < outW; x++, o += 4) {
      let r = 0, g = 0, b = 0, a = 0;
      for (let k = 0, s = xs.start[x] * 4; k < xs.count[x]; k++, s += 4) {
        const wx = xs.w[x][k];
        r += row[s] * wx;
        g += row[s + 1] * wx;
        b += row[s + 2] * wx;
        a += row[s + 3] * wx;
      }
      const scale = keepAlpha ? (a > 0 ? 1 / a : 0) : 1 / 255;
      out[o] = r * scale;
      out[o + 1] = g * scale;
      out[o + 2] = b * scale;
      out[o + 3] = keepAlpha ? a : 255;
    }
  }
  return out;
}
