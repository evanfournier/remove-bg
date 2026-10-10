// Edge refinement: low-resolution model mask -> crisp full-resolution alpha.
//
// The model predicts a 1024x1024 mask. Scaled up to a 24 MP photo, its edges
// are 5-7 px wide and carry a halo of background colour. Here an iterated
// colour guided filter (He, Sun & Tang, "Guided Image Filtering", TPAMI 2013;
// used as a joint upsampler as in He & Sun, "Fast Guided Filter", 2015) snaps
// the mask to the image's own edges. Everything runs on the 1024 grid except a
// final streamed pass over the full-resolution pixels. Only alpha is written.

const OPTIONS = {
  lo: 0.2, hi: 0.6, // hysteresis: keep pixels >= lo connected to a pixel >= hi
  minFrac: 0.005, keepPeak: 0.95, // drop blobs < 0.5% of the largest unless confident
  holeArea: 9, holeMean: 0.25, // fill enclosed holes of a few grid px (noise) or half-believed ones
  faint: 0.02, // mask values below this count as background when tracing regions
  r: 3, // guided filter window radius (grid px)
  eps: 1e-4, // guided filter regularisation (guide in 0..1)
  epsC: 1e-4, // confidence blend: flat windows fall back to the mask itself
  iters: 3, // filter passes, re-sharpening in between
  itLo: 0.25, // re-sharpening: clamp((q - itLo) / (1 - 2 itLo))
  outLo: 0.1, outHi: 0.9, // final levels: removes faint halo, keeps edges solid
};

/** O(N) mean over a (2r+1)^2 window, dividing by the in-image pixel count. */
function boxMean(src, dst, w, h, r, tmp, acc) {
  for (let y = 0; y < h; y++) {
    const o = y * w;
    let s = 0;
    for (let x = 0, e = Math.min(r, w - 1); x <= e; x++) s += src[o + x];
    for (let x = 0; x < w; x++) {
      tmp[o + x] = s / (Math.min(x + r, w - 1) - Math.max(x - r, 0) + 1);
      if (x + r + 1 < w) s += src[o + x + r + 1];
      if (x - r >= 0) s -= src[o + x - r];
    }
  }
  acc.fill(0);
  for (let y = 0, e = Math.min(r, h - 1); y <= e; y++) for (let x = 0, o = y * w; x < w; x++) acc[x] += tmp[o + x];
  for (let y = 0; y < h; y++) {
    const inv = 1 / (Math.min(y + r, h - 1) - Math.max(y - r, 0) + 1), o = y * w;
    for (let x = 0; x < w; x++) dst[o + x] = acc[x] * inv;
    if (y + r + 1 < h) for (let x = 0, a = (y + r + 1) * w; x < w; x++) acc[x] += tmp[a + x];
    if (y - r >= 0) for (let x = 0, b = (y - r) * w; x < w; x++) acc[x] -= tmp[b + x];
  }
}

/** Square dilation of a 0/1 map by r (separable distance passes). */
function dilate(bin, w, h, r) {
  const tmp = new Uint8Array(w * h), out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const o = y * w;
    for (let x = 0, last = -1e9; x < w; x++) { if (bin[o + x]) last = x; if (x - last <= r) tmp[o + x] = 1; }
    for (let x = w - 1, last = 1e9; x >= 0; x--) { if (bin[o + x]) last = x; if (last - x <= r) tmp[o + x] = 1; }
  }
  for (let x = 0; x < w; x++) {
    for (let y = 0, last = -1e9; y < h; y++) { if (tmp[y * w + x]) last = y; if (y - last <= r) out[y * w + x] = 1; }
    for (let y = h - 1, last = 1e9; y >= 0; y--) { if (tmp[y * w + x]) last = y; if (last - y <= r) out[y * w + x] = 1; }
  }
  return out;
}

/** Connected components of pixels equal to `value` (iterative flood fill). */
function components(bin, w, h, value, conn8, stack) {
  const n = w * h, label = new Int32Array(n), areas = [0];
  for (let s = 0; s < n; s++) {
    if (bin[s] !== value || label[s]) continue;
    const id = areas.length;
    let top = 0, area = 0;
    stack[top++] = s;
    label[s] = id;
    while (top) {
      const i = stack[--top], x = i % w, y = (i / w) | 0;
      area++;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -1; dx <= 1; dx++) {
          if ((!dx && !dy) || (!conn8 && dx && dy)) continue;
          const xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          const j = yy * w + xx;
          if (bin[j] === value && !label[j]) { label[j] = id; stack[top++] = j; }
        }
      }
    }
    areas.push(area);
  }
  return { label, areas };
}

/**
 * Cleans the 0..1 mask on its grid: removes faint stray blobs (shadows, bits of
 * background), fills small enclosed holes, zeroes haze away from the subject.
 * Soft values elsewhere are kept as they are, so hair and fur stay soft.
 */
function cleanMask(m, w, h, o) {
  const n = w * h, stack = new Int32Array(n);
  let max = 0;
  for (let i = 0; i < n; i++) if (m[i] > max) max = m[i];
  const hi = Math.min(o.hi, 0.8 * max); // a faint subject still gets a seed

  const S = new Uint8Array(n);
  let top = 0;
  for (let i = 0; i < n; i++) if (m[i] >= hi && hi > 0) { S[i] = 1; stack[top++] = i; }
  while (top) {
    const i = stack[--top], x = i % w, y = (i / w) | 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
      const j = yy * w + xx;
      if (!S[j] && m[j] >= o.lo) { S[j] = 1; stack[top++] = j; }
    }
  }

  const fg = components(S, w, h, 1, true, stack);
  let largest = 0;
  for (let c = 1; c < fg.areas.length; c++) largest = Math.max(largest, fg.areas[c]);
  const peak = new Float32Array(fg.areas.length);
  for (let i = 0; i < n; i++) { const l = fg.label[i]; if (l && m[i] > peak[l]) peak[l] = m[i]; }
  for (let i = 0; i < n; i++) {
    const l = fg.label[i];
    S[i] = l && (fg.areas[l] >= o.minFrac * largest || peak[l] >= o.keepPeak) ? 1 : 0;
  }

  // Background components not touching the border are holes (4-connected, the
  // dual of the 8-connected foreground).
  const bg = components(S, w, h, 0, false, stack);
  const border = new Uint8Array(bg.areas.length), sum = new Float64Array(bg.areas.length);
  for (let x = 0; x < w; x++) { border[bg.label[x]] = 1; border[bg.label[(h - 1) * w + x]] = 1; }
  for (let y = 0; y < h; y++) { border[bg.label[y * w]] = 1; border[bg.label[y * w + w - 1]] = 1; }
  for (let i = 0; i < n; i++) sum[bg.label[i]] += m[i];
  const p = new Float32Array(m);
  for (let i = 0; i < n; i++) {
    const l = bg.label[i];
    if (l && !border[l] && (bg.areas[l] <= o.holeArea || sum[l] / bg.areas[l] >= o.holeMean)) p[i] = 1;
  }
  // Zero the faint regions that are not attached to what we kept (stray blobs,
  // haze); soft tails attached to the subject (hair, fur) are left alone.
  const support = new Uint8Array(n);
  for (let i = 0; i < n; i++) support[i] = p[i] > o.faint ? 1 : 0;
  const parts = components(support, w, h, 1, true, stack);
  const attached = new Uint8Array(parts.areas.length);
  for (let i = 0; i < n; i++) if (S[i]) attached[parts.label[i]] = 1;
  for (let i = 0; i < n; i++) if (!attached[parts.label[i]]) p[i] = 0;
  return p;
}

/** One colour guided filter pass; returns window-averaged [ar, ag, ab, b, c]. */
function guidedFilter(I, p, w, h, o, ws) {
  const n = w * h, { tmp, acc, prod } = ws;
  const [Ir, Ig, Ib] = I;
  const [mr, mg, mb, mp, rp, gp, bp, rr, rg, rb, gg, gb, bb] = ws.planes;
  boxMean(Ir, mr, w, h, o.r, tmp, acc);
  boxMean(Ig, mg, w, h, o.r, tmp, acc);
  boxMean(Ib, mb, w, h, o.r, tmp, acc);
  boxMean(p, mp, w, h, o.r, tmp, acc);
  const corr = (A, B, out) => {
    for (let i = 0; i < n; i++) prod[i] = A[i] * B[i];
    boxMean(prod, out, w, h, o.r, tmp, acc);
  };
  corr(Ir, p, rp); corr(Ig, p, gp); corr(Ib, p, bp);
  corr(Ir, Ir, rr); corr(Ir, Ig, rg); corr(Ir, Ib, rb);
  corr(Ig, Ig, gg); corr(Ig, Ib, gb); corr(Ib, Ib, bb);
  for (let i = 0; i < n; i++) {
    const ur = mr[i], ug = mg[i], ub = mb[i], up = mp[i];
    const cr = rp[i] - ur * up, cg = gp[i] - ug * up, cb = bp[i] - ub * up;
    const vrr = rr[i] - ur * ur, vgg = gg[i] - ug * ug, vbb = bb[i] - ub * ub;
    // a = (Sigma + eps U)^-1 cov(I, p), via the symmetric 3x3 inverse
    const s11 = vrr + o.eps, s12 = rg[i] - ur * ug, s13 = rb[i] - ur * ub;
    const s22 = vgg + o.eps, s23 = gb[i] - ug * ub, s33 = vbb + o.eps;
    const c11 = s22 * s33 - s23 * s23, c12 = s13 * s23 - s12 * s33, c13 = s12 * s23 - s13 * s22;
    const c22 = s11 * s33 - s13 * s13, c23 = s12 * s13 - s11 * s23, c33 = s11 * s22 - s12 * s12;
    const inv = 1 / (s11 * c11 + s12 * c12 + s13 * c13);
    const ar = (c11 * cr + c12 * cg + c13 * cb) * inv;
    const ag = (c12 * cr + c22 * cg + c23 * cb) * inv;
    const ab = (c13 * cr + c23 * cg + c33 * cb) * inv;
    const v = (vrr + vgg + vbb) / 3;
    rp[i] = ar; gp[i] = ag; bp[i] = ab;
    mp[i] = up - ar * ur - ag * ug - ab * ub;
    rr[i] = v / (v + o.epsC);
  }
  for (const plane of [rp, gp, bp, mp, rr]) boxMean(plane, plane, w, h, o.r, tmp, acc);
  return [rp, gp, bp, mp, rr];
}

/**
 * Writes the refined alpha into `pixels` (RGBA, width x height), multiplying any
 * existing transparency. `mask` is the size x size model mask (0..1, after the
 * model's levels); `guide` is the RGBA size x size model input (the same image
 * squeezed to the grid).
 */
export function refineAlpha(pixels, width, height, mask, size, guide, options = {}) {
  const o = { ...OPTIONS, ...options };
  const w = size, h = size, n = w * h;
  const I = [new Float32Array(n), new Float32Array(n), new Float32Array(n)];
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    I[0][i] = guide[j] / 255; I[1][i] = guide[j + 1] / 255; I[2][i] = guide[j + 2] / 255;
  }

  let p = cleanMask(mask, w, h, o);
  const ws = {
    tmp: new Float32Array(n), prod: new Float32Array(n), acc: new Float64Array(w),
    planes: Array.from({ length: 13 }, () => new Float32Array(n)),
  };
  let planes;
  for (let it = 0; it < o.iters; it++) {
    planes = guidedFilter(I, p, w, h, o, ws);
    if (it === o.iters - 1) break;
    const [ar, ag, ab, b, c] = planes, q = new Float32Array(n), span = 1 - 2 * o.itLo;
    for (let i = 0; i < n; i++) {
      let v = ar[i] * I[0][i] + ag[i] * I[1][i] + ab[i] * I[2][i] + b[i];
      v = (c[i] * v + (1 - c[i]) * p[i] - o.itLo) / span;
      q[i] = v < 0 ? 0 : v > 1 ? 1 : v;
    }
    p = q;
  }
  planes = [...planes, p]; // [ar, ag, ab, b, c, p]

  // Full resolution, one row at a time: upsample the coefficients bilinearly and
  // evaluate alpha = c (a . RGB + b) + (1 - c) p with the image's own pixels.
  const x0 = new Int32Array(width), x1 = new Int32Array(width), fx = new Float32Array(width);
  for (let x = 0; x < width; x++) {
    const t = Math.min(Math.max((x + 0.5) * (w / width) - 0.5, 0), w - 1);
    x0[x] = t | 0;
    x1[x] = Math.min(x0[x] + 1, w - 1);
    fx[x] = t - x0[x];
  }
  const rows = planes.map(() => new Float32Array(w));
  const [RA, RG, RB, RBb, RC, RP] = rows;
  const inv255 = 1 / 255, outSpan = o.outHi - o.outLo;
  for (let y = 0; y < height; y++) {
    const t = Math.min(Math.max((y + 0.5) * (h / height) - 0.5, 0), h - 1);
    const r0 = (t | 0) * w, r1 = Math.min((t | 0) + 1, h - 1) * w, fy = t - (t | 0);
    for (let c = 0; c < 6; c++) {
      const P = planes[c], R = rows[c];
      for (let x = 0; x < w; x++) R[x] = P[r0 + x] + (P[r1 + x] - P[r0 + x]) * fy;
    }
    for (let x = 0, q = y * width * 4; x < width; x++, q += 4) {
      const i0 = x0[x], i1 = x1[x], f = fx[x];
      const c = RC[i0] + (RC[i1] - RC[i0]) * f;
      const pv = RP[i0] + (RP[i1] - RP[i0]) * f;
      let a = pv;
      if (c > 1e-3) {
        const ar = RA[i0] + (RA[i1] - RA[i0]) * f, ag = RG[i0] + (RG[i1] - RG[i0]) * f;
        const ab = RB[i0] + (RB[i1] - RB[i0]) * f, b = RBb[i0] + (RBb[i1] - RBb[i0]) * f;
        a = c * ((ar * pixels[q] + ag * pixels[q + 1] + ab * pixels[q + 2]) * inv255 + b) + (1 - c) * pv;
      }
      a = (a - o.outLo) / outSpan;
      pixels[q + 3] = Math.round(pixels[q + 3] * (a < 0 ? 0 : a > 1 ? 1 : a));
    }
  }
}
