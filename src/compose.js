// Masks on the model's grid: size x size bytes (0 = background, 255 = subject),
// the image squeezed to a square, as the background-removal model sees it.

/**
 * Applies the edits, in order, to the model's mask: "keep" adds the selected
 * object to the subject, "remove" cuts it out. Later edits win.
 * @param {Uint8Array} base
 * @param {{mode: 'keep' | 'remove', mask: Uint8Array}[]} edits
 */
export function composeMask(base, edits) {
  const out = Uint8Array.from(base);
  for (const { mode, mask } of edits) {
    if (mode === 'keep') {
      for (let i = 0; i < out.length; i++) if (mask[i] > out[i]) out[i] = mask[i];
    } else {
      for (let i = 0; i < out.length; i++) {
        const v = 255 - mask[i];
        if (v < out[i]) out[i] = v;
      }
    }
  }
  return out;
}

/** Fraction of the image that the mask selects. */
export function coverage(mask) {
  let n = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i] >= 128) n++;
  return n / mask.length;
}

/** Bilinear lookup of the mask (0..1) at a point given in fractions of the image. */
function sampler(mask, size) {
  return (xn, yn) => {
    const x = Math.min(size - 1, Math.max(0, xn * size - 0.5)), y = Math.min(size - 1, Math.max(0, yn * size - 0.5));
    const x0 = Math.floor(x), y0 = Math.floor(y);
    const x1 = Math.min(size - 1, x0 + 1), y1 = Math.min(size - 1, y0 + 1);
    const fx = x - x0, fy = y - y0;
    const top = mask[y0 * size + x0] * (1 - fx) + mask[y0 * size + x1] * fx;
    const bottom = mask[y1 * size + x0] * (1 - fx) + mask[y1 * size + x1] * fx;
    return (top * (1 - fy) + bottom * fy) / 255;
  };
}

/**
 * The cut-out as the AI agent sees it: the photo where the mask keeps it, solid
 * `color` (magenta) where it removes it.
 * @param {{width: number, height: number, data: Uint8ClampedArray}} view RGBA photo
 */
export function cutoutPreview(view, mask, size, color = [255, 0, 255]) {
  const { width, height, data } = view;
  const out = new Uint8ClampedArray(width * height * 4);
  const at = sampler(mask, size);
  for (let y = 0, i = 0; y < height; y++) {
    for (let x = 0; x < width; x++, i += 4) {
      const a = at((x + 0.5) / width, (y + 0.5) / height);
      out[i] = data[i] * a + color[0] * (1 - a);
      out[i + 1] = data[i + 1] * a + color[1] * (1 - a);
      out[i + 2] = data[i + 2] * a + color[2] * (1 - a);
      out[i + 3] = 255;
    }
  }
  return out;
}

/**
 * The editor's overlay (size x size RGBA, stretched over the photo): removed
 * areas tinted, the subject left clear.
 */
export function overlayPixels(mask, color = [214, 31, 92], opacity = 0.6) {
  const out = new Uint8ClampedArray(mask.length * 4);
  for (let i = 0, j = 0; i < mask.length; i++, j += 4) {
    out[j] = color[0];
    out[j + 1] = color[1];
    out[j + 2] = color[2];
    out[j + 3] = (255 - mask[i]) * opacity;
  }
  return out;
}

/**
 * Fills the holes of a selection that are smaller than `maxArea` pixels:
 * enclosed specks the segmentation model leaves in an object (the shadowed rim
 * of a plate). Openings that touch the image border are never filled.
 */
export function fillSmallHoles(mask, size, maxArea) {
  const seen = new Uint8Array(mask.length);
  const stack = new Int32Array(mask.length);
  const region = [];
  for (let start = 0; start < mask.length; start++) {
    if (seen[start] || mask[start] >= 128) continue;
    // Flood the background region (4-connected) that contains `start`.
    let top = 0, count = 0, border = false;
    region.length = 0;
    stack[top++] = start;
    seen[start] = 1;
    while (top) {
      const i = stack[--top];
      if (++count <= maxArea) region.push(i);
      const x = i % size, y = (i - x) / size;
      if (x === 0 || y === 0 || x === size - 1 || y === size - 1) border = true;
      if (x > 0 && !seen[i - 1] && mask[i - 1] < 128) { seen[i - 1] = 1; stack[top++] = i - 1; }
      if (x < size - 1 && !seen[i + 1] && mask[i + 1] < 128) { seen[i + 1] = 1; stack[top++] = i + 1; }
      if (y > 0 && !seen[i - size] && mask[i - size] < 128) { seen[i - size] = 1; stack[top++] = i - size; }
      if (y < size - 1 && !seen[i + size] && mask[i + size] < 128) { seen[i + size] = 1; stack[top++] = i + size; }
    }
    if (!border && count <= maxArea) for (const i of region) mask[i] = 255;
  }
  return mask;
}

/**
 * The separate parts of the subject (mask >= 128) that lie entirely inside a
 * box (fractions of the image), with their soft edge: a box drawn to remove
 * things then removes all of them (several tomatoes), not only the one object
 * the segmentation model picks (the plate under them).
 */
export function partsInside(mask, size, box) {
  const out = new Uint8Array(mask.length);
  const x0 = Math.max(0, Math.floor(box[0] * size)), x1 = Math.min(size - 1, Math.ceil(box[2] * size) - 1);
  const y0 = Math.max(0, Math.floor(box[1] * size)), y1 = Math.min(size - 1, Math.ceil(box[3] * size) - 1);
  const seen = new Uint8Array(mask.length);
  const stack = new Int32Array(mask.length);
  const region = [];
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const start = y * size + x;
      if (seen[start] || mask[start] < 128) continue;
      let top = 0, inside = true;
      region.length = 0;
      stack[top++] = start;
      seen[start] = 1;
      while (top) {
        const i = stack[--top];
        const px = i % size, py = (i - px) / size;
        if (px < x0 || px > x1 || py < y0 || py > y1) inside = false;
        if (inside) region.push(i);
        if (px > 0 && !seen[i - 1] && mask[i - 1] >= 128) { seen[i - 1] = 1; stack[top++] = i - 1; }
        if (px < size - 1 && !seen[i + 1] && mask[i + 1] >= 128) { seen[i + 1] = 1; stack[top++] = i + 1; }
        if (py > 0 && !seen[i - size] && mask[i - size] >= 128) { seen[i - size] = 1; stack[top++] = i - size; }
        if (py < size - 1 && !seen[i + size] && mask[i + size] >= 128) { seen[i + size] = 1; stack[top++] = i + size; }
      }
      if (!inside) continue;
      // The part and a 2-pixel ring around it (its soft edge), within the box.
      for (const i of region) {
        const px = i % size, py = (i - px) / size;
        for (let dy = -2; dy <= 2; dy++) {
          for (let dx = -2; dx <= 2; dx++) {
            const nx = px + dx, ny = py + dy;
            if (nx >= x0 && nx <= x1 && ny >= y0 && ny <= y1) out[ny * size + nx] = 255;
          }
        }
      }
    }
  }
  return out;
}

/** Pixel-wise maximum of two masks, in place in `a`. */
export function union(a, b) {
  for (let i = 0; i < a.length; i++) if (b[i] > a[i]) a[i] = b[i];
  return a;
}
