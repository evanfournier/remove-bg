// Reads the colour-space information embedded in a JPEG, PNG or WebP file and
// returns it as PNG chunks. Writing those chunks into the output PNG (together
// with the untouched decoded pixels) keeps the original colours, e.g. for
// Display P3 photos from phones, instead of squeezing them into sRGB.

import { zlib } from './png.js';

const ascii = (b, start, len) => String.fromCharCode(...b.subarray(start, start + len));

const isPng = (b) => b.length > 33 && b[0] === 0x89 && ascii(b, 1, 3) === 'PNG';
const isJpeg = (b) => b.length > 4 && b[0] === 0xff && b[1] === 0xd8;
const isWebp = (b) => b.length > 16 && ascii(b, 0, 4) === 'RIFF' && ascii(b, 8, 4) === 'WEBP';

// The output is RGBA, so only RGB profiles can be carried over (not CMYK/gray).
const isRgbProfile = (icc) => icc.length >= 128 && ascii(icc, 16, 4) === 'RGB ';

function concat(parts) {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

function pngColorChunks(b) {
  const colorType = b[25];
  if (colorType === 0 || colorType === 4) return []; // grayscale source
  const wanted = new Set(['iCCP', 'sRGB', 'gAMA', 'cHRM', 'cICP']);
  const chunks = [];
  const view = new DataView(b.buffer, b.byteOffset, b.byteLength);
  for (let i = 8; i + 8 <= b.length; ) {
    const len = view.getUint32(i);
    const type = ascii(b, i + 4, 4);
    if (type === 'IDAT' || type === 'IEND') break;
    if (wanted.has(type)) chunks.push({ type, data: b.slice(i + 8, i + 8 + len) });
    i += 12 + len;
  }
  return chunks;
}

function jpegIccProfile(b) {
  const parts = [];
  for (let i = 2; i + 4 <= b.length; ) {
    if (b[i] !== 0xff) break;
    const marker = b[i + 1];
    if (marker === 0xff) { i++; continue; } // fill byte
    if (marker === 0xd8 || marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) { i += 2; continue; }
    if (marker === 0xda || marker === 0xd9) break; // start of scan / end of image
    const end = i + 2 + ((b[i + 2] << 8) | b[i + 3]);
    if (marker === 0xe2 && ascii(b, i + 4, 12) === 'ICC_PROFILE\0') {
      parts.push({ seq: b[i + 16], data: b.subarray(i + 18, end) });
    }
    i = end;
  }
  if (!parts.length) return null;
  parts.sort((x, y) => x.seq - y.seq);
  return concat(parts.map((p) => p.data));
}

function webpIccProfile(b) {
  const view = new DataView(b.buffer, b.byteOffset, b.byteLength);
  for (let i = 12; i + 8 <= b.length; ) {
    const size = view.getUint32(i + 4, true);
    if (ascii(b, i, 4) === 'ICCP') return b.subarray(i + 8, i + 8 + size);
    i += 8 + size + (size & 1);
  }
  return null;
}

/** @returns {Promise<{type: string, data: Uint8Array}[]>} PNG chunks, empty if none. */
export async function extractColorChunks(bytes) {
  if (isPng(bytes)) return pngColorChunks(bytes);

  let icc = null;
  if (isJpeg(bytes)) icc = jpegIccProfile(bytes);
  else if (isWebp(bytes)) icc = webpIccProfile(bytes);
  if (!icc || !isRgbProfile(icc)) return [];

  // iCCP payload: profile name, NUL, compression method 0, zlib data.
  const name = new TextEncoder().encode('ICC Profile');
  return [{ type: 'iCCP', data: concat([name, new Uint8Array([0, 0]), await zlib(icc)]) }];
}
