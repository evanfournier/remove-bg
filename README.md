# Remove Background

A small website that removes the background from your photos. Everything runs
in your browser: images are never uploaded anywhere.

Drop images on the page (or click to pick them, or paste with Ctrl/Cmd+V). Each
result has a before/after slider and can be previewed on a checkerboard, white,
black or any colour to check the edges. Download one PNG or all of them as a ZIP.

## Smart cut-outs

- **Model**: [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) (lite) by default,
  a state-of-the-art segmentation model; [ISNet](https://github.com/xuebinqin/DIS)
  is available as a faster option. Both run on the CPU with WebAssembly, several
  threads at a time.
- **Edge refinement**: the model sees a 1024×1024 copy of the photo. Its mask is
  cleaned (stray blobs removed, small holes filled) and then snapped to the
  photo's real edges at full resolution with an iterated colour guided filter, so
  large photos get crisp edges without a halo of background colour. It can be
  switched off ("Sharpen edges") for very soft hair or fur.

## No quality loss

- **Same resolution**: the output has exactly the width and height of the input.
- **Original pixels**: only the alpha (transparency) channel is computed; every
  opaque pixel keeps its decoded RGB value bit for bit.
- **Lossless PNG**: written by our own encoder (not `canvas.toBlob`, which rounds
  semi-transparent pixels).
- **Colour profile kept**: wide-gamut photos (e.g. Display P3 from iPhones) keep
  their embedded profile in browsers that can decode without colour conversion
  (Chrome, Edge, Firefox). Safari converts to sRGB; the result then says so by
  not showing "colour profile kept".
- **EXIF rotation applied**, so photos come out the right way up.

Pixels that were already semi-transparent in the source (e.g. a PNG with soft
edges) can change by one level after the browser decodes them; they look the same.

## Run it locally

Requires Node.js 18+, and Python 3 with `onnx` and `numpy` to prepare the
BiRefNet model (without them the build still works, with ISNet only).

```sh
npm install
python3 -m venv .venv && .venv/bin/pip install onnx numpy   # Windows: .venv\Scripts\pip
npm start          # builds dist/ and serves it on http://localhost:8080
npm test           # unit tests (PNG/ZIP encoders, mask refinement, colour profiles)
```

The build uses `.venv` automatically; set `PYTHON=/path/to/python` to use another.

The first image downloads the chosen model (about 90 MB); it is cached in the
browser after that.

## How the BiRefNet model is prepared

`npm run build` downloads the public BiRefNet-lite ONNX export (published by the
[rembg](https://github.com/danielgatis/rembg) project, checked by SHA-256) and
converts it with `scripts/convert_birefnet.py`:

1. Its deformable convolutions are emulated with huge gather operations that
   need more than the 4 GB of memory a browser allows. They are replaced with
   ONNX Runtime's native `DeformConv`, which computes the same thing (maximum
   difference 1e-5) about 4× faster.
2. Weights are stored as float16 (computation stays float32), halving the
   download to about 90 MB.

The result is cached in `.cache/`.

## Deploy on GitHub Pages

1. Repository **Settings → Pages → Build and deployment → Source: GitHub Actions**.
2. Push to the branch listed in `.github/workflows/static.yml` (or run the
   **Deploy static content to Pages** workflow from the Actions tab). It tests,
   builds and publishes `dist/` at `https://<user>.github.io/remove-bg/`.

The site must be built before it is published: the repository has no
`index.html` at its root, and the models are only added to `dist/` by the build.
Any static host works: deploy the `dist/` folder produced by `npm run build`.

## Notes

- Multi-threading needs cross-origin isolation: the local server sends the
  headers; on GitHub Pages `coi-serviceworker` provides them, which causes one
  automatic reload on the first visit. Expect roughly 5–20 seconds per photo
  depending on the model and your CPU.
- Formats: whatever your browser can open (JPEG, PNG, WebP, AVIF, GIF, BMP; HEIC
  only in Safari). 16-bit PNGs are reduced to 8 bits per channel.

## Credits and licences

- BiRefNet (MIT), ONNX export from rembg (MIT).
- ISNet weights as packaged by IMG.LY in
  [`@imgly/background-removal-data`](https://www.npmjs.com/package/@imgly/background-removal-data) (AGPL-3.0).
- [ONNX Runtime Web](https://onnxruntime.ai/) (MIT).
- [coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker) (MIT).
- Edge refinement after He, Sun & Tang, *Guided Image Filtering* (TPAMI 2013) and
  He & Sun, *Fast Guided Filter* (2015).
