# Remove Background

A small website that removes the background from your photos. Everything runs
in your browser: images are never uploaded anywhere.

Drop images on the page (or click to pick them, or paste with Ctrl/Cmd+V), wait a
few seconds, then download the PNGs. Several images can be processed in a row.

## No quality loss

- **Same resolution**: the output has exactly the width and height of the input.
- **Original pixels**: every RGB value is the decoded original; only the alpha
  (transparency) channel is added. Verified bit-for-bit against the source.
- **Lossless PNG**: written by our own encoder (not `canvas.toBlob`, which rounds
  semi-transparent pixels).
- **Colour profile kept**: wide-gamut photos (e.g. Display P3 from iPhones) keep
  their embedded profile instead of being converted to sRGB.
- **EXIF rotation applied**, so photos come out the right way up.

How it works: the AI model ([ISNet](https://github.com/xuebinqin/DIS)) looks at
a 1024×1024 copy of the image and predicts a foreground mask. The mask is
cleaned up (solid subject fully opaque, background haze fully transparent),
scaled to the full resolution and applied to the original pixels.

## Run it locally

Requires Node.js 18+.

```sh
npm install
npm start          # builds dist/ and serves it on http://localhost:8080
```

The first image downloads the 84 MB model; it is cached in the browser after that.

## Deploy on GitHub Pages

1. Repository **Settings → Pages → Build and deployment → Source: GitHub Actions**.
2. Push to `main` (or run the **Deploy to GitHub Pages** workflow manually from the
   Actions tab). The site is published at `https://<user>.github.io/remove-bg/`.

Any static host works: deploy the `dist/` folder produced by `npm run build`.

## Notes

- The model runs on the CPU via WebAssembly and uses several threads when the page
  is cross-origin isolated (the local server sends the headers; on GitHub Pages
  `coi-serviceworker` enables it, which causes one automatic reload on the first
  visit). Expect a few seconds per photo.
- Formats: whatever your browser can open (JPEG, PNG, WebP, AVIF, GIF, BMP; HEIC
  only in Safari). 16-bit PNGs are reduced to 8 bits per channel.

## Credits and licences

- Model weights: ISNet as packaged by IMG.LY in
  [`@imgly/background-removal-data`](https://www.npmjs.com/package/@imgly/background-removal-data) (AGPL-3.0).
- [ONNX Runtime Web](https://onnxruntime.ai/) (MIT).
- [coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker) (MIT).
