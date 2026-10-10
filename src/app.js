import { makeZip } from './zip.js';

const worker = new Worker('worker.js', { type: 'module' });

const $ = (id) => document.getElementById(id);
const list = $('list');
const template = $('card-template');
const qualitySelect = $('quality');
const downloadAllButton = $('download-all');
const clearButton = $('clear');
const modelPanel = $('model-progress');

const items = new Map(); // id -> { card, name, originalUrl, resultUrl, previewUrl, blob }
let nextId = 0;

const mb = (bytes) => (bytes / 1024 / 1024).toFixed(1);

const store = {
  get(key, fallback) {
    try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(key, value); } catch { /* storage unavailable */ }
  },
};

// ---- Models -----------------------------------------------------------------

async function loadModels() {
  const res = await fetch('models/manifest.json', { cache: 'no-cache' });
  const { models } = await res.json();
  const saved = store.get('model', '');
  for (const m of models) {
    const option = new Option(`${m.label} · ${mb(m.bytes)} MB`, m.id);
    option.selected = saved ? m.id === saved : m.default;
    qualitySelect.append(option);
  }
}
loadModels().catch((err) => {
  console.error(err);
  qualitySelect.append(new Option('Model list unavailable', ''));
});
qualitySelect.addEventListener('change', () => store.set('model', qualitySelect.value));

const sharpenBox = $('sharpen');
sharpenBox.checked = store.get('sharpen', '1') === '1';
sharpenBox.addEventListener('change', () => store.set('sharpen', sharpenBox.checked ? '1' : '0'));

// ---- Preview background -----------------------------------------------------

const swatches = [...document.querySelectorAll('.swatch')];
const customColor = $('custom-color');
function setBackground(bg) {
  list.dataset.bg = bg;
  list.style.setProperty('--custom-bg', customColor.value);
  for (const s of swatches) s.setAttribute('aria-pressed', String(s.dataset.bg === bg));
  store.set('bg', bg);
  store.set('bg-color', customColor.value);
}
for (const s of swatches) s.addEventListener('click', () => setBackground(s.dataset.bg));
customColor.addEventListener('input', () => setBackground('custom'));
customColor.value = store.get('bg-color', customColor.value);
setBackground(store.get('bg', 'checker'));

// ---- Cards ------------------------------------------------------------------

function updateToolbar() {
  const done = [...items.values()].filter((i) => i.blob);
  downloadAllButton.disabled = done.length === 0;
  clearButton.disabled = items.size === 0;
  $('empty').hidden = items.size > 0;
}

function setProgress(progress, value) {
  if (value == null) progress.removeAttribute('value'); // indeterminate
  else progress.value = value;
}

function startProcessing(id) {
  const item = items.get(id);
  const { card } = item;
  item.blob = null; // excluded from "Download all" until the new result arrives
  card.querySelector('.compare').dataset.state = 'processing';
  card.querySelector('.overlay').hidden = false;
  card.querySelector('.status').textContent = 'Waiting…';
  setProgress(card.querySelector('.overlay progress'), 0);
  card.querySelector('.error')?.remove();
  card.querySelector('.download').hidden = true;
  card.querySelector('.rerun')?.remove();
  card.querySelector('.meta').textContent = '';
  worker.postMessage({ type: 'process', id, file: item.file, model: qualitySelect.value, sharpen: sharpenBox.checked });
}

function addFiles(files) {
  for (const file of files) {
    if (file.type && !file.type.startsWith('image/')) continue;
    const id = nextId++;
    const card = template.content.firstElementChild.cloneNode(true);
    const name = file.name || `pasted-image-${id + 1}.png`;
    card.querySelector('.name').textContent = name;

    const originalUrl = URL.createObjectURL(file);
    const before = card.querySelector('.before');
    before.addEventListener('load', () => {
      if (!items.get(id)?.width) setAspect(card, before.naturalWidth, before.naturalHeight);
    });
    before.src = originalUrl;

    const compare = card.querySelector('.compare');
    card.querySelector('input[type=range]').addEventListener('input', (event) => {
      compare.style.setProperty('--pos', `${event.target.value}%`);
    });

    list.append(card);
    items.set(id, { id, card, file, name, originalUrl });
    startProcessing(id);
  }
  updateToolbar();
}

function setAspect(card, width, height) {
  if (!width || !height) return;
  const compare = card.querySelector('.compare');
  compare.style.aspectRatio = `${width} / ${height}`;
  compare.style.width = `min(100%, calc(70vh * ${width / height}))`;
}

function showDone(item, data) {
  const { card } = item;
  if (item.previewUrl) URL.revokeObjectURL(item.previewUrl);
  if (item.resultUrl) URL.revokeObjectURL(item.resultUrl);
  item.blob = data.blob;
  item.width = data.width;
  item.resultUrl = URL.createObjectURL(data.blob);
  item.previewUrl = URL.createObjectURL(data.preview);

  setAspect(card, data.width, data.height);
  card.querySelector('.after').src = item.previewUrl;
  card.querySelector('.compare').dataset.state = 'done';
  card.querySelector('.overlay').hidden = true;

  const meta = [
    `${data.width} × ${data.height} px`,
    `PNG ${mb(data.blob.size)} MB`,
    `${data.seconds.toFixed(1)} s`,
    data.modelLabel,
  ];
  if (!data.sharpen) meta.push('edges not sharpened');
  if (data.colorProfile) meta.push('colour profile kept');
  card.querySelector('.meta').replaceChildren(...meta.map((text) => {
    const span = document.createElement('span');
    span.textContent = text;
    return span;
  }));

  const link = card.querySelector('.download');
  link.href = item.resultUrl;
  link.download = `${item.name.replace(/\.[^.]+$/, '')}-no-bg.png`;
  link.hidden = false;

  const rerun = document.createElement('button');
  rerun.className = 'rerun';
  rerun.textContent = 'Re-run';
  rerun.title = 'Process this image again with the current settings';
  rerun.addEventListener('click', () => startProcessing(item.id));
  link.before(rerun);
  updateToolbar();
}

function showError(item, message) {
  const { card } = item;
  card.querySelector('.compare').dataset.state = 'error';
  card.querySelector('.overlay').hidden = true;
  const error = document.createElement('p');
  error.className = 'error';
  error.textContent = `Could not process this image: ${message}`;
  card.append(error);
}

// ---- Worker messages --------------------------------------------------------

worker.onmessage = ({ data }) => {
  if (data.type === 'model-progress') {
    modelPanel.hidden = data.done;
    if (data.done) return;
    $('model-progress-label').textContent = `Downloading ${data.label} (first time only)`;
    $('model-progress-mb').textContent = `${mb(data.loaded)} / ${mb(data.total)} MB`;
    $('model-progress-bar').value = data.total ? data.loaded / data.total : 0;
    return;
  }
  if (data.type === 'engine') {
    $('engine').textContent = data.text;
    return;
  }

  const item = items.get(data.id);
  if (!item) return; // cleared meanwhile
  if (data.type === 'stage') {
    item.card.querySelector('.status').textContent = data.text;
    setProgress(item.card.querySelector('.overlay progress'), data.progress);
  } else if (data.type === 'done') {
    showDone(item, data);
  } else if (data.type === 'error') {
    showError(item, data.message);
  }
};

worker.onerror = (event) => {
  $('engine').textContent = `Worker failed to start: ${event.message || 'unknown error'}`;
};

// ---- Inputs -----------------------------------------------------------------

const fileInput = $('file');
fileInput.addEventListener('change', () => {
  addFiles(fileInput.files);
  fileInput.value = '';
});

document.addEventListener('paste', (event) => addFiles(event.clipboardData?.files || []));

window.addEventListener('dragover', (event) => {
  event.preventDefault();
  document.body.classList.add('dragging');
});
window.addEventListener('dragleave', (event) => {
  if (!event.relatedTarget) document.body.classList.remove('dragging');
});
window.addEventListener('drop', (event) => {
  event.preventDefault();
  document.body.classList.remove('dragging');
  addFiles(event.dataTransfer?.files || []);
});

downloadAllButton.addEventListener('click', async () => {
  const done = [...items.values()].filter((i) => i.blob);
  const used = new Map();
  const files = done.map((item) => {
    const base = item.name.replace(/\.[^.]+$/, '') + '-no-bg';
    const n = (used.get(base) || 0) + 1;
    used.set(base, n);
    return { name: n > 1 ? `${base} (${n}).png` : `${base}.png`, blob: item.blob };
  });

  const label = downloadAllButton.textContent;
  downloadAllButton.disabled = true;
  downloadAllButton.textContent = 'Preparing ZIP…';
  try {
    const zip = await makeZip(files);
    const a = document.createElement('a');
    a.href = URL.createObjectURL(zip);
    a.download = 'no-background.zip';
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 60_000);
  } catch (err) {
    alert(err.message);
  } finally {
    downloadAllButton.textContent = label;
    updateToolbar();
  }
});

clearButton.addEventListener('click', () => {
  worker.postMessage({ type: 'cancel', ids: [...items.keys()] });
  for (const item of items.values()) {
    for (const url of [item.originalUrl, item.resultUrl, item.previewUrl]) if (url) URL.revokeObjectURL(url);
  }
  items.clear();
  list.replaceChildren();
  updateToolbar();
});

updateToolbar();
