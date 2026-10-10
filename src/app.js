import { makeZip } from './zip.js';

const $ = (id) => document.getElementById(id);
const list = $('list');
const template = $('card-template');
const qualitySelect = $('quality');
const sharpenBox = $('sharpen');
const downloadAllButton = $('download-all');
const clearButton = $('clear');
const modelPanel = $('model-progress');
const fileInput = $('file');

const items = new Map(); // id -> item (see addFiles)
let nextId = 0;
let zipping = false;

const mb = (bytes) => (bytes / 1024 / 1024).toFixed(1);
const store = {
  get(key, fallback) {
    try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(key, value); } catch { /* storage unavailable */ }
  },
};

// ---- Settings ---------------------------------------------------------------

let models = [];
const modelsReady = (async () => {
  try {
    const res = await fetch('models/manifest.json', { cache: 'no-cache' });
    ({ models } = await res.json());
  } catch (err) {
    console.error(err);
  }
  const saved = store.get('model', '');
  for (const m of models) {
    const option = new Option(`${m.label} · ${mb(m.bytes)} MB`, m.id);
    option.selected = saved ? m.id === saved : m.default;
    qualitySelect.append(option);
  }
  if (!models.length) qualitySelect.append(new Option('Model list unavailable', ''));
})();
qualitySelect.addEventListener('change', () => store.set('model', qualitySelect.value));

sharpenBox.checked = store.get('sharpen', '1') === '1';
sharpenBox.addEventListener('change', () => store.set('sharpen', sharpenBox.checked ? '1' : '0'));

const swatches = [...document.querySelectorAll('.swatch')];
const customColor = $('custom-color');
function setBackground(bg) {
  list.dataset.bg = bg;
  list.style.setProperty('--custom-bg', customColor.value);
  for (const s of swatches) s.setAttribute('aria-checked', String(s.dataset.bg === bg));
  store.set('bg', bg);
  store.set('bg-color', customColor.value);
}
for (const s of swatches) s.addEventListener('click', () => setBackground(s.dataset.bg));
customColor.addEventListener('input', () => setBackground('custom'));
customColor.value = store.get('bg-color', customColor.value);
setBackground(store.get('bg', 'checker'));

// ---- Worker and job queue -----------------------------------------------------
// Jobs run one at a time. The worker is restarted when the model changes (each
// model then gets a fresh 4 GB WebAssembly heap), when a job is cancelled, and
// after a crash.

let worker = null;
let workerModel = null;
let running = null; // id of the job in the worker
const queue = []; // ids waiting

function stopWorker() {
  worker?.terminate();
  worker = null;
  workerModel = null;
}

function startWorker(model) {
  worker = new Worker('worker.js', { type: 'module' });
  workerModel = model;
  worker.onmessage = ({ data }) => onWorkerMessage(data);
  worker.onerror = (event) => {
    event.preventDefault();
    // A crash (usually out of memory) kills the job: report it as such.
    onWorkerMessage({ type: 'error', id: running, message: event.message || 'the processing stopped unexpectedly', outOfMemory: true });
  };
}

async function pump() {
  if (running !== null || !queue.length) return;
  const id = queue.shift();
  running = id;
  await modelsReady;
  if (running !== id) return; // cleared meanwhile
  const item = items.get(id);
  if (!item) {
    running = null;
    return pump();
  }
  if (worker && workerModel !== item.model) stopWorker();
  if (!worker) startWorker(item.model);
  setStatus(item, 'Starting…');
  worker.postMessage({ type: 'process', id: item.id, file: item.file, model: item.model, sharpen: item.sharpen });
}

function finishJob() {
  running = null;
  pump();
}

function enqueue(item) {
  item.model = qualitySelect.value || store.get('model', '');
  item.sharpen = sharpenBox.checked;
  item.fallback = null;
  showProcessing(item);
  queue.push(item.id);
  pump();
}

function onWorkerMessage(data) {
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
  if (data.id !== running) return; // from a job that was cancelled
  const item = items.get(data.id);
  if (!item) return;
  if (data.type === 'stage') {
    setStatus(item, data.text);
    setProgress(item.card.querySelector('.overlay progress'), data.progress);
  } else if (data.type === 'done') {
    showDone(item, data);
    finishJob();
  } else if (data.type === 'error') {
    modelPanel.hidden = true;
    stopWorker(); // its memory may be exhausted or corrupt
    const fallback = models.find((m) => m.id === item.model)?.fallback;
    if (data.outOfMemory && fallback && !item.fallback) {
      // Not enough memory for this model on this device: use the lighter one.
      item.fallback = item.model;
      item.model = fallback;
      queue.unshift(item.id);
      finishJob();
      return;
    }
    showError(item, data.message);
    finishJob();
  }
}

// ---- Cards ------------------------------------------------------------------

function updateToolbar() {
  const done = [...items.values()].filter((i) => i.blob && !i.processing);
  downloadAllButton.disabled = zipping || done.length === 0;
  downloadAllButton.textContent = zipping ? 'Preparing ZIP…' : 'Download all (.zip)';
  clearButton.disabled = items.size === 0;
  $('empty').hidden = items.size > 0;
}

function setProgress(progress, value) {
  if (value == null) progress.removeAttribute('value'); // indeterminate
  else progress.value = value;
}

function setStatus(item, text) {
  const status = item.card.querySelector('.status');
  if (status.textContent !== text) status.textContent = text;
}

function setAspect(card, width, height) {
  if (!width || !height) return;
  const compare = card.querySelector('.compare');
  compare.style.aspectRatio = `${width} / ${height}`;
  compare.style.width = `min(100%, calc(70vh * ${width / height}))`;
}

function showProcessing(item) {
  const { card } = item;
  item.processing = true;
  card.querySelector('.compare').dataset.state = 'processing';
  card.querySelector('.overlay').hidden = false;
  setStatus(item, 'Waiting…');
  setProgress(card.querySelector('.overlay progress'), 0);
  card.querySelector('.error')?.remove();
  card.querySelector('.download').hidden = true;
  const rerun = card.querySelector('.rerun');
  if (document.activeElement === rerun) card.focus(); // keep keyboard focus in the card
  rerun.hidden = true;
  updateToolbar();
}

function showDone(item, data) {
  const { card } = item;
  for (const url of [item.resultUrl, item.previewUrl]) if (url) URL.revokeObjectURL(url);
  item.processing = false;
  item.blob = data.blob;
  item.width = data.width;
  item.resultUrl = URL.createObjectURL(data.blob);
  item.previewUrl = URL.createObjectURL(data.preview);
  setAspect(card, data.width, data.height);
  card.querySelector('.after').src = item.previewUrl;
  card.querySelector('.compare').dataset.state = 'done';
  card.querySelector('.overlay').hidden = true;

  const meta = [`${data.width} × ${data.height} px`, `PNG ${mb(data.blob.size)} MB`, `${data.seconds.toFixed(1)} s`, data.modelLabel];
  if (item.fallback) meta.push('lighter model used: not enough memory for the best one');
  if (!data.sharpen) meta.push('edges not sharpened');
  if (data.colorProfile) meta.push('colour profile kept');
  card.querySelector('.meta').replaceChildren(
    ...meta.map((text) => Object.assign(document.createElement('span'), { textContent: text })),
  );

  const link = card.querySelector('.download');
  link.href = item.resultUrl;
  link.download = `${item.name.replace(/\.[^.]+$/, '')}-no-bg.png`;
  link.hidden = false;
  showRerun(item, 'Re-run');
  updateToolbar();
}

function showError(item, message) {
  const { card } = item;
  item.processing = false;
  card.querySelector('.overlay').hidden = true;
  if (item.blob) {
    // A failed re-run keeps the previous result.
    card.querySelector('.compare').dataset.state = 'done';
    card.querySelector('.download').hidden = false;
  } else {
    card.querySelector('.compare').dataset.state = 'error';
  }
  const error = document.createElement('p');
  error.className = 'error';
  error.setAttribute('role', 'alert');
  error.textContent = `Could not process this image: ${message}`;
  card.append(error);
  showRerun(item, 'Retry');
  updateToolbar();
}

function showRerun(item, label) {
  const rerun = item.card.querySelector('.rerun');
  rerun.textContent = label;
  rerun.hidden = false;
  if (document.activeElement === item.card) rerun.focus();
}

function addFiles(files) {
  for (const file of files) {
    if (file.type && !file.type.startsWith('image/')) continue;
    const id = nextId++;
    const card = template.content.firstElementChild.cloneNode(true);
    const name = file.name || `pasted-image-${id + 1}.png`;
    card.querySelector('.name').textContent = name;
    const item = { id, card, file, name, originalUrl: URL.createObjectURL(file) };

    const before = card.querySelector('.before');
    before.addEventListener('load', () => {
      if (!item.width) setAspect(card, before.naturalWidth, before.naturalHeight);
    });
    before.src = item.originalUrl;

    const compare = card.querySelector('.compare');
    const slider = card.querySelector('input[type=range]');
    slider.addEventListener('input', () => {
      compare.style.setProperty('--pos', `${slider.value}%`);
      slider.setAttribute('aria-valuetext', `${slider.value}% original`);
    });
    card.querySelector('.rerun').addEventListener('click', () => enqueue(item));

    list.append(card);
    items.set(id, item);
    enqueue(item);
  }
  updateToolbar();
}

// ---- Inputs -----------------------------------------------------------------

$('choose').addEventListener('click', (event) => {
  event.stopPropagation();
  fileInput.click();
});
$('drop').addEventListener('click', () => fileInput.click());
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
  const done = [...items.values()].filter((i) => i.blob && !i.processing);
  if (!done.length || zipping) return;
  const used = new Map();
  const files = done.map((item) => {
    const base = item.name.replace(/\.[^.]+$/, '') + '-no-bg';
    const n = (used.get(base) || 0) + 1;
    used.set(base, n);
    return { name: n > 1 ? `${base} (${n}).png` : `${base}.png`, blob: item.blob };
  });
  zipping = true;
  updateToolbar();
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
    zipping = false;
    updateToolbar();
  }
});

clearButton.addEventListener('click', () => {
  queue.length = 0;
  if (running !== null) {
    stopWorker(); // abort the job in progress
    running = null;
    modelPanel.hidden = true;
  }
  for (const item of items.values()) {
    for (const url of [item.originalUrl, item.resultUrl, item.previewUrl]) if (url) URL.revokeObjectURL(url);
  }
  items.clear();
  list.replaceChildren();
  updateToolbar();
});

updateToolbar();
