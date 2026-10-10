import { composeMask } from './compose.js';
import { agentSession, closeEditor, editingItem, openEditor } from './editor.js';
import { isOutOfMemoryMessage } from './errors.js';
import { openImage, resetSelection } from './segmenter.js';
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

// ---- AI agent settings --------------------------------------------------------
// Claude checks the cut-outs and fixes them (agent.js), with the user's own API
// key. Off until a key is entered; the key is only kept in this browser.

const ai = {
  key: store.get('ai-key', ''),
  auto: store.get('ai-auto', '1') === '1',
};
const aiButton = $('ai-button');
const aiDialog = $('ai-dialog');

function updateAiButton() {
  aiButton.textContent = !ai.key ? '✨ AI agent: off' : ai.auto ? '✨ AI agent: on' : '✨ AI agent: on request';
  aiButton.classList.toggle('on', Boolean(ai.key));
}

function openAiSettings() {
  $('ai-key').value = ai.key;
  $('ai-remember').checked = !ai.key || store.get('ai-key', '') === ai.key;
  $('ai-auto').checked = ai.auto;
  aiDialog.showModal();
}

aiButton.addEventListener('click', openAiSettings);
$('ai-form').addEventListener('submit', (event) => {
  if (event.submitter?.value !== 'save') return;
  ai.key = $('ai-key').value.trim();
  ai.auto = $('ai-auto').checked;
  store.set('ai-key', ai.key && $('ai-remember').checked ? ai.key : '');
  store.set('ai-auto', ai.auto ? '1' : '0');
  updateAiButton();
  // Check the photos that are already done.
  if (ai.key && ai.auto) for (const item of items.values()) if (item.baseMask && !item.aiChecked) scheduleAgent(item);
});
$('ai-close').addEventListener('click', () => aiDialog.close());
$('ai-forget').addEventListener('click', () => {
  ai.key = '';
  store.set('ai-key', '');
  updateAiButton();
  aiDialog.close();
});
updateAiButton();

let agentModule = null; // agent.js (with the Anthropic SDK), loaded on first use
async function askAI(session, instruction, onStatus, signal) {
  if (!ai.key) {
    openAiSettings();
    throw new Error('Add your Anthropic API key in the AI agent settings first.');
  }
  agentModule ??= await import('./agent.js');
  return agentModule.runAgent({ apiKey: ai.key, session, instruction, onStatus, signal });
}
const describeError = (err) => (agentModule ? agentModule.describeError(err) : err?.message || String(err));

// ---- Worker and job queue -----------------------------------------------------
// Jobs run one at a time. The worker is restarted when the model changes (each
// model then gets a fresh 4 GB WebAssembly heap), when a job is cancelled, and
// after a crash. A job is either a full run of the model, or the rendering of
// an edited mask (objects added or removed), which does not need the model.

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
    // A crash kills the job. A failed script load has no message: not memory.
    onWorkerMessage({
      type: 'error',
      id: running,
      message: event.message || 'the processing stopped unexpectedly; please retry',
      outOfMemory: isOutOfMemoryMessage(event.message),
    });
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
  // Added before the model list loaded: use the (now restored) selection.
  if (!models.some((m) => m.id === item.model)) item.model = qualitySelect.value;
  // Rendering an edited mask works in any worker: keep the loaded model.
  if (worker && workerModel !== item.model && !item.renderMask) stopWorker();
  if (!worker) startWorker(item.model);
  setStatus(item, 'Starting…');
  worker.postMessage({
    type: 'process',
    id: item.id,
    file: item.file,
    model: item.model,
    sharpen: item.sharpen,
    mask: item.renderMask,
  });
}

function finishJob() {
  running = null;
  pump();
}

/** Runs the model on the image (again), with the current settings. */
function enqueue(item) {
  if (editingItem() === item) closeEditor();
  item.agentAbort?.abort();
  item.model = qualitySelect.value || store.get('model', '');
  item.sharpen = sharpenBox.checked;
  item.fallback = null;
  item.renderMask = null;
  showProcessing(item);
  queue.push(item.id);
  pump();
}

/** Renders the image with the model's mask plus the edits. */
function enqueueRender(item, edits) {
  item.edits = edits;
  item.model = item.baseModel;
  item.sharpen = sharpenBox.checked;
  item.renderMask = composeMask(item.baseMask, edits);
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
    const fresh = Boolean(data.mask); // the model ran (not an edit)
    if (fresh) {
      item.baseMask = data.mask;
      item.baseModel = item.model;
      item.edits = [];
      item.aiChecked = false;
      setNote(item, '');
    }
    item.renderMask = null;
    showDone(item, data);
    finishJob();
    if (fresh && ai.key && ai.auto) scheduleAgent(item);
  } else if (data.type === 'error') {
    modelPanel.hidden = true;
    stopWorker(); // its memory may be exhausted or corrupt
    const fallback = models.find((m) => m.id === item.model)?.fallback;
    if (data.outOfMemory && fallback && !item.fallback && !item.renderMask) {
      // Not enough memory for this model on this device: use the lighter one.
      item.fallback = item.model;
      item.model = fallback;
      queue.unshift(item.id);
      finishJob();
      return;
    }
    item.renderMask = null;
    showError(item, data.message);
    finishJob();
  }
}

// ---- AI agent on every new photo ---------------------------------------------
// One photo at a time; the background removal of the next photos goes on
// meanwhile. The agent's changes are rendered like the user's edits.

const agentQueue = [];
let agentRunning = false;

function scheduleAgent(item) {
  if (agentQueue.includes(item.id) || (item.agentAbort && !item.agentAbort.signal.aborted)) return;
  agentQueue.push(item.id);
  setNote(item, '✨ AI agent: waiting…');
  runAgents();
}

async function runAgents() {
  if (agentRunning || !agentQueue.length) return;
  const item = items.get(agentQueue.shift());
  if (!item || !item.baseMask || item.processing || editingItem() === item) {
    if (item) setNote(item, '');
    return runAgents();
  }
  agentRunning = true;
  const controller = new AbortController();
  item.agentAbort = controller;
  updateCardActions(item);
  const status = (text) => setNote(item, `✨ AI agent: ${text}`);
  const edits = item.edits.slice();
  try {
    const view = await openImage(item.id, item.file, status);
    const result = await askAI(agentSession(item, view, edits), '', status, controller.signal);
    if (controller.signal.aborted || items.get(item.id) !== item) return;
    item.aiChecked = true;
    setNote(item, `✨ ${result.summary}`, result.cost);
    if (result.changes) enqueueRender(item, edits);
  } catch (err) {
    if (items.get(item.id) === item && !controller.signal.aborted) setNote(item, `✨ AI agent: ${describeError(err)}`, null, true);
    else if (items.get(item.id) === item) setNote(item, '');
  } finally {
    item.agentAbort = null;
    agentRunning = false;
    if (items.get(item.id) === item) updateCardActions(item);
    runAgents();
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

/** The line under the result that tells what the AI agent did. */
function setNote(item, text, cost = null, error = false) {
  const note = item.card.querySelector('.note');
  note.hidden = !text;
  note.classList.toggle('bad', error);
  note.textContent = text + (cost ? ` (≈ $${cost.toFixed(2)})` : '');
}

function setAspect(card, width, height) {
  if (!width || !height) return;
  const compare = card.querySelector('.compare');
  compare.style.aspectRatio = `${width} / ${height}`;
  compare.style.width = `min(100%, calc(70vh * ${width / height}))`;
}

/** Edit is offered once the model's mask is there, and not while the agent works on it. */
function updateCardActions(item) {
  const edit = item.card.querySelector('.edit');
  edit.hidden = item.processing || !item.baseMask;
  edit.disabled = Boolean(item.agentAbort);
  edit.title = item.agentAbort ? 'The AI agent is working on this photo' : 'Keep or remove objects: a plate, hands…';
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
  const actions = [...card.querySelectorAll('.rerun, .edit')];
  if (actions.includes(document.activeElement)) card.focus(); // keep keyboard focus in the card
  for (const b of actions) b.hidden = true;
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
  const kept = item.edits.filter((e) => e.mode === 'keep').length, removed = item.edits.length - kept;
  if (kept) meta.push(`${kept} object${kept > 1 ? 's' : ''} kept`);
  if (removed) meta.push(`${removed} object${removed > 1 ? 's' : ''} removed`);
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
  updateCardActions(item);
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
  updateCardActions(item);
  updateToolbar();
}

function showRerun(item, label) {
  const rerun = item.card.querySelector('.rerun');
  rerun.textContent = label;
  rerun.hidden = false;
  if (document.activeElement === item.card) rerun.focus();
}

const editorHooks = {
  apply: enqueueRender,
  askAI,
  describeError,
  closed: (item) => item.card.querySelector('.edit').focus({ preventScroll: true }),
};

function addFiles(files) {
  for (const file of files) {
    if (file.type && !file.type.startsWith('image/')) continue;
    const id = nextId++;
    const card = template.content.firstElementChild.cloneNode(true);
    const name = file.name || `pasted-image-${id + 1}.png`;
    card.querySelector('.name').textContent = name;
    const item = { id, card, file, name, originalUrl: URL.createObjectURL(file), edits: [] };

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
    card.querySelector('.edit').addEventListener('click', () => openEditor(item, editorHooks));

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
  closeEditor();
  queue.length = 0;
  if (running !== null) {
    stopWorker(); // abort the job in progress
    running = null;
    modelPanel.hidden = true;
  }
  agentQueue.length = 0;
  for (const item of items.values()) {
    item.agentAbort?.abort();
    for (const url of [item.originalUrl, item.resultUrl, item.previewUrl]) if (url) URL.revokeObjectURL(url);
  }
  resetSelection();
  items.clear();
  list.replaceChildren();
  updateToolbar();
});

updateToolbar();
