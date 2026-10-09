const worker = new Worker('worker.js', { type: 'module' });

const list = document.getElementById('list');
const engineLabel = document.getElementById('engine');
const downloadAllButton = document.getElementById('download-all');
const fileInput = document.getElementById('file');

const items = new Map(); // id -> { el, urls, link }
let nextId = 0;

const mb = (bytes) => (bytes / 1024 / 1024).toFixed(1);

function addFiles(files) {
  for (const file of files) {
    if (file.type && !file.type.startsWith('image/')) continue;
    const id = nextId++;
    const el = document.createElement('section');
    el.className = 'item';
    el.innerHTML = `
      <header>
        <span class="name"></span>
        <span class="status">Waiting…</span>
        <a class="download" hidden>Download PNG</a>
      </header>
      <div class="pair">
        <img class="original" alt="Original">
        <div class="result"><img alt="Background removed" hidden></div>
      </div>`;
    el.querySelector('.name').textContent = file.name;
    const originalUrl = URL.createObjectURL(file);
    el.querySelector('.original').src = originalUrl;
    list.append(el);
    items.set(id, { el, urls: [originalUrl], link: null });
    worker.postMessage({ id, file });
  }
}

worker.onmessage = ({ data }) => {
  if (data.type === 'model-progress') {
    const total = data.total ? ` / ${mb(data.total)}` : '';
    engineLabel.textContent = `Downloading AI model (first time only): ${mb(data.loaded)}${total} MB`;
    return;
  }
  if (data.type === 'engine') {
    engineLabel.textContent = data.text;
    return;
  }

  const item = items.get(data.id);
  if (!item) return; // cleared meanwhile
  const status = item.el.querySelector('.status');

  if (data.type === 'status') {
    status.textContent = data.text;
  } else if (data.type === 'error') {
    status.textContent = `Failed: ${data.message}`;
    status.classList.add('error');
  } else if (data.type === 'done') {
    const url = URL.createObjectURL(data.blob);
    item.urls.push(url);
    const img = item.el.querySelector('.result img');
    img.src = url;
    img.hidden = false;

    const link = item.el.querySelector('.download');
    const name = item.el.querySelector('.name').textContent;
    link.href = url;
    link.download = `${name.replace(/\.[^.]+$/, '')}-no-bg.png`;
    link.hidden = false;
    item.link = link;
    downloadAllButton.disabled = false;

    status.textContent =
      `${data.width}×${data.height} px · lossless PNG ${mb(data.blob.size)} MB · ` +
      `${data.seconds.toFixed(1)} s${data.colorProfile ? ' · colour profile kept' : ''}`;
  }
};

worker.onerror = (event) => {
  engineLabel.textContent = `Worker failed to start: ${event.message || 'unknown error'}`;
};

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
  for (const { link } of items.values()) {
    if (!link) continue;
    link.click();
    await new Promise((resolve) => setTimeout(resolve, 300));
  }
});

document.getElementById('clear').addEventListener('click', () => {
  for (const { urls } of items.values()) urls.forEach((u) => URL.revokeObjectURL(u));
  items.clear();
  list.replaceChildren();
  downloadAllButton.disabled = true;
});
