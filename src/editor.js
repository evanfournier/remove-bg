// "Edit selection" on a result: click an object (or drag a box around it) to
// keep it or to remove it, or ask the AI agent to do it. Objects are selected
// with segment-worker.js; the edits are combined with the model's mask on the
// page and drawn as an overlay, then rendered at full resolution on "Apply".

import { composeMask, overlayPixels, partsInside, union } from './compose.js';
import { openImage, selectObject } from './segmenter.js';

const HINT = 'Click an object, or drag a box around it.';
const DRAG = 6; // pixels before a press becomes a box

let current = null;

export const editingItem = () => current?.item ?? null;
export function closeEditor() {
  current?.close();
}

/**
 * @param item a processed image: { id, file, card, baseMask, edits }
 * @param hooks { apply(item, edits), askAI(session, instruction, onStatus, signal) -> {summary, cost} }
 */
export function openEditor(item, hooks) {
  current?.close();
  current = new Editor(item, hooks);
}

/**
 * The mask of one edit: the object the segmentation model selects; to remove
 * with a box, also every separate part of the cut-out lying inside the box.
 */
async function editMask(item, edits, prompt, mode, onProgress) {
  const size = Math.round(Math.sqrt(item.baseMask.length));
  const { mask } = await selectObject(item.id, item.file, prompt, size, onProgress);
  if (mode === 'remove' && prompt.box) union(mask, partsInside(composeMask(item.baseMask, edits), size, prompt.box));
  return mask;
}

/** What the AI agent works on: the photo, the mask and the list of edits. */
export function agentSession(item, view, edits, onChange = () => {}) {
  const size = Math.round(Math.sqrt(item.baseMask.length));
  return {
    view,
    size,
    mask: () => composeMask(item.baseMask, edits),
    segment: async (prompt, mode) => ({ mask: await editMask(item, edits, prompt, mode) }),
    push(edit) {
      edits.push(edit);
      onChange();
    },
    pop() {
      const edit = edits.pop();
      onChange();
      return edit;
    },
  };
}

class Editor {
  constructor(item, hooks) {
    this.item = item;
    this.hooks = hooks;
    this.size = Math.round(Math.sqrt(item.baseMask.length));
    this.draft = item.edits.slice();
    this.mode = 'keep';
    this.busy = false;
    this.agent = null; // AbortController of a running agent
    this.drag = null;
    this.events = new AbortController();

    const { card } = item;
    const $ = (selector) => card.querySelector(selector);
    this.compare = $('.compare');
    this.canvas = $('.mask-layer');
    this.box = $('.drag-box');
    this.bar = $('.editbar');
    this.statusEl = $('.edit-status');
    this.undoButton = $('.undo');
    this.resetButton = $('.reset');
    this.applyButton = $('.apply');
    this.askButton = $('.ask-ai button');
    this.instruction = $('.ask-ai input');
    this.modeButtons = [...card.querySelectorAll('.editbar [data-mode]')];

    this.canvas.width = this.canvas.height = this.size;
    this.ctx = this.canvas.getContext('2d');
    this.compare.dataset.state = 'edit';
    this.canvas.hidden = false;
    this.bar.hidden = false;
    $('.actions').hidden = true;

    const on = (el, type, fn) => el.addEventListener(type, fn, { signal: this.events.signal });
    for (const b of this.modeButtons) on(b, 'click', () => this.setMode(b.dataset.mode));
    on(this.undoButton, 'click', () => this.change(() => this.draft.pop()));
    on(this.resetButton, 'click', () => this.change(() => (this.draft.length = 0)));
    on($('.cancel'), 'click', () => this.close());
    on(this.applyButton, 'click', () => this.apply());
    on($('.ask-ai'), 'submit', (event) => {
      event.preventDefault();
      if (this.agent) this.agent.abort();
      else this.askAI();
    });
    on(card, 'keydown', (event) => {
      if (event.key === 'Escape' && !event.defaultPrevented) this.close();
    });
    on(this.canvas, 'pointerdown', (e) => this.pointerDown(e));
    on(this.canvas, 'pointermove', (e) => this.pointerMove(e));
    on(this.canvas, 'pointerup', (e) => this.pointerUp(e));
    on(this.canvas, 'pointercancel', () => this.endDrag());

    this.setMode('keep');
    this.redraw();
    this.status('Preparing the selection tool…');
    openImage(item.id, item.file, (text) => this.status(text)).then(
      () => !this.busy && this.status(HINT),
      (err) => this.status(`The selection tool is unavailable: ${err.message}`, true),
    );
    this.modeButtons[0].focus();
  }

  status(text, error = false) {
    if (this.closed) return;
    this.statusEl.textContent = text;
    this.statusEl.classList.toggle('bad', error);
  }

  setMode(mode) {
    this.mode = mode;
    for (const b of this.modeButtons) b.setAttribute('aria-checked', String(b.dataset.mode === mode));
    this.compare.dataset.mode = mode;
  }

  setBusy(busy) {
    this.busy = busy;
    this.compare.classList.toggle('busy', busy);
    this.updateButtons();
  }

  updateButtons() {
    const locked = this.busy;
    this.undoButton.disabled = locked || !this.draft.length;
    this.resetButton.disabled = locked || !this.draft.length;
    const changed = this.draft.length !== this.item.edits.length || this.draft.some((e, i) => e !== this.item.edits[i]);
    this.applyButton.disabled = locked || !changed;
    for (const b of this.modeButtons) b.disabled = locked;
    this.askButton.disabled = locked && !this.agent;
    this.askButton.textContent = this.agent ? 'Stop' : '✨ Ask AI';
    this.instruction.disabled = locked;
  }

  change(fn) {
    if (this.busy) return;
    fn();
    this.redraw();
  }

  redraw() {
    if (this.closed) return;
    const mask = composeMask(this.item.baseMask, this.draft);
    this.ctx.putImageData(new ImageData(overlayPixels(mask), this.size, this.size), 0, 0);
    this.updateButtons();
  }

  point(event) {
    const r = this.canvas.getBoundingClientRect();
    return {
      x: Math.min(1, Math.max(0, (event.clientX - r.left) / r.width)),
      y: Math.min(1, Math.max(0, (event.clientY - r.top) / r.height)),
      px: event.clientX,
      py: event.clientY,
    };
  }

  pointerDown(event) {
    if (this.busy || event.button !== 0) return;
    this.canvas.setPointerCapture(event.pointerId);
    this.drag = { start: this.point(event), end: null };
  }

  pointerMove(event) {
    if (!this.drag) return;
    const p = this.point(event), s = this.drag.start;
    if (!this.drag.end && Math.hypot(p.px - s.px, p.py - s.py) < DRAG) return;
    this.drag.end = p;
    Object.assign(this.box.style, {
      left: `${Math.min(s.x, p.x) * 100}%`,
      top: `${Math.min(s.y, p.y) * 100}%`,
      width: `${Math.abs(p.x - s.x) * 100}%`,
      height: `${Math.abs(p.y - s.y) * 100}%`,
    });
    this.box.hidden = false;
  }

  pointerUp(event) {
    if (!this.drag) return;
    this.pointerMove(event);
    const { start: s, end: e } = this.drag;
    this.endDrag();
    const prompt = e
      ? { box: [Math.min(s.x, e.x), Math.min(s.y, e.y), Math.max(s.x, e.x), Math.max(s.y, e.y)], points: [] }
      : { box: null, points: [[s.x, s.y, 1]] };
    this.select(prompt);
  }

  endDrag() {
    this.drag = null;
    this.box.hidden = true;
  }

  async select(prompt) {
    const mode = this.mode;
    this.setBusy(true);
    this.status('Selecting…');
    try {
      const mask = await editMask(this.item, this.draft, prompt, mode, (text) => this.status(text));
      if (this.closed) return;
      this.draft.push({ mode, mask, prompt, label: prompt.box ? 'box' : 'click', source: 'you' });
      this.status(HINT);
    } catch (err) {
      this.status(`Could not select: ${err.message}`, true);
    } finally {
      this.setBusy(false);
      this.redraw();
    }
  }

  async askAI() {
    const before = this.draft.length;
    this.agent = new AbortController();
    this.setBusy(true);
    try {
      const view = await openImage(this.item.id, this.item.file, (text) => this.status(text));
      const session = agentSession(this.item, view, this.draft, () => this.redraw());
      const result = await this.hooks.askAI(session, this.instruction.value, (text) => this.status(`✨ ${text}`), this.agent.signal);
      const cost = result.cost ? ` (≈ $${result.cost.toFixed(2)})` : '';
      this.status(`✨ ${result.summary}${cost}${result.changes ? ' Click Apply to keep the changes.' : ''}`);
    } catch (err) {
      this.draft.length = before; // keep nothing from a failed run
      this.status(`✨ ${this.hooks.describeError(err)}`, true);
    } finally {
      this.agent = null;
      this.setBusy(false);
      this.redraw();
    }
  }

  apply() {
    const edits = this.draft;
    this.close();
    this.hooks.apply(this.item, edits);
  }

  close() {
    if (this.closed) return;
    this.closed = true;
    this.agent?.abort();
    this.events.abort();
    this.endDrag();
    this.canvas.hidden = true;
    this.bar.hidden = true;
    this.item.card.querySelector('.actions').hidden = false;
    this.compare.dataset.state = 'done';
    this.compare.classList.remove('busy');
    if (current === this) current = null;
    this.hooks.closed?.(this.item);
  }
}
