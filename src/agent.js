// The AI agent. Claude looks at the photo and at the cut-out, decides what the
// subject is the way a person would (the plate holding the food, the hands
// holding the pizza...) and fixes the cut-out: it selects whole objects with
// the segmentation model (segment-worker.js), looks at the result after each
// change and corrects itself. Runs in the browser with the user's own
// Anthropic API key; only a 1024 px copy of the photo is sent.
//
// Bundled with the Anthropic SDK by the build (scripts/build.mjs).

import Anthropic, {
  APIConnectionError,
  APIError,
  APIUserAbortError,
  AuthenticationError,
  BadRequestError,
  PermissionDeniedError,
  RateLimitError,
} from '@anthropic-ai/sdk';
import { betaTool } from '@anthropic-ai/sdk/helpers/beta/json-schema';
import { coverage, cutoutPreview } from './compose.js';

export const MODEL = 'claude-opus-5-5';
const MAX_TURNS = 10;
// USD per million tokens, for the estimate shown to the user.
const PRICE = { input: 4, output: 20, cacheWrite: 5, cacheRead: 0.2 };

const SYSTEM = `You are the quality-control step of a background-removal tool. A segmentation model has cut out what it believes is the subject of the user's photo. Make sure the cut-out contains exactly what a person would call the subject of this photo, then stop.

What belongs to the subject:
- The main subject or subjects, whole, with nothing cut off.
- What holds or presents it and makes up the shot: the plate, bowl, cup, tray or board the food is served on; the hands and arms holding, touching or reaching for the subject (keep the visible arm up to the edge of the photo); the vase holding flowers; the stand of a product.
- For a photo of people: the whole people, with what they hold and wear.

What does not belong: the surroundings. The table or surface under the dish, tablecloths and napkins, the wall, the floor, the sky, and separate objects around the subject (other dishes, cutlery lying beside it, scattered ingredients), unless the user asks otherwise.

How to work:
- Image 1 is the photo. Image 2 is the current cut-out: everything removed is painted solid magenta.
- add_to_subject keeps a missing object; remove_from_subject cuts out something that should not be there. Each call selects ONE whole object with a segmentation model, from a box in pixel coordinates of Image 1 (origin at the top-left corner, x to the right, y down). Draw the box tightly around the whole object as it appears in the photo. The box alone usually works best; add inside_points or outside_points only when the box is ambiguous (for example a hand lying over a plate).
- remove_from_subject also removes every separate piece of the cut-out lying entirely inside its box, so one box can remove a group of things (several tomatoes) without touching what crosses the box's edge.
- After each call you get the updated cut-out. Check it. If a selection took too much or too little, fix it: undo_last_change and try a better box or points, or remove the extra part.
- Work at the level of whole objects: edges are refined later, so ignore small edge imperfections.
- If the cut-out is already right, change nothing.
- Finish with one short sentence for the user saying what you changed (for example "Kept the plate under the noodles."), or that the cut-out was already right.`;

const POINT = {
  type: 'object',
  properties: { x: { type: 'integer' }, y: { type: 'integer' } },
  required: ['x', 'y'],
  additionalProperties: false,
};

const SELECTION = {
  type: 'object',
  properties: {
    object: { type: 'string', description: 'What you are selecting, in a few words, e.g. "plate under the food".' },
    box: {
      type: 'object',
      description: 'Tight box around the whole object, in pixels of Image 1.',
      properties: {
        x_min: { type: 'integer' },
        y_min: { type: 'integer' },
        x_max: { type: 'integer' },
        y_max: { type: 'integer' },
      },
      required: ['x_min', 'y_min', 'x_max', 'y_max'],
      additionalProperties: false,
    },
    inside_points: { type: 'array', items: POINT, description: 'Points on the object, only to disambiguate. Usually empty.' },
    outside_points: { type: 'array', items: POINT, description: 'Points inside the box that are not part of the object. Usually empty.' },
  },
  required: ['object', 'box', 'inside_points', 'outside_points'],
  additionalProperties: false,
};

function base64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(binary);
}

async function jpeg(rgba, width, height) {
  const canvas = new OffscreenCanvas(width, height);
  canvas.getContext('2d').putImageData(new ImageData(rgba, width, height), 0, 0);
  const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality: 0.85 });
  return base64(await blob.arrayBuffer());
}

const pct = (x) => `${(x * 100).toFixed(x < 0.01 ? 1 : 0)}%`;

/**
 * Runs the agent on one image.
 * @param {object} options
 * @param {string} options.apiKey
 * @param {object} options.session the image being edited:
 *   view {width, height, data} (RGBA photo, longest side <= 1024), size (mask grid),
 *   mask() -> the current mask, segment(prompt) -> Promise<{mask}>,
 *   push(edit) and pop() -> edit to change the list of edits.
 * @param {string} [options.instruction] what the user asked for this photo
 * @param {(text: string) => void} [options.onStatus]
 * @param {AbortSignal} [options.signal]
 * @param {typeof fetch} [options.fetch] for tests
 * @param {(rgba, width, height) => Promise<string>} [options.encode] base64 JPEG, for tests
 * @returns {Promise<{summary: string, changes: number, tokens: number, cost: number}>}
 */
export async function runAgent({ apiKey, session, instruction = '', onStatus = () => {}, signal, fetch, encode = jpeg }) {
  // The key is the user's own, typed into this page: calling the API from the
  // browser is the point (there is no server).
  const client = new Anthropic({ apiKey, dangerouslyAllowBrowser: true, maxRetries: 2, ...(fetch && { fetch }) });
  const { view, size } = session;
  const W = view.width, H = view.height;
  const image = async (rgba) => ({
    type: 'image',
    source: { type: 'base64', media_type: 'image/jpeg', data: await encode(rgba, W, H) },
  });
  const cutout = () => image(cutoutPreview(view, session.mask(), size));

  // Tool calls run one at a time, in order; only the last of a turn returns the
  // updated cut-out (one image per turn is enough to check the result).
  let queue = Promise.resolve(), queued = 0;
  const serial = (fn) => {
    queued++;
    const run = queue.then(async () => {
      try {
        const text = await fn();
        if (queued > 1) return `${text} (The updated cut-out comes with the result of your last call.)`;
        return [{ type: 'text', text: `${text} The updated cut-out, removed areas in solid magenta:` }, await cutout()];
      } finally {
        queued--;
      }
    });
    queue = run.catch(() => {});
    return run;
  };

  let changes = 0;
  const toPrompt = ({ box, inside_points: inside = [], outside_points: outside = [] }) => {
    const clamp = (v, max) => Math.min(max, Math.max(0, Number(v)));
    const x0 = clamp(box?.x_min, W), x1 = clamp(box?.x_max, W), y0 = clamp(box?.y_min, H), y1 = clamp(box?.y_max, H);
    if (![x0, x1, y0, y1].every(Number.isFinite) || x1 - x0 < 2 || y1 - y0 < 2) {
      throw new Error(`Invalid box: inside the photo (${W} x ${H} pixels), x_min must be less than x_max and y_min less than y_max.`);
    }
    const point = (p, label) => [clamp(p.x, W) / W, clamp(p.y, H) / H, label];
    return {
      box: [x0 / W, y0 / H, x1 / W, y1 / H],
      points: [...inside.map((p) => point(p, 1)), ...outside.map((p) => point(p, 0))],
    };
  };
  const selectTool = (mode, name, description) => ({
    ...betaTool({
      name,
      description,
      inputSchema: SELECTION,
      run: (input) => serial(async () => {
        const prompt = toPrompt(input);
        const object = String(input.object || 'object').slice(0, 80);
        onStatus(`${mode === 'keep' ? 'Keeping' : 'Removing'} the ${object}…`);
        const { mask } = await session.segment(prompt, mode);
        const area = coverage(mask);
        if (area < 0.0002) {
          throw new Error('The segmentation model found no object in that box. Check the coordinates (pixels of Image 1).');
        }
        session.push({ mode, mask, prompt, label: object, source: 'ai' });
        changes++;
        return `${mode === 'keep' ? 'Kept' : 'Removed'} "${object}" (the selection covers ${pct(area)} of the photo).`;
      }),
    }),
    strict: true,
  });
  const tools = [
    selectTool('keep', 'add_to_subject', 'Keep a whole object that belongs to the subject but is missing from the cut-out.'),
    selectTool('remove', 'remove_from_subject', 'Cut out a whole object or area that the cut-out wrongly kept.'),
    {
      ...betaTool({
        name: 'undo_last_change',
        description: 'Undo your most recent add_to_subject or remove_from_subject.',
        inputSchema: { type: 'object', properties: {}, required: [], additionalProperties: false },
        run: () => serial(async () => {
          if (!changes) throw new Error('You have made no change to undo.');
          const edit = session.pop();
          changes--;
          return `Undid "${edit.label}".`;
        }),
      }),
      strict: true,
    },
  ];

  onStatus('Looking at the photo…');
  const content = [
    { type: 'text', text: `Image 1: the photo (${W} x ${H} pixels).` },
    await image(view.data),
    { type: 'text', text: 'Image 2: the current cut-out (removed areas in solid magenta).' },
    await cutout(),
    {
      type: 'text',
      text: instruction.trim()
        ? `The user's request for this photo, which takes precedence over the default rules: ${instruction.trim()}`
        : 'Check the cut-out and fix it if needed.',
    },
  ];

  const runner = client.beta.messages.toolRunner({
    model: MODEL,
    max_tokens: 16000,
    // Server-side fallback: if the model declines, the request is retried on
    // the model Anthropic recommends for that case, within the same call.
    betas: ['server-side-fallback-2026-07-01'],
    fallbacks: 'default',
    cache_control: { type: 'ephemeral' }, // each turn re-sends the conversation: cache it
    system: SYSTEM,
    tools,
    messages: [{ role: 'user', content }],
    max_iterations: MAX_TURNS,
  }, { signal });

  let last = null, cost = 0, tokens = 0;
  for await (const message of runner) {
    last = message;
    const u = message.usage || {};
    const cacheWrite = u.cache_creation_input_tokens || 0, cacheRead = u.cache_read_input_tokens || 0;
    tokens += (u.input_tokens || 0) + cacheWrite + cacheRead + (u.output_tokens || 0);
    cost += ((u.input_tokens || 0) * PRICE.input + cacheWrite * PRICE.cacheWrite +
      cacheRead * PRICE.cacheRead + (u.output_tokens || 0) * PRICE.output) / 1e6;
    if (message.stop_reason === 'tool_use') onStatus('Checking the result…');
  }
  if (last?.stop_reason === 'refusal') {
    // The page discards the edits of a run that fails.
    throw new AgentError('Claude declined to work on this photo.');
  }
  const summary = (last?.content || [])
    .filter((b) => b.type === 'text')
    .map((b) => b.text.trim())
    .filter(Boolean)
    .join(' ');
  return {
    summary: summary || (changes ? `Made ${changes} change${changes > 1 ? 's' : ''}.` : 'No change needed.'),
    changes,
    tokens,
    cost,
  };
}

export class AgentError extends Error {}

/** A message for the user about a failed run. */
export function describeError(err) {
  if (err instanceof APIUserAbortError) return 'Cancelled.';
  if (err instanceof AuthenticationError) return 'The API key was rejected: check it in the AI agent settings.';
  if (err instanceof PermissionDeniedError) return 'This API key is not allowed to use this model.';
  if (err instanceof RateLimitError) return 'Too many requests to the Claude API right now; try again in a minute.';
  if (err instanceof BadRequestError) return `The Claude API rejected the request: ${err.error?.error?.message || err.message}`;
  if (err instanceof APIConnectionError) return 'Could not reach the Claude API (network problem?).';
  if (err instanceof APIError) return `The Claude API failed (${err.status ?? 'error'}); try again.`;
  return err?.message || String(err);
}
