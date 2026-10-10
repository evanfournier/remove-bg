// The AI agent against a scripted Claude API (no network): request shape, tool
// execution through the segmentation session, self-correction, errors.
import assert from 'node:assert/strict';
import test from 'node:test';
import { AgentError, describeError, runAgent } from '../src/agent.js';
import { composeMask } from '../src/compose.js';

const SIZE = 16;

function fakeSession() {
  const base = new Uint8Array(SIZE * SIZE); // empty cut-out
  const edits = [];
  const prompts = [];
  return {
    edits,
    prompts,
    session: {
      view: { width: 100, height: 50, data: new Uint8ClampedArray(100 * 50 * 4) },
      size: SIZE,
      mask: () => composeMask(base, edits),
      async segment(prompt) {
        prompts.push(prompt);
        await new Promise((r) => setTimeout(r, 5)); // tool calls must still run in order
        return { mask: new Uint8Array(SIZE * SIZE).fill(255), score: 0.9 };
      },
      push: (edit) => edits.push(edit),
      pop: () => edits.pop(),
    },
  };
}

const message = (content, stop_reason) => ({
  id: `msg_${Math.random()}`,
  type: 'message',
  role: 'assistant',
  model: 'claude-opus-5-5',
  content,
  stop_reason,
  stop_sequence: null,
  usage: { input_tokens: 1000, output_tokens: 200, cache_creation_input_tokens: 0, cache_read_input_tokens: 0 },
});
const toolUse = (id, name, input) => ({ type: 'tool_use', id, name, input });
const box = (x_min, y_min, x_max, y_max) => ({ x_min, y_min, x_max, y_max });

function scriptedApi(responses) {
  const requests = [];
  const fetch = async (url, init) => {
    requests.push({ url: String(url), headers: new Headers(init.headers), body: JSON.parse(init.body) });
    const next = responses.shift();
    if (typeof next === 'number') {
      return new Response(JSON.stringify({ type: 'error', error: { type: 'authentication_error', message: 'invalid x-api-key' } }), {
        status: next,
        headers: { 'content-type': 'application/json' },
      });
    }
    return new Response(JSON.stringify(next), { status: 200, headers: { 'content-type': 'application/json' } });
  };
  return { fetch, requests };
}

const encode = async () => 'SlBFRw==';

test('the agent keeps objects, checks the result, corrects itself and reports', async () => {
  const { session, edits, prompts } = fakeSession();
  const { fetch, requests } = scriptedApi([
    message([
      toolUse('t1', 'add_to_subject', { object: 'plate', box: box(10, 20, 90, 60), inside_points: [], outside_points: [] }),
      toolUse('t2', 'remove_from_subject', { object: 'table', box: box(0, 0, 100, 10), inside_points: [], outside_points: [{ x: 50, y: 5 }] }),
    ], 'tool_use'),
    message([toolUse('t3', 'add_to_subject', { object: 'hand', box: box(80, 40, 20, 10), inside_points: [], outside_points: [] })], 'tool_use'),
    message([toolUse('t4', 'undo_last_change', {})], 'tool_use'),
    message([{ type: 'text', text: 'Kept the plate under the noodles.' }], 'end_turn'),
  ]);
  const statuses = [];
  const result = await runAgent({ apiKey: 'sk-test', session, fetch, encode, onStatus: (s) => statuses.push(s) });

  assert.equal(result.summary, 'Kept the plate under the noodles.');
  assert.equal(result.changes, 1);
  assert.deepEqual(edits.map((e) => [e.mode, e.label, e.source]), [['keep', 'plate', 'ai']]);
  // Pixel boxes of the 100 x 50 view -> fractions; tool calls ran in order.
  assert.deepEqual(prompts[0], { box: [0.1, 0.4, 0.9, 1], points: [] }); // y_max 60 clipped to the 50 px photo
  assert.deepEqual(prompts[1], { box: [0, 0, 1, 0.2], points: [[0.5, 0.1, 0]] });
  assert.equal(prompts.length, 2); // the invalid box never reached the model
  assert.ok(result.cost > 0 && result.tokens === 4 * 1200);
  assert.ok(statuses.includes('Keeping the plate…'));

  // Request shape.
  assert.equal(requests.length, 4);
  const [first, second, third] = requests;
  assert.match(first.url, /\/v1\/messages\?beta=true$/);
  assert.equal(first.headers.get('x-api-key'), 'sk-test');
  assert.equal(first.headers.get('anthropic-beta'), 'server-side-fallback-2026-07-01');
  assert.equal(first.headers.get('anthropic-dangerous-direct-browser-access'), 'true');
  assert.equal(first.body.model, 'claude-opus-5-5');
  assert.equal(first.body.fallbacks, 'default');
  assert.equal(first.body.thinking, undefined); // always on: adaptive by default
  assert.equal(first.body.tool_choice, undefined); // forced tool use is not allowed on this model
  assert.deepEqual(first.body.cache_control, { type: 'ephemeral' });
  assert.deepEqual(first.body.tools.map((t) => [t.name, t.strict, 'run' in t]), [
    ['add_to_subject', true, false],
    ['remove_from_subject', true, false],
    ['undo_last_change', true, false],
  ]);
  const images = first.body.messages[0].content.filter((b) => b.type === 'image');
  assert.equal(images.length, 2);
  assert.equal(images[0].source.media_type, 'image/jpeg');

  // Two calls in one turn: only the last result carries the updated cut-out.
  const results = second.body.messages.at(-1).content;
  assert.deepEqual(results.map((r) => r.tool_use_id), ['t1', 't2']);
  assert.equal(typeof results[0].content, 'string');
  assert.equal(results[1].content.filter((b) => b.type === 'image').length, 1);
  // The assistant turn is sent back unchanged (thinking blocks must be preserved).
  assert.deepEqual(second.body.messages[1].content.map((b) => b.id), ['t1', 't2']);

  // An invalid box is reported to the model as an error, not run.
  const error = third.body.messages.at(-1).content[0];
  assert.equal(error.is_error, true);
  assert.match(JSON.stringify(error.content), /Invalid box/);
});

test('a refusal throws, and API errors get a readable message', async () => {
  const { session } = fakeSession();
  const refused = scriptedApi([message([], 'refusal')]);
  await assert.rejects(runAgent({ apiKey: 'k', session, fetch: refused.fetch, encode }), AgentError);

  const unauthorised = scriptedApi([401]);
  const err = await runAgent({ apiKey: 'bad', session, fetch: unauthorised.fetch, encode }).catch((e) => e);
  assert.match(describeError(err), /API key was rejected/);
});
