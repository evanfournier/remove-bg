import assert from 'node:assert/strict';
import { test } from 'node:test';
import { isOutOfMemoryMessage } from '../src/errors.js';

test('memory errors are recognised, network failures are not', () => {
  assert.ok(isOutOfMemoryMessage('failed to call OrtRun(). ERROR_CODE: 6, ERROR_MESSAGE: std::bad_alloc'));
  assert.ok(isOutOfMemoryMessage('RangeError: Array buffer allocation failed'));
  assert.ok(isOutOfMemoryMessage('RuntimeError: Aborted()'));
  assert.ok(!isOutOfMemoryMessage(
    "no available backend found. ERR: [wasm] RuntimeError: Aborted(NetworkError: Failed to execute 'send' on 'XMLHttpRequest': Failed to load '/ort/ort-wasm-simd-threaded.wasm'.)",
  ));
  assert.ok(!isOutOfMemoryMessage('could not download the AI model (HTTP 404)'));
  assert.ok(!isOutOfMemoryMessage(undefined));
});
