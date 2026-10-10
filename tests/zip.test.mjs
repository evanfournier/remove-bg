import assert from 'node:assert/strict';
import { test } from 'node:test';
import { crc32 } from '../src/png.js';
import { makeZip } from '../src/zip.js';

test('makeZip stores files that read back intact', async () => {
  const files = [
    { name: 'a-no-bg.png', blob: new Blob([new Uint8Array([1, 2, 3, 4, 5])]) },
    { name: 'café (2).png', blob: new Blob([new Uint8Array(70000).fill(9)]) },
  ];
  const zip = new Uint8Array(await (await makeZip(files)).arrayBuffer());
  const view = new DataView(zip.buffer);
  const end = zip.length - 22;
  assert.equal(view.getUint32(end, true), 0x06054b50);
  assert.equal(view.getUint16(end + 10, true), 2);
  let cd = view.getUint32(end + 16, true);
  for (const { name, blob } of files) {
    const data = new Uint8Array(await blob.arrayBuffer());
    assert.equal(view.getUint32(cd, true), 0x02014b50);
    const nameLen = view.getUint16(cd + 28, true);
    assert.equal(new TextDecoder().decode(zip.subarray(cd + 46, cd + 46 + nameLen)), name);
    assert.equal(view.getUint32(cd + 16, true), crc32(data));
    const local = view.getUint32(cd + 42, true);
    assert.equal(view.getUint32(local, true), 0x04034b50);
    const start = local + 30 + view.getUint16(local + 26, true);
    assert.deepEqual(zip.subarray(start, start + data.length), data);
    cd += 46 + nameLen;
  }
});
