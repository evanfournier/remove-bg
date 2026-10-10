// Recognises memory exhaustion in ONNX Runtime / WebAssembly error messages.
// A bare "Aborted(<reason>)" is not one: Emscripten reports failed downloads of
// the .wasm that way. Only the empty "Aborted()" of a native abort counts.
const OUT_OF_MEMORY = /bad_alloc|out of memory|allocation failed|memory access out of bounds|Aborted\(\)|Maximum call stack/i;

export const isOutOfMemoryMessage = (message) => OUT_OF_MEMORY.test(String(message || ''));
