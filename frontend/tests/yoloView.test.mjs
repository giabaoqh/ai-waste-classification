import test from 'node:test';
import assert from 'node:assert/strict';
import { boxLayout, canAttemptDetection, fetchJson, OBJECT_DESCRIPTIONS, objectColor, validateDetectionResponse } from '../src/utils/yoloView.mjs';

// Contract fixtures only; never treated as model inference or accuracy evidence.
function fixture() {
  const mapping = Object.keys(OBJECT_DESCRIPTIONS).map((name, index) => ({ class_id: index, class_name: name, label_vi: name }));
  return { inference_status: 'completed', model: { task: 'detect', taxonomy: 'object_groups', class_mapping: mapping },
    bounding_box_format: 'xyxy_pixels', coordinate_space: 'exif_oriented_image', image_width: 800, image_height: 600,
    detections: [{ class_id: 0, class_name: 'bottle_jar', confidence: .3, bounding_box: [80, 60, 400, 300] }], total_objects: 1 };
}

test('accept zero/single/multiple detections without inventing material', () => {
  for (const count of [0, 1, 4]) {
    const body = fixture(); body.detections = Array.from({ length: count }, () => structuredClone(body.detections[0])); body.total_objects = count;
    assert.equal(validateDetectionResponse(body, 800, 600).detections.length, count);
    assert.ok(body.detections.every((row) => !('material' in row)));
  }
});

test('reject mismatched image dimensions and legacy material taxonomy', () => {
  assert.throws(() => validateDetectionResponse(fixture(), 600, 800));
  const body = fixture(); body.model.taxonomy = 'materials'; assert.throws(() => validateDetectionResponse(body, 800, 600));
});

test('reject invalid boxes/scores/class IDs and duplicate mapping', () => {
  for (const mutate of [
    (b) => { b.detections[0].bounding_box = [-1, 0, 50, 50]; },
    (b) => { b.detections[0].bounding_box = [0, 0, 900, 50]; },
    (b) => { b.detections[0].bounding_box = [20, 20, 10, 10]; },
    (b) => { b.detections[0].confidence = NaN; },
    (b) => { b.detections[0].class_id = 7; },
    (b) => { b.model.class_mapping[1].class_name = 'bottle_jar'; },
  ]) {
    const body = fixture(); mutate(body); assert.throws(() => validateDetectionResponse(body, 800, 600));
  }
});

test('box scales proportionally and labels keep readable screen size on mobile', () => {
  for (const renderedWidth of [320, 640, 1060]) {
    const layout = boxLayout([80, 60, 400, 300], 800, 600, renderedWidth, '1. Chai / lọ 30%');
    assert.equal(layout.x / 800 * renderedWidth, renderedWidth * .1);
    assert.equal(layout.width / 800 * renderedWidth, renderedWidth * .4);
    assert.ok(Math.abs(layout.fontSize * renderedWidth / 800 - 13) < .001);
    assert.ok(layout.labelX >= 0 && layout.labelX + layout.labelWidth <= 800);
  }
});

test('edge labels remain inside the oriented image', () => {
  const layout = boxLayout([770, 590, 800, 600], 800, 600, 320, 'Một nhãn dài ở cạnh ảnh');
  assert.ok(layout.labelX + layout.labelWidth <= 800 && layout.labelY + layout.labelHeight <= 600);
});

test('each object has deterministic color beyond the first nine', () => {
  assert.equal(new Set(Array.from({ length: 20 }, (_, i) => objectColor(i))).size, 20);
});

test('allow explicit transient retry without pretending missing/invalid artifacts are ready', () => {
  assert.equal(canAttemptDetection({ runtime_ready: true }), true);
  for (const status of ['inference_error', 'load_failed']) {
    assert.equal(canAttemptDetection({ status, artifact_verified: true, runtime_ready: false }), true);
    assert.equal(canAttemptDetection({ status, artifact_verified: false, runtime_ready: false }), false);
  }
  for (const status of ['unavailable', 'invalid_artifact', 'invalid_configuration']) {
    assert.equal(canAttemptDetection({ status, artifact_verified: false, runtime_ready: false }), false);
  }
  assert.equal(canAttemptDetection(null), false);
});

test('surface structured 503, server errors and malformed JSON', async () => {
  await assert.rejects(fetchJson('/test', {}, 100, async () => ({ ok: false, status: 503, json: async () => ({ detail: { message: 'Model unavailable' } }) })), /Model unavailable/);
  await assert.rejects(fetchJson('/test', {}, 100, async () => ({ ok: false, status: 500, json: async () => { throw new Error(); } })), /HTTP 500/);
  await assert.rejects(fetchJson('/test', {}, 100, async () => ({ ok: true, json: async () => { throw new Error(); } })), /JSON/);
});

test('timeout aborts pending transport and clears the request', async () => {
  let aborted = false;
  const pending = (_url, { signal }) => new Promise((_resolve, reject) => signal.addEventListener('abort', () => { aborted = true; reject(new DOMException('Abort', 'AbortError')); }));
  await assert.rejects(fetchJson('/test', {}, 10, pending), /thời gian chờ/);
  assert.ok(aborted);
});

test('caller cancellation aborts old requests, not a timeout error', async () => {
  const controller = new AbortController();
  const pending = (_url, { signal }) => new Promise((_resolve, reject) => signal.addEventListener('abort', () => reject(new DOMException('Abort', 'AbortError'))));
  const request = fetchJson('/test', { signal: controller.signal }, 1000, pending);
  controller.abort(); await assert.rejects(request, { name: 'AbortError' });
});
