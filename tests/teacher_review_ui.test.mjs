import test from 'node:test';
import assert from 'node:assert/strict';
import { boxFromPoints, toImagePoint } from '../scripts/teacher_student/web/geometry.mjs';

test('pointer conversion preserves pixel alignment at desktop/mobile zoom and offsets', () => {
  for (const width of [315.2, 640, 1060]) {
    const rect = { left: 21, top: 400, width, height: width * .75 };
    const point = toImagePoint(rect.left + width / 2, rect.top + rect.height / 4, rect, 800, 600);
    assert.ok(Math.abs(point[0] - 400) < 1e-9 && Math.abs(point[1] - 150) < 1e-9);
  }
});

test('reverse drag and out-of-frame pointer positions remain bounded', () => {
  assert.deepEqual(boxFromPoints([20, 40], [10, 5]), [10, 5, 20, 40]);
  assert.deepEqual(toImagePoint(-5, 1000, { left: 0, top: 0, width: 100, height: 100 }, 800, 600), [0, 600]);
  assert.throws(() => toImagePoint(0, 0, { width: 0, height: 0 }, 800, 600));
});
