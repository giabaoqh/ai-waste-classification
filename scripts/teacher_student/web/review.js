import { boxFromPoints, toImagePoint } from './geometry.mjs';

const $ = (id) => document.getElementById(id);
const LABELS = ['Chai / lọ', 'Lon', 'Cốc', 'Hộp carton', 'Nắp', 'Túi / màng / vỏ gói', 'Giấy', 'Đầu lọc thuốc lá', 'Vật thể rác khác'];
const FLAG_LABELS = { tiny_boxes: 'Box rất nhỏ', extreme_boxes: 'Box có hình học cực đoan', overlapping_duplicate_or_conflicting_boxes: 'Nghi box trùng / class xung đột', missing_image_license_reference: 'Thiếu tham chiếu license ảnh', unverified_negative: 'Ảnh nhãn trống chưa xác minh', scene_group_unknown: 'Chưa có scene ID', source_annotation_mismatch: 'Khác nhãn nguồn' };
let session, queue = [], current, boxes = [], issues = [], selected = -1, dirty = false, busy = false;
let sequence = 0, loadController, drawing = false, startPoint = null, endPoint = null;

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}

function error(message = '') { $('error').textContent = message; $('error').hidden = !message; }
function message(text) { $('message').textContent = text; }
function changed() { dirty = true; $('review-state').textContent = 'Có thay đổi chưa lưu — dấu duyệt cũ không áp dụng cho thay đổi này.'; }
function setBusy(value) {
  busy = value;
  for (const element of document.querySelectorAll('button,input,select,textarea')) element.disabled = value;
}

async function api(path, options = {}) {
  const controller = new AbortController();
  const relay = () => controller.abort();
  options.signal?.addEventListener('abort', relay, { once: true });
  const timer = setTimeout(relay, 30000);
  try {
    const response = await fetch(path, { ...options, signal: controller.signal,
      headers: { 'Content-Type': 'application/json', 'X-Review-Token': session?.csrf_token || '', ...options.headers } });
    const body = await response.json();
    if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : `HTTP ${response.status}`);
    return body;
  } finally { clearTimeout(timer); options.signal?.removeEventListener('abort', relay); }
}

function classOptions(select, value) {
  select.replaceChildren();
  const unknown = node('option', 'Chưa ánh xạ — chọn thủ công'); unknown.value = ''; select.append(unknown);
  session.taxonomy.forEach((name, index) => { const option = node('option', `${index} · ${LABELS[index]} (${name})`); option.value = String(index); select.append(option); });
  select.value = value === null || value === undefined ? '' : String(value);
}

function svgNode(tag, attributes) {
  const element = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
  return element;
}

function drawBox(box, index, layer) {
  const [x1, y1, x2, y2] = box.box;
  if (![x1,y1,x2,y2].every(Number.isFinite) || x2 <= x1 || y2 <= y1) return;
  const group = svgNode('g', {});
  const className = layer === 'draft' && selected === index ? 'box-selected' : `box-${layer}`;
  group.append(svgNode('rect', { x: x1, y: y1, width: x2 - x1, height: y2 - y1, class: className, 'stroke-width': 2, 'vector-effect': 'non-scaling-stroke' }));
  const width = $('overlay').getBoundingClientRect().width || 640;
  const size = 12 * current.source.width / width;
  const text = svgNode('text', { x: x1, y: Math.max(size, y1 - size * .2), 'font-size': size,
    fill: layer === 'original' ? '#38bdf8' : layer === 'teacher' ? '#fb923c' : '#4ade80', 'paint-order': 'stroke', stroke: '#07111c', 'stroke-width': size * .13 });
  text.textContent = `${layer === 'original' ? 'G' : layer === 'teacher' ? 'T' : 'N'}${index + 1}: ${box.class_id === null ? (box.raw_label || 'Chưa ánh xạ') : LABELS[box.class_id]}`;
  group.append(text);
  if (layer === 'draft') {
    group.setAttribute('role', 'button'); group.setAttribute('tabindex', '0');
    group.setAttribute('aria-label', `Chọn box nháp ${index + 1}`);
    const select = () => { selected = index; renderDraft(); renderOverlay(); };
    group.addEventListener('click', () => { if (!drawing) select(); });
    group.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); select(); } });
  }
  $('overlay').append(group);
}

function renderOverlay() {
  $('overlay').replaceChildren();
  if (!current) return;
  $('overlay').setAttribute('viewBox', `0 0 ${current.source.width} ${current.source.height}`);
  $('overlay').setAttribute('preserveAspectRatio', 'none');
  if ($('show-original').checked) current.original_boxes.forEach((box, index) => drawBox(box, index, 'original'));
  if ($('show-teacher').checked) (current.teacher?.proposals || []).forEach((box, index) => drawBox(box, index, 'teacher'));
  if ($('show-draft').checked) boxes.forEach((box, index) => drawBox(box, index, 'draft'));
  if (startPoint && endPoint) {
    const [x1, y1, x2, y2] = boxFromPoints(startPoint, endPoint);
    $('overlay').append(svgNode('rect', { x: x1, y: y1, width: x2 - x1, height: y2 - y1, class: 'box-drawing', 'stroke-width': 2, 'vector-effect': 'non-scaling-stroke' }));
  }
}

function renderComparison() {
  $('comparison').replaceChildren();
  current.original_boxes.forEach((box, index) => {
    const row = node('tr');
    [ `Gốc ${index + 1}`, `${LABELS[box.class_id]} (${box.class_name})`, '— nhãn gốc không có score', '—', box.box.map(v => v.toFixed(1)).join(', ') ].forEach(text => row.append(node('td', text)));
    $('comparison').append(row);
  });
  $('teacher-state').textContent = current.teacher ? `Teacher: ${current.teacher.metadata.model_id} · ${current.teacher.metadata.version} · import chưa duyệt; ${current.teacher.proposals.length} đề xuất.` : 'Chưa có kết quả Teacher. Không có box/score Teacher để so sánh; nhãn gốc không được giả thành Teacher.';
  $('comparison-note').textContent = current.teacher?.comparison.note || 'IoU và bất đồng class sẽ xuất hiện sau khi import output Teacher có provenance.';
  (current.teacher?.proposals || []).forEach((box, index) => {
    const comparison = current.teacher.comparison.proposals[index];
    const row = node('tr');
    const score = box.confidence === null ? '—' : `${box.confidence.toFixed(3)} (${box.confidence_kind})`;
    [`Teacher ${index + 1}`, box.class_id === null ? `Chưa ánh xạ: ${box.raw_label}` : `${LABELS[box.class_id]} (${box.class_name})`, score,
      comparison.iou.toFixed(3)].forEach(text => row.append(node('td', text)));
    const action = node('td', comparison.flag + ' '); const use = node('button', 'Thêm đề xuất vào nháp'); use.type = 'button';
    use.addEventListener('click', () => { boxes.push({ class_id: box.class_id, box: [...box.box], origin: 'teacher', proposal_id: box.proposal_id }); selected = boxes.length - 1; changed(); renderDraft(); renderOverlay(); });
    action.append(use); row.append(action); $('comparison').append(row);
  });
}

function renderDraft() {
  $('draft').replaceChildren();
  boxes.forEach((box, index) => {
    const row = node('tr', undefined, selected === index ? 'selected' : ''); const chooseCell = node('td'); const choose = node('button', `Box ${index + 1}`); choose.type = 'button';
    choose.addEventListener('click', () => { selected = index; renderDraft(); renderOverlay(); }); chooseCell.append(choose); row.append(chooseCell);
    const classCell = node('td'); const select = node('select'); select.setAttribute('aria-label', `Class box nháp ${index + 1}`); classOptions(select, box.class_id);
    select.addEventListener('change', () => { box.class_id = select.value === '' ? null : Number(select.value); box.origin = 'human'; row.lastElementChild.textContent = 'human'; changed(); renderOverlay(); }); classCell.append(select); row.append(classCell);
    ['x1', 'y1', 'x2', 'y2'].forEach((name, axis) => {
      const cell = node('td'); const input = node('input'); input.type = 'number'; input.step = '0.1'; input.value = String(box.box[axis]); input.setAttribute('aria-label', `${name} box nháp ${index + 1}`);
      input.addEventListener('input', () => { box.box[axis] = input.value === '' ? null : Number(input.value); box.origin = 'human'; row.lastElementChild.textContent = 'human'; changed(); renderOverlay(); }); cell.append(input); row.append(cell);
    });
    row.append(node('td', box.origin)); $('draft').append(row);
  });
}

function renderIssues() {
  $('issues').replaceChildren();
  issues.forEach((issue, index) => {
    const frame = node('div', undefined, 'issue'); frame.append(node('strong', `${index + 1}. ${issue.type}`));
    const label = node('label', 'Đã giải quyết (cần ghi căn cứ) '); const resolved = node('input'); resolved.type = 'checkbox'; resolved.checked = issue.resolved;
    resolved.setAttribute('aria-label', `Vấn đề ${index + 1} đã giải quyết`); resolved.addEventListener('change', () => { issue.resolved = resolved.checked; changed(); }); label.append(resolved); frame.append(label);
    const note = node('input'); note.type = 'text'; note.value = issue.note || ''; note.setAttribute('aria-label', `Ghi chú vấn đề ${index + 1}`);
    note.addEventListener('input', () => { issue.note = note.value; changed(); }); frame.append(note); $('issues').append(frame);
  });
}

async function loadItem(imageId) {
  if (dirty && current) { $('item').value = current.image_id; error('Có thay đổi chưa lưu. Lưu nháp/hold trước khi chuyển ảnh.'); return; }
  loadController?.abort(); loadController = new AbortController(); const request = ++sequence;
  setBusy(true); error();
  try {
    const data = await api(`/api/items/${imageId}`, { signal: loadController.signal });
    if (request !== sequence) return;
    current = data; boxes = structuredClone(data.draft_boxes); issues = structuredClone(data.issues); selected = -1; drawing = false; startPoint = null; endPoint = null;
    $('item').value = imageId; $('image').src = `/api/image/${imageId}`;
    $('source').textContent = `${data.source.image_file} · ${data.source.source} · TRAIN · ${data.source.width}×${data.source.height}px · SHA-256 ${data.source.image_sha256}`;
    $('flags').textContent = 'Dấu hiệu QA (chưa xác nhận sai): ' + (data.source.flags.map(flag => FLAG_LABELS[flag] || flag).join('; ') || 'Không có dấu hiệu hình học tự động. Vẫn cần xem toàn ảnh.');
    $('review-state').textContent = `Trạng thái: ${data.status} · revision ${data.revision}`;
    $('reviewer').value = data.review?.reviewer || $('reviewer').value;
    $('notes').value = data.review?.notes || ''; $('rights').value = data.review?.rights_evidence || '';
    for (const [id, field] of [['check-all','all_objects_reviewed'],['check-qa','heuristics_reviewed'],['check-provenance','provenance_reviewed'],['check-hazards','hazards_reviewed']]) $(id).checked = data.review?.checks[field] === true;
    $('provenance').textContent = JSON.stringify({ source: data.source, teacher: data.teacher?.metadata || null, history: data.history }, null, 2);
    $('draw').setAttribute('aria-pressed', 'false'); renderComparison(); renderDraft(); renderIssues(); renderOverlay();
    message('Đã mở ảnh. Không có thao tác tự duyệt hoặc huấn luyện.');
  } catch (err) { if (err.name !== 'AbortError') error(err.message); }
  finally { if (request === sequence) setBusy(false); }
}

async function save(status) {
  if (!current || busy) return;
  setBusy(true); error();
  const payload = { expected_revision: current.revision, boxes, issues, status, reviewer: $('reviewer').value,
    notes: $('notes').value, rights_evidence: $('rights').value, checks: {
      all_objects_reviewed: $('check-all').checked, heuristics_reviewed: $('check-qa').checked,
      provenance_reviewed: $('check-provenance').checked, hazards_reviewed: $('check-hazards').checked } };
  try {
    const updated = await api(`/api/items/${current.image_id}/review`, { method: 'POST', body: JSON.stringify(payload) });
    dirty = false; current = updated;
    const listItem = queue.find(item => item.image_id === updated.image_id); listItem.status = updated.status;
    updateQueue(); await loadItem(updated.image_id); message(`Đã lưu ${updated.status}, revision ${updated.revision}. Chưa export hoặc train.`);
  } catch (err) { error(err.name === 'AbortError' ? 'Request quá thời gian chờ. Tải lại để kiểm tra revision trước khi lưu lại.' : err.message); }
  finally { setBusy(false); }
}

function updateQueue() {
  $('item').replaceChildren();
  queue.forEach((item, index) => { const option = node('option', `${index + 1}. ${item.file.split('/').pop()} · ${item.source} · ${item.status}`); option.value = item.image_id; $('item').append(option); });
  if (current) $('item').value = current.image_id;
  $('progress').textContent = `${queue.length} ảnh train · ${queue.filter(i => i.status === 'approved').length} đã duyệt`;
}

function point(event) { return toImagePoint(event.clientX, event.clientY, $('overlay').getBoundingClientRect(), current.source.width, current.source.height); }
$('overlay').addEventListener('pointerdown', event => {
  if (!drawing || !current || busy) return;
  if ($('new-class').value === '') { error('Chọn class cho box mới trước. Không tự suy class.'); return; }
  event.preventDefault(); startPoint = point(event); endPoint = startPoint; $('overlay').setPointerCapture(event.pointerId); renderOverlay();
});
$('overlay').addEventListener('pointermove', event => { if (startPoint) { endPoint = point(event); renderOverlay(); } });
$('overlay').addEventListener('pointerup', event => {
  if (!startPoint) return;
  const box = boxFromPoints(startPoint, point(event)); startPoint = null; endPoint = null;
  if (box[2] - box[0] >= 1 && box[3] - box[1] >= 1) { boxes.push({ class_id: Number($('new-class').value), box, origin: 'human' }); selected = boxes.length - 1; changed(); }
  drawing = false; $('draw').setAttribute('aria-pressed', 'false'); renderDraft(); renderOverlay();
});
$('overlay').addEventListener('pointercancel', () => { startPoint = null; endPoint = null; renderOverlay(); });
$('image').addEventListener('load', renderOverlay);
new ResizeObserver(renderOverlay).observe($('overlay'));
for (const id of ['show-original','show-teacher','show-draft']) $(id).addEventListener('change', renderOverlay);
for (const id of ['reviewer','notes','rights','check-all','check-qa','check-provenance','check-hazards']) $(id).addEventListener('input', changed);
$('item').addEventListener('change', () => loadItem($('item').value));
for (const [id, step] of [['previous',-1],['next',1]]) $(id).addEventListener('click', () => { const index = queue.findIndex(i => i.image_id === current.image_id); const next = queue[index + step]; if (next) loadItem(next.image_id); });
$('draw').addEventListener('click', () => { drawing = !drawing; $('draw').setAttribute('aria-pressed', String(drawing)); message(drawing ? 'Chọn class rồi kéo một box trên ảnh.' : 'Đã tắt chế độ vẽ.'); });
$('delete').addEventListener('click', () => { if (selected < 0) { error('Chọn box nháp cần xóa.'); return; } boxes.splice(selected, 1); selected = -1; changed(); renderDraft(); renderOverlay(); });
$('reset').addEventListener('click', () => { boxes = structuredClone(current.original_boxes); selected = -1; changed(); renderDraft(); renderOverlay(); });
$('reload').addEventListener('click', () => { dirty = false; loadItem(current.image_id); });
$('add-issue').addEventListener('click', () => { issues.push({ type: $('issue-type').value, resolved: false, note: '' }); changed(); renderIssues(); });
for (const [id, state] of [['save-draft','draft'],['hold','hold'],['reject','rejected'],['approve','approved']]) $(id).addEventListener('click', () => save(state));

try {
  session = await api('/api/session'); queue = await api('/api/items');
  $('session').textContent = `${session.test_only ? 'WORKSPACE KIỂM THỬ — không được duyệt/export training. ' : 'Workspace review thật — cần người kiểm duyệt. '}${session.workspace} · fingerprint ${session.dataset_fingerprint}`;
  classOptions($('new-class'), null); updateQueue();
  if (queue.length) await loadItem(queue[0].image_id); else message('Pilot chưa có ảnh.');
} catch (err) { error(err.message); }
