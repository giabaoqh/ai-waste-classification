import { useEffect, useRef, useState } from 'react';
import { fetchJson, OBJECT_DESCRIPTIONS } from '../utils/yoloView.mjs';

const MATERIALS = [['unknown', 'Chưa xác định'], ['paper', 'Giấy / carton'], ['plastic', 'Nhựa'], ['metal', 'Kim loại'], ['glass', 'Thủy tinh'], ['organic', 'Thực phẩm / hữu cơ'], ['mixed', 'Hỗn hợp / nhiều lớp']];

export default function WasteObjectCard({ detection, index, color, selected, onSelect, registerRef, backendUrl, onOpenSafety }) {
  const [itemType, setItemType] = useState('unknown');
  const [material, setMaterial] = useState('unknown');
  const [condition, setCondition] = useState('not_assessed');
  const [guidance, setGuidance] = useState(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const work = useRef({ sequence: 0, controller: null });

  useEffect(() => () => { work.current.sequence += 1; work.current.controller?.abort(); }, []);

  function invalidate() {
    work.current.sequence += 1; work.current.controller?.abort(); setGuidance(null); setError(''); setBusy(false);
  }

  async function getGuidance() {
    invalidate();
    const sequence = work.current.sequence;
    const controller = new AbortController(); work.current.controller = controller; setBusy(true);
    try {
      const data = await fetchJson(`${backendUrl}/api/safety/assess`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: controller.signal,
        body: JSON.stringify({ item_type: itemType, ordinary_material: material, battery_condition: condition, city: 'Đà Nẵng' }),
      });
      if (sequence === work.current.sequence) setGuidance(data);
    } catch (err) {
      if (sequence === work.current.sequence && err.name !== 'AbortError') setError(err.message);
    } finally { if (sequence === work.current.sequence) setBusy(false); }
  }

  return <article ref={registerRef} tabIndex={-1} id={`yolo-card-${index}`} className={`yolo-card ${selected ? 'is-selected' : ''}`} style={{ '--object-color': color }} aria-label={`Kết quả vật thể ${index + 1}`}
    onClick={(event) => { if (!event.target.closest('button,input,select,summary,a,details')) onSelect(); }}
    onKeyDown={(event) => { if (event.target === event.currentTarget && event.key === 'Enter') onSelect(); }}>
    <button type="button" className="yolo-card-title" onClick={onSelect} aria-pressed={selected}>
      <span className="yolo-number">{index + 1}</span><strong>{detection.label_vi}</strong><span>{(detection.confidence * 100).toFixed(1)}%</span>
    </button>
    <p className="yolo-class">Class YOLO: <code>{detection.class_name}</code></p>
    <p className="yolo-review">{detection.is_uncertain ? 'Score thấp — cần người kiểm tra.' : 'Model thử nghiệm — vẫn cần xác nhận.'}</p>
    <p>{OBJECT_DESCRIPTIONS[detection.class_name]}</p>
    <dl><dt>Nhóm vật liệu</dt><dd>{itemType === 'ordinary' && material !== 'unknown' ? `${MATERIALS.find(([key]) => key === material)?.[1]} (do bạn cung cấp, chưa xác minh)` : 'Chưa xác định'}</dd>
      <dt>Phân loại / xử lý</dt><dd>{guidance ? 'Hướng dẫn phòng ngừa theo thông tin bạn khai báo, có nguồn bên dưới.' : 'Chưa xác định — cần bổ sung thông tin.'}</dd></dl>
    <details><summary>Bổ sung thông tin vật thể {index + 1}</summary>
      <p>Thông tin sau do bạn quan sát, không phải kết luận vật liệu hay nguy hại của YOLO.</p>
      <label>Loại vật thể bạn khai báo<select value={itemType} onChange={(event) => { invalidate(); setItemType(event.target.value); }}>
        <option value="unknown">Chưa xác định</option><option value="ordinary">Vật thông thường — bổ sung vật liệu</option>
        <option value="battery">Pin / ắc quy</option><option value="medical_sharp">Vật sắc nhọn y tế</option>
        <option value="medical_suspected">Nghi rác y tế</option><option value="e_waste">Thiết bị điện tử</option><option value="radiation_suspected">Nghi nguồn phóng xạ</option>
      </select></label>
      {itemType === 'ordinary' && <label>Vật liệu theo bạn quan sát<select value={material} onChange={(event) => { invalidate(); setMaterial(event.target.value); }}>{MATERIALS.map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select></label>}
      {itemType === 'battery' && <><label>Tình trạng pin do bạn cung cấp<select value={condition} onChange={(event) => { invalidate(); setCondition(event.target.value); }}>
        <option value="not_assessed">Chưa biết</option><option value="intact_cool">Nguyên vẹn, nguội</option><option value="damaged">Hỏng / phồng / rò rỉ</option><option value="hot_or_smoking">Nóng / bốc khói</option>
      </select></label><p>Không tự chạy classifier pin trên ảnh nhiều vật. Nếu cần, chụp riêng một vật và <button className="yolo-text-button" type="button" onClick={onOpenSafety}>mở luồng sàng lọc pin</button>.</p></>}
      <button type="button" disabled={busy || itemType === 'unknown'} onClick={getGuidance}>{busy ? 'Đang lấy hướng dẫn…' : 'Xem hướng dẫn theo khai báo'}</button>
    </details>
    {error && <p role="alert">{error}</p>}
    {guidance && <div className="yolo-guidance"><h4>{guidance.title}</h4><p>Nguồn thông tin vật thể: bạn cung cấp. Chưa xác nhận an toàn.</p>
      {!!guidance.risks?.length && <ul>{guidance.risks.map((text) => <li key={text}>{text}</li>)}</ul>}
      <ol>{guidance.steps.map((text) => <li key={text}>{text}</li>)}</ol>
      {!!guidance.avoid.length && <ul>{guidance.avoid.map((text) => <li key={text}>{text}</li>)}</ul>}
      {guidance.sources.map((url, position) => <a key={url} href={url} target="_blank" rel="noreferrer">Nguồn {position + 1} </a>)}
      {!!guidance.collection_contacts.length && <><p>Điểm từng công bố thu gom — cần liên hệ xác nhận còn tiếp nhận:</p><ul>{guidance.collection_contacts.map((point) => <li key={point.id}>{point.address} · <a href={point.source_url} target="_blank" rel="noreferrer">Nguồn / liên hệ</a></li>)}</ul></>}
    </div>}
  </article>;
}
