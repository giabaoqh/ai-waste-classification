import { useState } from 'react';
import './SafetyPanel.css';

const ITEM_TYPES = [
  ['unknown', 'Chưa xác định'], ['battery', 'Pin / ắc quy'],
  ['e_waste', 'Thiết bị điện tử'], ['medical_sharp', 'Kim tiêm / vật sắc nhọn y tế'],
  ['medical_suspected', 'Nghi rác y tế'], ['radiation_suspected', 'Nghi nguồn phóng xạ'],
  ['ordinary', 'Rác thông thường'],
];

export default function SafetyPanel({ backendUrl }) {
  const [itemType, setItemType] = useState('unknown');
  const [condition, setCondition] = useState('not_assessed');
  const [city, setCity] = useState('Đà Nẵng');
  const [material, setMaterial] = useState('unknown');
  const [file, setFile] = useState(null);
  const [scan, setScan] = useState(null);
  const [guidance, setGuidance] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  async function submit(mode) {
    setBusy(true);
    setError('');
    setGuidance(null);
    setScan(null);
    try {
      let response;
      if (mode === 'scan') {
        if (!file) throw new Error('Chọn ảnh cận cảnh một vật trước khi sàng lọc.');
        if (file.size > 10 * 1024 * 1024) throw new Error('Ảnh cần nhỏ hơn 10 MB.');
        const body = new FormData();
        body.append('file', file);
        response = await fetch(`${backendUrl}/api/safety/scan`, { method: 'POST', body });
      } else {
        response = await fetch(`${backendUrl}/api/safety/assess`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ item_type: itemType, battery_condition: condition, city, ordinary_material: material }),
        });
      }
      const result = await response.json();
      if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'Không thể xử lý yêu cầu.');
      if (mode === 'scan') {
        setScan(result);
        setGuidance(result.guidance);
      } else setGuidance(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Không kết nối được dịch vụ.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="safety-panel" aria-labelledby="safety-title">
      <h2 id="safety-title">Pin và rác có nguy cơ</h2>
      <p>Ưu tiên Đà Nẵng. Ảnh không xác nhận được nhiễm bệnh, mức bức xạ hay việc một vật an toàn.</p>
      <div className="safety-grid">
        <fieldset disabled={busy}>
          <legend>Sàng lọc pin qua ảnh — thử nghiệm</legend>
          <p>Chọn ảnh cận cảnh một vật. Mô hình vẫn có thể bỏ sót pin nhìn thấy được. Chưa hỗ trợ tìm pin ẩn hoặc khoanh vùng pin trong đống rác.</p>
          <label>Ảnh JPEG, PNG hoặc WebP, tối đa 10 MB
            <input type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => {
              setFile(event.target.files?.[0] || null); setScan(null); setGuidance(null); setError('');
            }} />
          </label>
          <button type="button" disabled={!file} onClick={() => submit('scan')}>Sàng lọc hình dạng pin</button>
        </fieldset>
        <fieldset disabled={busy}>
          <legend>Hướng dẫn theo điều bạn quan sát</legend>
          <label>Loại vật thể
            <select value={itemType} onChange={(event) => { setItemType(event.target.value); setGuidance(null); setScan(null); }}>
              {ITEM_TYPES.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
            </select>
          </label>
          {itemType === 'battery' && <label>Tình trạng pin (do bạn cung cấp)
            <select value={condition} onChange={(event) => { setCondition(event.target.value); setGuidance(null); }}>
              <option value="not_assessed">Chưa biết</option>
              <option value="intact_cool">Nguyên vẹn, nguội</option>
              <option value="damaged">Phồng, hỏng hoặc rò rỉ</option>
              <option value="hot_or_smoking">Đang nóng hoặc bốc khói</option>
            </select>
          </label>}
          {itemType === 'ordinary' && <label>Vật liệu bạn quan sát
            <select value={material} onChange={(event) => { setMaterial(event.target.value); setGuidance(null); }}>
              {[['unknown', 'Chưa rõ'], ['paper', 'Giấy / carton'], ['plastic', 'Nhựa'], ['metal', 'Lon / kim loại'], ['glass', 'Chai lọ thủy tinh'], ['organic', 'Thực phẩm / hữu cơ'], ['mixed', 'Bao bì hỗn hợp']].map(([value, label]) => <option key={value} value={value}>{label}</option>)}
            </select>
          </label>}
          <label>Tỉnh / thành phố
            <input value={city} maxLength={100} onChange={(event) => { setCity(event.target.value); setGuidance(null); }} />
          </label>
          <button type="button" onClick={() => submit('assess')}>Xem hướng dẫn xử lý</button>
        </fieldset>
      </div>
      {busy && <p role="status">Đang xử lý…</p>}
      {error && <p role="alert" className="safety-alert">{error}</p>}
      <div aria-live="polite">
        {scan && <div className="safety-result">
          <h3>{scan.label_vi}</h3>
          <p>Score hình dạng pin: {scan.battery_score.toFixed(3)} · Ngưỡng: {scan.threshold.toFixed(2)}. Đây chưa phải xác suất đã hiệu chỉnh.</p>
          <p>{scan.uncertainty_note}</p>
          {scan.battery_suspected && <p>Chọn “Pin / ắc quy” và tình trạng pin bên trên để xem hướng dẫn phù hợp.</p>}
        </div>}
        {guidance && <article className="safety-result">
          <h3>{guidance.title}</h3>
          <p>{guidance.evidence_source === 'user_report' ? 'Dựa trên thông tin bạn chọn.' : 'Gợi ý từ mô hình hình ảnh thử nghiệm.'} Chưa xác nhận vật thể an toàn.</p>
          {!!guidance.risks.length && <><h4>Nguy cơ cần lưu ý</h4><ul>{guidance.risks.map((text) => <li key={text}>{text}</li>)}</ul></>}
          <h4>Cách xử lý trước mắt</h4><ol>{guidance.steps.map((text) => <li key={text}>{text}</li>)}</ol>
          {!!guidance.avoid.length && <><h4>Điều cần tránh</h4><ul>{guidance.avoid.map((text) => <li key={text}>{text}</li>)}</ul></>}
          {!!guidance.collection_contacts.length && <><h4>Điểm từng công bố thu gom pin — cần xác nhận</h4>
            <p>Chưa xác nhận hoạt động hiện tại, giờ mở cửa hay loại pin tiếp nhận. Liên hệ trước khi mang tới.</p>
            <ul>{guidance.collection_contacts.map((point) => <li key={point.id}>
              <strong>{point.name}</strong>: {point.address}. <a href={point.source_url} target="_blank" rel="noreferrer">Nguồn và liên hệ</a>
            </li>)}</ul></>}
          {!!guidance.sources.length && <p>Nguồn: {guidance.sources.map((url, index) => <a className="safety-source" key={url} href={url} target="_blank" rel="noreferrer">Tài liệu {index + 1}</a>)}</p>}
        </article>}
      </div>
    </section>
  );
}
