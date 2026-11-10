import { useEffect, useRef, useState } from 'react';
import { objectColor, boxLayout, canAttemptDetection, fetchJson, prepareUpload, validateDetectionResponse } from '../utils/yoloView.mjs';
import WasteObjectCard from './WasteObjectCard';
import './YoloDetectionPanel.css';

export default function YoloDetectionPanel({ backendUrl, onOpenSafety }) {
  const [upload, setUpload] = useState(null);
  const [preview, setPreview] = useState('');
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState('idle');
  const [error, setError] = useState('');
  const [selected, setSelected] = useState(null);
  const [capabilities, setCapabilities] = useState(null);
  const [capError, setCapError] = useState('');
  const [refresh, setRefresh] = useState(0);
  const [renderedWidth, setRenderedWidth] = useState(640);
  const work = useRef({ sequence: 0, controller: null, url: '' });
  const imageFrame = useRef(null);
  const cards = useRef([]);
  const boxes = useRef([]);
  const input = useRef(null);

  useEffect(() => {
    const controller = new AbortController();
    fetchJson(`${backendUrl}/api/yolo/capabilities`, { signal: controller.signal }).then((data) => {
      if (!controller.signal.aborted) { setCapabilities(data); setCapError(''); }
    }).catch((err) => { if (!controller.signal.aborted) setCapError(err.message); });
    return () => controller.abort();
  }, [backendUrl, refresh]);

  useEffect(() => () => { work.current.sequence += 1; work.current.controller?.abort(); URL.revokeObjectURL(work.current.url); }, []);

  useEffect(() => {
    const element = imageFrame.current;
    if (!element) return undefined;
    const observer = new ResizeObserver(([entry]) => setRenderedWidth(entry.contentRect.width));
    observer.observe(element);
    return () => observer.disconnect();
  }, [preview]);

  function cancel() { work.current.sequence += 1; work.current.controller?.abort(); setStatus('idle'); }

  async function chooseFile(file) {
    cancel(); const sequence = work.current.sequence;
    URL.revokeObjectURL(work.current.url); work.current.url = '';
    setUpload(null); setPreview(''); setResult(null); setSelected(null); setError(''); setStatus('preparing');
    if (!file) { setStatus('idle'); return; }
    try {
      const prepared = await prepareUpload(file);
      if (sequence !== work.current.sequence) return;
      const url = URL.createObjectURL(prepared.file); work.current.url = url;
      setUpload(prepared); setPreview(url); setStatus('idle');
    } catch (err) { if (sequence === work.current.sequence) { setError(err.message); setStatus('error'); } }
  }

  async function analyse() {
    if (!upload) return;
    cancel(); const sequence = work.current.sequence;
    const controller = new AbortController(); work.current.controller = controller;
    setStatus('loading'); setError(''); setResult(null); setSelected(null);
    try {
      const body = new FormData(); body.append('file', upload.file);
      const response = await fetchJson(`${backendUrl}/api/yolo/detect`, { method: 'POST', body, signal: controller.signal });
      if (sequence !== work.current.sequence) return;
      setResult(validateDetectionResponse(response, upload.width, upload.height)); setStatus('done'); setRefresh((value) => value + 1);
    } catch (err) {
      if (sequence === work.current.sequence && err.name !== 'AbortError') { setError(err.message || 'Không kết nối được API.'); setStatus('error'); setRefresh((value) => value + 1); }
    }
  }

  function selectFromBox(index) {
    setSelected(index); cards.current[index]?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); cards.current[index]?.focus({ preventScroll: true });
  }

  function selectFromCard(index) {
    setSelected(index); imageFrame.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); boxes.current[index]?.focus({ preventScroll: true });
  }

  return <section className="yolo-panel" aria-labelledby="yolo-title">
    <header className="yolo-intro"><div><h2 id="yolo-title">Nhận diện vật thể bằng YOLO</h2><p>Chín nhóm vật thể · bản thử nghiệm · chưa xác nhận vật liệu hoặc an toàn.</p></div>
      <button type="button" onClick={onOpenSafety}>Pin và hướng dẫn nguy hại</button></header>
    <p className="yolo-model-state">{capError ? `Không kiểm tra được model: ${capError}` : capabilities ? `Model: ${capabilities.model?.model_version || 'chưa sẵn sàng'} · ${capabilities.status}` : 'Đang kiểm tra model…'} <button type="button" className="yolo-text-button" onClick={() => setRefresh((value) => value + 1)}>Kiểm tra lại</button></p>
    {capabilities && !capabilities.runtime_ready && <p role="alert" className="yolo-alert">{capabilities.reason}{canAttemptDetection(capabilities) && ' Bạn có thể gửi yêu cầu mới để thử lại; model hiện chưa sẵn sàng.'}</p>}
    <label className="yolo-upload">Chọn ảnh rác để phân tích<input ref={input} type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => { chooseFile(event.target.files?.[0]); event.target.value = ''; }} /></label>
    {status === 'preparing' && <p role="status">Đang chuẩn bị ảnh…</p>}
    {preview && <>
      <p className="yolo-image-caption">{upload?.originalName} · {upload?.width} × {upload?.height} px</p>
      <div ref={imageFrame} className="yolo-image-frame">
        <img src={preview} alt="Ảnh phân tích YOLO" />
        {result && <svg className="yolo-box-layer" viewBox={`0 0 ${result.image_width} ${result.image_height}`} preserveAspectRatio="none" aria-label="Các vật thể YOLO phát hiện">
          {result.detections.map((detection, index) => {
            const color = objectColor(index);
            const label = `${index + 1}. ${detection.label_vi} ${(detection.confidence * 100).toFixed(1)}%`;
            const layout = boxLayout(detection.bounding_box, result.image_width, result.image_height, renderedWidth, label);
            return <g key={index} ref={(element) => { boxes.current[index] = element; }} role="button" tabIndex={0} aria-label={`Chọn vật thể ${index + 1}: ${detection.label_vi}`} aria-pressed={selected === index}
              onClick={() => selectFromBox(index)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); selectFromBox(index); } }} className={selected === index ? 'box-selected' : ''}>
              <rect x={layout.x} y={layout.y} width={layout.width} height={layout.height} fill={color} fillOpacity={selected === index ? 0.22 : 0.04} stroke={color} strokeWidth={selected === index ? 4 : 2} vectorEffect="non-scaling-stroke" />
              <rect x={layout.labelX} y={layout.labelY} width={layout.labelWidth} height={layout.labelHeight} fill={color} />
              <text x={layout.labelX + layout.fontSize * 0.4} y={layout.labelY + layout.fontSize * 1.3} fontSize={layout.fontSize} fill="#07111c" fontWeight="700">{label}</text>
            </g>;
          })}
        </svg>}
      </div>
    </>}
    <div className="yolo-actions"><button type="button" disabled={!upload || status === 'loading' || status === 'preparing' || !canAttemptDetection(capabilities)} onClick={analyse}>Phân tích bằng YOLO</button>
      {status === 'loading' && <button type="button" onClick={cancel}>Hủy yêu cầu</button>}
      {(upload || error) && <button type="button" onClick={() => chooseFile(null)}>Xóa ảnh / kết quả</button>}</div>
    {status === 'loading' && <p role="status">Đang nhận diện… Bạn có thể hủy hoặc chọn ảnh khác; kết quả request cũ sẽ được bỏ qua.</p>}
    {error && <p role="alert" className="yolo-alert">{error}</p>}
    <section className="yolo-results" aria-labelledby="yolo-results-title" aria-live="polite">
      <h3 id="yolo-results-title">Kết quả nhận diện{result ? ` · ${result.total_objects} vật thể` : ''}</h3>
      {!result && <p>Chọn ảnh rồi phân tích để xem box và từng vật thể bên dưới.</p>}
      {result && <><p className="yolo-model-state">Model suy luận: {result.model.model_version} · Ngưỡng lọc: {result.confidence_threshold} · Ngưỡng cần kiểm tra: {result.uncertainty_threshold}. Confidence là score của model, chưa phải xác suất đã hiệu chỉnh.</p>
        <p className="yolo-review">{result.uncertainty_note} Không có cảnh báo không có nghĩa ảnh an toàn.</p>
        {result.total_objects === 0 && <p className="yolo-empty">Không có vật thể vượt ngưỡng hiện tại. Không kết luận ảnh không có rác, pin hay vật nguy hại.</p>}
        {selected !== null && <p>Đang chọn vật thể {selected + 1}. Click box hoặc tiêu đề card để đối chiếu.</p>}
        <div className="yolo-card-grid">{result.detections.map((detection, index) => <WasteObjectCard key={index} detection={detection} index={index} color={objectColor(index)}
          selected={selected === index} onSelect={() => selectFromCard(index)} registerRef={(element) => { cards.current[index] = element; }} backendUrl={backendUrl} onOpenSafety={onOpenSafety} />)}</div>
      </>}
    </section>
  </section>;
}
