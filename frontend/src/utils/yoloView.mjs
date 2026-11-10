// Object semantics only; material/hazard guidance requires a separate user report.
export const OBJECT_DESCRIPTIONS = {
  bottle_jar: 'Nhóm chai hoặc lọ. Hình dạng này có thể thuộc nhiều vật liệu khác nhau.',
  can: 'Nhóm vật thể dạng lon. Cần kiểm tra thêm thành phần và phần còn chứa bên trong.',
  cup: 'Nhóm cốc hoặc ly. Nhãn này chưa xác định giấy, nhựa hay thủy tinh.',
  carton: 'Nhóm hộp dạng carton, có thể gồm bao bì nhiều lớp. Chưa xác nhận khả năng tái chế.',
  cap_lid: 'Nhóm nắp chai hoặc nắp hộp. Chưa xác định thành phần vật liệu.',
  bag_film_wrapper: 'Nhóm túi, màng hoặc vỏ gói. Không mặc định tất cả là nhựa hoặc được thu gom chung.',
  paper: 'Nhóm vật thể mô hình gợi ý là giấy. Cần xác nhận vật liệu, lớp phủ và tình trạng nhiễm bẩn.',
  cigarette: 'Nhóm đầu lọc hoặc phần còn lại của thuốc lá. Không suy ra thành phần hoặc mức nguy hại định lượng.',
  other_litter: 'Nhóm vật thể rác khác. Có thể chứa vật nguy hại; không dùng nhãn này để chọn thùng rác thông thường.',
};

export const BOX_COLORS = ['#38bdf8', '#fbbf24', '#c084fc', '#4ade80', '#fb7185', '#2dd4bf', '#fb923c', '#a3e635', '#e879f9'];
export const objectColor = (index) => BOX_COLORS[index] || `hsl(${(index * 137.508) % 360} 85% 65%)`;

// The contract keeps a transient failure visible until a new POST succeeds.
// GET refresh alone cannot recover it; allow an explicit retry, not a ready claim.
export function canAttemptDetection(capabilities) {
  return capabilities?.runtime_ready === true || (capabilities?.artifact_verified === true &&
    ['inference_error', 'load_failed'].includes(capabilities.status));
}

export function boxLayout(box, imageWidth, imageHeight, renderedWidth, text) {
  const [x1, y1, x2, y2] = box;
  const scale = imageWidth / Math.max(1, renderedWidth);
  const fontSize = 13 * scale;
  const labelHeight = 24 * scale;
  const labelWidth = Math.min(imageWidth, (text.length * 7 + 16) * scale);
  return { x: x1, y: y1, width: x2 - x1, height: y2 - y1, fontSize, labelHeight, labelWidth,
    labelX: Math.max(0, Math.min(x1, imageWidth - labelWidth)),
    labelY: Math.max(0, Math.min(y1 >= labelHeight ? y1 - labelHeight : y1, imageHeight - labelHeight)) };
}

export function validateDetectionResponse(body, width, height) {
  if (!body || body.inference_status !== 'completed' || body.model?.taxonomy !== 'object_groups' || body.model?.task !== 'detect' ||
      body.bounding_box_format !== 'xyxy_pixels' || body.coordinate_space !== 'exif_oriented_image' ||
      body.image_width !== width || body.image_height !== height || !Array.isArray(body.detections) ||
      body.total_objects !== body.detections.length || !Array.isArray(body.model.class_mapping) || body.model.class_mapping.length !== 9) {
    throw new Error('Phản hồi YOLO không đúng contract hoặc kích thước ảnh không khớp. Không vẽ box trên ảnh này.');
  }
  body.model.class_mapping.forEach((row, index) => {
    if (row.class_id !== index || !Object.hasOwn(OBJECT_DESCRIPTIONS, row.class_name)) throw new Error('Mapping YOLO chưa được hỗ trợ.');
  });
  if (new Set(body.model.class_mapping.map((row) => row.class_name)).size !== 9) throw new Error('Mapping YOLO có nhãn lặp.');
  body.detections.forEach((row) => {
    const mapping = body.model.class_mapping[row.class_id];
    const box = row.bounding_box;
    if (!mapping || mapping.class_name !== row.class_name || !Number.isFinite(row.confidence) || row.confidence < 0 || row.confidence > 1 ||
        !Array.isArray(box) || box.length !== 4 || !box.every(Number.isFinite) ||
        box[0] < 0 || box[1] < 0 || box[2] > width || box[3] > height || box[2] <= box[0] || box[3] <= box[1]) {
      throw new Error('Box hoặc mapping trong phản hồi không hợp lệ.');
    }
  });
  return body;
}

export async function fetchJson(url, options = {}, timeoutMs = 30000, fetcher = fetch) {
  const controller = new AbortController();
  let timedOut = false;
  const relay = () => controller.abort();
  options.signal?.addEventListener('abort', relay, { once: true });
  if (options.signal?.aborted) controller.abort();
  const timer = setTimeout(() => { timedOut = true; controller.abort(); }, timeoutMs);
  try {
    const response = await fetcher(url, { ...options, signal: controller.signal });
    const body = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = body?.detail;
      throw new Error(typeof detail === 'string' ? detail : detail?.message || `API trả lỗi HTTP ${response.status}.`);
    }
    if (!body) throw new Error('API không trả JSON hợp lệ.');
    return body;
  } catch (error) {
    if (timedOut) throw new Error('Yêu cầu quá thời gian chờ 30 giây. Vui lòng thử lại.');
    throw error;
  } finally {
    clearTimeout(timer);
    options.signal?.removeEventListener('abort', relay);
  }
}

export async function prepareUpload(file) {
  if (!file || file.size === 0) throw new Error('Chọn một tệp ảnh có nội dung.');
  if (file.size > 10 * 1024 * 1024) throw new Error('Ảnh vượt quá 10 MB.');
  if (file.type && !['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) throw new Error('Chỉ nhận JPEG, PNG hoặc WebP.');
  let bitmap;
  try { bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' }); }
  catch { throw new Error('Không đọc được ảnh. Tệp có thể bị hỏng hoặc sai định dạng.'); }
  try {
    const { width, height } = bitmap;
    if (Math.min(width, height) < 64 || width * height > 20000000) throw new Error('Ảnh cần ít nhất 64 px mỗi chiều và không vượt 20 megapixel.');
    const canvas = document.createElement('canvas');
    canvas.width = width; canvas.height = height;
    const context = canvas.getContext('2d');
    if (!context) throw new Error('Trình duyệt không hỗ trợ chuẩn hóa hướng ảnh.');
    context.fillStyle = '#ffffff'; context.fillRect(0, 0, width, height); context.drawImage(bitmap, 0, 0);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.95));
    if (!blob || blob.size > 10 * 1024 * 1024) throw new Error('Ảnh sau chuẩn hóa vượt giới hạn hoặc không tạo được preview. Chọn ảnh nhỏ hơn.');
    return { file: new File([blob], 'upload-oriented.jpg', { type: 'image/jpeg' }), width, height, originalName: file.name };
  } finally { bitmap.close(); }
}
