import React, { useState, useEffect, useRef } from 'react';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://127.0.0.1:8000';

const CLASS_META = {
  plastic: { icon: '🧴', label: 'Nhựa' },
  paper: { icon: '📦', label: 'Giấy / Carton' },
  metal: { icon: '🥫', label: 'Kim loại' },
  glass: { icon: '🍾', label: 'Thủy tinh' },
  organic: { icon: '🍎', label: 'Hữu cơ' },
  other: { icon: '🗑️', label: 'Khác / Hỗn hợp' },
};

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [backendHealth, setBackendHealth] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  // Check backend health on initial load
  useEffect(() => {
    async function checkHealth() {
      try {
        const res = await fetch(`${BACKEND_URL}/api/health`);
        if (res.ok) {
          const data = await res.json();
          setBackendHealth({ online: true, data });
        } else {
          setBackendHealth({ online: false, status: res.status });
        }
      } catch (err) {
        setBackendHealth({ online: false, error: err.message });
      }
    }
    checkHealth();
  }, []);

  const handleFileChange = (file) => {
    if (!file) return;
    setError(null);
    setResult(null);

    // Basic frontend check
    if (!file.type.startsWith('image/')) {
      setError('Tệp được chọn không phải là hình ảnh hợp lệ (vui lòng chọn file .jpg, .png, .webp...).');
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setError('Kích thước ảnh vượt quá giới hạn cho phép (tối đa 10 MB).');
      return;
    }

    setSelectedFile(file);
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
  };

  const onFileInputChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileChange(e.target.files[0]);
    }
  };

  const onDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = () => {
    setIsDragging(false);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedFile) {
      setError('Vui lòng chọn hoặc kéo thả một hình ảnh rác thải trước khi nhận diện.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`${BACKEND_URL}/api/predict`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || `Lỗi máy chủ HTTP ${response.status}`);
      }

      setResult(data);
    } catch (err) {
      console.error('Lỗi khi gửi yêu cầu nhận diện:', err);
      setError(err.message || 'Không thể kết nối đến máy chủ Backend API. Hãy kiểm tra xem backend đã được khởi động chưa.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="brand-badge">
          <span>🌿 Capstone Project</span>
          <span>•</span>
          <span>Phase 5.4.3: AI Inference (MobileNetV3-Large)</span>
        </div>
        <h1>Hệ Thống Nhận Diện Rác Thải & Gợi Ý Tái Chế</h1>
        <p>
          AI-Based Waste Recognition and Recycling Recommendation System
        </p>
      </header>

      {/* Main Grid */}
      <div className="main-grid">
        {/* Input Column */}
        <div className="card">
          <div className="card-title">
            <span>Tải Ảnh Rác Thải</span>
            {backendHealth && (
              <span className={`backend-status-pill ${backendHealth.online ? 'backend-online' : 'backend-offline'}`}>
                <span className="status-dot"></span>
                {backendHealth.online ? (
                  backendHealth.data?.ai_model_loaded ? (
                    `Model Sẵn Sàng (${backendHealth.data.device})`
                  ) : (
                    'Backend Online (Chưa load model)'
                  )
                ) : (
                  'Backend Offline'
                )}
              </span>
            )}
          </div>

          <form onSubmit={handleSubmit}>
            {/* Dropzone */}
            <div
              className={`dropzone ${isDragging ? 'active' : ''}`}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onDrop={onDrop}
              onClick={() => fileInputRef.current && fileInputRef.current.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden-file-input"
                onChange={onFileInputChange}
              />
              <div className="dropzone-icon">📷</div>
              <div className="dropzone-text">
                {selectedFile ? 'Nhấp hoặc kéo thả để đổi ảnh khác' : 'Chọn hoặc kéo thả ảnh vào đây'}
              </div>
              <div className="dropzone-hint">
                Hỗ trợ định dạng JPEG, PNG, WebP (Tối đa 10 MB)
              </div>
            </div>

            {/* Preview */}
            {previewUrl && (
              <div className="preview-container">
                <img src={previewUrl} alt="Xem trước rác thải" className="preview-image" />
                <div className="preview-info">
                  <span className="file-name">{selectedFile?.name}</span>
                  <span className="file-size">
                    {(selectedFile?.size / 1024).toFixed(1)} KB
                  </span>
                </div>
              </div>
            )}

            {/* Error Message */}
            {error && (
              <div className="alert alert-error">
                <span className="alert-icon">⚠️</span>
                <div>{error}</div>
              </div>
            )}

            {/* Actions */}
            <div className="action-group">
              <button
                type="submit"
                className="btn-primary"
                disabled={!selectedFile || isLoading}
              >
                {isLoading ? (
                  <>
                    <span className="spinner"></span>
                    <span>Đang phân tích hình ảnh AI...</span>
                  </>
                ) : (
                  <span>Gửi Nhận Diện AI (POST /api/predict)</span>
                )}
              </button>

              {selectedFile && (
                <button
                  type="button"
                  className="btn-secondary"
                  onClick={handleReset}
                  disabled={isLoading}
                >
                  Xóa
                </button>
              )}
            </div>
          </form>
        </div>

        {/* Results Column */}
        <div className="card">
          <div className="card-title">
            <span>Kết Quả Phân Tích AI</span>
            {result && (
              result.mock ? (
                <span className="mock-badge">Mock Response</span>
              ) : (
                <span className="real-badge">MobileNetV3-Large</span>
              )
            )}
          </div>

          {!result && !isLoading && (
            <div className="results-empty">
              <div style={{ fontSize: '40px', marginBottom: '12px' }}>♻️</div>
              <p>Chưa có kết quả nhận diện.</p>
              <p style={{ fontSize: '13px', marginTop: '6px' }}>
                Hãy chọn một ảnh rác thải bên trái và bấm <strong>"Gửi Nhận Diện AI"</strong> để phân loại.
              </p>
            </div>
          )}

          {isLoading && (
            <div className="results-empty">
              <div className="spinner" style={{ borderColor: 'rgba(5, 150, 105, 0.3)', borderTopColor: '#059669', width: '32px', height: '32px', margin: '0 auto 16px' }}></div>
              <p>Đang gửi ảnh tới mạng neural MobileNetV3-Large để dự đoán...</p>
            </div>
          )}

          {result && (
            <div>
              {result.mock ? (
                <div className="alert alert-warning">
                  <span className="alert-icon">ℹ️</span>
                  <div>
                    <strong>Thông báo:</strong> {result.message}
                  </div>
                </div>
              ) : (
                <>
                  {/* Hero Prediction Result */}
                  <div className="prediction-hero">
                    <div className="prediction-hero-header">
                      <div className="prediction-class-title">
                        <span>{CLASS_META[result.predicted_class]?.icon || '🏷️'}</span>
                        <span>{result.label_vi}</span>
                      </div>
                      <span className="confidence-badge">
                        Độ tin cậy: {(result.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div style={{ fontSize: '11.5px', color: '#047857', opacity: 0.85, marginTop: '4px' }}>
                      * Độ tin cậy thể hiện xác suất của mô hình AI đối với 6 nhóm mục tiêu, không phải cam kết tuyệt đối.
                    </div>
                  </div>

                  {/* Warning banner for "other" category */}
                  {result.predicted_class === 'other' && (
                    <div className="alert alert-warning" style={{ marginTop: '12px' }}>
                      <span className="alert-icon">⚠️</span>
                      <div style={{ fontSize: '13px' }}>
                        <strong>Danh mục tồn dư:</strong> Nhóm "Khác / Hỗn hợp" gồm các vật liệu không xác định hoặc không thuộc 5 nhóm tái chế/hữu cơ chuẩn. Vui lòng phân loại thủ công hoặc kiểm tra quy tắc rác thải tại địa phương.
                      </div>
                    </div>
                  )}

                  {/* Recommendation Card */}
                  {result.recommendation && (
                    <div className="recommendation-box">
                      <div
                        className="bin-tag"
                        style={{ backgroundColor: result.recommendation.bin_color || '#059669' }}
                      >
                        🗑️ {result.recommendation.bin}
                      </div>
                      <p className="guidance-text">{result.recommendation.guidance}</p>
                      
                      {result.recommendation.tips && result.recommendation.tips.length > 0 && (
                        <ul className="tips-list">
                          {result.recommendation.tips.map((tip, idx) => (
                            <li key={idx} className="tip-item">
                              <span className="tip-bullet">✓</span>
                              <span>{tip}</span>
                            </li>
                          ))}
                        </ul>
                      )}
                      <div style={{ fontSize: '11.5px', color: '#64748b', marginTop: '10px', fontStyle: 'italic' }}>
                        ℹ️ Hướng dẫn mang tính tham khảo thực tế. Hãy luôn ưu tiên quy định phân loại rác của chính quyền địa phương.
                      </div>
                    </div>
                  )}

                  {/* Probability Breakdown */}
                  {result.probabilities && (
                    <div className="probabilities-card">
                      <div className="prob-title">
                        <span>Phân Bố Xác Suất (6 Nhóm Rác)</span>
                        <span style={{ fontSize: '11px', color: '#64748b' }}>Softmax Output</span>
                      </div>
                      {Object.entries(result.probabilities).map(([cls, prob]) => {
                        const isTop = cls === result.predicted_class;
                        const pct = (prob * 100).toFixed(1);
                        const meta = CLASS_META[cls] || { icon: '🏷️', label: cls };
                        const numPct = Number(pct);
                        return (
                          <div key={cls} className="prob-row">
                            <div className={`prob-meta ${isTop ? 'highlight' : ''}`}>
                              <span>{meta.icon} {meta.label}</span>
                              <span>{pct}%</span>
                            </div>
                            <div className="prob-bar-bg">
                              <div
                                className={`prob-bar-fill ${isTop ? 'active' : ''}`}
                                style={{ width: numPct > 0 ? `${Math.max(numPct, 0.8)}%` : '0%' }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}

                  {/* Image Metadata */}
                  <div className="result-card">
                    <div className="result-row">
                      <span className="result-label">Tên tệp:</span>
                      <span className="result-val">{result.filename}</span>
                    </div>
                    <div className="result-row">
                      <span className="result-label">Kích thước ảnh:</span>
                      <span className="result-val">
                        {result.image_dimensions?.width} × {result.image_dimensions?.height} px ({Math.round(result.image_size_bytes / 1024)} KB)
                      </span>
                    </div>
                    <div className="result-row">
                      <span className="result-label">Định dạng ảnh:</span>
                      <span className="result-val">{result.image_format} ({result.content_type})</span>
                    </div>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
