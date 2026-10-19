import React, { useState, useEffect, useRef } from 'react';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://127.0.0.1:8000';

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
      setError('Tệp được chọn không phải là hình ảnh hợp lệ (vui lòng chọn file .jpg, .png...).');
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
          <span>Phase 4: Minimal Skeleton</span>
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
                Backend: {backendHealth.online ? 'Online' : 'Offline'}
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
                    <span>Đang gửi phân tích...</span>
                  </>
                ) : (
                  <span>Gửi Phân Tích (POST /api/predict)</span>
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
            <span>Kết Quả Phân Tích</span>
            {result?.mock && <span className="mock-badge">Mock Response</span>}
          </div>

          {!result && !isLoading && (
            <div className="results-empty">
              <div style={{ fontSize: '40px', marginBottom: '12px' }}>♻️</div>
              <p>Chưa có kết quả nhận diện.</p>
              <p style={{ fontSize: '13px', marginTop: '6px' }}>
                Hãy chọn một ảnh rác thải bên trái và bấm <strong>"Gửi Phân Tích"</strong> để kiểm tra kết nối API.
              </p>
            </div>
          )}

          {isLoading && (
            <div className="results-empty">
              <div className="spinner" style={{ borderColor: 'rgba(5, 150, 105, 0.3)', borderTopColor: '#059669', width: '32px', height: '32px', margin: '0 auto 16px' }}></div>
              <p>Đang gửi ảnh tới máy chủ backend và xác thực định dạng...</p>
            </div>
          )}

          {result && (
            <div>
              <div className="alert alert-warning">
                <span className="alert-icon">ℹ️</span>
                <div>
                  <strong>Thông báo:</strong> {result.message}
                </div>
              </div>

              <div className="result-card">
                <div className="result-row">
                  <span className="result-label">Trạng thái API:</span>
                  <span className="result-val" style={{ color: '#059669' }}>{result.status}</span>
                </div>
                <div className="result-row">
                  <span className="result-label">Tên tệp nhận diện:</span>
                  <span className="result-val">{result.filename}</span>
                </div>
                <div className="result-row">
                  <span className="result-label">Định dạng xác thực:</span>
                  <span className="result-val">{result.image_format} ({result.content_type})</span>
                </div>
                <div className="result-row">
                  <span className="result-label">Kích thước ảnh:</span>
                  <span className="result-val">
                    {result.image_dimensions?.width} x {result.image_dimensions?.height} px ({Math.round(result.image_size_bytes / 1024)} KB)
                  </span>
                </div>
              </div>

              <div style={{ marginTop: '16px' }}>
                <span style={{ fontSize: '13px', fontWeight: '600', color: '#475569' }}>
                  Các nhóm rác dự kiến phân loại trong tương lai:
                </span>
                <div className="categories-pills">
                  {result.supported_categories?.map((cat) => (
                    <span key={cat} className="cat-pill">🏷️ {cat}</span>
                  ))}
                </div>
              </div>

              <div className="notice-banner">
                💡 <strong>Giai đoạn Khung Tối Thiểu (Minimal Skeleton):</strong> Endpoint phản hồi xác nhận luồng upload multipart, kiểm tra kích thước và tính toàn vẹn của ảnh thành công. Mô hình AI thực tế sẽ được tích hợp trong giai đoạn sau.
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
