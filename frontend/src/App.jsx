import React, { useState, useEffect, useRef } from 'react';

const getBackendUrl = () => {
  if (import.meta.env.VITE_BACKEND_URL) return import.meta.env.VITE_BACKEND_URL;
  return '';
};

const BACKEND_URL = getBackendUrl();

const CLASS_META = {
  plastic: { icon: '🧴', label: 'Nhựa', color: '#eab308' },
  paper: { icon: '📦', label: 'Giấy / Carton', color: '#3b82f6' },
  metal: { icon: '🥫', label: 'Kim loại', color: '#ef4444' },
  glass: { icon: '🍾', label: 'Thủy tinh', color: '#06b6d4' },
  organic: { icon: '🍎', label: 'Hữu cơ', color: '#22c55e' },
  other: { icon: '🗑️', label: 'Khác / Hỗn hợp', color: '#64748b' },
  unsupported_non_waste: { icon: '⚠️', label: 'Không phải rác / Ngoại lai', color: '#dc2626' },
};

function formatDate(isoString) {
  if (!isoString) return '';
  try {
    const d = new Date(isoString);
    return d.toLocaleString('vi-VN', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      day: '2-digit',
      month: '2-digit',
      year: 'numeric'
    });
  } catch {
    return isoString;
  }
}

export default function App() {
  const [activeTab, setActiveTab] = useState('predict'); // 'predict' | 'history' | 'analytics' | 'metrics' | 'centers'
  const [darkMode, setDarkMode] = useState(() => {
    return localStorage.getItem('theme') === 'dark' ||
      window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', darkMode ? 'dark' : 'light');
    localStorage.setItem('theme', darkMode ? 'dark' : 'light');
  }, [darkMode]);

  // Global states
  const [backendHealth, setBackendHealth] = useState(null);
  const [notification, setNotification] = useState(null);

  // Input Mode
  const [inputMode, setInputMode] = useState('upload'); // 'upload' | 'webcam' | 'url' | 'batch'
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);
  const mobileCameraInputRef = useRef(null);

  // Grad-CAM View: 'overlay' | 'heatmap' | 'original'
  const [xaiView, setXaiView] = useState('overlay');

  // User Feedback states
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);
  const [feedbackStatusChoice, setFeedbackStatusChoice] = useState(null); // 'correct' | 'incorrect' | 'uncertain'
  const [showFeedbackForm, setShowFeedbackForm] = useState(false);
  const [correctionClass, setCorrectionClass] = useState('other');
  const [correctionNote, setCorrectionNote] = useState('');
  const [isSubmittingFeedback, setIsSubmittingFeedback] = useState(false);

  // Webcam states
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  // URL prediction states
  const [imageUrlInput, setImageUrlInput] = useState('');

  // Batch prediction states
  const [batchFiles, setBatchFiles] = useState([]);
  const [batchResults, setBatchResults] = useState(null);
  const [isBatchLoading, setIsBatchLoading] = useState(false);
  const batchInputRef = useRef(null);

  // History states
  const [historyList, setHistoryList] = useState([]);
  const [historyTotal, setHistoryTotal] = useState(0);
  const [historyPage, setHistoryPage] = useState(1);
  const [historyPageSize] = useState(8);
  const [historyTotalPages, setHistoryTotalPages] = useState(1);
  const [isHistoryLoading, setIsHistoryLoading] = useState(false);
  const [filterClass, setFilterClass] = useState('');
  const [filterStartDate, setFilterStartDate] = useState('');
  const [filterEndDate, setFilterEndDate] = useState('');
  const [historyStats, setHistoryStats] = useState(null);

  // Analytics states (Dashboard)
  const [analyticsData, setAnalyticsData] = useState(null);
  const [isAnalyticsLoading, setIsAnalyticsLoading] = useState(false);
  const [timelineGranularity, setTimelineGranularity] = useState('daily'); // 'daily' | 'weekly'

  // Model Metrics states (Tab 4)
  const [modelMetrics, setModelMetrics] = useState(null);

  // Recycling Centers states (Tab 5)
  const [recyclingCenters, setRecyclingCenters] = useState([]);
  const [filterCity, setFilterCity] = useState('');
  const [filterMaterial, setFilterMaterial] = useState('');
  // Multi-Object Detection States (Tab: 'detect')
  const [detectFile, setDetectFile] = useState(null);
  const [detectPreview, setDetectPreview] = useState(null);
  const [isDetectLoading, setIsDetectLoading] = useState(false);
  const [detectResult, setDetectResult] = useState(null);
  const [detectError, setDetectError] = useState(null);
  const [detectViewMode, setDetectViewMode] = useState('annotated'); // 'annotated' | 'original'
  const [detectInputMode, setDetectInputMode] = useState('upload'); // 'upload' | 'url'
  const [detectUrlInput, setDetectUrlInput] = useState('');
  const [detectFilterCategory, setDetectFilterCategory] = useState(null);
  const detectFileInputRef = useRef(null);


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
    fetchStats();
  }, []);

  const showToast = (msg) => {
    setNotification(msg);
    setTimeout(() => setNotification(null), 3500);
  };

  const fetchStats = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/history/stats`);
      if (res.ok) {
        const data = await res.json();
        setHistoryStats(data);
      }
    } catch (err) {
      console.error('Lỗi tải thống kê:', err);
    }
  };

  const fetchHistory = async (page = 1, cls = filterClass, start = filterStartDate, end = filterEndDate) => {
    setIsHistoryLoading(true);
    try {
      const params = new URLSearchParams({
        page: String(page),
        page_size: String(historyPageSize)
      });
      if (cls) params.append('class_name', cls);
      if (start) params.append('start_date', start);
      if (end) params.append('end_date', end);

      const res = await fetch(`${BACKEND_URL}/api/history?${params.toString()}`);
      if (!res.ok) throw new Error('Không thể tải danh sách lịch sử.');
      const data = await res.json();
      setHistoryList(data.items || []);
      setHistoryTotal(data.total || 0);
      setHistoryPage(data.page || 1);
      setHistoryTotalPages(data.total_pages || 1);
    } catch (err) {
      console.error(err);
    } finally {
      setIsHistoryLoading(false);
    }
  };

  const fetchAnalytics = async () => {
    setIsAnalyticsLoading(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/analytics/dashboard`);
      if (!res.ok) throw new Error('Lỗi tải dữ liệu dashboard.');
      const data = await res.json();
      setAnalyticsData(data);
    } catch (err) {
      console.error(err);
    } finally {
      setIsAnalyticsLoading(false);
    }
  };

  const fetchModelMetrics = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/model/metrics`);
      if (res.ok) {
        const data = await res.json();
        setModelMetrics(data);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const fetchRecyclingCenters = async (city = filterCity, mat = filterMaterial) => {
    try {
      const params = new URLSearchParams();
      if (city) params.append('city', city);
      if (mat) params.append('material', mat);
      const res = await fetch(`${BACKEND_URL}/api/recycling-centers?${params.toString()}`);
      if (res.ok) {
        const data = await res.json();
        setRecyclingCenters(data);
      }
    } catch (err) {
      console.error(err);
    }
  };

    const handleDetectFileSelect = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      setDetectError('Vui lòng chọn tệp hình ảnh hợp lệ (JPEG, PNG, WEBP).');
      return;
    }
    setDetectFile(file);
    setDetectError(null);
    setDetectResult(null);
    setDetectFilterCategory(null);
    const reader = new FileReader();
    reader.onload = () => setDetectPreview(reader.result);
    reader.readAsDataURL(file);
  };

  const handleRunDetect = async () => {
    if (!detectFile) {
      setDetectError('Vui lòng chọn ảnh trước khi phát hiện.');
      return;
    }
    setIsDetectLoading(true);
    setDetectError(null);
    setDetectFilterCategory(null);
    const formData = new FormData();
    formData.append('file', detectFile);

    try {
      const res = await fetch(`${BACKEND_URL}/api/detect`, {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Lỗi máy chủ (${res.status})`);
      }
      const data = await res.json();
      setDetectResult(data);
      setDetectViewMode(data.annotated_image ? 'annotated' : 'original');
      showToast(`Đã quét xong: Phát hiện ${data.total_objects} vật thể rác.`);
    } catch (err) {
      setDetectError(err.message || 'Không thể thực hiện phát hiện đa vật thể.');
    } finally {
      setIsDetectLoading(false);
    }
  };

  const handleRunDetectUrl = async () => {
    if (!detectUrlInput.trim()) {
      setDetectError('Vui lòng nhập đường dẫn URL ảnh.');
      return;
    }
    setIsDetectLoading(true);
    setDetectError(null);
    setDetectResult(null);
    setDetectFilterCategory(null);
    setDetectPreview(detectUrlInput.trim());

    try {
      const res = await fetch(`${BACKEND_URL}/api/detect/url`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_url: detectUrlInput.trim() }),
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Lỗi máy chủ (${res.status})`);
      }
      const data = await res.json();
      setDetectResult(data);
      setDetectViewMode(data.annotated_image ? 'annotated' : 'original');
      showToast(`Đã quét xong: Phát hiện ${data.total_objects} vật thể rác.`);
    } catch (err) {
      setDetectError(err.message || 'Không thể tải ảnh từ URL để phát hiện.');
    } finally {
      setIsDetectLoading(false);
    }
  };

  const handleResetDetect = () => {
    setDetectFile(null);
    setDetectPreview(null);
    setDetectResult(null);
    setDetectError(null);
    setDetectUrlInput('');
    setDetectFilterCategory(null);
    if (detectFileInputRef.current) {
      detectFileInputRef.current.value = '';
    }
  };

  const handleTabChange = (tab) => {
    setActiveTab(tab);
    if (tab === 'history') {
      fetchHistory(1);
      fetchStats();
    } else if (tab === 'analytics') {
      fetchAnalytics();
    } else if (tab === 'metrics') {
      fetchModelMetrics();
    } else if (tab === 'centers') {
      fetchRecyclingCenters();
    }
  };

  // Webcam Controls
  const startCamera = async () => {
    setCameraError(null);
    setIsCameraActive(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } }
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
    } catch (err) {
      setCameraError('Không thể mở camera hoặc quyền camera bị từ chối.');
      setIsCameraActive(false);
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setIsCameraActive(false);
  };

  const capturePhoto = () => {
    if (!videoRef.current) return;
    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth || 640;
    canvas.height = videoRef.current.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], `camera_${Date.now()}.jpg`, { type: 'image/jpeg' });
        handleFileChange(file);
        stopCamera();
      }
    }, 'image/jpeg', 0.9);
  };

  const handleFileChange = (file) => {
    if (!file) return;
    setError(null);
    setResult(null);
    setFeedbackSubmitted(false);
    setShowFeedbackForm(false);

    if (!file.type.startsWith('image/')) {
      setError('Tệp được chọn không phải là hình ảnh hợp lệ.');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setError('Kích thước ảnh vượt quá giới hạn 10 MB.');
      return;
    }

    setSelectedFile(file);
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
  };

  const handleSubmitPrediction = async (e) => {
    if (e) e.preventDefault();
    if (!selectedFile) {
      setError('Vui lòng chọn hoặc chụp ảnh trước khi gửi nhận diện.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);
    setFeedbackSubmitted(false);
    setShowFeedbackForm(false);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res = await fetch(`${BACKEND_URL}/api/predict?include_gradcam=true`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `Lỗi máy chủ (${res.status})`);
      setResult(data);
      fetchStats();
    } catch (err) {
      setError(err.message || 'Không thể kết nối đến máy chủ nhận diện.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmitUrlPrediction = async (e) => {
    e.preventDefault();
    if (!imageUrlInput.trim()) {
      setError('Vui lòng nhập đường link URL hình ảnh.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);
    setFeedbackSubmitted(false);

    try {
      const res = await fetch(`${BACKEND_URL}/api/predict/url?include_gradcam=true`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_url: imageUrlInput.trim() })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `Lỗi xử lý URL (${res.status})`);
      setResult(data);
      setPreviewUrl(imageUrlInput.trim());
      fetchStats();
    } catch (err) {
      setError(err.message || 'Không thể tải hoặc xử lý ảnh từ URL cung cấp.');
    } finally {
      setIsLoading(false);
    }
  };

  // Batch prediction handler
  const handleBatchFileSelect = (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 10) {
      alert('Tối đa 10 ảnh trong 1 lần phân loại hàng loạt.');
      return;
    }
    setBatchFiles(files);
    setBatchResults(null);
  };

  const handleBatchSubmit = async () => {
    if (batchFiles.length === 0) return;
    setIsBatchLoading(true);
    const formData = new FormData();
    batchFiles.forEach((file) => formData.append('files', file));

    try {
      const res = await fetch(`${BACKEND_URL}/api/predict/batch`, {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Lỗi khi xử lý hàng loạt.');
      setBatchResults(data);
      showToast(`Đã hoàn tất phân loại hàng loạt: ${data.successful}/${data.total} thành công.`);
      fetchStats();
    } catch (err) {
      alert('Lỗi: ' + err.message);
    } finally {
      setIsBatchLoading(false);
    }
  };

  // Feedback Submission: 'correct' | 'incorrect' | 'uncertain'
  const handleSendFeedback = async (statusChoice, correctedClass = null, note = null) => {
    if (!result || !result.record_id) return;
    setIsSubmittingFeedback(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/history/${result.record_id}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          feedback_status: statusChoice,
          is_correct: statusChoice === 'correct',
          correct_class: correctedClass,
          note: note
        })
      });
      if (res.ok) {
        setFeedbackSubmitted(true);
        setShowFeedbackForm(false);
        if (statusChoice === 'correct') {
          showToast('✅ Cảm ơn bạn đã xác nhận AI dự đoán chuẩn xác!');
        } else if (statusChoice === 'uncertain') {
          showToast('❓ Đã ghi nhận ca không chắc chắn để nhóm phát triển thẩm định!');
        } else {
          showToast('📝 Đã lưu ca báo sai kèm nhãn thực tế để cải thiện mô hình!');
        }
        fetchStats();
      }
    } catch (err) {
      console.error('Feedback error:', err);
    } finally {
      setIsSubmittingFeedback(false);
    }
  };

  const handleExportCSV = () => {
    const params = new URLSearchParams();
    if (filterClass) params.append('class_name', filterClass);
    if (filterStartDate) params.append('start_date', filterStartDate);
    if (filterEndDate) params.append('end_date', filterEndDate);
    window.open(`${BACKEND_URL}/api/history/export?${params.toString()}`, '_blank');
  };

  const handleExportErrorCases = () => {
    window.open(`${BACKEND_URL}/api/analytics/errors/export`, '_blank');
  };

  return (
    <div className="app-container">
      {/* Header with Dark Mode Toggle */}
      <div className="header-actions">
        <button
          type="button"
          className="theme-toggle-btn"
          onClick={() => setDarkMode(!darkMode)}
          title="Chuyển đổi giao diện Sáng / Tối"
        >
          <span>{darkMode ? '☀️ Giao diện Sáng' : '🌙 Giao diện Tối'}</span>
        </button>
      </div>

      <header className="app-header">
        <div className="brand-badge">
          <span>🌿 AI Capstone Project</span>
          <span>•</span>
          <span>Phân Loại Rác Thông Minh & Hướng Dẫn Tái Chế</span>
        </div>
        <h1>Hệ Thống Nhận Diện Rác Thải & Phân Tích Tái Chế</h1>
        <p>Ứng dụng mạng nơ-ron MobileNetV3-Large, phân tích thuộc tính bắt cháy, mức độ tái chế và giải thích AI (Grad-CAM).</p>
      </header>

      {/* Navigation Tabs (5 Tabs) */}
      <div className="tab-nav-wrapper">
        <nav className="tab-nav">
          <button
            type="button"
            className={`tab-btn ${activeTab === 'predict' ? 'active' : ''}`}
            onClick={() => handleTabChange('predict')}
          >
            <span className="tab-icon">🔍</span>
            <span>Phân Loại Rác</span>
          </button>
          <button
            type="button"
            className={`tab-btn ${activeTab === 'detect' ? 'active' : ''}`}
            onClick={() => handleTabChange('detect')}
          >
            <span className="tab-icon">🎯</span>
            <span>Phát Hiện Đa Vật Thể</span>
          </button>

          <button
            type="button"
            className={`tab-btn ${activeTab === 'history' ? 'active' : ''}`}
            onClick={() => handleTabChange('history')}
          >
            <span className="tab-icon">📜</span>
            <span>Lịch Sử</span>
            {historyStats && historyStats.total_predictions > 0 && (
              <span className="tab-counter">{historyStats.total_predictions}</span>
            )}
          </button>
          <button
            type="button"
            className={`tab-btn ${activeTab === 'analytics' ? 'active' : ''}`}
            onClick={() => handleTabChange('analytics')}
          >
            <span className="tab-icon">📊</span>
            <span>Bảng Thống Kê (Dashboard)</span>
          </button>
          <button
            type="button"
            className={`tab-btn ${activeTab === 'metrics' ? 'active' : ''}`}
            onClick={() => handleTabChange('metrics')}
          >
            <span className="tab-icon">📈</span>
            <span>Hiệu Suất AI (XAI)</span>
          </button>
          <button
            type="button"
            className={`tab-btn ${activeTab === 'centers' ? 'active' : ''}`}
            onClick={() => handleTabChange('centers')}
          >
            <span className="tab-icon">🗺️</span>
            <span>Điểm Thu Gom</span>
          </button>
        </nav>
      </div>

      {notification && (
        <div className="toast-notification">
          <span>{notification}</span>
        </div>
      )}

      {/* ==================================================================== */}
      {/* TAB 1: PREDICTION VIEW                                              */}
      {/* ==================================================================== */}
      {activeTab === 'predict' && (
        <div className="main-grid">
          {/* Left Column: Input Selection */}
          <div className="card">
            <div className="card-title">
              <span>Đầu Vào Hình Ảnh</span>
              {backendHealth && (
                <span className={`backend-status-pill ${backendHealth.online ? 'backend-online' : 'backend-offline'}`}>
                  <span className="status-dot"></span>
                  {backendHealth.online ? (backendHealth.data?.ai_model_loaded ? 'Model Ready' : 'Online') : 'Offline'}
                </span>
              )}
            </div>

            {/* Input Mode Switcher */}
            <div className="input-mode-tabs">
              <button
                type="button"
                className={`mode-tab-btn ${inputMode === 'upload' ? 'active' : ''}`}
                onClick={() => { setInputMode('upload'); stopCamera(); }}
              >
                📁 Tải Tệp
              </button>
              <button
                type="button"
                className={`mode-tab-btn ${inputMode === 'webcam' ? 'active' : ''}`}
                onClick={() => { setInputMode('webcam'); startCamera(); }}
              >
                📸 Camera
              </button>
              <button
                type="button"
                className={`mode-tab-btn ${inputMode === 'url' ? 'active' : ''}`}
                onClick={() => { setInputMode('url'); stopCamera(); }}
              >
                🌐 Link Ảnh
              </button>
              <button
                type="button"
                className={`mode-tab-btn ${inputMode === 'batch' ? 'active' : ''}`}
                onClick={() => { setInputMode('batch'); stopCamera(); }}
              >
                🔁 Hàng Loạt
              </button>
            </div>

            {/* MODE 1: FILE UPLOAD */}
            {inputMode === 'upload' && (
              <form onSubmit={handleSubmitPrediction}>
                <div
                  className={`dropzone ${isDragging ? 'active' : ''}`}
                  onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                  onDragLeave={() => setIsDragging(false)}
                  onDrop={(e) => {
                    e.preventDefault();
                    setIsDragging(false);
                    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                      handleFileChange(e.dataTransfer.files[0]);
                    }
                  }}
                  onClick={() => fileInputRef.current && fileInputRef.current.click()}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    className="hidden-file-input"
                    onChange={(e) => e.target.files && handleFileChange(e.target.files[0])}
                  />
                  <div className="dropzone-icon">📷</div>
                  <div className="dropzone-text">
                    {selectedFile ? 'Nhấp hoặc kéo thả để đổi ảnh khác' : 'Chọn hoặc kéo thả ảnh vào đây'}
                  </div>
                  <div className="dropzone-hint">Hỗ trợ JPEG, PNG, WebP (Tối đa 10 MB)</div>
                </div>

                {/* Nút chụp camera native dành cho điện thoại */}
                <input
                  ref={mobileCameraInputRef}
                  type="file"
                  accept="image/*"
                  capture="environment"
                  className="hidden-file-input"
                  onChange={(e) => e.target.files && handleFileChange(e.target.files[0])}
                />
                <div style={{ marginTop: '10px' }}>
                  <button
                    type="button"
                    className="btn-secondary"
                    style={{
                      width: '100%',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: '8px',
                      padding: '10px 14px',
                      fontWeight: 600
                    }}
                    onClick={() => mobileCameraInputRef.current && mobileCameraInputRef.current.click()}
                  >
                    📱 Chụp Bằng Camera Điện Thoại
                  </button>
                </div>

                {previewUrl && (
                  <div className="preview-container">
                    <img src={previewUrl} alt="Xem trước" className="preview-image" />
                    <div className="preview-info">
                      <span className="file-name">{selectedFile?.name}</span>
                      <span className="file-size">{(selectedFile?.size / 1024).toFixed(1)} KB</span>
                    </div>
                  </div>
                )}

                {error && (
                  <div className="alert alert-error" style={{ marginTop: '12px' }}>
                    <span className="alert-icon">⚠️</span>
                    <div>{error}</div>
                  </div>
                )}

                <div className="action-group">
                  <button
                    type="submit"
                    className="btn-primary"
                    disabled={!selectedFile || isLoading}
                  >
                    {isLoading ? 'Đang phân tích AI & Thuộc tính...' : 'Phân Tích AI & Đề Xuất Xử Lý'}
                  </button>
                  {selectedFile && (
                    <button
                      type="button"
                      className="btn-secondary"
                      onClick={() => { setSelectedFile(null); setPreviewUrl(null); setResult(null); }}
                    >
                      Xóa
                    </button>
                  )}
                </div>
              </form>
            )}

            {/* MODE 2: WEBCAM */}
            {inputMode === 'webcam' && (
              <div className="webcam-container">
                {cameraError ? (
                  <div className="alert alert-error">{cameraError}</div>
                ) : (
                  <>
                    <video ref={videoRef} autoPlay playsInline muted className="webcam-video" />
                    <div className="webcam-controls">
                      <button type="button" className="btn-snap" onClick={capturePhoto}>
                        📸 Chụp Ảnh Rác Thải
                      </button>
                      <button type="button" className="btn-cancel-cam" onClick={stopCamera}>
                        Dừng Camera
                      </button>
                    </div>
                  </>
                )}
                {selectedFile && previewUrl && (
                  <div className="preview-container" style={{ width: '100%', marginTop: '10px' }}>
                    <img src={previewUrl} alt="Ảnh chụp" className="preview-image" />
                    <button
                      type="button"
                      className="btn-primary"
                      style={{ marginTop: '10px' }}
                      onClick={handleSubmitPrediction}
                      disabled={isLoading}
                    >
                      {isLoading ? 'Đang phân tích...' : 'Gửi Nhận Diện Ảnh Vừa Chụp'}
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* MODE 3: URL INPUT */}
            {inputMode === 'url' && (
              <form onSubmit={handleSubmitUrlPrediction} className="url-input-container">
                <input
                  type="url"
                  className="url-input-field"
                  placeholder="Dán link ảnh tại đây (vd: https://images.unsplash.com/...)"
                  value={imageUrlInput}
                  onChange={(e) => setImageUrlInput(e.target.value)}
                />
                <button
                  type="submit"
                  className="btn-primary"
                  disabled={!imageUrlInput.trim() || isLoading}
                >
                  {isLoading ? 'Đang tải và nhận diện...' : 'Tải Ảnh & Phân Loại AI'}
                </button>
                {previewUrl && (
                  <div className="preview-container" style={{ marginTop: '10px' }}>
                    <img src={previewUrl} alt="Từ URL" className="preview-image" />
                  </div>
                )}
                {error && (
                  <div className="alert alert-error">
                    <span>⚠️</span>
                    <div>{error}</div>
                  </div>
                )}
              </form>
            )}

            {/* MODE 4: BATCH PROCESSING */}
            {inputMode === 'batch' && (
              <div>
                <div
                  className="dropzone"
                  onClick={() => batchInputRef.current && batchInputRef.current.click()}
                >
                  <input
                    ref={batchInputRef}
                    type="file"
                    multiple
                    accept="image/*"
                    className="hidden-file-input"
                    onChange={handleBatchFileSelect}
                  />
                  <div className="dropzone-icon">🗂️</div>
                  <div className="dropzone-text">Chọn nhiều ảnh cùng lúc (Tối đa 10 ảnh)</div>
                  <div className="dropzone-hint">Phù hợp khi cần phân loại giỏ rác nhiều món</div>
                </div>

                {batchFiles.length > 0 && (
                  <div style={{ marginTop: '12px' }}>
                    <div style={{ fontSize: '13px', fontWeight: 600, marginBottom: '6px' }}>
                      Đã chọn {batchFiles.length} tệp:
                    </div>
                    <div className="batch-preview-list">
                      {batchFiles.map((f, i) => (
                        <div key={i} className="batch-preview-item">
                          <img src={URL.createObjectURL(f)} alt="" className="batch-thumb" />
                          <div className="batch-thumb-info">{f.name}</div>
                        </div>
                      ))}
                    </div>
                    <button
                      type="button"
                      className="btn-primary"
                      style={{ marginTop: '12px' }}
                      onClick={handleBatchSubmit}
                      disabled={isBatchLoading}
                    >
                      {isBatchLoading ? 'Đang xử lý hàng loạt...' : `Phân Loại Cả ${batchFiles.length} Ảnh`}
                    </button>
                  </div>
                )}

                {batchResults && (
                  <div className="batch-results-list">
                    <div className="alert alert-success">
                      Hoàn thành: {batchResults.successful}/{batchResults.total} ảnh phân loại thành công!
                    </div>
                    {batchResults.items.map((item, idx) => (
                      <div key={idx} className="batch-result-card">
                        <div>
                          <strong>{item.filename}</strong>
                          {item.success && item.result ? (
                            <div style={{ fontSize: '12.5px', color: '#059669', marginTop: '2px' }}>
                              → {CLASS_META[item.result.predicted_class]?.icon} {item.result.label_vi} ({(item.result.confidence * 100).toFixed(1)}%)
                            </div>
                          ) : (
                            <div style={{ fontSize: '12px', color: '#ef4444' }}>Lỗi: {item.error}</div>
                          )}
                        </div>
                        {item.success && item.result && (
                          <span className="bin-tag" style={{ fontSize: '11px', padding: '3px 8px' }}>
                            {item.result.recommendation.bin}
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Right Column: Prediction Results & Analysis */}
          <div className="card">
            <div className="card-title">
              <span>Kết Quả Nhận Diện AI</span>
              {result && <span className="real-badge">MobileNetV3-Large</span>}
            </div>

            {!result && !isLoading && (
              <div className="results-empty">
                <div style={{ fontSize: '42px', marginBottom: '12px' }}>♻️</div>
                <p>Chưa có kết quả phân loại.</p>
                <p style={{ fontSize: '13px', marginTop: '6px', color: '#64748b' }}>
                  Hãy chọn tải tệp, mở camera hoặc dán URL bên trái để bắt đầu.
                </p>
              </div>
            )}

            {isLoading && (
              <div className="results-empty">
                <div className="spinner" style={{ width: '36px', height: '36px', margin: '0 auto 16px' }}></div>
                <p>Đang đưa ảnh qua mạng nơ-ron MobileNetV3, trích xuất thuộc tính bắt cháy & tái chế...</p>
              </div>
            )}

            {result && (
              <div>
                {/* Non-Waste / OOD Warning Alert */}
                {result.is_waste_detected === false ? (
                  <div style={{ padding: "20px", borderRadius: "12px", background: "var(--card-bg, #fff)", border: "2px solid #ef4444", margin: "8px 0" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "14px", marginBottom: "14px" }}>
                      <div style={{ fontSize: "40px" }}>
                        {result.detection_status === "person_detected" ? "🧍‍♂️" :
                         result.detection_status === "pet_detected" ? "🐕" :
                         result.detection_status === "furniture_detected" ? "🛋️" :
                         result.detection_status === "scenery_detected" ? "🏞️" : "⚠️"}
                      </div>
                      <div>
                        <div style={{ fontSize: "18px", fontWeight: 700, color: "#dc2626" }}>
                          Phát Hiện Đối Tượng Ngoại Lai (Không Phải Rác Thải)
                        </div>
                        <div style={{ fontSize: "13px", color: "#64748b" }}>
                          Trạng thái kiểm định: <span className="uncertain-tag" style={{ marginLeft: "4px" }}>{result.detection_status}</span>
                        </div>
                      </div>
                    </div>

                    <div className="alert alert-danger" style={{ marginBottom: "12px", lineHeight: "1.6" }}>
                      <strong>Lý do từ chối:</strong> {result.rejection_reason || "Hình ảnh chứa người, động vật hoặc đồ vật gia đình không thuộc danh mục rác thải."}
                    </div>

                    <div className="alert alert-warning" style={{ marginBottom: "16px", lineHeight: "1.6" }}>
                      💡 <strong>Khuyến nghị chụp lại:</strong> {result.suggested_action || "Vui lòng chỉ chụp cận cảnh một vật thể rác thải sinh hoạt (chai lọ, vỏ lon, túi giấy, thức ăn thừa...)."} 
                    </div>

                    <div style={{ padding: "14px", background: "var(--bg-secondary, #f8fafc)", borderRadius: "8px", fontSize: "13px", color: "#475569", lineHeight: "1.6" }}>
                      <div style={{ fontWeight: 600, color: "#334155", marginBottom: "6px" }}>📌 Quy tắc an toàn & giải thích kỹ thuật:</div>
                      <div>• Để bảo đảm độ tin cậy và không nhận diện sai người, thú cưng hoặc nội thất thành rác tái chế, hệ thống kích hoạt cơ chế <strong>Semantic Gatekeeper</strong> để từ chối các phân loại không có cơ sở.</div>
                      <div>• Hãy đặt riêng vật thể rác lên mặt phẳng (bàn hoặc sàn) và chụp trực diện, chiếm ít nhất 50% khung hình.</div>
                    </div>

                    <button
                      type="button"
                      className="btn-primary"
                      style={{ marginTop: "16px", width: "100%" }}
                      onClick={() => {
                        setResult(null);
                        setSelectedFile(null);
                        setPreviewUrl(null);
                        setImageUrlInput("");
                      }}
                    >
                      🔄 Chụp / Chọn Lại Ảnh Rác Khác
                    </button>
                  </div>
                ) : (
                  <div>
                    {/* Hero Prediction Header */}
                <div className="prediction-hero">
                  <div className="prediction-hero-header">
                    <div className="prediction-class-title">
                      <span>{CLASS_META[result.predicted_class]?.icon || '🏷️'}</span>
                      <span>{result.label_vi}</span>
                    </div>
                    <div className="hero-badges-group">
                      {result.is_uncertain && (
                        <span className="uncertain-tag">⚠️ Cần kiểm tra lại</span>
                      )}
                      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '3px' }}>
                        <span className={`confidence-badge ${result.is_uncertain ? 'uncertain' : ''}`} title="Độ tin cậy thô từ hàm Softmax">
                          {(result.confidence * 100).toFixed(1)}% (Raw)
                        </span>
                        {result.calibrated_confidence !== undefined && result.calibrated_confidence !== null && (
                          <span 
                            style={{ 
                              fontSize: '11px', 
                              color: '#065f46', 
                              fontWeight: 600, 
                              background: '#d1fae5', 
                              padding: '2px 7px', 
                              borderRadius: '4px', 
                              border: '1px solid #6ee7b7' 
                            }} 
                            title="Độ tin cậy sau hiệu chuẩn nhiệt độ (Temperature Scaling T=1.6366, giảm sai số ECE còn 1.19%)"
                          >
                            🎯 {(result.calibrated_confidence * 100).toFixed(1)}% (Hiệu chuẩn)
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                  {result.recommendation?.is_override_applied && (
                    <div style={{ marginTop: '10px', padding: '10px 14px', background: '#fef2f2', border: '1px solid #fecaca', borderRadius: '8px', fontSize: '12.5px', color: '#991b1b', lineHeight: 1.5 }}>
                      <strong>🚨 Cơ chế an toàn kích hoạt:</strong> {result.recommendation.override_reason}
                    </div>
                  )}
                  {result.uncertainty_note && (
                    <div style={{ marginTop: '8px', padding: '8px 12px', background: '#fffbeb', border: '1px solid #fde68a', borderRadius: '6px', fontSize: '12px', color: '#92400e' }}>
                      <strong>ℹ️ Lưu ý phân loại:</strong> {result.uncertainty_note}
                    </div>
                  )}
                  <div style={{ fontSize: '11px', color: '#047857', marginTop: '6px', display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: '4px' }}>
                    <span>* Phân tích phân bổ xác suất trên 6 nhóm danh mục rác tiêu chuẩn.</span>
                    {result.calibrated_confidence && <span>* Hiệu chuẩn ECE 1.19% (T*=1.6366).</span>}
                  </div>
                </div>

                {/* Advanced Combustibility & Recyclability Analysis */}
                {result.recommendation && (
                  <div className="analysis-grid">
                    {/* Combustibility Card */}
                    {result.recommendation.combustibility && (
                      <div className="analysis-card">
                        <div className="analysis-header">
                          <span className="analysis-label">🔥 Khả Năng Cháy:</span>
                          <span
                            className="analysis-badge"
                            style={{ backgroundColor: result.recommendation.combustibility.badge_color }}
                          >
                            {result.recommendation.combustibility.status}
                          </span>
                        </div>
                        <div className="analysis-desc">
                          {result.recommendation.combustibility.analysis}
                        </div>
                        {result.recommendation.combustibility.warning && (
                          <div className="analysis-warning">
                            ⚠️ {result.recommendation.combustibility.warning}
                          </div>
                        )}
                      </div>
                    )}

                    {/* Recyclability Card */}
                    {result.recommendation.recyclability && (
                      <div className="analysis-card">
                        <div className="analysis-header">
                          <span className="analysis-label">♻️ Mức Độ Tái Chế:</span>
                          <span
                            className="analysis-badge"
                            style={{ backgroundColor: result.recommendation.recyclability.badge_color }}
                          >
                            {result.recommendation.recyclability.level} ({result.recommendation.recyclability.difficulty_score})
                          </span>
                        </div>
                        <div className="analysis-desc">
                          {result.recommendation.recyclability.analysis}
                        </div>
                        {result.recommendation.recyclability.economic_value && (
                          <div style={{ fontSize: '11.5px', color: '#059669', fontWeight: 600 }}>
                            💰 Giá trị kinh tế: {result.recommendation.recyclability.economic_value}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}

                {/* 3-Step Disposal & Recycling Workflow */}
                {result.recommendation?.processing_steps && (
                  <div className="steps-container">
                    <div className="steps-title">📋 Quy Trình Xử Lý & Tái Chế Tiêu Chuẩn:</div>
                    {result.recommendation.processing_steps.map((st, i) => (
                      <div key={i} className="step-item">{st}</div>
                    ))}
                    {result.recommendation.environmental_impact && (
                      <div style={{ marginTop: '10px', paddingTop: '8px', borderTop: '1px dashed var(--border)', fontSize: '12px' }}>
                        <div>⏳ <strong>Thời gian phân hủy tự nhiên:</strong> {result.recommendation.environmental_impact.decomposition_time}</div>
                        <div style={{ color: '#059669', marginTop: '3px' }}>🌱 <strong>Lợi ích môi trường:</strong> {result.recommendation.environmental_impact.carbon_saving}</div>
                      </div>
                    )}
                  </div>
                )}

                
                {/* Special Destination (Hazardous / E-Waste) */}
                {result.recommendation?.special_destination && (
                  <div className="alert alert-danger" style={{ marginTop: '12px', background: '#fef2f2', border: '1px solid #f87171', color: '#991b1b' }}>
                    <strong>📍 Điểm thu gom đặc thù bắt buộc:</strong>
                    <div style={{ marginTop: '4px', fontSize: '13px' }}>{result.recommendation.special_destination}</div>
                  </div>
                )}

                {/* Preparation Requirements */}
                {result.recommendation?.preparation_requirements && result.recommendation.preparation_requirements.length > 0 && (
                  <div className="steps-container" style={{ marginTop: '12px' }}>
                    <div className="steps-title">🧤 Yêu Cầu Chuẩn Bị & An Toàn Trước Khi Bỏ:</div>
                    {result.recommendation.preparation_requirements.map((req, rIdx) => (
                      <div key={rIdx} className="step-item" style={{ fontSize: '12.5px' }}>✓ {req}</div>
                    ))}
                  </div>
                )}

                {/* Mixed Material Notes */}
                {result.recommendation?.mixed_material_notes && (
                  <div style={{ marginTop: '10px', fontSize: '12px', color: 'var(--text-muted)', fontStyle: 'italic', background: 'var(--surface)', padding: '8px 12px', borderRadius: '6px', border: '1px solid var(--border)' }}>
                    ℹ️ <strong>Lưu ý vật liệu hỗn hợp:</strong> {result.recommendation.mixed_material_notes}
                  </div>
                )}

                {/* Legal Jurisdiction & Source Attribution */}
                {result.recommendation?.jurisdiction && (
                  <div style={{ marginTop: '10px', display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: '6px', fontSize: '11px', color: 'var(--text-muted)', padding: '6px 10px', background: '#f8fafc', borderRadius: '6px', border: '1px solid var(--border)' }}>
                    <span>⚖️ <strong>Căn cứ:</strong> {result.recommendation.jurisdiction}</span>
                    {result.recommendation.source && (
                      <span>📖 <strong>Nguồn:</strong> {result.recommendation.source}</span>
                    )}
                  </div>
                )}

                {/* Explainable AI (Grad-CAM) Visualizer */}
                {result.gradcam_overlay && (
                  <div className="xai-card">
                    <div className="xai-header">
                      <div className="xai-title">
                        <span>🧠 Trực Quan Hóa AI (Grad-CAM Explainable AI)</span>
                      </div>
                      <div className="xai-view-tabs">
                        <button
                          type="button"
                          className={`xai-tab-btn ${xaiView === 'overlay' ? 'active' : ''}`}
                          onClick={() => setXaiView('overlay')}
                        >
                          Lớp Phủ Nhiệt
                        </button>
                        <button
                          type="button"
                          className={`xai-tab-btn ${xaiView === 'heatmap' ? 'active' : ''}`}
                          onClick={() => setXaiView('heatmap')}
                        >
                          Heatmap
                        </button>
                        <button
                          type="button"
                          className={`xai-tab-btn ${xaiView === 'original' ? 'active' : ''}`}
                          onClick={() => setXaiView('original')}
                        >
                          Ảnh Gốc
                        </button>
                      </div>
                    </div>

                    <div className="xai-image-wrapper">
                      {xaiView === 'overlay' && (
                        <img src={result.gradcam_overlay} alt="Grad-CAM Overlay" className="xai-display-img" />
                      )}
                      {xaiView === 'heatmap' && (
                        <img src={result.gradcam_heatmap} alt="Grad-CAM Heatmap" className="xai-display-img" />
                      )}
                      {xaiView === 'original' && (
                        <img src={previewUrl} alt="Ảnh gốc" className="xai-display-img" />
                      )}
                    </div>

                    <div className="xai-explainer">
                      💡 <strong>Giải thích vùng kích hoạt:</strong> Các vùng màu <strong>Đỏ - Vàng</strong> là nơi mô hình AI tập trung trích xuất đặc trưng để quyết định nhóm <strong>{result.label_vi}</strong>.
                    </div>
                  </div>
                )}

                {/* User Feedback & Error Reporting Loop */}
                <div className="feedback-box">
                  <div className="feedback-header">
                    <span className="feedback-title">🎯 Đánh giá kết quả nhận diện:</span>
                    {!feedbackSubmitted ? (
                      <div className="feedback-actions">
                        <button
                          type="button"
                          className="btn-feedback-thumb yes"
                          onClick={() => handleSendFeedback('correct')}
                          disabled={isSubmittingFeedback}
                          title="Xác nhận mô hình nhận diện chính xác"
                        >
                          👍 Chuẩn xác
                        </button>
                        <button
                          type="button"
                          className="btn-feedback-thumb no"
                          onClick={() => {
                            setFeedbackStatusChoice('incorrect');
                            setShowFeedbackForm(true);
                          }}
                          disabled={isSubmittingFeedback}
                          title="Báo kết quả sai để nhóm phát triển sửa đổi"
                        >
                          👎 Báo sai
                        </button>
                        <button
                          type="button"
                          className="btn-feedback-thumb"
                          style={{ borderColor: '#f59e0b', color: '#d97706' }}
                          onClick={() => {
                            setFeedbackStatusChoice('uncertain');
                            setShowFeedbackForm(true);
                          }}
                          disabled={isSubmittingFeedback}
                          title="Không chắc chắn về vật thể này"
                        >
                          ❓ Nghi ngờ
                        </button>
                      </div>
                    ) : (
                      <span style={{ fontSize: '12px', color: '#059669', fontWeight: 600 }}>
                        ✓ Đã ghi nhận phản hồi đóng góp
                      </span>
                    )}
                  </div>

                  {showFeedbackForm && !feedbackSubmitted && (
                    <div className="feedback-form-correction">
                      <div style={{ fontSize: '12px', color: '#64748b' }}>
                        {feedbackStatusChoice === 'incorrect' ? 'Nhãn thực tế của vật thể này là gì?' : 'Bạn muốn nhóm phát triển kiểm tra điều gì?'}
                      </div>
                      {feedbackStatusChoice === 'incorrect' && (
                        <select
                          className="filter-select"
                          value={correctionClass}
                          onChange={(e) => setCorrectionClass(e.target.value)}
                        >
                          {Object.entries(CLASS_META).map(([k, v]) => (
                            <option key={k} value={k}>{v.icon} {v.label}</option>
                          ))}
                        </select>
                      )}
                      <input
                        type="text"
                        className="url-input-field"
                        placeholder="Mô tả chi tiết (vd: chai thủy tinh đen bị nhầm thành nhựa, rác dính dầu mỡ...)"
                        value={correctionNote}
                        onChange={(e) => setCorrectionNote(e.target.value)}
                      />
                      <button
                        type="button"
                        className="btn-primary"
                        onClick={() => handleSendFeedback(feedbackStatusChoice, correctionClass, correctionNote)}
                        disabled={isSubmittingFeedback}
                      >
                        Gửi Báo Cáo Để Nhóm Cải Thiện Mô Hình
                      </button>
                    </div>
                  )}
                </div>

                {/* Top 3 Predictions */}
                {result.top_predictions && (
                  <div className="top3-card" style={{ marginTop: '14px' }}>
                    <div className="top3-title">
                      <span>Top 3 Khả Năng Cao Nhất</span>
                    </div>
                    <div className="top3-list">
                      {result.top_predictions.map((item, idx) => {
                        const meta = CLASS_META[item.class_name] || { icon: '🏷️', label: item.label_vi };
                        const pct = (item.probability * 100).toFixed(1);
                        return (
                          <div key={idx} className={`top3-item ${idx === 0 ? 'rank-1' : ''}`}>
                            <div className="top3-rank">#{idx + 1}</div>
                            <div className="top3-info">
                              <div className="top3-header">
                                <span>{meta.icon} {item.label_vi}</span>
                                <strong>{pct}%</strong>
                              </div>
                              <div className="top3-bar-bg">
                                <div
                                  className="top3-bar-fill"
                                  style={{
                                    width: `${Math.max(Number(pct), 1)}%`,
                                    backgroundColor: idx === 0 ? '#10b981' : '#94a3b8'
                                  }}
                                />
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

            {/* ==================================================================== */}
      {/* TAB: MULTI-OBJECT DETECTION VIEW                                    */}
      {/* ==================================================================== */}
      {activeTab === 'detect' && (
        <div className="detect-tab-container">
          <div className="detect-header-card">
            <div className="detect-header-title">
              <span style={{ fontSize: '2rem' }}>🎯</span>
              <div>
                <h2>Phát Hiện & Khoanh Vùng Đa Vật Thể Rác Thải</h2>
                <p style={{ color: 'var(--text-muted)', fontSize: '13.5px', marginTop: '4px' }}>
                  Định vị đồng thời nhiều vật thể rác trong cùng một bức ảnh, gán bounding box màu đỏ và phân loại theo 6 danh mục rác thải chuẩn.
                </p>
              </div>
            </div>
          </div>

          <div className="main-grid">
            {/* Cột trái: Tải ảnh & Xem ảnh chú thích */}
            <div className="upload-section">
              <div className="card">
                <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
                  <button
                    type="button"
                    className={`btn ${detectInputMode === 'upload' ? 'btn-primary' : 'btn-outline'}`}
                    style={{ flex: 1, padding: '8px 12px', fontSize: '13px' }}
                    onClick={() => setDetectInputMode('upload')}
                  >
                    📁 Tải tệp lên
                  </button>
                  <button
                    type="button"
                    className={`btn ${detectInputMode === 'url' ? 'btn-primary' : 'btn-outline'}`}
                    style={{ flex: 1, padding: '8px 12px', fontSize: '13px' }}
                    onClick={() => setDetectInputMode('url')}
                  >
                    🔗 Nhập URL ảnh
                  </button>
                </div>

                {detectInputMode === 'upload' ? (
                  <div
                    className="dropzone"
                    style={{ minHeight: '220px', cursor: 'pointer' }}
                    onClick={() => detectFileInputRef.current?.click()}
                  >
                    <input
                      ref={detectFileInputRef}
                      type="file"
                      accept="image/*"
                      style={{ display: 'none' }}
                      onChange={handleDetectFileSelect}
                    />
                    {detectPreview ? (
                      <div style={{ width: '100%', textAlign: 'center' }}>
                        <div className="detect-view-toggle-bar">
                          <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-muted)' }}>
                            {detectResult?.annotated_image ? 'Chế độ xem ảnh:' : 'Ảnh xem trước:'}
                          </span>
                          {detectResult?.annotated_image && (
                            <div className="xai-view-tabs">
                              <button
                                type="button"
                                className={`xai-tab-btn ${detectViewMode === 'annotated' ? 'active' : ''}`}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setDetectViewMode('annotated');
                                }}
                              >
                                🔴 Ảnh Bounding Box
                              </button>
                              <button
                                type="button"
                                className={`xai-tab-btn ${detectViewMode === 'original' ? 'active' : ''}`}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setDetectViewMode('original');
                                }}
                              >
                                🖼️ Ảnh Gốc
                              </button>
                            </div>
                          )}
                        </div>
                        <img
                          src={detectViewMode === 'annotated' && detectResult?.annotated_image ? detectResult.annotated_image : detectPreview}
                          alt="Preview"
                          style={{
                            maxWidth: '100%',
                            maxHeight: '360px',
                            objectFit: 'contain',
                            borderRadius: '8px',
                            border: '1px solid var(--border)'
                          }}
                        />
                        <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '8px' }}>
                          Nhấp để chọn ảnh khác
                        </p>
                      </div>
                    ) : (
                      <div className="dropzone-content">
                        <span className="dropzone-icon">🎯</span>
                        <p className="dropzone-text">Kéo thả ảnh chứa nhiều vật thể rác vào đây</p>
                        <p className="dropzone-hint">Hỗ trợ ảnh chai nhựa, lon nhôm, thức ăn thừa, hộp carton... (PNG, JPG, WEBP)</p>
                      </div>
                    )}
                  </div>
                ) : (
                  <div style={{ marginBottom: '16px' }}>
                    <div style={{ display: 'flex', gap: '8px' }}>
                      <input
                        type="url"
                        className="form-input"
                        placeholder="Dán link ảnh (http:// hoặc https://)..."
                        value={detectUrlInput}
                        onChange={(e) => setDetectUrlInput(e.target.value)}
                        style={{ flex: 1, padding: '10px 14px', borderRadius: '8px', border: '1px solid var(--border)' }}
                      />
                      <button
                        type="button"
                        className="btn btn-primary"
                        onClick={handleRunDetectUrl}
                        disabled={isDetectLoading}
                      >
                        {isDetectLoading ? 'Đang tải...' : 'Phát hiện'}
                      </button>
                    </div>
                    {detectPreview && (
                      <div style={{ marginTop: '14px', textAlign: 'center' }}>
                        {detectResult?.annotated_image && (
                          <div className="detect-view-toggle-bar">
                            <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-muted)' }}>Chế độ xem:</span>
                            <div className="xai-view-tabs">
                              <button
                                type="button"
                                className={`xai-tab-btn ${detectViewMode === 'annotated' ? 'active' : ''}`}
                                onClick={() => setDetectViewMode('annotated')}
                              >
                                🔴 Ảnh Bounding Box
                              </button>
                              <button
                                type="button"
                                className={`xai-tab-btn ${detectViewMode === 'original' ? 'active' : ''}`}
                                onClick={() => setDetectViewMode('original')}
                              >
                                🖼️ Ảnh Gốc
                              </button>
                            </div>
                          </div>
                        )}
                        <img
                          src={detectViewMode === 'annotated' && detectResult?.annotated_image ? detectResult.annotated_image : detectPreview}
                          alt="URL preview"
                          style={{ maxWidth: '100%', maxHeight: '360px', objectFit: 'contain', borderRadius: '8px' }}
                        />
                      </div>
                    )}
                  </div>
                )}

                {detectInputMode === 'upload' && detectFile && (
                  <div style={{ marginTop: '16px', display: 'flex', gap: '10px' }}>
                    <button
                      type="button"
                      className="btn btn-primary"
                      style={{ flex: 2, padding: '12px', fontWeight: 700 }}
                      onClick={handleRunDetect}
                      disabled={isDetectLoading}
                    >
                      {isDetectLoading ? '⏳ Đang quét & phân loại đa vật thể...' : '🎯 Quét & Phân Loại Ngay'}
                    </button>
                    <button
                      type="button"
                      className="btn btn-outline"
                      style={{ flex: 1 }}
                      onClick={handleResetDetect}
                      disabled={isDetectLoading}
                    >
                      🔄 Đổi ảnh
                    </button>
                  </div>
                )}

                {detectError && (
                  <div className="alert alert-danger" style={{ marginTop: '14px' }}>
                    <span>⚠️ {detectError}</span>
                  </div>
                )}
              </div>
            </div>

            {/* Cột phải: Kết quả nhận diện & Phân loại đa vật thể */}
            <div className="results-section">
              {isDetectLoading ? (
                <div className="card loading-card" style={{ textAlign: 'center', padding: '48px 24px' }}>
                  <div className="spinner" style={{ margin: '0 auto 16px' }} />
                  <h3>Đang chạy mô hình AI hai giai đoạn...</h3>
                  <p style={{ color: 'var(--text-muted)', fontSize: '13.5px', marginTop: '6px' }}>
                    Giai đoạn 1: SSDLite320 định vị bounding box & Non-Maximum Suppression<br/>
                    Giai đoạn 2: MobileNetV3-Large phân loại chính xác danh mục rác thải
                  </p>
                </div>
              ) : detectResult ? (
                <div>
                  {/* Banner tóm tắt */}
                  <div className="detect-summary-banner">
                    <div>
                      <div className="detect-total-count">
                        <span>🎯</span>
                        <span>Phát hiện {detectResult.total_objects} vật thể rác</span>
                      </div>
                      <div className="detect-chips-wrapper">
                        {Object.entries(detectResult.category_counts).map(([catKey, count]) => {
                          const labels = {
                            plastic: 'Nhựa',
                            paper: 'Giấy',
                            metal: 'Kim loại',
                            glass: 'Thủy tinh',
                            organic: 'Hữu cơ',
                            other: 'Khác'
                          };
                          const isFiltered = detectFilterCategory === catKey;
                          return (
                            <button
                              key={catKey}
                              type="button"
                              className={`detect-category-chip ${count > 0 ? 'active' : ''} ${isFiltered ? 'filter-active' : ''}`}
                              style={{
                                cursor: count > 0 ? 'pointer' : 'default',
                                opacity: count > 0 ? 1 : 0.6,
                                outline: isFiltered ? '2px solid #ef4444' : 'none'
                              }}
                              onClick={() => {
                                if (count > 0) {
                                  setDetectFilterCategory(isFiltered ? null : catKey);
                                }
                              }}
                              title={count > 0 ? `Nhấp để lọc chỉ hiển thị nhóm ${labels[catKey]}` : ''}
                            >
                              <span>{labels[catKey] || catKey}:</span>
                              <strong>{count}</strong>
                            </button>
                          );
                        })}
                      </div>
                    </div>
                    {detectResult.processing_time_ms && (
                      <div className="detect-time-badge">
                        ⚡ {detectResult.processing_time_ms} ms
                      </div>
                    )}
                  </div>

                  {/* Cảnh báo phi rác nếu có người hoặc động vật */}
                  {detectResult.has_non_waste && (
                    <div className="alert alert-warning" style={{ marginBottom: '16px' }}>
                      <strong>ℹ️ Lưu ý đối tượng phi rác:</strong> Phát hiện {detectResult.non_waste_objects.length} đối tượng không phải rác
                      ({detectResult.non_waste_objects.map(o => o.class_name).join(', ')}). Hệ thống đã tự động lọc bỏ an toàn.
                    </div>
                  )}

                  {/* Trạng thái không tìm thấy rác */}
                  {detectResult.total_objects === 0 && (
                    <div className="card" style={{ textAlign: 'center', padding: '36px 20px' }}>
                      <span style={{ fontSize: '2.5rem' }}>🔍</span>
                      <h3 style={{ marginTop: '12px' }}>Không phát hiện vật thể rác nào trong ảnh</h3>
                      <p style={{ color: 'var(--text-muted)', fontSize: '13.5px', marginTop: '6px' }}>
                        Ảnh có thể bị mờ, góc chụp quá xa hoặc không chứa các vật thể rác có hình dạng định hình. Vui lòng chụp gần hơn và đủ sáng.
                      </p>
                    </div>
                  )}

                  {/* Danh sách từng vật thể rác được phát hiện */}
                  {detectResult.objects
                    .filter(item => !detectFilterCategory || item.waste_class === detectFilterCategory)
                    .map((item) => (
                      <div key={item.index} className="detect-object-card">
                        <div className="detect-object-card-header">
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <span className="detect-index-tag">#{item.index}</span>
                            <h4 style={{ margin: 0, fontSize: '1.05rem', color: 'var(--text-main)' }}>
                              {item.label_vi}
                            </h4>
                            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                              ({item.detector_label_vi})
                            </span>
                          </div>
                          <span
                            className="badge"
                            style={{
                              background: item.recommendation.bin_color || '#10b981',
                              color: '#ffffff',
                              padding: '4px 10px',
                              borderRadius: '6px',
                              fontSize: '11.5px',
                              fontWeight: 700
                            }}
                          >
                            {item.recommendation.bin}
                          </span>
                        </div>

                        {/* Phân tách 2 chỉ số Confidence */}
                        <div className="detect-confidence-grid">
                          <div className="confidence-metric-box">
                            <span className="confidence-metric-label">🎯 Định vị (SSDLite)</span>
                            <span className="confidence-metric-val">{(item.detector_confidence * 100).toFixed(1)}%</span>
                          </div>
                          <div className="confidence-metric-box">
                            <span className="confidence-metric-label">🧠 Phân loại (MobileNetV3)</span>
                            <span className="confidence-metric-val" style={{ color: '#059669' }}>
                              {(item.classifier_confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="confidence-metric-box">
                            <span className="confidence-metric-label">📐 Tọa độ Bounding Box</span>
                            <span className="confidence-metric-val" style={{ fontSize: '11px', fontFamily: 'monospace' }}>
                              [{item.bbox.join(', ')}]
                            </span>
                          </div>
                        </div>

                        {/* Hướng dẫn xử lý & Mẹo phân loại */}
                        <div style={{ fontSize: '13px', color: 'var(--text-muted)', marginTop: '8px' }}>
                          <p style={{ marginBottom: '6px' }}>
                            <strong style={{ color: 'var(--text-main)' }}>💡 Hướng dẫn:</strong> {item.recommendation.guidance}
                          </p>
                          {item.recommendation.tips && item.recommendation.tips.length > 0 && (
                            <ul style={{ paddingLeft: '18px', margin: '4px 0 0' }}>
                              {item.recommendation.tips.map((tip, tIdx) => (
                                <li key={tIdx} style={{ fontSize: '12px', marginBottom: '2px' }}>{tip}</li>
                              ))}
                            </ul>
                          )}
                        </div>
                      </div>
                    ))}
                </div>
              ) : (
                <div className="card" style={{ textAlign: 'center', padding: '40px 20px' }}>
                  <span style={{ fontSize: '2.5rem' }}>🎯</span>
                  <h3 style={{ marginTop: '12px' }}>Chưa có ảnh nào được phân tích</h3>
                  <p style={{ color: 'var(--text-muted)', fontSize: '13.5px', marginTop: '6px' }}>
                    Tải lên một bức ảnh chứa một hoặc nhiều món rác thải để xem hệ thống định vị từng vật thể và gán nhãn phân loại tự động.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}


      {/* ==================================================================== */}
      {/* TAB 2: PREDICTION HISTORY VIEW                                       */}
      {/* ==================================================================== */}
      {activeTab === 'history' && (
        <div className="history-container">
          <div className="history-filter-bar" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              <select
                className="filter-select"
                value={filterClass}
                onChange={(e) => {
                  setFilterClass(e.target.value);
                  fetchHistory(1, e.target.value, filterStartDate, filterEndDate);
                }}
              >
                <option value="">Tất cả các nhóm rác</option>
                {Object.entries(CLASS_META).map(([k, v]) => (
                  <option key={k} value={k}>{v.icon} {v.label}</option>
                ))}
              </select>
              <input
                type="date"
                className="filter-input"
                value={filterStartDate}
                onChange={(e) => setFilterStartDate(e.target.value)}
              />
              <input
                type="date"
                className="filter-input"
                value={filterEndDate}
                onChange={(e) => setFilterEndDate(e.target.value)}
              />
              <button
                type="button"
                className="btn-primary"
                style={{ padding: '6px 12px', fontSize: '13px' }}
                onClick={() => fetchHistory(1, filterClass, filterStartDate, filterEndDate)}
              >
                Lọc
              </button>
            </div>

            <button type="button" className="btn-export-csv" onClick={handleExportCSV}>
              📥 Xuất File Lịch Sử (CSV)
            </button>
          </div>

          <div className="history-table-wrapper" style={{ marginTop: '16px' }}>
            {isHistoryLoading ? (
              <div style={{ padding: '30px', textAlign: 'center' }}>Đang tải lịch sử...</div>
            ) : historyList.length === 0 ? (
              <div style={{ padding: '40px', textAlign: 'center', color: '#64748b' }}>
                Chưa có bản ghi lịch sử nào phù hợp.
              </div>
            ) : (
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
                <thead>
                  <tr style={{ background: '#f8fafc', borderBottom: '1px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '10px' }}>ID</th>
                    <th style={{ padding: '10px' }}>Thời Gian</th>
                    <th style={{ padding: '10px' }}>Tên Tệp</th>
                    <th style={{ padding: '10px' }}>Dự Đoán AI</th>
                    <th style={{ padding: '10px' }}>Độ Tin Cậy</th>
                    <th style={{ padding: '10px' }}>Phản Hồi User</th>
                    <th style={{ padding: '10px', textAlign: 'right' }}>Ghi Chú</th>
                  </tr>
                </thead>
                <tbody>
                  {historyList.map((row) => (
                    <tr key={row.id} className="history-row" style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '10px' }}>#{row.id}</td>
                      <td style={{ padding: '10px' }}>{formatDate(row.created_at)}</td>
                      <td style={{ padding: '10px' }}>{row.filename}</td>
                      <td style={{ padding: '10px' }}>
                        {CLASS_META[row.predicted_class]?.icon} <strong>{row.label_vi}</strong>
                      </td>
                      <td style={{ padding: '10px' }}>
                        {(row.confidence * 100).toFixed(1)}%
                      </td>
                      <td style={{ padding: '10px' }}>
                        {row.feedback_status === 'correct' && <span style={{ color: '#059669', fontWeight: 600 }}>👍 Chuẩn xác</span>}
                        {row.feedback_status === 'incorrect' && (
                          <span style={{ color: '#ef4444', fontWeight: 600 }}>
                            👎 Sửa: {CLASS_META[row.feedback_correct_class]?.label || row.feedback_correct_class}
                          </span>
                        )}
                        {row.feedback_status === 'uncertain' && <span style={{ color: '#d97706', fontWeight: 600 }}>❓ Không chắc</span>}
                        {!row.feedback_status && (
                          row.feedback_is_correct === true ? <span style={{ color: '#059669' }}>👍 Đúng</span> :
                          row.feedback_is_correct === false ? <span style={{ color: '#ef4444' }}>👎 Sai</span> :
                          <span style={{ color: '#94a3b8' }}>-</span>
                        )}
                      </td>
                      <td style={{ padding: '10px', textAlign: 'right', color: '#64748b' }}>
                        {row.feedback_note || '-'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      )}

      {/* ==================================================================== */}
      {/* TAB 3: REDESIGNED ANALYTICS DASHBOARD VIEW                           */}
      {/* ==================================================================== */}
      {activeTab === 'analytics' && (
        <div className="analytics-container">
          {analyticsData && (
            <>
              {/* Mandatory Disclaimer Alert */}
              <div className="dashboard-disclaimer">
                <span style={{ fontSize: '18px' }}>⚠️</span>
                <span>
                  <strong>Lưu ý quan trọng:</strong> {analyticsData.disclaimer}
                </span>
              </div>

              {/* 1. TOP METRIC OVERVIEW CARDS */}
              <div className="stats-dashboard">
                <div className="stats-metric-card">
                  <div className="metric-number">{analyticsData.total_predictions}</div>
                  <div className="metric-label">Tổng Số Lần Nhận Diện</div>
                  <div style={{ fontSize: '11px', color: '#64748b', marginTop: '4px' }}>
                    Độ tin cậy TB: {(analyticsData.average_confidence * 100).toFixed(1)}%
                  </div>
                </div>

                <div className="stats-metric-card">
                  <div className="metric-number">
                    {analyticsData.most_frequent_class ? (
                      `${CLASS_META[analyticsData.most_frequent_class.class_name]?.icon} ${analyticsData.most_frequent_class.label_vi}`
                    ) : 'Chưa có'}
                  </div>
                  <div className="metric-label">Loại Rác Nhận Diện Nhiều Nhất</div>
                  {analyticsData.most_frequent_class && (
                    <div style={{ fontSize: '11.5px', color: '#059669', marginTop: '4px', fontWeight: 600 }}>
                      {analyticsData.most_frequent_class.count} lượt ({analyticsData.most_frequent_class.percentage}% tổng số)
                    </div>
                  )}
                </div>

                <div className="stats-metric-card">
                  <div className="metric-number" style={{ color: '#059669' }}>
                    {analyticsData.feedback_metrics ? `${analyticsData.feedback_metrics.accuracy_rate}%` : 'N/A'}
                  </div>
                  <div className="metric-label">Tỷ Lệ Đúng Theo Phản Hồi</div>
                  <div style={{ fontSize: '11px', color: '#64748b', marginTop: '4px' }}>
                    Từ {analyticsData.feedback_metrics?.total_feedbacks || 0} lượt người dùng đánh giá
                  </div>
                </div>

                <div className="stats-metric-card">
                  <div className="metric-number" style={{ color: '#d97706' }}>
                    {analyticsData.uncertain_count}
                  </div>
                  <div className="metric-label">Ca Không Chắc Chắn / Nghi Ngờ</div>
                  <div style={{ fontSize: '11px', color: '#64748b', marginTop: '4px' }}>
                    Chiếm {analyticsData.uncertain_percentage}% tổng lượt
                  </div>
                </div>
              </div>

              {/* 2. TỶ LỆ TỪNG LOẠI RÁC (WASTE CATEGORY BREAKDOWN) */}
              <div className="analytic-card" style={{ marginTop: '20px' }}>
                <div className="analytic-card-title">Phân Bố & Tỷ Lệ Từng Loại Rác Thải (6 Nhóm)</div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '12px', marginTop: '14px' }}>
                  {analyticsData.class_distribution.map((item) => (
                    <div key={item.class_name} style={{ padding: '14px', background: 'var(--surface)', borderRadius: '8px', border: '1px solid var(--border)' }}>
                      <div style={{ fontSize: '22px' }}>{CLASS_META[item.class_name]?.icon}</div>
                      <div style={{ fontWeight: 700, marginTop: '6px' }}>{item.label_vi}</div>
                      <div style={{ fontSize: '17px', color: '#059669', fontWeight: 800 }}>{item.count} lượt</div>
                      <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{item.percentage}% tổng số</div>
                      <div style={{ marginTop: '6px', background: '#e2e8f0', height: '6px', borderRadius: '3px', overflow: 'hidden' }}>
                        <div style={{ width: `${item.percentage}%`, height: '100%', backgroundColor: CLASS_META[item.class_name]?.color || '#10b981' }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* 3. BIỂU ĐỒ THEO NGÀY HOẶC THEO TUẦN (TIMELINE TRENDS) */}
              <div className="analytic-card" style={{ marginTop: '20px' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px', marginBottom: '14px' }}>
                  <div className="analytic-card-title" style={{ margin: 0 }}>
                    Biểu Đồ Xu Hướng Nhận Diện Rác Theo Thời Gian
                  </div>
                  <div className="timeline-toggle-group">
                    <button
                      type="button"
                      className={`timeline-toggle-btn ${timelineGranularity === 'daily' ? 'active' : ''}`}
                      onClick={() => setTimelineGranularity('daily')}
                    >
                      📅 Theo Ngày (Daily)
                    </button>
                    <button
                      type="button"
                      className={`timeline-toggle-btn ${timelineGranularity === 'weekly' ? 'active' : ''}`}
                      onClick={() => setTimelineGranularity('weekly')}
                    >
                      📆 Theo Tuần (Weekly)
                    </button>
                  </div>
                </div>

                {/* Timeline Render */}
                <div style={{ display: 'flex', gap: '14px', alignItems: 'flex-end', height: '140px', padding: '10px 0', overflowX: 'auto' }}>
                  {timelineGranularity === 'daily' ? (
                    analyticsData.daily_trends && analyticsData.daily_trends.length > 0 ? (
                      analyticsData.daily_trends.map((t) => {
                        const maxCount = Math.max(...analyticsData.daily_trends.map((x) => x.count), 1);
                        const h = Math.max((t.count / maxCount) * 90, 8);
                        return (
                          <div key={t.date} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', minWidth: '46px' }}>
                            <span style={{ fontSize: '11px', fontWeight: 700, color: '#059669', marginBottom: '4px' }}>{t.count}</span>
                            <div style={{ width: '26px', height: h + 'px', background: 'linear-gradient(180deg, #10b981 0%, #059669 100%)', borderRadius: '4px 4px 0 0' }} />
                            <span style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '4px', whiteSpace: 'nowrap' }}>
                              {t.date.slice(5)}
                            </span>
                          </div>
                        );
                      })
                    ) : (
                      <div style={{ color: 'var(--text-muted)', fontSize: '13px' }}>Chưa có dữ liệu theo ngày.</div>
                    )
                  ) : (
                    analyticsData.weekly_trends && analyticsData.weekly_trends.length > 0 ? (
                      analyticsData.weekly_trends.map((w) => {
                        const maxCount = Math.max(...analyticsData.weekly_trends.map((x) => x.count), 1);
                        const h = Math.max((w.count / maxCount) * 90, 8);
                        return (
                          <div key={w.week} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', minWidth: '55px' }}>
                            <span style={{ fontSize: '11px', fontWeight: 700, color: '#3b82f6', marginBottom: '4px' }}>{w.count}</span>
                            <div style={{ width: '30px', height: h + 'px', background: 'linear-gradient(180deg, #38bdf8 0%, #0284c7 100%)', borderRadius: '4px 4px 0 0' }} />
                            <span style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '4px', whiteSpace: 'nowrap' }}>
                              {w.week}
                            </span>
                          </div>
                        );
                      })
                    ) : (
                      <div style={{ color: 'var(--text-muted)', fontSize: '13px' }}>Chưa có dữ liệu theo tuần.</div>
                    )
                  )}
                </div>
              </div>

              {/* 4. THỐNG KÊ KẾT QUẢ ĐÚNG / SAI TỪ PHẢN HỒI NGƯỜI DÙNG */}
              <div className="analytic-card" style={{ marginTop: '20px' }}>
                <div className="analytic-card-title">Thống Kê Đánh Giá Kết Quả Từ Phản Hồi Người Dùng</div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '14px', marginTop: '12px' }}>
                  <div style={{ padding: '12px', background: '#ecfdf5', borderRadius: '8px', border: '1px solid #a7f3d0' }}>
                    <div style={{ fontSize: '12px', color: '#065f46', fontWeight: 700 }}>👍 Số Lượt Xác Nhận Đúng:</div>
                    <div style={{ fontSize: '22px', fontWeight: 800, color: '#059669', marginTop: '4px' }}>
                      {analyticsData.feedback_metrics?.agreed_count || 0}
                    </div>
                  </div>
                  <div style={{ padding: '12px', background: '#fef2f2', borderRadius: '8px', border: '1px solid #fecaca' }}>
                    <div style={{ fontSize: '12px', color: '#991b1b', fontWeight: 700 }}>👎 Số Lượt Báo Kết Quả Sai:</div>
                    <div style={{ fontSize: '22px', fontWeight: 800, color: '#dc2626', marginTop: '4px' }}>
                      {analyticsData.feedback_metrics?.disagreed_count || 0}
                    </div>
                  </div>
                  <div style={{ padding: '12px', background: '#fffbeb', borderRadius: '8px', border: '1px solid #fef3c7' }}>
                    <div style={{ fontSize: '12px', color: '#92400e', fontWeight: 700 }}>❓ Số Lượt Báo Nghi Ngờ:</div>
                    <div style={{ fontSize: '22px', fontWeight: 800, color: '#d97706', marginTop: '4px' }}>
                      {analyticsData.feedback_metrics?.uncertain_feedback_count || 0}
                    </div>
                  </div>
                  <div style={{ padding: '12px', background: 'var(--surface)', borderRadius: '8px', border: '1px solid var(--border)' }}>
                    <div style={{ fontSize: '12px', color: 'var(--text-muted)', fontWeight: 700 }}>🎯 Độ Hài Lòng Chung:</div>
                    <div style={{ fontSize: '22px', fontWeight: 800, color: '#059669', marginTop: '4px' }}>
                      {analyticsData.feedback_metrics?.accuracy_rate || 0}%
                    </div>
                  </div>
                </div>
              </div>

              {/* 5. KHU VỰC DÀNH CHO NHÓM PHÁT TRIỂN: TỔNG HỢP CA LỖI ĐỂ CẢI THIỆN MÔ HÌNH (ACTIVE LEARNING) */}
              <div className="analytic-card" style={{ marginTop: '20px' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '10px' }}>
                  <div>
                    <div className="analytic-card-title" style={{ margin: 0 }}>
                      Tổng Hợp Các Ca Lỗi Để Cải Thiện Mô Hình (Error Cases for Active Learning)
                    </div>
                    <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '2px' }}>
                      Danh sách các ca bị báo sai hoặc không chắc chắn để nhóm AI bổ sung vào dataset huấn luyện lại.
                    </div>
                  </div>
                  <button type="button" className="btn-export-csv" onClick={handleExportErrorCases}>
                    📥 Xuất File Ca Lỗi (CSV Tái Huấn Luyện)
                  </button>
                </div>

                <div className="error-cases-wrapper">
                  {analyticsData.error_cases && analyticsData.error_cases.length > 0 ? (
                    <table className="error-cases-table">
                      <thead>
                        <tr>
                          <th>ID</th>
                          <th>Tên Tệp</th>
                          <th>AI Dự Đoán</th>
                          <th>Độ Tin Cậy</th>
                          <th>Phản Hồi User</th>
                          <th>Sửa Nhãn Thực Tế</th>
                          <th>Ghi Chú Đóng Góp</th>
                        </tr>
                      </thead>
                      <tbody>
                        {analyticsData.error_cases.map((e) => (
                          <tr key={e.id}>
                            <td>#{e.id}</td>
                            <td>{e.filename}</td>
                            <td>
                              {CLASS_META[e.predicted_class]?.icon} {e.label_vi}
                            </td>
                            <td>{(e.confidence * 100).toFixed(1)}%</td>
                            <td>
                              {e.feedback_status === 'incorrect' ? (
                                <span style={{ color: '#dc2626', fontWeight: 600 }}>👎 Báo sai</span>
                              ) : e.feedback_status === 'uncertain' ? (
                                <span style={{ color: '#d97706', fontWeight: 600 }}>❓ Nghi ngờ</span>
                              ) : (
                                <span style={{ color: '#d97706' }}>⚠️ AI không chắc</span>
                              )}
                            </td>
                            <td>
                              {e.feedback_correct_class ? (
                                <strong>{CLASS_META[e.feedback_correct_class]?.icon} {CLASS_META[e.feedback_correct_class]?.label || e.feedback_correct_class}</strong>
                              ) : '-'}
                            </td>
                            <td style={{ color: 'var(--text-muted)' }}>{e.feedback_note || '-'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : (
                    <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '13px' }}>
                      Chưa ghi nhận ca lỗi nào từ người dùng.
                    </div>
                  )}
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {/* ==================================================================== */}
      {/* TAB 4: MODEL EVALUATION & METRICS VIEW                               */}
      {/* ==================================================================== */}
      {activeTab === 'metrics' && (
        <div className="metrics-container">
          {modelMetrics && (
            <>
              <div className="metrics-spec-card">
                <div className="spec-item">
                  <span className="spec-label">Kiến Trúc AI</span>
                  <span className="spec-val">{modelMetrics.model_spec.architecture}</span>
                </div>
                <div className="spec-item">
                  <span className="spec-label">Kích Thước Mô Hình</span>
                  <span className="spec-val">{modelMetrics.model_spec.parameters_count}</span>
                </div>
                <div className="spec-item">
                  <span className="spec-label">Độ Chính Xác Tổng (Accuracy)</span>
                  <span className="spec-val" style={{ color: '#059669' }}>
                    {(modelMetrics.evaluation.accuracy * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="spec-item">
                  <span className="spec-label">Thời Gian Suy Luận (Latency)</span>
                  <span className="spec-val">{modelMetrics.model_spec.average_inference_latency_ms} ms</span>
                </div>
              </div>

              {/* Per-class Metrics Table */}
              <div className="analytic-card">
                <div className="analytic-card-title">Chỉ Số Đánh Giá Chi Tiết Từng Lớp (Test Set)</div>
                <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: '12px', fontSize: '13px' }}>
                  <thead>
                    <tr style={{ background: '#f8fafc', borderBottom: '1px solid #e2e8f0', textAlign: 'left' }}>
                      <th style={{ padding: '8px' }}>Nhóm Rác</th>
                      <th style={{ padding: '8px' }}>Precision</th>
                      <th style={{ padding: '8px' }}>Recall</th>
                      <th style={{ padding: '8px' }}>F1-Score</th>
                      <th style={{ padding: '8px' }}>Số Mẫu (Support)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(modelMetrics.evaluation.per_class || {}).map(([cls, m]) => (
                      <tr key={cls} style={{ borderBottom: '1px solid #e2e8f0' }}>
                        <td style={{ padding: '8px', fontWeight: 600 }}>
                          {CLASS_META[cls]?.icon} {CLASS_META[cls]?.label || cls}
                        </td>
                        <td style={{ padding: '8px' }}>{(m.precision * 100).toFixed(1)}%</td>
                        <td style={{ padding: '8px' }}>{(m.recall * 100).toFixed(1)}%</td>
                        <td style={{ padding: '8px', fontWeight: 700, color: '#059669' }}>
                          {(m.f1_score * 100).toFixed(1)}%
                        </td>
                        <td style={{ padding: '8px' }}>{m.support}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Confusion Matrix 6x6 */}
              <div className="analytic-card">
                <div className="analytic-card-title">Ma Trận Nhầm Lẫn (Confusion Matrix 6x6)</div>
                <div className="confusion-matrix-wrapper">
                  <table className="confusion-table">
                    <thead>
                      <tr>
                        <th>Thực tế \ Dự đoán</th>
                        {modelMetrics.model_spec.classes_order.map((c) => (
                          <th key={c}>{CLASS_META[c]?.label}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {modelMetrics.evaluation.confusion_matrix?.map((row, rIdx) => {
                        const trueClass = modelMetrics.model_spec.classes_order[rIdx];
                        return (
                          <tr key={rIdx}>
                            <td style={{ fontWeight: 700, textAlign: 'left' }}>
                              {CLASS_META[trueClass]?.label}
                            </td>
                            {row.map((val, cIdx) => (
                              <td
                                key={cIdx}
                                className={rIdx === cIdx ? 'confusion-cell-diag' : 'confusion-cell-off'}
                              >
                                {val}
                              </td>
                            ))}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {/* ==================================================================== */}
      {/* TAB 5: RECYCLING CENTERS DIRECTORY                                  */}
      {/* ==================================================================== */}
      {activeTab === 'centers' && (
        <div className="recycling-container">
          <div className="recycling-filter-bar">
            <select
              className="filter-select"
              value={filterCity}
              onChange={(e) => {
                setFilterCity(e.target.value);
                fetchRecyclingCenters(e.target.value, filterMaterial);
              }}
            >
              <option value="">Tất cả thành phố (Toàn quốc)</option>
              <option value="Hà Nội">Hà Nội</option>
              <option value="Hồ Chí Minh">TP. Hồ Chí Minh</option>
              <option value="Đà Nẵng">Đà Nẵng</option>
              <option value="Cần Thơ">Cần Thơ</option>
            </select>

            <select
              className="filter-select"
              value={filterMaterial}
              onChange={(e) => {
                setFilterMaterial(e.target.value);
                fetchRecyclingCenters(filterCity, e.target.value);
              }}
            >
              <option value="">Tất cả loại vật liệu tiếp nhận</option>
              <option value="plastic">Nhựa (Plastic)</option>
              <option value="paper">Giấy & Vỏ hộp sữa</option>
              <option value="metal">Kim loại & Vỏ lon</option>
              <option value="glass">Thủy tinh</option>
              <option value="organic">Rác hữu cơ vi sinh</option>
              <option value="other">Pin cũ & Rác nguy hại điện tử</option>
            </select>
          </div>

          <div className="recycling-grid">
            {recyclingCenters.map((c) => (
              <div key={c.id} className="center-card">
                <div>
                  <span className="center-city-pill">{c.city} • {c.district}</span>
                  <div className="center-name">{c.name}</div>
                  <div className="center-address">📍 {c.address}</div>
                  <div className="center-materials">
                    {c.accepted_materials.map((m) => (
                      <span key={m} className="mat-tag">
                        {CLASS_META[m]?.icon} {CLASS_META[m]?.label || m}
                      </span>
                    ))}
                  </div>
                </div>

                <div>
                  <div className="center-meta-row">
                    <span>🕒 {c.opening_hours}</span>
                  </div>
                  {c.contact && (
                    <div className="center-meta-row">
                      <span>📞 {c.contact}</span>
                    </div>
                  )}
                  {c.notes && (
                    <div style={{ fontSize: '11.5px', color: '#059669', marginTop: '8px', fontStyle: 'italic' }}>
                      💡 {c.notes}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
