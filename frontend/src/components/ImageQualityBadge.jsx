import React from 'react';

export function ClientQualityIndicator({ quality }) {
  if (!quality) return null;

  if (quality.status === 'optimal') {
    return (
      <div className="client-quality-badge quality-good">
        <span className="quality-dot dot-good" />
        <span>Ánh sáng đạt chuẩn ({quality.brightnessPct}%) • Ảnh rõ nét</span>
      </div>
    );
  }

  if (quality.status === 'dark') {
    return (
      <div className="client-quality-badge quality-warn">
        <span className="quality-dot dot-warn" />
        <span>⚠️ {quality.warning}</span>
      </div>
    );
  }

  if (quality.status === 'bright') {
    return (
      <div className="client-quality-badge quality-warn">
        <span className="quality-dot dot-warn" />
        <span>⚠️ {quality.warning}</span>
      </div>
    );
  }

  return null;
}

export function BackendQualityCard({ assessment }) {
  if (!assessment) return null;

  const { brightness_score, sharpness_score, has_warnings, warnings } = assessment;

  return (
    <div className={`quality-card ${has_warnings ? 'has-warnings' : 'is-clean'}`}>
      <div className="quality-card-header">
        <span className="quality-card-title">
          {has_warnings ? '⚠️ Đánh Giá Kỹ Thuật Ảnh' : '✅ Đánh Giá Kỹ Thuật Ảnh (Đạt Chuẩn)'}
        </span>
        <div className="quality-metrics-mini">
          <span>Độ sáng: <strong>{brightness_score}/255</strong></span>
          <span>•</span>
          <span>Độ nét (Laplacian): <strong>{sharpness_score}</strong></span>
        </div>
      </div>
      {has_warnings && warnings && warnings.length > 0 && (
        <ul className="quality-warnings-list">
          {warnings.map((w, idx) => (
            <li key={idx}>💡 {w}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

