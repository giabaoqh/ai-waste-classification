import React from 'react';

const RANKS = [
  { min: 0, max: 50, title: 'Mầm Non Tái Chế', icon: '🌱', next: 50, color: '#10b981' },
  { min: 51, max: 150, title: 'Tình Nguyện Viên Xanh', icon: '🌿', next: 150, color: '#059669' },
  { min: 151, max: 300, title: 'Chiến Binh Sinh Thái', icon: '🌲', next: 300, color: '#047857' },
  { min: 301, max: 600, title: 'Chuyên Gia Phân Loại', icon: '🌟', next: 600, color: '#0284c7' },
  { min: 601, max: Infinity, title: 'Đại Sứ Môi Trường', icon: '🌍', next: null, color: '#7c3aed' },
];

const ALL_BADGES = [
  { id: 'first_scan', name: 'Bước Chân Đầu Tiên', desc: 'Nhận diện & phân loại món rác đầu tiên', icon: '🥇' },
  { id: 'streak_3', name: 'Thói Quen Vàng', desc: 'Duy trì chuỗi 3 ngày phân loại xanh liên tiếp', icon: '🔥' },
  { id: 'pts_100', name: 'Hiệp Sĩ Xanh', desc: 'Tích lũy đạt cột mốc 100 Điểm Xanh', icon: '💯' },
  { id: 'hazard_hero', name: 'Hộ Vệ Sinh Thái', desc: 'Thu gom an toàn pin hoặc rác điện tử nguy hại', icon: '🛡️' },
  { id: 'co2_1kg', name: 'Lá Chắn Trái Đất', desc: 'Giảm thiểu tương đương hơn 1.0 kg CO₂ phát thải', icon: '🌿' },
];

export default function EcoProfileModal({ profile, onClose, onReset }) {
  if (!profile) return null;

  const currentRank = RANKS.find((r) => profile.points >= r.min && profile.points <= r.max) || RANKS[0];
  const nextTarget = currentRank.next;
  const progressPct = nextTarget
    ? Math.min(100, Math.round(((profile.points - currentRank.min) / (nextTarget - currentRank.min)) * 100))
    : 100;

  return (
    <div className="eco-modal-overlay" onClick={onClose}>
      <div className="eco-modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="eco-modal-header">
          <div className="eco-modal-title">
            <span className="eco-modal-icon">{currentRank.icon}</span>
            <div>
              <h3>Hồ Sơ Sinh Thái & Danh Hiệu Xanh</h3>
              <p>Thành tích bảo vệ môi trường từ thói quen phân loại rác</p>
            </div>
          </div>
          <button type="button" className="eco-modal-close" onClick={onClose} aria-label="Đóng">
            ✕
          </button>
        </div>

        {/* Current Rank & Progress Bar */}
        <div className="eco-rank-section">
          <div className="eco-rank-info">
            <span className="eco-rank-badge" style={{ backgroundColor: currentRank.color }}>
              {currentRank.icon} Cấp bậc: {currentRank.title}
            </span>
            <span className="eco-rank-points">
              <strong>{profile.points}</strong> {nextTarget ? `/ ${nextTarget} Điểm` : 'Điểm (Đạt tối đa)'}
            </span>
          </div>
          {nextTarget && (
            <div className="eco-progress-track">
              <div
                className="eco-progress-bar"
                style={{ width: `${progressPct}%`, backgroundColor: currentRank.color }}
              />
            </div>
          )}
          {nextTarget && (
            <div className="eco-progress-hint">
              Cần thêm {nextTarget - profile.points} Điểm Xanh để nâng cấp danh hiệu tiếp theo!
            </div>
          )}
        </div>

        {/* 4 Stats Grid */}
        <div className="eco-stats-grid">
          <div className="eco-stat-box">
            <span className="eco-stat-val">🌱 {profile.points}</span>
            <span className="eco-stat-lbl">Tổng Điểm Xanh</span>
          </div>
          <div className="eco-stat-box">
            <span className="eco-stat-val">💧 {profile.co2SavedKg} kg</span>
            <span className="eco-stat-lbl">Giảm phát thải CO₂</span>
          </div>
          <div className="eco-stat-box">
            <span className="eco-stat-val">📦 {profile.totalScans}</span>
            <span className="eco-stat-lbl">Vật thể đã quét</span>
          </div>
          <div className="eco-stat-box">
            <span className="eco-stat-val">🔥 {profile.streak} ngày</span>
            <span className="eco-stat-lbl">Chuỗi phân loại</span>
          </div>
        </div>

        {/* Badges Gallery */}
        <div className="eco-badges-section">
          <h4>🏅 Bộ Sưu Tập Huy Hiệu</h4>
          <div className="eco-badges-list">
            {ALL_BADGES.map((b) => {
              const isUnlocked = profile.unlockedBadges?.includes(b.id);
              return (
                <div key={b.id} className={`eco-badge-item ${isUnlocked ? 'unlocked' : 'locked'}`}>
                  <div className="eco-badge-icon">{b.icon}</div>
                  <div className="eco-badge-content">
                    <div className="eco-badge-name">
                      {b.name} {isUnlocked && <span className="eco-unlocked-tag">Đã đạt</span>}
                    </div>
                    <div className="eco-badge-desc">{b.desc}</div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="eco-modal-footer">
          <button type="button" className="btn-eco-reset" onClick={onReset}>
            Đặt lại thành tích
          </button>
          <button type="button" className="btn-eco-done" onClick={onClose}>
            Tuyệt vời, đóng lại
          </button>
        </div>
      </div>
    </div>
  );
}

