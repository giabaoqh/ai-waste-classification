import React, { useState, useEffect } from 'react';

export default function PwaInstallBanner() {
  const [deferredPrompt, setDeferredPrompt] = useState(null);
  const [isVisible, setIsVisible] = useState(false);
  const [isInstalled, setIsInstalled] = useState(false);

  useEffect(() => {
    // Check if app is already running as standalone PWA
    if (window.matchMedia('(display-mode: standalone)').matches || window.navigator.standalone) {
      setIsInstalled(true);
      return;
    }

    const handler = (e) => {
      e.preventDefault();
      setDeferredPrompt(e);
      setIsVisible(true);
    };

    window.addEventListener('beforeinstallprompt', handler);

    window.addEventListener('appinstalled', () => {
      setIsInstalled(true);
      setIsVisible(false);
      setDeferredPrompt(null);
    });

    return () => {
      window.removeEventListener('beforeinstallprompt', handler);
    };
  }, []);

  const handleInstallClick = async () => {
    if (!deferredPrompt) return;
    deferredPrompt.prompt();
    const { outcome } = await deferredPrompt.userChoice;
    if (outcome === 'accepted') {
      setIsVisible(false);
    }
    setDeferredPrompt(null);
  };

  if (isInstalled || !isVisible) return null;

  return (
    <div className="pwa-install-banner">
      <div className="pwa-banner-content">
        <div className="pwa-icon">📱</div>
        <div className="pwa-text">
          <div className="pwa-title">Cài đặt Ứng dụng Phân Loại Rác</div>
          <div className="pwa-subtitle">Mở toàn màn hình, chụp ảnh trực tiếp và dùng như app mobile</div>
        </div>
      </div>
      <div className="pwa-actions">
        <button type="button" className="btn-pwa-install" onClick={handleInstallClick}>
          Cài đặt ngay
        </button>
        <button type="button" className="btn-pwa-dismiss" onClick={() => setIsVisible(false)} aria-label="Bỏ qua">
          ✕
        </button>
      </div>
    </div>
  );
}

