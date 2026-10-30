import React, { useState, useEffect, useRef } from 'react';

// Haversine formula to calculate distance in km
function calculateDistance(lat1, lon1, lat2, lon2) {
  const R = 6371; // Earth radius in km
  const dLat = (lat2 - lat1) * (Math.PI / 180);
  const dLon = (lon2 - lon1) * (Math.PI / 180);
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(lat1 * (Math.PI / 180)) * Math.cos(lat2 * (Math.PI / 180)) *
    Math.sin(dLon / 2) * Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  const d = R * c;
  return d < 1 ? `${Math.round(d * 1000)} m` : `${d.toFixed(1)} km`;
}

// 100% Free Tile Providers without API key requirement
const TILE_LAYERS = {
  google_streets: {
    name: 'Google Đường Phố',
    url: 'https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
    subdomains: ['mt0', 'mt1', 'mt2', 'mt3'],
    maxZoom: 20,
    attribution: '&copy; Google Maps'
  },
  google_satellite: {
    name: 'Google Vệ Tinh',
    url: 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
    subdomains: ['mt0', 'mt1', 'mt2', 'mt3'],
    maxZoom: 20,
    attribution: '&copy; Google Maps'
  },
  esri: {
    name: 'ArcGIS Street',
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
    subdomains: [],
    maxZoom: 19,
    attribution: '&copy; Esri &bull; OpenStreetMap'
  },
  osm: {
    name: 'OpenStreetMap',
    url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    subdomains: ['a', 'b', 'c'],
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap'
  }
};

export default function SmartLitterMap({ backendUrl, darkMode, onAwardEcoPoints, showNotification }) {
  const [reports, setReports] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [statusFilter, setStatusFilter] = useState('all');
  const [severityFilter, setSeverityFilter] = useState('all');
  const [showReportModal, setShowReportModal] = useState(false);
  const [activeTileType, setActiveTileType] = useState('google_streets');

  // User location for navigation & directions
  const [userCoords, setUserCoords] = useState(null);
  const [activeRoutingReport, setActiveRoutingReport] = useState(null);
  const [routingMetrics, setRoutingMetrics] = useState(null);
  const [isRoutingLoading, setIsRoutingLoading] = useState(false);

  // Filters for Proximity & News Intelligence
  const [sourceFilter, setSourceFilter] = useState('all'); // 'all', 'citizen', 'news', 'social'
  const [radiusFilter, setRadiusFilter] = useState('all'); // 'all', 5, 15, 30, 50 (km)
  const [isSyncingNews, setIsSyncingNews] = useState(false);

  // Form states for new report
  const [reportLat, setReportLat] = useState('');
  const [reportLng, setReportLng] = useState('');
  const [reportAddress, setReportAddress] = useState('');
  const [reportDesc, setReportDesc] = useState('');
  const [reportSeverity, setReportSeverity] = useState('medium');
  const [reportReporter, setReportReporter] = useState('');
  const [reportFile, setReportFile] = useState(null);
  const [reportPreview, setReportPreview] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isGettingLocation, setIsGettingLocation] = useState(false);
  const [formError, setFormError] = useState(null);

  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const tileLayerRef = useRef(null);
  const markersGroupRef = useRef(null);
  const routeLineRef = useRef(null);
  const userMarkerRef = useRef(null);
  const fileInputRef = useRef(null);
  const radarCircleRef = useRef(null);
  const markersMapRef = useRef(new Map());

  // Fetch reports & stats
  const fetchReports = async () => {
    try {
      setLoading(true);
      const query = new URLSearchParams();
      if (statusFilter !== 'all') query.append('status', statusFilter);
      if (severityFilter !== 'all') query.append('severity', severityFilter);
      if (sourceFilter !== 'all') query.append('source_type', sourceFilter);
      if (userCoords) {
        query.append('user_lat', userCoords.lat);
        query.append('user_lng', userCoords.lng);
        if (radiusFilter !== 'all') {
          query.append('radius_km', radiusFilter);
          query.append('sort_by', 'distance');
        }
      }
      const qs = query.toString() ? `?${query.toString()}` : '';

      const [resReports, resStats] = await Promise.all([
        fetch(`${backendUrl || ''}/api/litter-reports${qs}`),
        fetch(`${backendUrl || ''}/api/litter-reports/stats`)
      ]);

      if (resReports.ok) {
        const data = await resReports.json();
        setReports(data);
      }
      if (resStats.ok) {
        const statsData = await resStats.json();
        setStats(statsData);
      }
    } catch (err) {
      console.error('Error fetching litter reports:', err);
      showNotification && showNotification('Lỗi khi tải dữ liệu bản đồ rác.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchReports();
  }, [statusFilter, severityFilter, sourceFilter, radiusFilter, userCoords]);

  // Initialize Leaflet Map
  useEffect(() => {
    if (!mapContainerRef.current || !window.L) return;

    if (!mapInstanceRef.current) {
      const map = window.L.map(mapContainerRef.current, {
        center: [16.047079, 108.206230],
        zoom: 6,
        scrollWheelZoom: true
      });

      // Default to Google Maps Street Tiles (Ultra-fast & zero API key in VN)
      const currentConfig = TILE_LAYERS[activeTileType] || TILE_LAYERS.google_streets;
      tileLayerRef.current = window.L.tileLayer(currentConfig.url, {
        maxZoom: currentConfig.maxZoom,
        subdomains: currentConfig.subdomains,
        attribution: currentConfig.attribution
      }).addTo(map);

      markersGroupRef.current = window.L.layerGroup().addTo(map);

      map.on('click', (e) => {
        setReportLat(e.latlng.lat.toFixed(5));
        setReportLng(e.latlng.lng.toFixed(5));
      });

      mapInstanceRef.current = map;
    }

    const t1 = setTimeout(() => {
      if (mapInstanceRef.current) mapInstanceRef.current.invalidateSize();
    }, 150);
    const t2 = setTimeout(() => {
      if (mapInstanceRef.current) mapInstanceRef.current.invalidateSize();
    }, 600);

    const handleResize = () => {
      if (mapInstanceRef.current) mapInstanceRef.current.invalidateSize();
    };
    window.addEventListener('resize', handleResize);

    // Auto-detect user geolocation once on mount
    if (navigator.geolocation && !userCoords) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          const uCoords = { lat: pos.coords.latitude, lng: pos.coords.longitude };
          setUserCoords(uCoords);
          addUserMarker(uCoords.lat, uCoords.lng);
        },
        () => {},
        { timeout: 8000 }
      );
    }

    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  // Switch Tile Layer
  const switchTileLayer = (layerKey) => {
    if (!mapInstanceRef.current || !window.L) return;
    setActiveTileType(layerKey);

    const config = TILE_LAYERS[layerKey];
    if (tileLayerRef.current) {
      mapInstanceRef.current.removeLayer(tileLayerRef.current);
    }
    tileLayerRef.current = window.L.tileLayer(config.url, {
      maxZoom: config.maxZoom,
      subdomains: config.subdomains,
      attribution: config.attribution
    }).addTo(mapInstanceRef.current);

    mapInstanceRef.current.invalidateSize();
  };

  // Add/Update User Marker
  const addUserMarker = (lat, lng) => {
    if (!mapInstanceRef.current || !window.L) return;

    if (userMarkerRef.current) {
      mapInstanceRef.current.removeLayer(userMarkerRef.current);
    }

    const userIcon = window.L.divIcon({
      className: 'custom-user-pin',
      html: `
        <div style="
          background: #3b82f6;
          width: 28px;
          height: 28px;
          border-radius: 50%;
          border: 3px solid #ffffff;
          box-shadow: 0 0 12px rgba(59, 130, 246, 0.8);
          display: flex;
          align-items: center;
          justify-content: center;
          color: #ffffff;
          font-size: 13px;
        ">
          📍
        </div>
      `,
      iconSize: [28, 28],
      iconAnchor: [14, 14]
    });

    userMarkerRef.current = window.L.marker([lat, lng], { icon: userIcon })
      .addTo(mapInstanceRef.current)
      .bindPopup('<div style="font-weight:700;color:#1e293b;padding:4px;">📍 Vị trí hiện tại của bạn</div>');

    // Draw Radar Scanning Circle around User GPS
    if (radarCircleRef.current) {
      mapInstanceRef.current.removeLayer(radarCircleRef.current);
    }
    const rKm = radiusFilter === 'all' ? 15 : Number(radiusFilter);
    radarCircleRef.current = window.L.circle([lat, lng], {
      radius: rKm * 1000,
      color: '#3b82f6',
      fillColor: '#3b82f6',
      fillOpacity: 0.07,
      weight: 1.5,
      dashArray: '6, 8'
    }).addTo(mapInstanceRef.current);
  };

  // Real-Road Routing Algorithm using OSRM (Open Source Routing Machine)
  // Traces the actual curves, highways, turns, and streets like Google Maps
  const drawRouteToHotspot = async (destLat, destLng, item) => {
    if (!mapInstanceRef.current || !window.L) return;

    if (!userCoords) {
      handleGetCurrentLocation();
      showNotification && showNotification('Đang lấy vị trí GPS của bạn để tìm đường...');
      return;
    }

    setIsRoutingLoading(true);
    showNotification && showNotification('Đang tính toán tuyến đường giao thông thực tế theo thuật toán OSRM...');

    // Remove existing route lines if any
    if (routeLineRef.current) {
      mapInstanceRef.current.removeLayer(routeLineRef.current);
      routeLineRef.current = null;
    }

    try {
      // Call OSRM Road Routing API: coordinates in [longitude, latitude] format
      const osrmUrl = `https://router.project-osrm.org/route/v1/driving/${userCoords.lng},${userCoords.lat};${destLng},${destLat}?overview=full&geometries=geojson`;
      const res = await fetch(osrmUrl);
      
      if (!res.ok) throw new Error('Không thể kết nối đến máy chủ điều hướng OSRM.');
      const data = await res.json();

      if (data.code === 'Ok' && data.routes && data.routes.length > 0) {
        const route = data.routes[0];
        // Convert GeoJSON [lon, lat] coordinates to Leaflet [lat, lon]
        const latlngs = route.geometry.coordinates.map(pt => [pt[1], pt[0]]);

        // 1. Route Shadow / Glow outline (Google Maps style)
        const shadowPolyline = window.L.polyline(latlngs, {
          color: '#1e40af',
          weight: 8,
          opacity: 0.35,
          lineCap: 'round',
          lineJoin: 'round'
        });

        // 2. Main navigation line (Google Maps vivid blue)
        const mainPolyline = window.L.polyline(latlngs, {
          color: '#2563eb',
          weight: 5,
          opacity: 0.95,
          lineCap: 'round',
          lineJoin: 'round'
        });

        const routeFeatureGroup = window.L.featureGroup([shadowPolyline, mainPolyline]).addTo(mapInstanceRef.current);
        routeLineRef.current = routeFeatureGroup;

        // Calculate real road distance and driving duration
        const realDistanceKm = (route.distance / 1000).toFixed(1);
        const distStr = route.distance < 1000 ? `${Math.round(route.distance)} m` : `${realDistanceKm} km`;
        
        const totalMinutes = Math.round(route.duration / 60);
        let durationStr = `${totalMinutes} phút`;
        if (totalMinutes >= 60) {
          const hours = Math.floor(totalMinutes / 60);
          const mins = totalMinutes % 60;
          durationStr = mins > 0 ? `${hours} giờ ${mins} phút` : `${hours} giờ`;
        }

        setRoutingMetrics({ distanceStr: distStr, durationStr: durationStr });
        setActiveRoutingReport(item);

        // Smoothly fit bounds to embrace the whole road network
        mapInstanceRef.current.fitBounds(routeFeatureGroup.getBounds(), { padding: [60, 60] });

        showNotification && showNotification(`🚗 Tuyến đường thực tế: ${distStr} • Dự kiến đi xe ~${durationStr}`);
      } else {
        throw new Error('OSRM không tìm được tuyến đường bộ nối liền.');
      }
    } catch (err) {
      console.warn('OSRM routing failed, falling back to direct line:', err);
      // Fallback to direct dashed line if road routing is unreachable
      const startPoint = [userCoords.lat, userCoords.lng];
      const endPoint = [destLat, destLng];
      const directLine = window.L.polyline([startPoint, endPoint], {
        color: '#f59e0b',
        weight: 4,
        opacity: 0.8,
        dashArray: '8, 8',
        lineCap: 'round'
      }).addTo(mapInstanceRef.current);
      routeLineRef.current = directLine;
      
      const directDist = calculateDistance(userCoords.lat, userCoords.lng, destLat, destLng);
      setRoutingMetrics({ distanceStr: `~${directDist} (Đường chim bay)`, durationStr: 'N/A' });
      setActiveRoutingReport(item);
      mapInstanceRef.current.fitBounds([startPoint, endPoint], { padding: [60, 60] });
      showNotification && showNotification(`Đã vẽ đường nối: ${directDist}`);
    } finally {
      setIsRoutingLoading(false);
    }
  };

  // Render Markers
  useEffect(() => {
    if (!mapInstanceRef.current || !markersGroupRef.current || !window.L) return;

    markersGroupRef.current.clearLayers();

    markersMapRef.current.clear();
    reports.forEach((item) => {
      const isResolved = item.status === 'resolved';
      const isInProgress = item.status === 'in_progress';
      const isCritical = item.severity === 'critical';
      const isNews = item.source_type === 'news';
      const isSocial = item.source_type === 'social';

      let pinColor = '#f59e0b';
      let pinEmoji = '⚠️';
      let pinClass = 'litter-pin-pending';

      if (isResolved) {
        pinColor = '#10b981';
        pinEmoji = '✅';
        pinClass = 'litter-pin-resolved';
      } else if (isCritical) {
        pinColor = '#ef4444';
        pinEmoji = '🚨';
        pinClass = 'litter-pin-critical';
      } else if (isNews) {
        pinColor = '#7c3aed';
        pinEmoji = '📰';
        pinClass = 'litter-pin-news';
      } else if (isSocial) {
        pinColor = '#0284c7';
        pinEmoji = '📱';
        pinClass = 'litter-pin-social';
      } else if (item.severity === 'high') {
        pinColor = '#ea580c';
        pinEmoji = '🔥';
        pinClass = 'litter-pin-high';
      } else if (isInProgress) {
        pinColor = '#3b82f6';
        pinEmoji = '🚚';
        pinClass = 'litter-pin-progress';
      }

      const customIcon = window.L.divIcon({
        className: `custom-litter-pin ${pinClass}`,
        html: `
          <div style="
            background: ${pinColor};
            width: 34px;
            height: 34px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #ffffff;
            font-size: 16px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
            border: 2px solid #ffffff;
            cursor: pointer;
            transition: transform 0.2s;
          ">
            ${pinEmoji}
          </div>
        `,
        iconSize: [34, 34],
        iconAnchor: [17, 34],
        popupAnchor: [0, -34]
      });

      const marker = window.L.marker([item.latitude, item.longitude], { icon: customIcon });

      const statusText = isResolved
        ? '<span style="color:#10b981;font-weight:700;">✅ Đã dọn dẹp sạch sẽ</span>'
        : isInProgress
        ? '<span style="color:#3b82f6;font-weight:700;">🚚 Đang điều phối dọn dẹp</span>'
        : '<span style="color:#ef4444;font-weight:700;">⏳ Chờ thu gom xử lý</span>';

      const wasteTags = (item.detected_waste_classes || [])
        .map((c) => `<span style="background:var(--bg-subtle, #334155);color:var(--text-main, #f8fafc);padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600;display:inline-block;margin:2px;">#${c}</span>`)
        .join(' ');

      const imageHtml = item.image_base64
        ? `<div style="margin-bottom:8px;border-radius:6px;overflow:hidden;max-height:160px;background:#000;">
            <img src="${item.image_base64}" alt="Hotspot" style="width:100%;height:140px;object-fit:cover;" />
           </div>`
        : '';

      const distInfo = userCoords
        ? `<div style="font-size: 12px; font-weight: 600; color: #10b981; margin: 4px 0;">
            📏 Cách bạn: <strong>${calculateDistance(userCoords.lat, userCoords.lng, item.latitude, item.longitude)}</strong>
           </div>`
        : '';

      const googleMapsUrl = userCoords
        ? `https://www.google.com/maps/dir/?api=1&origin=${userCoords.lat},${userCoords.lng}&destination=${item.latitude},${item.longitude}`
        : `https://www.google.com/maps/dir/?api=1&destination=${item.latitude},${item.longitude}`;

      const sourceBadge = isNews
        ? `<div style="display:inline-flex;align-items:center;gap:4px;background:#7c3aed;color:#ffffff;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:700;margin-bottom:6px;">📰 ${item.source_name || 'Báo chí phản ánh'}</div>`
        : isSocial
        ? `<div style="display:inline-flex;align-items:center;gap:4px;background:#0284c7;color:#ffffff;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:700;margin-bottom:6px;">📱 ${item.source_name || 'Mạng xã hội cư dân'}</div>`
        : `<div style="display:inline-flex;align-items:center;gap:4px;background:#10b981;color:#ffffff;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:700;margin-bottom:6px;">👤 Người dân phản ánh</div>`;

      const newsTitle = item.title
        ? `<div style="font-size: 13px; font-weight: 700; line-height: 1.35; margin-bottom: 4px; color: var(--text-main, #0f172a);">
            ${item.title}
           </div>`
        : '';

      const newsLink = item.source_url
        ? `<div style="margin: 4px 0 6px 0;">
            <a href="${item.source_url}" target="_blank" rel="noopener noreferrer" style="color: #2563eb; font-size: 11.5px; font-weight: 600; text-decoration: underline; display: inline-flex; align-items: center; gap: 3px;">
              🔗 Đọc bài báo gốc trên ${item.source_name || 'Báo chí'} &rarr;
            </a>
           </div>`
        : '';

      const popupHtml = `
        <div class="litter-popup-card" style="font-family: inherit; font-size: 13px; min-width: 250px; max-width: 300px; padding: 4px;">
          ${sourceBadge}
          ${imageHtml}
          ${newsTitle}
          <div style="font-size: 13.5px; font-weight: 700; margin-bottom: 4px; color: var(--text-main, #0f172a);">
            📍 ${item.address || 'Điểm rác phát sinh'}
          </div>
          <div style="font-size: 12px; margin-bottom: 6px;">
            ${statusText} &bull; Mức độ: <strong>${item.severity.toUpperCase()}</strong>
          </div>
          ${distInfo}
          <p style="font-size: 12px; margin: 4px 0 8px 0; line-height: 1.4; color: var(--text-muted, #475569);">
            ${item.description || 'Không có mô tả chi tiết'}
          </p>
          ${newsLink}
          <div style="margin-bottom: 8px;">
            ${wasteTags}
          </div>
          <div style="font-size: 11px; opacity: 0.8; margin-bottom: 8px;">
            Báo cáo: ${item.reported_by || 'Người dân'} &bull; 👍 ${item.upvotes} xác nhận
          </div>
          
          <!-- Direction & Action Buttons -->
          <div style="display: flex; flex-direction: column; gap: 6px; margin-top: 8px;">
            <div style="display: flex; gap: 6px;">
              <button id="draw-route-btn-${item.id}" style="
                flex: 1;
                background: #2563eb;
                color: #ffffff;
                border: none;
                padding: 7px 8px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 11.5px;
                font-weight: 700;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 4px;
              ">🚗 Vẽ Đường Đi Thực Tế</button>

              <a href="${googleMapsUrl}" target="_blank" rel="noopener noreferrer" style="
                flex: 1;
                background: #3b82f6;
                color: #ffffff;
                text-decoration: none;
                padding: 7px 8px;
                border-radius: 6px;
                font-size: 11.5px;
                font-weight: 700;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 4px;
              ">🗺️ Google Maps</a>
            </div>

            <div style="display: flex; gap: 6px;">
              <button id="upvote-btn-${item.id}" style="
                flex: 1;
                padding: 6px 8px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 11px;
                font-weight: 600;
              ">👍 Xác nhận (${item.upvotes})</button>
              ${!isResolved ? `
                <button id="resolve-btn-${item.id}" style="
                  flex: 1;
                  background: #059669;
                  border: none;
                  color: #ffffff;
                  padding: 6px 8px;
                  border-radius: 6px;
                  cursor: pointer;
                  font-size: 11px;
                  font-weight: 600;
                ">✅ Đã dọn</button>
              ` : ''}
            </div>
          </div>
        </div>
      `;

      marker.bindPopup(popupHtml);
      markersMapRef.current.set(item.id, marker);

      marker.on('popupopen', () => {
        const routeBtn = document.getElementById(`draw-route-btn-${item.id}`);
        if (routeBtn) {
          routeBtn.onclick = () => drawRouteToHotspot(item.latitude, item.longitude, item);
        }
        const upBtn = document.getElementById(`upvote-btn-${item.id}`);
        if (upBtn) {
          upBtn.onclick = () => handleUpvote(item.id);
        }
        const resBtn = document.getElementById(`resolve-btn-${item.id}`);
        if (resBtn) {
          resBtn.onclick = () => handleResolve(item.id);
        }
      });

      markersGroupRef.current.addLayer(marker);
    });

    if (mapInstanceRef.current) {
      mapInstanceRef.current.invalidateSize();
    }
  }, [reports, userCoords]);

  // Handle Upvote
  const handleUpvote = async (reportId) => {
    try {
      const res = await fetch(`${backendUrl || ''}/api/litter-reports/${reportId}/upvote`, {
        method: 'POST'
      });
      if (res.ok) {
        showNotification && showNotification('Đã ghi nhận xác nhận điểm nóng của bạn! 👍');
        fetchReports();
      }
    } catch (e) {
      console.error(e);
    }
  };

  // Handle Resolve
  const handleResolve = async (reportId) => {
    try {
      const res = await fetch(`${backendUrl || ''}/api/litter-reports/${reportId}/status`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          status: 'resolved',
          resolved_note: 'Người dân / Đội tình nguyện đã dọn dẹp sạch sẽ.'
        })
      });
      if (res.ok) {
        showNotification && showNotification('Cảm ơn bạn! Điểm nóng đã được đánh dấu ĐÃ DỌN DẸP SẠCH SẼ! 🌱');
        onAwardEcoPoints && onAwardEcoPoints(30, 'Đã hỗ trợ dọn dẹp sạch điểm rác đô thị');
        fetchReports();
      }
    } catch (e) {
      console.error(e);
    }
  };

  // Sync news & social media reports
  const handleSyncNews = async () => {
    setIsSyncingNews(true);
    showNotification && showNotification('📰 Đang quét tin tức rác và ô nhiễm từ các cơ quan báo chí & mạng xã hội...');
    try {
      const res = await fetch(`${backendUrl || ''}/api/litter-reports/sync-news`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        showNotification && showNotification(data.message || 'Đã đồng bộ tin tức rác từ báo chí thành công! 📰');
        await fetchReports();
      } else {
        showNotification && showNotification('Không thể kết nối máy chủ quét tin tức.');
      }
    } catch (err) {
      console.error('Error syncing news reports:', err);
      showNotification && showNotification('Lỗi khi quét tin tức rác từ báo chí.');
    } finally {
      setIsSyncingNews(false);
    }
  };

  // Jump to the nearest hotspot around user GPS
  const handleFocusNearestReport = () => {
    if (!reports || reports.length === 0) {
      showNotification && showNotification('Không có điểm rác nào trong khu vực quét.');
      return;
    }
    let nearest = null;
    let minD = Infinity;
    reports.forEach((r) => {
      const d = r.distance_km ?? (userCoords ? parseFloat(calculateDistance(userCoords.lat, userCoords.lng, r.latitude, r.longitude)) : Infinity);
      if (d < minD) {
        minD = d;
        nearest = r;
      }
    });

    if (nearest && mapInstanceRef.current) {
      mapInstanceRef.current.flyTo([nearest.latitude, nearest.longitude], 16, { duration: 1.2 });
      setTimeout(() => {
        const marker = markersMapRef.current.get(nearest.id);
        if (marker) marker.openPopup();
      }, 1300);
      showNotification && showNotification(`🎯 Đã định vị điểm rác gần bạn nhất: ${nearest.address || ''} (${minD} km)`);
    }
  };

  // Quick City Jumps
  const jumpToCity = (lat, lng, zoom = 13) => {
    if (mapInstanceRef.current) {
      mapInstanceRef.current.flyTo([lat, lng], zoom, { duration: 1.2 });
      setTimeout(() => mapInstanceRef.current.invalidateSize(), 300);
    }
  };

  // Geolocation
  const handleGetCurrentLocation = () => {
    if (!navigator.geolocation) {
      showNotification && showNotification('Trình duyệt không hỗ trợ Geolocation.');
      return;
    }
    setIsGettingLocation(true);
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const lat = pos.coords.latitude;
        const lng = pos.coords.longitude;
        const coords = { lat, lng };
        setUserCoords(coords);
        addUserMarker(lat, lng);
        setReportLat(lat.toFixed(5));
        setReportLng(lng.toFixed(5));
        jumpToCity(lat, lng, 15);
        setIsGettingLocation(false);
        showNotification && showNotification('Đã định vị tọa độ GPS của bạn thành công! 📍');
      },
      (err) => {
        console.error(err);
        setIsGettingLocation(false);
        showNotification && showNotification('Không thể lấy tọa độ GPS. Vui lòng cấp quyền truy cập vị trí trên trình duyệt.');
      },
      { timeout: 10000, enableHighAccuracy: true }
    );
  };

  // Handle Image File Selection for Report
  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      setFormError('Vui lòng chọn tệp hình ảnh hợp lệ (JPG, PNG, WebP).');
      return;
    }
    setReportFile(file);
    setFormError(null);
    const reader = new FileReader();
    reader.onload = () => setReportPreview(reader.result);
    reader.readAsDataURL(file);
  };

  // Handle Form Submit
  const handleReportSubmit = async (e) => {
    e.preventDefault();
    if (!reportLat || !reportLng) {
      setFormError('Vui lòng nhập tọa độ GPS hoặc bấm "Lấy vị trí của tôi" / bấm lên bản đồ.');
      return;
    }

    setIsSubmitting(true);
    setFormError(null);

    try {
      const formData = new FormData();
      formData.append('latitude', reportLat);
      formData.append('longitude', reportLng);
      formData.append('address', reportAddress || 'Vị trí người dân báo cáo');
      formData.append('description', reportDesc || 'Bãi rác tự phát phát sinh');
      formData.append('severity', reportSeverity);
      formData.append('reported_by', reportReporter || 'Người dân phản ánh');
      if (reportFile) {
        formData.append('file', reportFile);
      }

      const res = await fetch(`${backendUrl || ''}/api/litter-reports`, {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || 'Lỗi khi gửi báo cáo.');
      }

      showNotification && showNotification('🎉 Báo cáo điểm nóng rác thải thành công! Bạn nhận được +50 Điểm Sinh Thái 🌱');
      onAwardEcoPoints && onAwardEcoPoints(50, 'Đã báo cáo điểm nóng rác thải vì thành phố xanh sạch đẹp');

      setReportAddress('');
      setReportDesc('');
      setReportFile(null);
      setReportPreview(null);
      setShowReportModal(false);

      fetchReports();
      jumpToCity(parseFloat(reportLat), parseFloat(reportLng), 14);
    } catch (err) {
      console.error(err);
      setFormError(err.message || 'Lỗi khi gửi báo cáo điểm rác.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="litter-map-tab-container">
      {/* Top Header Card */}
      <div className="litter-map-header-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <div className="brand-badge" style={{ display: 'inline-flex', marginBottom: '8px' }}>
              <span>🏙️ Smart City Crowdsourcing</span>
              <span>&bull;</span>
              <span>Bản Đồ Điểm Nóng Rác Thải Đô Thị</span>
            </div>
            <h2 style={{ fontSize: '22px', fontWeight: 700, margin: '4px 0', color: 'var(--text-main)' }}>
              Bản Đồ Báo Cáo Điểm Nóng Rác Thải & Môi Trường Đô Thị
            </h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '13px', margin: 0, maxWidth: '680px' }}>
              Chung tay vì thành phố xanh: Báo cáo bãi rác tự phát, thùng rác quá tải hoặc rác thải nguy hại để kết nối với cộng đồng và đội ngũ vệ sinh môi trường xử lý kịp thời.
            </p>
          </div>

          <div style={{ display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' }}>
            <button
              type="button"
              className="btn btn-secondary"
              style={{
                padding: '10px 16px',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                fontWeight: 600,
                borderColor: '#7c3aed',
                color: '#7c3aed',
                background: 'rgba(124, 58, 237, 0.08)'
              }}
              onClick={handleSyncNews}
              disabled={isSyncingNews}
            >
              <span style={{ display: 'inline-block', animation: isSyncingNews ? 'spin 1s linear infinite' : 'none' }}>📰</span>
              <span>{isSyncingNews ? 'Đang Quét Báo Chí...' : 'Quét Tin Rác Báo Chí & MXH'}</span>
            </button>

            <button
              type="button"
              className="btn btn-primary"
              style={{ padding: '10px 18px', display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 600 }}
              onClick={() => {
                setShowReportModal(true);
                handleGetCurrentLocation();
              }}
            >
              <span>📸</span>
              <span>Báo Cáo Điểm Rác Mới</span>
            </button>
          </div>
        </div>

        {/* Smart City Statistics Counter */}
        {stats && (
          <div className="smart-city-stats-grid" style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
            gap: '12px',
            marginTop: '20px',
            paddingTop: '16px',
            borderTop: '1px solid var(--border)'
          }}>
            <div className="stat-pill">
              <span className="stat-pill-label">📍 Tổng điểm báo cáo</span>
              <span className="stat-pill-val">{stats.total_reports}</span>
            </div>
            <div className="stat-pill">
              <span className="stat-pill-label">⏳ Đang chờ dọn dẹp</span>
              <span className="stat-pill-val" style={{ color: '#f59e0b' }}>{stats.pending_reports}</span>
            </div>
            <div className="stat-pill">
              <span className="stat-pill-label">🚨 Khẩn cấp / Y tế</span>
              <span className="stat-pill-val" style={{ color: '#ef4444' }}>{stats.critical_reports}</span>
            </div>
            <div className="stat-pill">
              <span className="stat-pill-label">🚚 Đang điều phối</span>
              <span className="stat-pill-val" style={{ color: '#3b82f6' }}>{stats.in_progress_reports}</span>
            </div>
            <div className="stat-pill">
              <span className="stat-pill-label">✅ Tỉ lệ đã dọn sạch</span>
              <span className="stat-pill-val" style={{ color: '#10b981' }}>{stats.resolution_rate}%</span>
            </div>
          </div>
        )}
      </div>

      {/* Control Toolbar: Filters, Map Layer Toggle & City Jumps */}
      <div className="litter-map-controls-card">
        <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'center', gap: '12px' }}>
          {/* Source Filter: Citizen vs News vs Social */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-muted)' }}>Nguồn:</span>
            <div className="xai-view-tabs">
              <button
                type="button"
                className={`xai-tab-btn ${sourceFilter === 'all' ? 'active' : ''}`}
                onClick={() => setSourceFilter('all')}
              >
                Tất cả nguồn
              </button>
              <button
                type="button"
                className={`xai-tab-btn ${sourceFilter === 'citizen' ? 'active' : ''}`}
                onClick={() => setSourceFilter('citizen')}
              >
                👤 Người dân
              </button>
              <button
                type="button"
                className={`xai-tab-btn ${sourceFilter === 'news' ? 'active' : ''}`}
                onClick={() => setSourceFilter('news')}
                style={{ color: sourceFilter === 'news' ? '#ffffff' : '#7c3aed' }}
              >
                📰 Báo chí
              </button>
              <button
                type="button"
                className={`xai-tab-btn ${sourceFilter === 'social' ? 'active' : ''}`}
                onClick={() => setSourceFilter('social')}
                style={{ color: sourceFilter === 'social' ? '#ffffff' : '#0284c7' }}
              >
                📱 MXH
              </button>
            </div>
          </div>

          {/* GPS Proximity / Radius Filter */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-muted)' }}>Bán kính GPS:</span>
            <div className="xai-view-tabs">
              <button
                type="button"
                className={`xai-tab-btn ${radiusFilter === 'all' ? 'active' : ''}`}
                onClick={() => setRadiusFilter('all')}
              >
                Toàn quốc
              </button>
              <button
                type="button"
                className={`xai-tab-btn ${radiusFilter === 5 ? 'active' : ''}`}
                onClick={() => {
                  if (!userCoords) handleGetCurrentLocation();
                  setRadiusFilter(5);
                }}
              >
                🎯 &lt; 5 km
              </button>
              <button
                type="button"
                className={`xai-tab-btn ${radiusFilter === 15 ? 'active' : ''}`}
                onClick={() => {
                  if (!userCoords) handleGetCurrentLocation();
                  setRadiusFilter(15);
                }}
              >
                🎯 &lt; 15 km
              </button>
              <button
                type="button"
                className={`xai-tab-btn ${radiusFilter === 30 ? 'active' : ''}`}
                onClick={() => {
                  if (!userCoords) handleGetCurrentLocation();
                  setRadiusFilter(30);
                }}
              >
                🎯 &lt; 30 km
              </button>
            </div>
          </div>

          {/* Status Filter */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-muted)' }}>Trạng thái:</span>
            <div className="xai-view-tabs">
              <button
                type="button"
                className={`xai-tab-btn ${statusFilter === 'all' ? 'active' : ''}`}
                onClick={() => setStatusFilter('all')}
              >
                Tất cả ({reports.length})
              </button>
              <button
                type="button"
                className={`xai-tab-btn ${statusFilter === 'pending' ? 'active' : ''}`}
                onClick={() => setStatusFilter('pending')}
              >
                ⏳ Chờ xử lý
              </button>
              <button
                type="button"
                className={`xai-tab-btn ${statusFilter === 'in_progress' ? 'active' : ''}`}
                onClick={() => setStatusFilter('in_progress')}
              >
                🚚 Đang dọn
              </button>
              <button
                type="button"
                className={`xai-tab-btn ${statusFilter === 'resolved' ? 'active' : ''}`}
                onClick={() => setStatusFilter('resolved')}
              >
                ✅ Đã sạch
              </button>
            </div>
          </div>

          {/* Map Layer Switcher: Google Street / Google Satellite / OSM */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-muted)' }}>Bản đồ:</span>
            <div className="xai-view-tabs">
              <button
                type="button"
                className={`xai-tab-btn ${activeTileType === 'google_streets' ? 'active' : ''}`}
                onClick={() => switchTileLayer('google_streets')}
              >
                🗺️ Đường Phố
              </button>
              <button
                type="button"
                className={`xai-tab-btn ${activeTileType === 'google_satellite' ? 'active' : ''}`}
                onClick={() => switchTileLayer('google_satellite')}
              >
                🛰️ Vệ Tinh
              </button>
              <button
                type="button"
                className={`xai-tab-btn ${activeTileType === 'osm' ? 'active' : ''}`}
                onClick={() => switchTileLayer('osm')}
              >
                🌍 OSM
              </button>
            </div>
          </div>

          {/* Quick City Jumps */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-muted)' }}>Xem nhanh:</span>
            <button
              type="button"
              className="city-jump-btn"
              onClick={handleGetCurrentLocation}
              disabled={isGettingLocation}
            >
              📍 {isGettingLocation ? 'Đang dò GPS...' : 'Vị trí của tôi'}
            </button>
            <button
              type="button"
              className="city-jump-btn"
              onClick={() => jumpToCity(21.0285, 105.8542, 13)}
            >
              Hà Nội
            </button>
            <button
              type="button"
              className="city-jump-btn"
              onClick={() => jumpToCity(16.0678, 108.2208, 13)}
            >
              Đà Nẵng
            </button>
            <button
              type="button"
              className="city-jump-btn"
              onClick={() => jumpToCity(10.7769, 106.7009, 13)}
            >
              TP. Hồ Chí Minh
            </button>
            <button
              type="button"
              className="city-jump-btn"
              onClick={() => jumpToCity(16.047079, 108.206230, 6)}
            >
              Toàn quốc
            </button>
          </div>
        </div>
      </div>

      {/* Active Navigation Bar with Road Route Metrics */}
      {activeRoutingReport && (
        <div style={{
          background: 'linear-gradient(135deg, rgba(37, 99, 235, 0.12), rgba(16, 185, 129, 0.12))',
          border: '1px solid #3b82f6',
          borderRadius: '10px',
          padding: '12px 18px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '12px',
          boxShadow: '0 4px 12px rgba(37, 99, 235, 0.1)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <span style={{ fontSize: '26px' }}>🚗</span>
            <div>
              <div style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-main)' }}>
                Tuyến đường thực tế: {activeRoutingReport.address}
              </div>
              {routingMetrics && (
                <div style={{ fontSize: '12.5px', color: '#2563eb', fontWeight: 600, marginTop: '2px', display: 'flex', gap: '14px', flexWrap: 'wrap' }}>
                  <span>📏 Quãng đường bộ: <strong>{routingMetrics.distanceStr}</strong></span>
                  <span>⏱️ Dự kiến đi xe: <strong>~{routingMetrics.durationStr}</strong></span>
                </div>
              )}
            </div>
          </div>

          <div style={{ display: 'flex', gap: '8px' }}>
            <a
              href={`https://www.google.com/maps/dir/?api=1&origin=${userCoords?.lat || ''},${userCoords?.lng || ''}&destination=${activeRoutingReport.latitude},${activeRoutingReport.longitude}`}
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-primary"
              style={{ fontSize: '12px', padding: '8px 14px', display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 600 }}
            >
              <span>🧭</span>
              <span>Mở Dẫn Đường Google Maps</span>
            </a>
            <button
              type="button"
              className="btn btn-secondary"
              style={{ fontSize: '12px', padding: '8px 14px' }}
              onClick={() => {
                if (routeLineRef.current && mapInstanceRef.current) {
                  mapInstanceRef.current.removeLayer(routeLineRef.current);
                  routeLineRef.current = null;
                }
                setActiveRoutingReport(null);
                setRoutingMetrics(null);
              }}
            >
              Đóng tuyến đường
            </button>
          </div>
        </div>
      )}

      {/* GPS Radar Proximity Alert Banner */}
      {userCoords && reports && reports.length > 0 && (
        <div style={{
          background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(124, 58, 237, 0.08))',
          border: '1px solid rgba(59, 130, 246, 0.3)',
          borderRadius: '10px',
          padding: '10px 16px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '10px',
          fontSize: '13px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ fontSize: '20px' }}>📡</span>
            <div>
              <span style={{ fontWeight: 700, color: 'var(--text-main)' }}>Radar định vị GPS:</span>
              <span style={{ color: 'var(--text-muted)', marginLeft: '6px' }}>
                Đang quét trong bán kính {radiusFilter === 'all' ? '15 km' : `${radiusFilter} km`}. 
                Tìm thấy <strong>{reports.length}</strong> điểm rác xung quanh bạn.
              </span>
            </div>
          </div>
          <button
            type="button"
            className="btn btn-secondary"
            style={{ fontSize: '12px', padding: '6px 14px', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '6px' }}
            onClick={handleFocusNearestReport}
          >
            <span>🎯</span>
            <span>Xem Điểm Rác Gần Nhất</span>
          </button>
        </div>
      )}

      {/* Main Map Viewport */}
      <div className="litter-map-viewport-wrapper">
        <div
          ref={mapContainerRef}
          style={{
            width: '100%',
            height: '520px',
            borderRadius: 'var(--radius-lg, 12px)',
            border: '1px solid var(--border)',
            boxShadow: '0 4px 16px rgba(0,0,0,0.06)',
            zIndex: 1
          }}
        />

        {/* Map Legend Overlay */}
        <div className="litter-map-legend" style={{
          position: 'absolute',
          bottom: '20px',
          right: '20px',
          padding: '10px 14px',
          borderRadius: '8px',
          boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
          fontSize: '11px',
          zIndex: 999,
          display: 'flex',
          flexDirection: 'column',
          gap: '6px'
        }}>
          <div style={{ fontWeight: 700, marginBottom: '2px', color: 'var(--text-main)' }}>Chú giải điểm rác & Nguồn tin:</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--text-main)' }}>
            <span style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#ef4444' }}></span>
            <span>🚨 Khẩn cấp (Rác y tế / Nguy hại)</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--text-main)' }}>
            <span style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#7c3aed' }}></span>
            <span>📰 Báo chí phản ánh (Dân Trí, VnExpress...)</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--text-main)' }}>
            <span style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#0284c7' }}></span>
            <span>📱 Mạng xã hội cư dân</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--text-main)' }}>
            <span style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#f59e0b' }}></span>
            <span>⚠️ Điểm rác người dân báo</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--text-main)' }}>
            <span style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#3b82f6' }}></span>
            <span>🚚 Đang điều phối thu gom</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--text-main)' }}>
            <span style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#10b981' }}></span>
            <span>✅ Đã dọn dẹp sạch sẽ</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--text-main)' }}>
            <span style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#3b82f6', border: '2px solid #fff' }}></span>
            <span>📍 Vị trí GPS của bạn</span>
          </div>
        </div>
      </div>

      {/* Recent Hotspots List (Bottom Feed) */}
      <div className="litter-feed-section" style={{ marginTop: '24px' }}>
        <h3 style={{ fontSize: '16px', fontWeight: 700, marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-main)' }}>
          <span>📋</span>
          <span>Danh Sách Điểm Nóng Mới Nhất ({reports.length} điểm)</span>
        </h3>

        <div className="litter-feed-grid" style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
          gap: '12px'
        }}>
          {reports.map((item) => {
            const dist = userCoords ? calculateDistance(userCoords.lat, userCoords.lng, item.latitude, item.longitude) : null;
            return (
              <div
                key={item.id}
                className="history-card"
                style={{
                  cursor: 'pointer',
                  transition: 'transform 0.15s ease',
                  padding: '14px',
                  borderLeft: `4px solid ${
                    item.status === 'resolved' ? '#10b981' : item.severity === 'critical' ? '#ef4444' : '#f59e0b'
                  }`
                }}
                onClick={() => {
                  jumpToCity(item.latitude, item.longitude, 16);
                  drawRouteToHotspot(item.latitude, item.longitude, item);
                  window.scrollTo({ top: 320, behavior: 'smooth' });
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '6px' }}>
                  <span style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-main)' }}>
                    📍 {item.address}
                  </span>
                  <span className={`badge ${item.status === 'resolved' ? 'badge-safe' : item.severity === 'critical' ? 'badge-danger' : 'badge-warning'}`} style={{ fontSize: '10px' }}>
                    {item.status === 'resolved' ? 'ĐÃ DỌN' : item.severity.toUpperCase()}
                  </span>
                </div>
                {dist && (
                  <div style={{ fontSize: '11px', fontWeight: 600, color: '#10b981', marginBottom: '4px' }}>
                    📏 Cách bạn: {dist} &bull; Nhấp để vẽ đường đi
                  </div>
                )}
                <p style={{ fontSize: '12px', color: 'var(--text-muted)', margin: '4px 0 8px 0', lineHeight: 1.4 }}>
                  {item.description}
                </p>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '11px', color: 'var(--text-light)' }}>
                  <span>Báo cáo: {item.reported_by}</span>
                  <span>👍 {item.upvotes}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Modal: Báo Cáo Điểm Rác Mới */}
      {showReportModal && (
        <div className="modal-backdrop" onClick={() => setShowReportModal(false)}>
          <div
            className="modal-card"
            style={{ maxWidth: '540px', width: '92%' }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h3 style={{ fontSize: '18px', fontWeight: 700, margin: 0, display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-main)' }}>
                <span>📸</span>
                <span>Báo Cáo Điểm Nóng Rác Thải</span>
              </h3>
              <button
                type="button"
                className="modal-close-btn"
                onClick={() => setShowReportModal(false)}
              >
                &times;
              </button>
            </div>

            {formError && (
              <div className="alert alert-danger" style={{ marginBottom: '12px', fontSize: '12px' }}>
                {formError}
              </div>
            )}

            <form onSubmit={handleReportSubmit}>
              {/* Photo Upload Area */}
              <div style={{ marginBottom: '14px' }}>
                <label style={{ fontWeight: 600, fontSize: '13px', display: 'block', marginBottom: '6px', color: 'var(--text-main)' }}>
                  Ảnh hiện trường bãi rác (Tùy chọn nhưng khuyến khích):
                </label>
                <div
                  className="dropzone"
                  style={{ minHeight: '120px', padding: '16px', cursor: 'pointer' }}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    style={{ display: 'none' }}
                    onChange={handleFileChange}
                  />
                  {reportPreview ? (
                    <div style={{ textAlign: 'center' }}>
                      <img
                        src={reportPreview}
                        alt="Preview"
                        style={{ maxHeight: '140px', maxWidth: '100%', borderRadius: '6px', objectFit: 'contain' }}
                      />
                      <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
                        Nhấp để chọn ảnh khác
                      </p>
                    </div>
                  ) : (
                    <div>
                      <span style={{ fontSize: '24px' }}>📷</span>
                      <p style={{ fontSize: '12px', fontWeight: 600, margin: '4px 0', color: 'var(--text-main)' }}>
                        Kéo thả ảnh hoặc nhấp để tải ảnh lên
                      </p>
                      <p style={{ fontSize: '11px', color: 'var(--text-muted)', margin: 0 }}>
                        AI sẽ tự động quét thành phần rác thải trong ảnh
                      </p>
                    </div>
                  )}
                </div>
              </div>

              {/* GPS Coordinates & Auto Location */}
              <div style={{ marginBottom: '14px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                  <label style={{ fontWeight: 600, fontSize: '13px', margin: 0, color: 'var(--text-main)' }}>
                    Tọa độ GPS <span style={{ color: '#ef4444' }}>*</span>:
                  </label>
                  <button
                    type="button"
                    className="city-jump-btn"
                    onClick={handleGetCurrentLocation}
                    disabled={isGettingLocation}
                  >
                    📍 {isGettingLocation ? 'Đang lấy tọa độ...' : 'Lấy vị trí GPS của tôi'}
                  </button>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                  <input
                    type="text"
                    className="form-input"
                    placeholder="Vĩ độ (Lat, e.g. 21.0285)"
                    value={reportLat}
                    onChange={(e) => setReportLat(e.target.value)}
                    required
                  />
                  <input
                    type="text"
                    className="form-input"
                    placeholder="Kinh độ (Lng, e.g. 105.8542)"
                    value={reportLng}
                    onChange={(e) => setReportLng(e.target.value)}
                    required
                  />
                </div>
                <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
                  Mẹo: Bạn cũng có thể nhấp trực tiếp vào bất kỳ vị trí nào trên bản đồ để lấy tọa độ.
                </p>
              </div>

              {/* Address / Location */}
              <div style={{ marginBottom: '14px' }}>
                <label style={{ fontWeight: 600, fontSize: '13px', display: 'block', marginBottom: '4px', color: 'var(--text-main)' }}>
                  Địa chỉ hoặc Mô tả điểm nhận diện:
                </label>
                <input
                  type="text"
                  className="form-input"
                  placeholder="Ví dụ: Trước ngõ 12 đường Trần Phú, P. Mộ Lao, Hà Đông..."
                  value={reportAddress}
                  onChange={(e) => setReportAddress(e.target.value)}
                />
              </div>

              {/* Severity Selection */}
              <div style={{ marginBottom: '14px' }}>
                <label style={{ fontWeight: 600, fontSize: '13px', display: 'block', marginBottom: '4px', color: 'var(--text-main)' }}>
                  Mức độ nghiêm trọng:
                </label>
                <select
                  className="form-input"
                  value={reportSeverity}
                  onChange={(e) => setReportSeverity(e.target.value)}
                >
                  <option value="low">🟢 Mức độ thấp (Vài chai lọ vứt rải rác)</option>
                  <option value="medium">🟡 Trung bình (Đống rác tự phát / Thùng rác tràn)</option>
                  <option value="high">🟠 Nghiêm trọng (Bãi rác lớn bốc mùi, ô nhiễm nguồn nước)</option>
                  <option value="critical">🚨 Khẩn cấp (Có rác y tế, kim tiêm, pin/ắc quy nguy hại)</option>
                </select>
              </div>

              {/* Description */}
              <div style={{ marginBottom: '14px' }}>
                <label style={{ fontWeight: 600, fontSize: '13px', display: 'block', marginBottom: '4px', color: 'var(--text-main)' }}>
                  Mô tả tình trạng chi tiết:
                </label>
                <textarea
                  className="form-input"
                  rows={2}
                  placeholder="Mô tả cụ thể để đội thu gom dễ dàng tiếp cận..."
                  value={reportDesc}
                  onChange={(e) => setReportDesc(e.target.value)}
                />
              </div>

              {/* Reporter name */}
              <div style={{ marginBottom: '18px' }}>
                <label style={{ fontWeight: 600, fontSize: '13px', display: 'block', marginBottom: '4px', color: 'var(--text-main)' }}>
                  Người báo cáo:
                </label>
                <input
                  type="text"
                  className="form-input"
                  placeholder="Tên hoặc biệt danh của bạn (mặc định: Người dân)"
                  value={reportReporter}
                  onChange={(e) => setReportReporter(e.target.value)}
                />
              </div>

              <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end' }}>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => setShowReportModal(false)}
                >
                  Hủy
                </button>
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={isSubmitting}
                >
                  {isSubmitting ? 'Đang gửi...' : 'Gửi Báo Cáo (+50 Điểm)'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
