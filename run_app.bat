@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
title AI Waste Recognition System - Khởi Chạy Hệ Thống

cd /d "%~dp0"

echo ==============================================================================
echo    HỆ THỐNG NHẬN DIỆN RÁC THẢI VÀ GỢI Ý TÁI CHẾ THÔNG MINH (AI-WASTE)
echo    MobileNetV3-Large + FastAPI Backend + React / Vite Web & Mobile
echo ==============================================================================
echo.

:: 1. Kiểm tra môi trường Python Backend
if not exist "backend\.venv\Scripts\python.exe" (
    echo [LỖI] Không tìm thấy môi trường Python tại backend\.venv\
    echo Vui lòng tạo môi trường và cài đặt thư viện:
    echo   python -m venv backend\.venv
    echo   backend\.venv\Scripts\pip install -r backend\requirements.txt
    echo.
    pause
    exit /b 1
)

:: 2. Kiểm tra mã nguồn Frontend
if not exist "frontend\package.json" (
    echo [LỖI] Không tìm thấy thư mục frontend hoặc package.json!
    pause
    exit /b 1
)

:: Kiểm tra node_modules, nếu chưa cài thì tự động cài đặt
if not exist "frontend\node_modules" (
    echo [THÔNG BÁO] Đang cài đặt thư viện Frontend (npm install)...
    cd frontend && call npm install && cd ..
)

:: 3. Tự động nhận diện địa chỉ IP Wi-Fi / LAN thực tế
set "LOCAL_IP=127.0.0.1"
for /f "tokens=2 delims=:" %%A in ('ipconfig ^| find "IPv4"') do (
    for /f "tokens=1" %%B in ("%%A") do (
        if not "%%B"=="127.0.0.1" (
            set "LOCAL_IP=%%B"
        )
    )
)

echo [1/3] Đang khởi động Backend API (FastAPI - Port 8000)...
start "AI Backend [FastAPI - Port 8000]" cmd /k "chcp 65001 >nul && cd /d "%~dp0" && title AI Backend [Port 8000] && echo ======================================== && echo   AI BACKEND FASTAPI (Port 8000) && echo ======================================== && echo. && backend\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload"

echo [2/3] Đang khởi động Frontend Web (React/Vite - Port 5173)...
start "AI Frontend [React/Vite - Port 5173]" cmd /k "chcp 65001 >nul && cd /d "%~dp0frontend" && title AI Frontend [Port 5173] && echo ======================================== && echo   AI FRONTEND REACT/VITE (Port 5173) && echo ======================================== && echo. && npm run dev"

echo [3/3] Đang mở trình duyệt giao diện hệ thống...
timeout /t 3 /nobreak >nul 2>&1
if errorlevel 1 ping 127.0.0.1 -n 4 >nul

start "" "http://localhost:5173"

echo.
echo ==============================================================================
echo   ĐÃ KHỞI CHẠY THÀNH CÔNG ĐỒNG THỜI CẢ BACKEND VÀ FRONTEND!
echo ==============================================================================
echo.
echo   [1] TRÊN MÁY TÍNH NÀY:
echo       * Giao diện người dùng: http://localhost:5173
echo       * Tài liệu API Swagger: http://localhost:8000/docs
echo       * Kiểm tra sức khỏe:    http://localhost:8000/api/health
echo.
echo   [2] TRÊN ĐIỆN THOẠI (Kết nối chung mạng Wi-Fi):
echo       * Mở trình duyệt trên điện thoại và truy cập:
echo         👉 http://!LOCAL_IP!:5173
echo.
echo   [Lưu ý quan trọng]:
echo   - Hai cửa sổ terminal (Backend và Frontend) đang chạy song song.
echo   - Không đóng 2 cửa sổ đó trong suốt quá trình thử nghiệm hệ thống.
echo   - Nhấn bất kỳ phím nào để đóng cửa sổ quản lý này khi cần.
echo ==============================================================================
echo.
pause
