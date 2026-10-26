@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
title AI Waste Recognition System

cd /d "%~dp0"

echo ==============================================================================
echo    HE THONG NHAN DIEN RAC THAI VA GOI Y TAI CHE THONG MINH - AI WASTE
echo    MobileNetV3-Large + FastAPI Backend + React / Vite Web and Mobile
echo ==============================================================================
echo.

REM 1. Kiem tra moi truong Python Backend
if not exist "backend\.venv\Scripts\python.exe" (
    echo [LOI] Khong tim thay moi truong Python tai backend\.venv\
    echo Vui long tao moi truong va cai dat thu vien:
    echo   python -m venv backend\.venv
    echo   backend\.venv\Scripts\pip install -r backend\requirements.txt
    echo.
    pause
    exit /b 1
)

REM 2. Kiem tra ma nguon Frontend
if not exist "frontend\package.json" (
    echo [LOI] Khong tim thay thu muc frontend hoac package.json!
    pause
    exit /b 1
)

if not exist "frontend\node_modules" (
    echo [THONG BAO] Dang cai dat thu vien Frontend qua npm install...
    cd /d "%~dp0frontend"
    call npm install
    cd /d "%~dp0"
)

REM 3. Tu dong tim dia chi IP Wi-Fi hoac LAN
set "LOCAL_IP=127.0.0.1"
for /f "tokens=2 delims=:" %%A in ('ipconfig ^| find "IPv4"') do (
    for /f "tokens=1" %%B in ("%%A") do (
        if not "%%B"=="127.0.0.1" set "LOCAL_IP=%%B"
    )
)

echo [1/3] Dang khoi dong Backend API - FastAPI tren Port 8000...
start "AI Backend - Port 8000" cmd /k "chcp 65001 >nul & cd /d "%~dp0" & echo ======================================== & echo   AI BACKEND FASTAPI - Port 8000 & echo ======================================== & echo. & backend\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload"

echo [2/3] Dang khoi dong Frontend Web - React Vite tren Port 5173...
start "AI Frontend - Port 5173" cmd /k "chcp 65001 >nul & cd /d "%~dp0frontend" & echo ======================================== & echo   AI FRONTEND REACT VITE - Port 5173 & echo ======================================== & echo. & npm run dev"

echo [3/3] Dang mo trinh duyet den giao dien he thong...
timeout /t 3 /nobreak >nul 2>&1
if errorlevel 1 ping 127.0.0.1 -n 4 >nul

start "" "http://localhost:5173"

echo.
echo ==============================================================================
echo   DA KHOI CHAY THANH CONG CA BACKEND VA FRONTEND!
echo ==============================================================================
echo.
echo   [1] TREN MAY TINH NAY:
echo       * Giao dien Web:        http://localhost:5173
echo       * Backend API Swagger:  http://localhost:8000/docs
echo       * Kiem tra he thong:    http://localhost:8000/api/health
echo.
echo   [2] TREN DIEN THOAI (Ket noi cung mang Wi-Fi):
echo       * Mo trinh duyet tren dien thoai va truy cap:
echo         http://!LOCAL_IP!:5173
echo.
echo   [Luu y]:
echo   - Hai cua so terminal Backend va Frontend dang chay song song.
echo   - Khong dong 2 cua so do trong suot qua trinh su dung he thong.
echo   - Nhan phim bat ky de dong cua so quan ly nay khi can.
echo ==============================================================================
echo.
pause
