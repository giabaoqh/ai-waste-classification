@echo off
setlocal
title AI Waste Recognition System

cd /d "%~dp0"

echo ============================================================
echo   AI WASTE RECOGNITION AND RECYCLING SYSTEM
echo ============================================================
echo.

if not exist "backend\.venv\Scripts\python.exe" (
    echo ERROR: Backend Python environment not found.
    pause
    exit /b 1
)

if not exist "frontend\package.json" (
    echo ERROR: Frontend package.json not found.
    pause
    exit /b 1
)

echo [1/3] Starting FastAPI backend (Listening on all interfaces 0.0.0.0)...
start "AI Backend - Port 8000" cmd /k "cd /d ""%~dp0"" && ""%~dp0backend\.venv\Scripts\python.exe"" -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload"

echo [2/3] Waiting for backend...
set "READY=0"

for /L %%i in (1,1,30) do (
    powershell -NoProfile -Command "try { $r = Invoke-RestMethod 'http://127.0.0.1:8000/api/health' -TimeoutSec 2; if ($r.status -eq 'ok' -and $r.ai_model_loaded -eq $true) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
    if not errorlevel 1 (
        set "READY=1"
        goto backend_ready
    )
    timeout /t 2 /nobreak >nul
)

:backend_ready
if "%READY%"=="0" (
    echo ERROR: Backend did not become ready within 60 seconds.
    echo Check the Backend window for errors.
    pause
    exit /b 1
)

echo Backend is ready.
echo.
echo [3/3] Starting React frontend (Listening on 0.0.0.0)...
start "AI Frontend - Port 5173" cmd /k "cd /d ""%~dp0frontend"" && npm run dev -- --host 0.0.0.0"

timeout /t 3 /nobreak >nul
start "" "http://localhost:5173"

echo.
echo ============================================================
echo   SYSTEM STARTUP COMPLETE!
echo.
echo   [Tren May Tinh]:
echo     Frontend: http://localhost:5173
echo     Backend:  http://localhost:8000
echo.
echo   [Tren Dien Thoai - Ket noi chung mang Wi-Fi]:
echo     Mo trinh duyet tren dien thoai va truy cap:
echo     👉 http://192.168.1.56:5173
echo ============================================================
pause
