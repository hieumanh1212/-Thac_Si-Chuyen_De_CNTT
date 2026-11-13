@echo off
rem Khoi dong Hybrid RAG: mo 2 cua so (Backend + Frontend) roi mo trinh duyet.
rem Dong ung dung: dong 2 cua so do (hoac Ctrl+C trong tung cua so).
chcp 65001 >nul
set "ROOT=%~dp0"

if not exist "%ROOT%.venv\Scripts\python.exe" (
  echo [Loi] Khong tim thay .venv - hay cai dat theo README truoc.
  pause
  exit /b 1
)
if not exist "%ROOT%frontend\node_modules" (
  echo Chua co node_modules, dang chay npm install...
  pushd "%ROOT%frontend" && call npm install && popd
)

start "Hybrid RAG - Backend (8000)" /D "%ROOT%backend" cmd /k ""%ROOT%.venv\Scripts\python.exe" -m uvicorn api.main:app --port 8000 --reload"
start "Hybrid RAG - Frontend (5173)" /D "%ROOT%frontend" cmd /k "npm run dev"

echo Dang cho backend nap mo hinh (lan dau mat khoang 1 phut)...
set /a WAITED=0
:wait_backend
curl.exe -s -o nul -f http://127.0.0.1:8000/api/health && goto ready
if %WAITED% geq 180 goto ready
timeout /t 3 /nobreak >nul
set /a WAITED+=3
goto wait_backend

:ready
echo Backend san sang sau khoang %WAITED% giay. Dang mo trinh duyet...
start "" http://localhost:5173
timeout /t 3 /nobreak >nul
