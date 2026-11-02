@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"
set "ROOT=%CD%"
set "BACKEND_DIR=%ROOT%\backend"
set "FRONTEND_DIR=%ROOT%\frontend"

where py >nul 2>nul
if errorlevel 1 (
  echo Python was not found on PATH. Install Python 3.10+ and try again.
  pause
  exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
  echo Node.js and npm were not found on PATH. Install Node.js 18+ and try again.
  pause
  exit /b 1
)

if not exist "%BACKEND_DIR%\.venv\Scripts\python.exe" (
  echo Creating Python virtual environment for the backend...
  py -3 -m venv "%BACKEND_DIR%\.venv"
)

call "%BACKEND_DIR%\.venv\Scripts\activate.bat"
python -m pip install --upgrade pip >nul
python -m pip install -r "%BACKEND_DIR%\requirements.txt"

if not exist "%BACKEND_DIR%\.env" if exist "%BACKEND_DIR%\.env.example" (
  copy "%BACKEND_DIR%\.env.example" "%BACKEND_DIR%\.env" >nul
)

echo Checking local Ollama service...
curl.exe --fail --silent http://127.0.0.1:11434/api/tags >nul
if errorlevel 1 (
  echo Ollama is not reachable at http://127.0.0.1:11434.
  echo Start Ollama before beginning an interview. The app will report the actual connection status.
)

start "AI Interview Coach Backend" /D "%BACKEND_DIR%" "%BACKEND_DIR%\.venv\Scripts\python.exe" app.py

timeout /t 3 /nobreak > nul

cd /d "%FRONTEND_DIR%"
if not exist "node_modules" (
  echo Installing frontend dependencies...
  npm install
)

start "AI Interview Coach Frontend" /D "%FRONTEND_DIR%" cmd /k "npm run dev -- --host 127.0.0.1"

start "" http://127.0.0.1:5173

echo.
echo AI Interview Coach is starting...
echo Backend: http://127.0.0.1:5000
echo Frontend: http://127.0.0.1:5173
echo.
echo Ollama model: qwen2.5:3b
echo Confirm the Ollama status in the app before starting an interview.
echo.
pause
exit /b 0
