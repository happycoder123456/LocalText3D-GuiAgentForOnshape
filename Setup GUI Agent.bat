@echo off
rem Setup the Onshape GUI Agent (one time): installs deps + Chromium into .venv-agent
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher "py" not found. Install Python 3.13 from python.org first.
  pause
  exit /b 1
)

if not exist .venv-agent (
  echo Creating .venv-agent...
  py -3.13 -m venv .venv-agent || exit /b 1
)

.venv-agent\Scripts\python.exe -m pip install --upgrade pip || exit /b 1
.venv-agent\Scripts\python.exe -m pip install -r requirements.txt || exit /b 1
.venv-agent\Scripts\playwright install chromium || exit /b 1

echo.
echo Setup complete.
echo   1. Install Ollama and run:  ollama pull qwen2.5vl:3b
echo   2. Start the sidecar with:  Start GUI Agent.bat
echo.
pause
