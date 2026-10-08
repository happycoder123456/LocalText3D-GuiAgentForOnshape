@echo off
rem Start the Onshape GUI Agent sidecar on http://127.0.0.1:8767
setlocal
cd /d "%~dp0"

if not exist .venv-agent\Scripts\python.exe (
  echo .venv-agent missing. Run "Setup GUI Agent.bat" first.
  pause
  exit /b 1
)

echo Starting sidecar on http://127.0.0.1:8767 (Ctrl+C to quit)...
.venv-agent\Scripts\python.exe -m agent serve
pause
