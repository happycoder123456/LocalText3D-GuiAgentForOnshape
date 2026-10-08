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

rem Desktop + Start Menu shortcuts (idempotent), so the app launches like
rem any installed Windows application afterwards.
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$repo = '%CD%';" ^
  "$pyw = Join-Path $repo '.venv-agent\Scripts\pythonw.exe';" ^
  "$ico = Join-Path $repo 'agent_icon.ico';" ^
  "$desktop = [Environment]::GetFolderPath('Desktop');" ^
  "$menu = [Environment]::GetFolderPath('Programs');" ^
  "$ws = New-Object -ComObject WScript.Shell;" ^
  "$targets = @((Join-Path $desktop 'Onshape Agent.lnk'), (Join-Path $menu 'Onshape Agent.lnk'));" ^
  "foreach ($t in $targets) { if (-not (Test-Path $t)) { $lnk = $ws.CreateShortcut($t); $lnk.TargetPath = $pyw; $lnk.Arguments = '-m agent gui'; $lnk.WorkingDirectory = $repo; $lnk.IconLocation = ($ico + ',0'); $lnk.Description = 'Local GUI agent that 3D-models in Onshape'; $lnk.Save(); Write-Output ('created ' + $t) } }"

echo.
echo Setup complete.
echo   1. Install Ollama and run:  ollama pull qwen2.5vl:3b
echo   2. Launch "Onshape Agent" from your Desktop or Start Menu (or double-click "Onshape Agent Desktop.bat")
echo.
pause
