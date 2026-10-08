@echo off
rem ============================================================
rem  Onshape Agent - double-click launcher (Windows)
rem
rem  First run: installs everything into .venv-agent, then
rem  creates "Onshape Agent" shortcuts on your Desktop and in
rem  the Start Menu. After that, launch it like any app from
rem  those shortcuts - you never need this file again.
rem ============================================================
setlocal EnableExtensions
cd /d "%~dp0"

set "PYW=.venv-agent\Scripts\pythonw.exe"
set "PY=.venv-agent\Scripts\python.exe"

if exist "%PYW%" goto shortcuts

echo ==============================================
echo    Onshape Agent - first run setup
echo ==============================================
echo Creating .venv-agent and installing dependencies (this can take a few minutes)...
where py >nul 2>nul
if not errorlevel 1 (
  py -3.13 -m venv .venv-agent
) else (
  python -m venv .venv-agent
)
if not exist "%PY%" goto failed
"%PY%" -m pip install --upgrade pip >nul
"%PY%" -m pip install -r requirements.txt
"%PY%" -m playwright install chromium
if not exist "%PYW%" goto failed

:shortcuts
rem Create Desktop + Start Menu shortcuts so the app behaves like any
rem installed Windows application. Idempotent: skipped if they exist.
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$repo = '%CD%';" ^
  "$pyw = Join-Path $repo '.venv-agent\Scripts\pythonw.exe';" ^
  "$ico = Join-Path $repo 'agent_icon.ico';" ^
  "$desktop = [Environment]::GetFolderPath('Desktop');" ^
  "$menu = [Environment]::GetFolderPath('Programs');" ^
  "$ws = New-Object -ComObject WScript.Shell;" ^
  "$targets = @((Join-Path $desktop 'Onshape Agent.lnk'), (Join-Path $menu 'Onshape Agent.lnk'));" ^
  "foreach ($t in $targets) { if (-not (Test-Path $t)) { $lnk = $ws.CreateShortcut($t); $lnk.TargetPath = $pyw; $lnk.Arguments = '-m agent gui'; $lnk.WorkingDirectory = $repo; $lnk.IconLocation = ($ico + ',0'); $lnk.Description = 'Local GUI agent that 3D-models in Onshape'; $lnk.Save(); Write-Output ('created ' + $t) } }" ^
  >nul

:launch
rem start with an explicit title so the window never flashes a console.
start "Onshape Agent" "%CD%\%PYW%" -m agent gui
exit /b 0

:failed
echo.
echo ERROR: could not start the Onshape Agent.
echo.
echo  - Install Python 3.13 from python.org, then double-click this file again.
echo  - Or run "Start Onshape Agent.bat" to see the full error output.
echo.
pause
exit /b 1
