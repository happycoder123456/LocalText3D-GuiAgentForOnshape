@echo off
rem Launch the Onshape Agent desktop app (tkinter GUI).
rem The desktop shortcut "Onshape Agent.lnk" points straight at pythonw.exe;
rem this file is the visible / first-run path documented in the README.
setlocal EnableExtensions
cd /d "%~dp0"

set "PYW=.venv-agent\Scripts\pythonw.exe"
set "PY=.venv-agent\Scripts\python.exe"

if exist "%PYW%" goto launch

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
