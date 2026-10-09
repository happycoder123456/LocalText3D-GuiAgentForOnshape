@echo off
rem Inno payload: finalize the install (venv, deps, Chromium).
rem %1 = install directory (passed as "{app}" resolved by [Run] WorkingDir,
rem but Inno gives us the dir via the current directory, so use %CD%).
setlocal EnableExtensions
rem DEST is the app install root: the parent of this payload\ folder.
set "DEST=%~dp0..\"

rem --- 1. Python 3.13 check -------------------------------------------
where py >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python 3.13 is required. Install from python.org, tick
  echo "Add python.exe to PATH", then reinstall this app.
  exit /b 1
)
py -3.13 --version >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python 3.13 is required ^(found only other versions^).
  exit /b 1
)

rem --- 2. venv + pip ---------------------------------------------------
if not exist "%DEST%.venv" (
  py -3.13 -m venv "%DEST%.venv" || exit /b 1
)
"%DEST%.venv\Scripts\python.exe" -m pip install --upgrade pip >nul 2>nul
"%DEST%.venv\Scripts\python.exe" -m pip install -r "%DEST%requirements.txt" || exit /b 1

rem --- 3. Chromium for the agent browser --------------------------------
"%DEST%.venv\Scripts\playwright.exe" install chromium || echo WARN: Chromium install failed; the agent will fall back to installed Chrome/Edge.

exit /b 0
