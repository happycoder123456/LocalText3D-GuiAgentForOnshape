@echo off
setlocal EnableExtensions
title Onshape GUI Agent
cd /d "%~dp0"

rem Must be the PROJECT folder (agent\cli.py sits next to this file), not the dataset folder.
if not exist "agent\cli.py" (
  echo.
  echo ERROR: This launcher is not in the OnshapeGuiAgent project folder.
  echo Expected to find agent\cli.py next to this file.
  echo If you cloned/unzipped the project somewhere, move this file there.
  echo.
  set /p "=Press Enter to close... "
  exit /b 1
)

echo ==============================================
echo    Onshape GUI Agent - starting everything
echo ==============================================

set "PY=.venv-agent\Scripts\python.exe"

rem --- [1/4] Python environment (auto-setup on first run) ----------------
if exist "%PY%" goto env_ok
echo [1/4] First run: creating .venv-agent and installing dependencies...
where py >nul 2>nul
if not errorlevel 1 (
  py -3.13 -m venv .venv-agent
) else (
  python -m venv .venv-agent
)
if not exist "%PY%" (
  echo      ERROR: could not create the virtual environment.
  echo      Install Python 3.13 from python.org, then double-click this again.
  set /p "=Press Enter to close... "
  exit /b 1
)
"%PY%" -m pip install --upgrade pip >nul
"%PY%" -m pip install -r requirements.txt
echo      Downloading Chromium (may fail on some networks - that is OK,
echo      the agent falls back to your installed Chrome/Edge automatically^):
"%PY%" -m playwright install chromium
:env_ok
echo [1/4] Python environment: OK

rem --- [2/4] Ollama ------------------------------------------------------
echo [2/4] Checking Ollama on 127.0.0.1:11434...
curl -s --max-time 2 http://127.0.0.1:11434/api/tags >nul 2>&1
if not errorlevel 1 goto ollama_ok
where ollama >nul 2>nul
if errorlevel 1 goto ollama_missing
echo      Starting Ollama...
start "Ollama" /min ollama serve
set /a _tries=0
:wait_ollama
ping -n 2 127.0.0.1 >nul
set /a _tries+=1
curl -s --max-time 2 http://127.0.0.1:11434/api/tags >nul 2>&1
if not errorlevel 1 goto ollama_ok
if %_tries% lss 20 goto wait_ollama
echo      WARNING: Ollama did not start. Vision features will not work.
goto ollama_done
:ollama_missing
echo      WARNING: Ollama not found. Install it from https://ollama.com
echo      then double-click this file again.
goto ollama_done
:ollama_ok
echo [2/4] Ollama: OK
:ollama_done

rem --- [3/4] Sidecar -----------------------------------------------------
echo [3/4] Checking sidecar on 127.0.0.1:8767...
curl -s --max-time 2 http://127.0.0.1:8767/status >nul 2>&1
if not errorlevel 1 goto sidecar_ok
echo      Starting sidecar in a minimized window...
start "Onshape Agent sidecar" /min cmd /k ""%CD%\%PY%" -m agent serve"
set /a _tries=0
:wait_sidecar
ping -n 2 127.0.0.1 >nul
set /a _tries+=1
curl -s --max-time 2 http://127.0.0.1:8767/status >nul 2>&1
if not errorlevel 1 goto sidecar_ok
if %_tries% lss 20 goto wait_sidecar
echo      ERROR: sidecar did not start. See the "Onshape Agent sidecar" window.
goto menu
:sidecar_ok
echo [3/4] Sidecar: OK  (http://127.0.0.1:8767)

rem --- [4/4] Vision model -------------------------------------------------
echo [4/4] Checking vision model for Ollama...
ollama list 2>nul | findstr /i /c:"qwen2.5vl" /c:"llava" /c:"llama3.2-vision" /c:"moondream" /c:"minicpm" /c:"qwen3-vl" >nul
if not errorlevel 1 goto model_ok
echo      No vision model found. Installing qwen2.5vl:3b (about 3.2 GB^)...
ollama pull qwen2.5vl:3b
if errorlevel 1 (
  echo      WARNING: model download failed. Retry later with:  ollama pull qwen2.5vl:3b
)
:model_ok
echo [4/4] Vision model: OK

rem --- Menu ---------------------------------------------------------------
:menu
echo.
echo ==============================================
echo   Ready. Actions:
echo ==============================================
echo   [L] Sign in to Onshape (do this once first)
echo   [R] Run a modeling goal
echo   [P] Plan a goal without running it
echo   [V] Learn techniques from a YouTube URL
echo   [S] Show status
echo   [Q] Quit this menu (sidecar keeps running)
echo.
set "SEL="
set /p "SEL=Choose [L/R/P/V/S/Q]: "
if errorlevel 1 goto do_quit
rem Compare only the first character so a stray CR from piped input can't break parsing.
if /i "%SEL:~0,1%"=="L" goto do_login
if /i "%SEL:~0,1%"=="R" goto do_run
if /i "%SEL:~0,1%"=="P" goto do_plan
if /i "%SEL:~0,1%"=="V" goto do_video
if /i "%SEL:~0,1%"=="S" goto do_status
if /i "%SEL:~0,1%"=="Q" goto do_quit
goto menu

:do_login
echo.
echo Opening the Onshape window - sign in there, then close it with the X button.
"%PY%" -m agent login <nul
goto menu

:do_run
set "GOAL="
set /p "GOAL=Enter your modeling goal: "
if not defined GOAL goto menu
echo.
"%PY%" -m agent run "%GOAL%" <nul
set /p "=Press Enter to continue... "
echo.
goto menu

:do_plan
set "GOAL="
set /p "GOAL=Enter your modeling goal: "
if not defined GOAL goto menu
echo.
"%PY%" -m agent plan "%GOAL%" <nul
set /p "=Press Enter to continue... "
echo.
goto menu

:do_video
set "VIDURL="
set /p "VIDURL=Paste a YouTube URL: "
if not defined VIDURL goto menu
echo.
"%PY%" -m agent learn --url "%VIDURL%" <nul
set /p "=Press Enter to continue... "
echo.
goto menu

:do_status
echo.
"%PY%" -m agent status <nul
set /p "=Press Enter to continue... "
echo.
goto menu

:do_quit
echo.
echo Sidecar keeps running. Next time just double-click this file again.
exit /b 0
