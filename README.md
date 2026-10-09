# Onshape GUI Agent

A **local, free GUI agent that 3D-models in Onshape** — driven by an **Ollama** vision model running entirely on your machine. No cloud AI, no API keys, no paid service. It can **learn techniques from YouTube videos** and record/replay your own demos, in the spirit of the Blender GUI agent from [LocalText3D](https://github.com/happycoder123456/LocalText3D-Blender5.1GuiAgent).

The agent drives a **Playwright-controlled Chromium tab** on `cad.onshape.com`: it screenshots the page, asks the local vision model for the next action, and executes clicks/keys/drags in that tab. Only that tab is ever controlled.

```
OnshapeGuiAgent sidecar  -->  http://127.0.0.1:8767  -->  Ollama at http://127.0.0.1:11434
       (this computer only)                              (this computer only)
       Playwright Chromium on cad.onshape.com
```

Each person runs the sidecar and Ollama **on their own machine**. The sidecar binds `127.0.0.1` only; it refuses `0.0.0.0` and LAN binds, and rejects requests whose `Host`/`Origin` is not loopback.

## Share safely

- Share the code. Do **not** share remote access to a running install.
- Do **not** port-forward `8767` (or Ollama's `11434`).
- The browser profile in the dataset folder holds your Onshape login — never commit or send it (it is gitignored).

## Requirements

- Python 3.13 (64-bit) from [python.org](https://www.python.org/downloads/)
- [Ollama](https://ollama.com) with a vision model:
  ```powershell
  ollama pull qwen2.5vl:3b
  ```
  `llama3.2-vision` and `llava` also work; the agent auto-picks the best installed vision model.
- A Chromium browser. Install Playwright's own:
  ```powershell
  pip install -r requirements.txt
  playwright install chromium
  ```
  or skip that step — the driver falls back to your installed **Google Chrome**, then **Microsoft Edge**.
- [ffmpeg](https://ffmpeg.org) on your PATH for video learning (optional; frames fall back to OpenCV if you prefer `pip install opencv-python-headless`).

## Download the installer (recommended)

[![Download](https://img.shields.io/badge/Download-Onshape--Agent--Setup.exe-2ea44f?style=for-the-badge&logo=windows&logoColor=white)](https://github.com/happycoder123456/LocalText3D-GuiAgentForOnshape/releases/latest/download/Onshape-Agent-Setup.exe)

1. Click the button above — you get **`Onshape-Agent-Setup.exe`**, a normal Windows installer (not a zip).
2. Run it. It installs the app to your user folder, sets up Python packages and the agent browser, and creates a **Start Menu + Desktop shortcut** automatically.
3. If Python 3.13 is missing, install it once from [python.org](https://www.python.org/downloads/) (tick *Add python.exe to PATH*) and re-run the installer.
4. Launch **Onshape Agent** from the Start Menu; first launch takes a couple of minutes while the local browser is prepared.
5. In the app: **Session → Sign in to Onshape**.

> Uninstall any time via *Settings → Apps*, like a normal Windows app.

## Install it like a Windows app (from source)

You do **not** need to open a console, find scripts, or know where anything lives. From the folder you downloaded/cloned this project to:

1. **Install [Python 3.13](https://www.python.org/downloads/)** (64-bit, tick *Add python.exe to PATH* during install) and **[Ollama](https://ollama.com)**.
2. Double-click **`Onshape Agent Desktop.bat`** in this folder — once. It installs everything (Python packages, a browser, shortcuts) and then opens the app. Takes a few minutes on the first run only.
3. From then on, launch **“Onshape Agent”** like any installed program:
   - an **“Onshape Agent” icon on your Desktop**, and
   - **“Onshape Agent” in your Start Menu** (both are created for you in step 2 — you will never touch the `.bat` again).
4. In the app, go to the **Session** tab → **Sign in to Onshape** → sign in once in the browser window that opens. Done.

> Cloned via `git clone https://github.com/happycoder123456/LocalText3D-GuiAgentForOnshape.git`? Same steps — everything above works straight out of the clone.

**Where things live** (for reference only — the shortcuts handle all of this):

| Thing | Location |
| --- | --- |
| The app itself | the folder you cloned/downloaded (contains `agent\`); launched by `.venv-agent\Scripts\pythonw.exe -m agent gui` |
| Desktop / Start Menu shortcut | `Desktop\Onshape Agent.lnk` / `Start Menu\Programs\Onshape Agent.lnk` (auto-created) |
| Your Onshape login + data | `%USERPROFILE%\OnshapeGuiAgent\agent_dataset` (outside the program folder; gitignored, never uploaded) |
| Ollama models | managed by Ollama itself |

## Quick start (what the app looks like)

Launching **Onshape Agent** opens a dark-themed window with three tabs:

- **Model** — type a goal, preview the plan, run/stop the agent, watch live progress
- **Learn from video** — paste a YouTube URL and watch it distill techniques into local memory
- **Session** — sign in to Onshape, record/replay demos, see memory stats

The app starts the sidecar by itself; no console wrangling needed.

## Console / command-line use (optional)

Prefer a terminal? Everything is also drivable by hand — run every command **from this project folder** (the one containing `agent\`), not from `~/OnshapeGuiAgent` (that is only the data folder). In PowerShell, commands in the current folder need a leading `.\`:

```powershell
# 0. desktop app
.\.venv-agent\Scripts\python.exe -m agent gui

# 1. sidecar (leave running)
.\.venv-agent\Scripts\python.exe -m agent serve
# -> Onshape GUI Agent sidecar listening on http://127.0.0.1:8767
```

> If PowerShell says *The module '.venv-agent' could not be loaded*, you are missing the leading `.\` — type `.\.venv-agent\Scripts\python.exe ...`.

Open `http://127.0.0.1:8767/health` to confirm Ollama is reachable (`ollama_ok: true`) and see your installed models.

```powershell
# 2. sign in once (opens a browser window; sign in, close it with the X button)
.\.venv-agent\Scripts\python.exe -m agent login

# 3. run a modeling goal (opens a Chromium tab, then works)
.\.venv-agent\Scripts\python.exe -m agent run "make a 40mm flange with 6 bolt holes"

# 4. plan only — print the plan, don't execute
.\.venv-agent\Scripts\python.exe -m agent plan "revolve a sphere into a bowl"

# 5. learn techniques from a YouTube tutorial
.\.venv-agent\Scripts\python.exe -m agent learn --url "https://www.youtube.com/watch?v=..." --minutes 20

# 6. check a running sidecar
.\.venv-agent\Scripts\python.exe -m agent status
```

On Linux/macOS use `.venv-agent/bin/python` instead (or `./start.sh` for the sidecar).

## What it does

| Mode | What happens |
| --- | --- |
| **Run** (plan + vision) | Local text model writes a 3–14 step CAD plan; the vision model executes each step from screenshots |
| **Run** (`--no-plan`) | Pure vision: every step decided straight from the screen |
| **Record** | Watches the agent's own tab; saves your clicks/keys + screenshots as an episode |
| **Replay** | Replays the last recorded episode in a fresh tab |
| **Learn from video** | yt-dlp → ffmpeg keyframes + captions → local VLM extracts technique cards into memory |

Learned **techniques** (for example *circular pattern of holes*, *shell to 2 mm*) and recorded **skills** are stored in `memory.jsonl` / `episodes.jsonl` in your dataset folder. Techniques from videos are injected into **every plan** — even when the goal's wording doesn't overlap the technique names — so what the agent learns from a tutorial is actually used when you later ask it to model something.

## Learn from video

```powershell
.\.venv-agent\Scripts\python.exe -m agent learn --url "https://youtu.be/VIDEO_ID" --minutes 30
```

- Up to **1200 minutes (20 h)** of video per lesson.
- A local file works too but must live **inside** the agent dataset folder (`~/OnshapeGuiAgent/agent_dataset/videos` by default); paths outside it are refused.
- Concepts are stored as named techniques with summary, preconditions, and GUI steps — not whole-plan clones, so they graft onto different goals later.

## Agent API

Sidecar listens on `127.0.0.1:8767` only. Non-loopback binds are refused at startup.

- `GET /health` — Ollama reachability, models, preferred vision model, agent status
- `GET /status` — running / recording / step / goal / mode / plan preview
- `GET /skills` — recorded skills + learned concepts
- `POST /plan` — `{goal, planner_model?}` → plan preview
- `POST /agent/start` — `{goal, model?, planner_model?, max_steps?, use_plan?, run_mode?}`
- `POST /agent/stop`
- `POST /record/start` — `{goal?, interval_ms?}`
- `POST /record/stop` — `{goal?, success?}`
- `POST /replay/start` — `{goal?}`
- `POST /video/learn` — `{url?, path?, goal?, model?, max_minutes?}`

## Dataset

Default: `~/OnshapeGuiAgent/agent_dataset` (override with `ONSHAPE_AGENT_DATASET`).

```
agent_dataset/
  episodes.jsonl     # recorded demos
  memory.jsonl       # skills + video concepts
  screenshots/       # episode + recording frames
  videos/            # downloaded tutorials (pruned automatically)
  frames/            # extracted keyframes
  browser_profile/   # persistent Chromium profile (your Onshape login) — gitignored
```

Nothing leaves your machine: the dataset is gitignored, and only your local Ollama model ever sees screenshots.

## Tests

No GPU, no Ollama, no browser needed:

```powershell
python -m unittest discover -s tests
```

- 252 tests covering loopback security, action/plan parsing, memory, the video teacher pipeline, browser allowlists, the agent loop (with an injected fake browser), the HTTP sidecar, and the desktop GUI's sidecar client.
- The GitHub Actions workflow (`.github/workflows/tests.yml`) is **optional** — it only re-runs this suite on GitHub's servers. Nothing in the app depends on it; the tests above run identically on any machine.

Optional end-to-end smoke tests (need the optional deps):

```powershell
python scripts/smoke_video.py     # ffmpeg keyframes + captions + concept merge
python scripts/smoke_browser.py   # real browser: launch, Onshape, screenshot, actions
```

## Design notes

- **Browser, not desktop.** Onshape is a web app, so the agent drives one Playwright tab instead of global mouse/keyboard. No focus stealing, no keyloggers, and typing in other windows is never captured.
- **Strict URL allowlist.** Navigation is refused for anything that is not `*.onshape.com` — a confused model cannot browse elsewhere.
- **Loopback-only**, same rules as LocalText3D: `require_loopback_bind` / `require_loopback_port` (port 0 allowed for tests), Host/Origin checks, body-size caps, redirect refusal, and strict Ollama model-name validation.
- **Local models only.** The sidecar talks to `http://127.0.0.1:11434` and refuses HTTP redirects, so a poisoned `Host` header cannot redirect model traffic.

## License

MIT
