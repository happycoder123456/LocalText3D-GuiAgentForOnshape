"""Desktop app for the Onshape GUI Agent (tkinter, stdlib only).

Talks to the local sidecar on 127.0.0.1:8767 and starts it on demand, so
double-clicking one launcher is enough. Nothing leaves this machine.
"""

from __future__ import annotations

import json
import queue
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
import urllib.error
import urllib.request
from typing import Any

SIDECAR_URL = "http://127.0.0.1:8767"
OLLAMA_URL = "http://127.0.0.1:11434"
POLL_SEC = 1.0
MAX_LOG_LINES = 400


def _fatal(message: str) -> None:
    """Show a fatal error even with no console (pythonw swallows stderr)."""
    print(message, file=sys.stderr)
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, message, "Onshape Agent", 0x10)
    except Exception:
        pass

# -- palette ---------------------------------------------------------------
# Palette tuned for a calm VS Code-style dark IDE surface (professional,
# low-glare, print-quality contrast). Layers: BG < SURFACE < CARD.
BG = "#181a1f"        # app background (deepest)
SURFACE = "#1f2126"   # sidebar / header bands
CARD = "#26282e"      # raised panel
FIELD = "#2d3038"     # text entry fields
FG = "#e9ecf2"        # primary text
MUTED = "#9aa2b1"     # secondary text
ACCENT = "#5c8dff"    # brand blue
ACCENT_HOVER = "#729dff"
RUN_BG = "#2f9e6e"    # success green
GOOD = RUN_BG         # backwards-compatible alias used by the status chips
BAD = "#e0564e"       # danger red
MONO = ("Consolas", 9)
TITLE_FONT = ("Segoe UI Semibold", 15)
BODY_FONT = ("Segoe UI", 10)
SMALL_FONT = ("Segoe UI", 9)


class SidecarError(RuntimeError):
    """Sidecar unreachable or answered with an error."""


class SidecarClient:
    """Tiny localhost HTTP client (stdlib only) — unit-testable without Tk."""

    def __init__(self, base_url: str = SIDECAR_URL, timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get(self, path: str) -> dict[str, Any]:
        return self._request("GET", path, None)

    def post(self, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._request("POST", path, body or {})

    def _request(self, method: str, path: str, body: dict[str, Any] | None) -> dict[str, Any]:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace") or ""
            except Exception:
                pass
            try:
                parsed = json.loads(detail) if detail else {}
                detail = str(parsed.get("error") or detail or exc.reason)
            except (ValueError, AttributeError):
                pass
            raise SidecarError(detail or f"HTTP {exc.code}") from exc
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            reason = getattr(exc, "reason", exc)
            raise SidecarError(f"sidecar unreachable ({reason})") from exc
        try:
            payload = json.loads(raw) if raw else {}
        except ValueError as exc:
            raise SidecarError("sidecar returned invalid JSON") from exc
        # A 200 body may carry a non-empty "error" field: that is the agent's
        # RUN STATE (e.g. NEEDS_LOGIN) inside status/health payloads, not a
        # transport failure. Raising on it made every poll report "sidecar
        # stopped" after any run error, spawn restart attempts, and freeze the
        # whole UI. Real failures arrive as HTTP 4xx/5xx and are handled by
        # the HTTPError branch above; run errors are read from status.error.
        return payload if isinstance(payload, dict) else {}

    def reachable(self) -> bool:
        try:
            self.get("/status")
            return True
        except SidecarError:
            return False


def project_root():
    from pathlib import Path

    return Path(__file__).resolve().parent.parent


def start_sidecar_process() -> subprocess.Popen | None:
    """Spawn `python -m agent serve` in the background; None if we can't.

    stdout/stderr go to scripts/sidecar.log — swallowing them into DEVNULL
    made every sidecar crash unauditable (the GUI could only guess why it
    died). The file is truncated once it grows past ~2 MB.
    """
    log_fh = None
    try:
        try:
            log_path = project_root() / "scripts" / "sidecar.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            if log_path.exists() and log_path.stat().st_size > 2_000_000:
                log_path.write_text("", encoding="utf-8")
            log_fh = open(log_path, "a", encoding="utf-8", errors="replace")
            log_fh.write(f"--- sidecar start {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            log_fh.flush()
        except OSError:
            log_fh = None
        proc = subprocess.Popen(
            [sys.executable, "-m", "agent", "serve"],
            cwd=str(project_root()),
            stdout=log_fh if log_fh is not None else subprocess.DEVNULL,
            stderr=log_fh if log_fh is not None else subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return proc
    except OSError:
        return None
    finally:
        if log_fh is not None:
            # The child owns its own duplicated handle; close ours.
            log_fh.close()


def start_ollama_process() -> subprocess.Popen | None:
    exe = shutil.which("ollama")
    if not exe:
        return None
    try:
        return subprocess.Popen(
            [exe, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError:
        return None


def ollama_reachable(timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Onshape Agent")
        self.geometry("1100x800")
        self.minsize(920, 660)
        self.configure(bg=BG)
        try:
            # Windows 10 1809+: dark native title bar to match the theme.
            import ctypes

            hwnd = ctypes.windll.user32.GetParent(self.winfo_id())
            if hwnd:
                for attr_value in (0x00000001,):
                    ctypes.windll.dwmapi.DwmSetWindowAttribute(
                        hwnd, 20, ctypes.byref(ctypes.c_int(attr_value)), 4
                    )
        except Exception:
            pass
        self.client = SidecarClient()
        self.events: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        self._status: dict[str, Any] = {}
        self._health: dict[str, Any] = {}
        self._learn_seen: set[str] = set()
        self._sidecar_proc: subprocess.Popen | None = None
        self._restarting = False
        # None = never reached yet (bootstrap is still starting it), so a
        # slow first boot is not mistaken for a crash.
        self._sidecar_up: bool | None = None

        self._apply_style()
        self._build_header()
        self._build_body()
        self._build_footer()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.log_line("Starting — checking sidecar on 127.0.0.1:8767 …")
        self.after(200, self._bootstrap)
        self.after(300, self._drain_events)

    # -- chrome ---------------------------------------------------------------

    def _apply_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=BG, foreground=FG, font=BODY_FONT)
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD)
        style.configure("CardInner.TFrame", background=CARD)
        style.configure("TLabel", background=BG, foreground=FG, font=BODY_FONT)
        style.configure("Card.TLabel", background=CARD, foreground=FG)
        style.configure("Muted.TLabel", background=CARD, foreground=MUTED, font=SMALL_FONT)
        style.configure("Section.TLabel", background=CARD, foreground=MUTED,
                        font=("Segoe UI", 8, "bold"))
        style.configure("Title.TLabel", background=SURFACE, foreground=FG, font=TITLE_FONT)
        style.configure(
            "TButton",
            background=FIELD,
            foreground=FG,
            font=BODY_FONT,
            padding=(16, 8),
            borderwidth=0,
            focuscolor=FIELD,
        )
        style.map("TButton", background=[("active", "#3a3e47"), ("disabled", FIELD)],
                  foreground=[("disabled", MUTED)])
        style.configure(
            "Run.TButton", background=RUN_BG, foreground="#0d1b13", font=("Segoe UI", 10, "bold")
        )
        style.map("Run.TButton", background=[("active", ACCENT_HOVER), ("disabled", "#284b3d")],
                  foreground=[("active", "#ffffff"), ("disabled", "#6d8a7d")])
        style.configure("Stop.TButton", background=BAD, foreground="#ffffff",
                        font=("Segoe UI", 10, "bold"))
        style.map("Stop.TButton", background=[("active", "#f06a62"), ("disabled", "#4a2b29")],
                  foreground=[("disabled", "#a37874")])
        style.configure("Ghost.TButton", background=CARD, foreground=FG,
                        font=("Segoe UI", 10))
        style.map("Ghost.TButton", background=[("active", FIELD), ("disabled", CARD)],
                  foreground=[("disabled", MUTED)])
        style.configure("TNotebook", background=BG, borderwidth=0, tabmargins=[0, 0, 0, 0])
        style.configure("TNotebook.Tab", background=BG, foreground=BG, padding=(0, 0))
        style.map("TNotebook.Tab", background=[("selected", BG)])
        style.layout("TNotebook.Tab", [])  # hide tab chrome; sidebar navigates
        style.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=FIELD,
                        bordercolor=FIELD, lightcolor=ACCENT, darkcolor=ACCENT)
        style.configure(
            "Treeview",
            background=FIELD,
            fieldbackground=FIELD,
            foreground=FG,
            font=BODY_FONT,
            rowheight=28,
            borderwidth=0,
        )
        style.configure("Treeview.Heading", background=CARD, foreground=MUTED,
                        font=("Segoe UI", 8, "bold"), padding=(8, 6), relief="flat")
        style.map("Treeview", background=[("selected", "#37507d")])
        style.configure("TEntry", fieldbackground=FIELD, foreground=FG, insertcolor=FG,
                        bordercolor=FIELD, lightcolor=FIELD, darkcolor=FIELD)
        style.map("TEntry", bordercolor=[("focus", ACCENT)])
        style.configure("TSpinbox", fieldbackground=FIELD, foreground=FG,
                        bordercolor=FIELD, lightcolor=FIELD, darkcolor=FIELD,
                        arrowcolor=MUTED)
        style.configure("Card.TCheckbutton", background=CARD, foreground=FG)
        style.map("Card.TCheckbutton", background=[("active", CARD)])
        style.configure("TCheckbutton", background=BG, foreground=FG)
        style.configure("TLabelframe", background=BG, foreground=MUTED)
        style.configure("TLabelframe.Label", background=BG, foreground=MUTED, font=SMALL_FONT)

    def _build_header(self) -> None:
        head = tk.Frame(self, bg=SURFACE, padx=16, pady=12)
        head.pack(fill="x")
        inner = tk.Frame(head, bg=SURFACE)
        inner.pack(fill="x")
        left = tk.Frame(inner, bg=SURFACE)
        left.pack(side="left")
        tk.Label(left, text="◆  Onshape Agent", bg=SURFACE, fg=ACCENT,
                 font=TITLE_FONT).pack(side="left")
        self.sidecar_chip = tk.Label(head, text="● sidecar …", bg=SURFACE, fg=MUTED, font=SMALL_FONT)
        self.sidecar_chip.pack(side="right", padx=(10, 0))
        self.ollama_chip = tk.Label(head, text="● ollama …", bg=SURFACE, fg=MUTED, font=SMALL_FONT)
        self.ollama_chip.pack(side="right", padx=(10, 0))
        self.model_chip = tk.Label(head, text="", bg=SURFACE, fg=MUTED, font=SMALL_FONT)
        self.model_chip.pack(side="right", padx=(10, 0))

    def _build_body(self) -> None:
        """Sidebar navigation (VS Code / modern app feel) + page container."""
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        nav = tk.Frame(body, bg=SURFACE, width=196, padx=10, pady=14)
        nav.pack(side="left", fill="y")
        nav.pack_propagate(False)

        self._nav_buttons = []
        for i, (label, page) in enumerate(
            (("◆  Model", 0), ("▶  Learn", 1), ("⚙  Session", 2))
        ):
            btn = tk.Button(
                nav,
                text=label,
                anchor="w",
                padx=12,
                pady=8,
                bd=0,
                bg=SURFACE,
                fg=MUTED,
                activebackground=CARD,
                activeforeground=FG,
                font=("Segoe UI", 10, "bold"),
                cursor="hand2",
                relief="flat",
                command=lambda idx=i: self.show_page(idx),
            )
            btn.pack(fill="x", pady=2)
            self._nav_buttons.append(btn)

        self.show_page = self._notebook_select  # type: ignore[method-assign]
        main = tk.Frame(body, bg=BG, padx=18, pady=14)
        main.pack(side="left", fill="both", expand=True)
        self.notebook = ttk.Notebook(main)
        self.notebook.pack(fill="both", expand=True)
        self._build_model_tab()
        self._build_learn_tab()
        self._build_session_tab()
        self.show_page(0)

    def _notebook_select(self, idx: int) -> None:
        self.notebook.select(idx)
        for j, btn in enumerate(self._nav_buttons):
            selected = j == idx
            btn.configure(fg=FG if selected else MUTED, bg=CARD if selected else SURFACE)

    def _build_model_tab(self) -> None:
        tab = ttk.Frame(self.notebook, style="Card.TFrame", padding=18)
        self.notebook.add(tab, text="  Model  ")

        ttk.Label(tab, text="MODELLING GOAL", style="Section.TLabel").pack(anchor="w")
        self.goal_text = tk.Text(
            tab, height=4, wrap="word", bg=FIELD, fg=FG, insertbackground=FG,
            font=BODY_FONT, relief="flat", padx=12, pady=10,
        )
        self.goal_text.pack(fill="x", pady=(5, 12))
        self.goal_text.insert("1.0", "make me a simple cube")

        opts = ttk.Frame(tab, style="Card.TFrame")
        opts.pack(fill="x", pady=(0, 12))
        self.use_plan_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Use plan", style="Card.TCheckbutton",
                        variable=self.use_plan_var).pack(side="left")
        ttk.Label(opts, text="  Max steps:", style="Card.TLabel").pack(side="left", padx=(16, 0))
        self.max_steps_var = tk.StringVar(value="160")
        ttk.Spinbox(opts, from_=1, to=2500, width=6, textvariable=self.max_steps_var).pack(
            side="left", padx=(4, 0)
        )

        actions = ttk.Frame(tab, style="Card.TFrame")
        actions.pack(fill="x", pady=(0, 14))
        self.run_btn = ttk.Button(
            actions, text="▶  Run agent", style="Run.TButton", command=self.on_run
        )
        self.run_btn.pack(side="left")
        self.stop_btn = ttk.Button(
            actions, text="■  Stop", style="Stop.TButton", command=self.on_stop
        )
        self.stop_btn.pack(side="left", padx=(10, 0))
        self.plan_btn = ttk.Button(actions, text="⌁  Preview plan", style="Ghost.TButton",
                                   command=self.on_preview)
        self.plan_btn.pack(side="left", padx=(10, 0))

        prog = ttk.Frame(tab, style="Card.TFrame")
        prog.pack(fill="x", pady=(0, 10))
        self.progress = ttk.Progressbar(prog, mode="determinate", maximum=100, value=0)
        self.progress.pack(fill="x")
        status_row = ttk.Frame(prog, style="Card.TFrame")
        status_row.pack(fill="x", pady=(8, 0))
        self.status_lbl = ttk.Label(status_row, text="Idle", style="Card.TLabel")
        self.status_lbl.pack(side="left")
        self.step_lbl = ttk.Label(status_row, text="", style="Muted.TLabel")
        self.step_lbl.pack(side="right")

        ttk.Label(tab, text="PLAN PREVIEW", style="Section.TLabel").pack(anchor="w", pady=(8, 5))
        cols = ("#", "Step", "Expect")
        self.plan_tree = ttk.Treeview(tab, columns=cols, show="headings", height=7)
        for col, width in (("#", 40), ("Step", 470), ("Expect", 420)):
            self.plan_tree.heading(col, text=col)
            self.plan_tree.column(col, width=width, anchor="w")
        self.plan_tree.pack(fill="both", expand=True)

        self.detail_lbl = tk.Label(
            tab, text="", bg=CARD, fg=MUTED, font=SMALL_FONT, anchor="w",
            padx=12, pady=8, justify="left",
        )
        self.detail_lbl.pack(fill="x", pady=(10, 0))

    def _build_learn_tab(self) -> None:
        tab = ttk.Frame(self.notebook, style="Card.TFrame", padding=18)
        self.notebook.add(tab, text="  Learn from video  ")

        ttk.Label(tab, text="YOUTUBE URL", style="Section.TLabel").pack(anchor="w")
        self.url_entry = ttk.Entry(tab)
        self.url_entry.pack(fill="x", pady=(5, 12))

        ttk.Label(
            tab, text="Focus goal (optional, e.g. “make a bracket”)", style="Section.TLabel"
        ).pack(anchor="w")
        self.learn_goal = ttk.Entry(tab)
        self.learn_goal.pack(fill="x", pady=(5, 12))

        opts = ttk.Frame(tab, style="Card.TFrame")
        opts.pack(fill="x", pady=(0, 12))
        ttk.Label(opts, text="Minutes to study:", style="Card.TLabel").pack(side="left")
        self.minutes_var = tk.StringVar(value="6")
        ttk.Spinbox(opts, from_=1, to=1200, width=6, textvariable=self.minutes_var).pack(
            side="left", padx=(4, 0)
        )

        actions = ttk.Frame(tab, style="Card.TFrame")
        actions.pack(fill="x", pady=(0, 14))
        self.learn_btn = ttk.Button(
            actions, text="▶  Learn from video", style="Run.TButton", command=self.on_learn
        )
        self.learn_btn.pack(side="left")

        self.learn_progress = ttk.Progressbar(
            tab, mode="determinate", maximum=100, value=0
        )
        self.learn_progress.pack(fill="x")
        self.learn_phase_lbl = ttk.Label(tab, text="", style="Card.TLabel")
        self.learn_phase_lbl.pack(fill="x", pady=(6, 10))

        ttk.Label(
            tab,
            text=(
                "Learned techniques are saved into local memory and injected into later "
                "plans automatically. Everything stays on this machine."
            ),
            style="Muted.TLabel",
            wraplength=860,
        ).pack(anchor="w", pady=(0, 10))

        ttk.Label(tab, text="LEARNING HISTORY", style="Section.TLabel").pack(anchor="w", pady=(4, 5))
        self.learn_log = tk.Text(
            tab, height=9, wrap="word", bg=FIELD, fg=FG, font=MONO,
            relief="flat", padx=12, pady=10, state="disabled",
        )
        self.learn_log.pack(fill="both", expand=True)

    def _build_session_tab(self) -> None:
        tab = ttk.Frame(self.notebook, style="Card.TFrame", padding=18)
        self.notebook.add(tab, text="  Session  ")

        row1 = ttk.Frame(tab, style="Card.TFrame")
        row1.pack(fill="x", pady=(0, 12))
        self.login_btn = ttk.Button(row1, text="🔑  Sign in to Onshape",
                                    style="Run.TButton", command=self.on_login)
        self.login_btn.pack(side="left")
        self.dataset_btn = ttk.Button(
            row1, text="📁  Open dataset folder", style="Ghost.TButton",
            command=self.on_open_dataset
        )
        self.dataset_btn.pack(side="left", padx=(10, 0))

        ttk.Label(
            tab,
            text="Sign in once per machine: a browser window opens, sign in there, then close it.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(0, 16))

        card = ttk.Frame(tab, style="CardInner.TFrame", padding=14)
        card.pack(fill="x", pady=(0, 12))
        ttk.Label(card, text="RECORD / REPLAY", style="Section.TLabel").pack(anchor="w")
        self.record_goal = ttk.Entry(card)
        self.record_goal.pack(fill="x", pady=(6, 10), ipady=3)
        actions = ttk.Frame(card, style="CardInner.TFrame")
        actions.pack(fill="x")
        self.record_btn = ttk.Button(actions, text="●  Start recording", style="Ghost.TButton",
                                     command=self.on_record_toggle)
        self.record_btn.pack(side="left")
        self.replay_btn = ttk.Button(actions, text="↻  Replay last demo", style="Ghost.TButton",
                                     command=self.on_replay)
        self.replay_btn.pack(side="left", padx=(10, 0))

        mem = ttk.Frame(tab, style="CardInner.TFrame", padding=14)
        mem.pack(fill="x", pady=(0, 12))
        ttk.Label(mem, text="LOCAL MEMORY", style="Section.TLabel").pack(anchor="w")
        self.memory_lbl = ttk.Label(mem, text="—", background=CARD, foreground=FG, font=BODY_FONT)
        self.memory_lbl.pack(anchor="w", pady=(6, 0))

        about = ttk.Frame(tab, style="CardInner.TFrame", padding=14)
        about.pack(fill="x")
        ttk.Label(
            about,
            text=(
                "Local-only GUI agent: Playwright tab on cad.onshape.com + Ollama models on this PC.\n"
                "No cloud AI, no API keys, no telemetry. Sidecar binds 127.0.0.1 only."
            ),
            background=CARD, foreground=MUTED, font=SMALL_FONT, justify="left",
        ).pack(anchor="w")

    def _build_footer(self) -> None:
        holder = ttk.Frame(self)
        holder.pack(fill="both", expand=False, padx=18, pady=(0, 14))
        ttk.Label(holder, text="ACTIVITY LOG", style="Section.TLabel").pack(anchor="w", pady=(0, 5))
        log_card = tk.Frame(holder, bg=CARD, padx=2, pady=2)
        log_card.pack(fill="both", expand=True)
        self.log = tk.Text(
            log_card, height=8, wrap="word", bg="#131417", fg=MUTED, font=MONO,
            relief="flat", padx=12, pady=8, state="disabled", bd=0,
            highlightthickness=0,
        )
        self.log.tag_configure("err", foreground=BAD)
        self.log.tag_configure("ok", foreground=GOOD)
        self.log.tag_configure("info", foreground="#c8cdd6")
        self.log.pack(fill="both", expand=True)

    # -- logging ---------------------------------------------------------------

    def log_line(self, text: str, tag: str = "info") -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log.configure(state="normal")
        self.log.insert("end", f"[{stamp}] {text}\n", tag)
        count = int(self.log.index("end-1c").split(".")[0])
        if count > MAX_LOG_LINES:
            self.log.delete("1.0", f"{count - MAX_LOG_LINES}.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def learn_line(self, text: str) -> None:
        self.learn_log.configure(state="normal")
        self.learn_log.insert("end", text.rstrip() + "\n")
        self.learn_log.see("end")
        self.learn_log.configure(state="disabled")

    # -- button actions -----------------------------------------------------------

    def _goal(self) -> str:
        return self.goal_text.get("1.0", "end").strip()

    def _call(self, path: str, body: dict[str, Any] | None = None, ok_msg: str = "") -> None:
        def work() -> None:
            try:
                self.client.post(path, body)
                if ok_msg:
                    self.events.put(("log", (ok_msg, "ok")))
            except SidecarError as exc:
                self.events.put(("log", (f"{path} failed: {exc}", "err")))

        threading.Thread(target=work, daemon=True).start()

    def on_run(self) -> None:
        goal = self._goal()
        if not goal:
            messagebox.showinfo("Goal needed", "Type a modeling goal first, e.g. “make a cube”.")
            return
        try:
            steps = max(1, min(2500, int(self.max_steps_var.get())))
        except ValueError:
            steps = 160
        self._call(
            "/agent/start",
            {"goal": goal, "max_steps": steps, "use_plan": bool(self.use_plan_var.get())},
            ok_msg=f"Run started: {goal}",
        )

    def on_stop(self) -> None:
        self._call("/agent/stop", {}, ok_msg="Stop requested")

    def on_preview(self) -> None:
        goal = self._goal()
        if not goal:
            messagebox.showinfo("Goal needed", "Type a modeling goal first.")
            return
        self.plan_btn.configure(state="disabled")
        self.log_line("Planning… (local text model)")

        def work() -> None:
            try:
                data = self.client.post("/plan", {"goal": goal})
                self.events.put(("plan", data))
            except SidecarError as exc:
                self.events.put(("log", (f"plan failed: {exc}", "err")))
                self.events.put(("plan_done", None))

        threading.Thread(target=work, daemon=True).start()

    def on_learn(self) -> None:
        url = self.url_entry.get().strip()
        if not url:
            messagebox.showinfo("URL needed", "Paste a YouTube URL first.")
            return
        try:
            minutes = float(self.minutes_var.get())
        except ValueError:
            minutes = 6.0
        self._call(
            "/video/learn",
            {"url": url, "goal": self.learn_goal.get().strip(), "max_minutes": minutes},
            ok_msg="Video teacher started",
        )

    def on_record_toggle(self) -> None:
        if self._status.get("recording"):
            self._call(
                "/record/stop",
                {"goal": self.record_goal.get().strip(), "success": True},
                ok_msg="Recording saved to local memory",
            )
        else:
            self._call(
                "/record/start",
                {"goal": self.record_goal.get().strip()},
                ok_msg="Recording started",
            )

    def on_replay(self) -> None:
        self._call("/replay/start", {"goal": self.record_goal.get().strip()}, ok_msg="Replaying…")

    def on_login(self) -> None:
        try:
            subprocess.Popen(
                [sys.executable, "-m", "agent", "login"],
                cwd=str(project_root()),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self.log_line("Sign-in window opened — sign in there, then close it.", "ok")
        except OSError as exc:
            messagebox.showerror("Cannot start browser", str(exc))

    def on_open_dataset(self) -> None:
        root_path = str((self._status.get("memory") or {}).get("root") or "")
        if not root_path:
            self.log_line("Dataset folder unknown yet (sidecar not ready).", "err")
            return
        try:
            import os

            os.startfile(root_path)
        except OSError as exc:
            self.log_line(f"Cannot open folder: {exc}", "err")

    # -- background bootstrap / polling -------------------------------------------

    def _bootstrap(self) -> None:
        def work() -> None:
            if self.client.reachable():
                self._sidecar_up = True
                self.events.put(("boot", "ok"))
                return
            self.events.put(("log", ("Sidecar not running — starting it…", "info")))
            self._sidecar_proc = start_sidecar_process()
            for _ in range(40):
                time.sleep(0.5)
                if self.client.reachable():
                    self._sidecar_up = True
                    self.events.put(("log", ("Sidecar ready.", "ok")))
                    self.events.put(("boot", "ok"))
                    return
            self.events.put(("boot", "fail"))

        threading.Thread(target=work, daemon=True).start()

    def _drain_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    text, tag = payload
                    self.log_line(text, tag)
                elif kind == "plan":
                    self._show_plan(payload)
                    self.plan_btn.configure(state="normal")
                elif kind == "plan_done":
                    self.plan_btn.configure(state="normal")
                elif kind == "boot":
                    if payload == "ok":
                        self.log_line("Connected to sidecar. Ready.", "ok")
                    else:
                        self.log_line(
                            "Sidecar failed to start — run “Start Onshape Agent.bat” once, then reopen.",
                            "err",
                        )
                elif kind == "sidecar_down":
                    # A dead sidecar used to leave the buttons live and silent.
                    # Bring it back so Run/Learn work again without a restart.
                    if not self._restarting:
                        self._restarting = True
                        self.log_line("Sidecar stopped — restarting it…", "err")
                        threading.Thread(target=self._restart_sidecar, daemon=True).start()
                elif kind == "status":
                    self._render_status(payload)
                elif kind == "health":
                    self._health = payload
                    self._render_chips()
        except queue.Empty:
            pass
        self.after(300, self._drain_events)

    def _restart_sidecar(self) -> None:
        try:
            proc = start_sidecar_process()
            if proc is not None:
                self._sidecar_proc = proc
            for _ in range(40):
                time.sleep(0.5)
                if self.client.reachable():
                    self._sidecar_up = True
                    self.events.put(("log", ("Sidecar is back.", "ok")))
                    return
            self.events.put(
                (
                    "log",
                    (
                        "Sidecar did not come back — double-click “Start Onshape Agent.bat” "
                        "once, then reopen this app.",
                        "err",
                    ),
                )
            )
        finally:
            self._restarting = False

    def _poll(self) -> None:
        def work() -> None:
            try:
                st = self.client.get("/status")
                self._sidecar_up = True
                self.events.put(("status", st))
            except SidecarError:
                # Only the up -> down transition is news; while the sidecar is
                # still booting (or already restarting) it must stay quiet,
                # otherwise a second sidecar gets spawned against a busy port.
                if self._sidecar_up is True:
                    self._sidecar_up = False
                    self.events.put(("sidecar_down", None))
                elif self._sidecar_up is None:
                    self._sidecar_up = False
            try:
                self.events.put(("health", self.client.get("/health")))
            except SidecarError:
                pass

        threading.Thread(target=work, daemon=True).start()
        self.after(int(POLL_SEC * 1000), self._poll)

    # -- rendering ---------------------------------------------------------------

    def _show_plan(self, data: dict[str, Any]) -> None:
        preview = data.get("plan_preview")
        if isinstance(preview, dict):
            steps = preview.get("steps") or []
            model = str(preview.get("planner_model") or "")
        else:
            steps = preview or []
            model = ""
        self.plan_tree.delete(*self.plan_tree.get_children())
        if not steps:
            self.log_line("Planner returned no steps.", "err")
            return
        for i, step in enumerate(steps, 1):
            if not isinstance(step, dict):
                continue
            self.plan_tree.insert(
                "", "end", values=(i, str(step.get("step") or ""), str(step.get("expect") or ""))
            )
        self.log_line(f"Plan ready ({len(steps)} steps{', ' + model if model else ''}).", "ok")

    def _render_status(self, st: dict[str, Any]) -> None:
        prev_error = str(self._status.get("error") or "")
        prev_detail = str(self._status.get("detail") or "")
        prev_running = bool(self._status.get("running"))
        self._status = st

        running = bool(st.get("running"))
        recording = bool(st.get("recording"))
        learning = bool(st.get("learning_video"))
        busy = running or recording or learning

        self.run_btn.configure(state="disabled" if busy else "normal")
        self.learn_btn.configure(state="disabled" if busy else "normal")
        self.stop_btn.configure(state="normal" if busy else "disabled")
        self.record_btn.configure(
            text="■  Stop & save recording" if recording else "●  Start recording"
        )
        self.replay_btn.configure(state="disabled" if busy else "normal")

        self.status_lbl.configure(text=str(st.get("status") or "Idle"))
        detail = str(st.get("detail") or "")
        if detail:
            self.detail_lbl.configure(text=detail[:240])

        if running:
            step = int(st.get("step") or 0)
            max_steps = max(1, int(st.get("max_steps") or 1))
            self.progress.configure(
                mode="determinate", maximum=max_steps, value=min(step, max_steps)
            )
            self.step_lbl.configure(text=f"step {step}/{max_steps}")
        elif learning:
            pct = float(st.get("learn_percent") or 0)
            self.learn_progress.configure(mode="determinate", maximum=100, value=pct)
            self.learn_phase_lbl.configure(
                text=f"{str(st.get('learn_phase') or '')} — {pct:.0f}%"
            )
        elif recording:
            self.step_lbl.configure(text="recording…")
        else:
            self.progress.configure(mode="determinate", value=0)
            self.step_lbl.configure(text="")

        # A running session's plan appears shortly after start.
        if running and st.get("plan_preview") and not self.plan_tree.get_children():
            self._show_plan({"plan_preview": st.get("plan_preview")})

        err = str(st.get("error") or "")
        if err and err != prev_error:
            self.log_line(f"ERROR: {err}", "err")
        elif not err and prev_error and not busy:
            self.log_line("Previous error cleared.", "ok")

        if detail and detail != prev_detail:
            if learning:
                self.learn_line(detail)
            elif running:
                self.log_line(detail[:220])

        if learning:
            for line in st.get("learn_history") or []:
                text = str(line)
                if text not in self._learn_seen:
                    self._learn_seen.add(text)
                    self.learn_line(text)
        elif not learning and prev_detail and not busy and not running:
            pass

        mem = st.get("memory") or {}
        self.memory_lbl.configure(
            text=(
                f"{mem.get('concepts', 0)} techniques learned from video   ·   "
                f"{mem.get('skills', 0)} recorded skills   ·   "
                f"{mem.get('episodes', 0)} episodes"
            )
        )
        if prev_running and not running and not recording and not learning:
            final = str(st.get("status") or "Idle")
            self.log_line(f"Run ended: {final} — {detail[:160]}", "ok")

        self._render_chips()

    def _render_chips(self) -> None:
        health = self._health
        if health:
            ollama_ok = bool(health.get("ollama_ok"))
            self.ollama_chip.configure(
                text="● ollama OK" if ollama_ok else "● ollama DOWN",
                fg=GOOD if ollama_ok else BAD,
            )
            self.model_chip.configure(text=str(health.get("preferred_model") or ""))
            if not ollama_ok and not getattr(self, "_ollama_warned", False):
                self._ollama_warned = True
                if not ollama_reachable():
                    self.log_line(
                        "Ollama is not reachable — starting it. If it stays down, install it "
                        "from https://ollama.com then run: ollama pull qwen2.5vl:3b",
                        "err",
                    )
                    start_ollama_process()
                else:
                    self.log_line("Ollama reachable but /api/tags says down?", "err")
        if self._status:
            if self._status.get("browser_url"):
                self.sidecar_chip.configure(text="● browser open", fg=ACCENT)
            else:
                self.sidecar_chip.configure(text="● sidecar OK", fg=GOOD)

    def _on_close(self) -> None:
        st = self._status or {}
        if st.get("running") or st.get("recording") or st.get("learning_video"):
            # Closing the window kills the sidecar and whatever it was doing.
            if not messagebox.askyesno(
                "Work in progress",
                "A run / recording / lesson is still going. Closing now stops it.\n\n"
                "Close anyway?",
            ):
                return
        try:
            if self._sidecar_proc is not None and self._sidecar_proc.poll() is None:
                self._sidecar_proc.terminate()
        except Exception:
            pass
        self.destroy()


def main() -> int:
    """Open the app. Under pythonw there is no console, so a failure here used
    to look like "double-click and it instantly closes" — always say why."""
    try:
        app = App()
    except tk.TclError as exc:
        _fatal(
            "Cannot open the Onshape Agent window:\n\n"
            f"{exc}\n\n"
            "Fallback: run  python -m agent serve  then open http://127.0.0.1:8767"
        )
        return 1
    except Exception as exc:  # noqa: BLE001 - anything else must still be visible
        _fatal(f"The Onshape Agent could not start:\n\n{type(exc).__name__}: {exc}")
        return 1
    app._poll()
    try:
        app.mainloop()
    except Exception as exc:  # noqa: BLE001
        _fatal(f"The Onshape Agent window crashed:\n\n{type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
