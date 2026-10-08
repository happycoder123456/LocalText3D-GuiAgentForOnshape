"""Command-line entry: python -m agent <serve|run|learn|plan|status>."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _cmd_serve(args: argparse.Namespace) -> int:
    from agent.server import serve

    serve(host=args.host, port=args.port, dataset_dir=Path(args.dataset) if args.dataset else None)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from agent.loop import AgentLoop

    loop = AgentLoop(dataset_root=Path(args.dataset) if args.dataset else None)
    try:
        loop.start_run(
            goal=args.goal,
            model=args.model,
            planner_model=args.planner,
            max_steps=args.max_steps,
            use_plan=not args.no_plan,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        while loop.busy():
            st = loop.status()
            print(f"\r[{st['status']}] step {st['step']}/{st['max_steps']} — {st['detail'][:100]}", end="", flush=True)
            time.sleep(0.5)
    except KeyboardInterrupt:
        loop.stop()
        print("\nstopped.")
    st = loop.status()
    print(f"\n{st['status']}: {st['detail']}")
    if st.get("error"):
        print(f"error: {st['error']}", file=sys.stderr)
        return 1
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    from agent.loop import AgentLoop
    from agent.ollama import OllamaError

    loop = AgentLoop(dataset_root=Path(args.dataset) if args.dataset else None)
    try:
        plan = loop.preview_plan(goal=args.goal, planner_model=args.planner)
    except (ValueError, OllamaError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for i, step in enumerate(plan["steps"], 1):
        expect = f"  (expect: {step['expect']})" if step.get("expect") else ""
        print(f"{i:2d}. {step['step']}{expect}")
    return 0


def _cmd_learn(args: argparse.Namespace) -> int:
    from agent.loop import AgentLoop

    loop = AgentLoop(dataset_root=Path(args.dataset) if args.dataset else None)
    try:
        loop.learn_from_video(
            url=args.url or "",
            path=args.path or "",
            goal=args.goal or "",
            model=args.model or "",
            max_minutes=args.minutes,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        while loop.busy():
            st = loop.status()
            print(f"\r[{st['status']}] {st['detail'][:110]}", end="", flush=True)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n(cancel the download with Ctrl+C; partial downloads stay in the dataset)")
        return 130
    st = loop.status()
    print(f"\n{st['status']}: {st['detail']}")
    return 1 if st.get("error") else 0


def _cmd_login(args: argparse.Namespace) -> int:
    """Open the agent's isolated browser so the user can sign in to Onshape once."""
    from agent.browser import BrowserDriver
    from agent.paths import browser_profile_dir

    root = Path(args.dataset) if args.dataset else None
    try:
        driver = BrowserDriver(headless=False, profile_dir=str(browser_profile_dir(root)))
        driver.start()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Opened {driver.url()} in a new window.")
    print("Sign in to Onshape there. The login is saved in your local browser profile.")
    print("Press Ctrl+C here when done (closing the window also works).")
    try:
        while True:
            time.sleep(0.5)
            if not driver.alive():
                # Window closed (by the user or by a run closing the profile).
                break
    except KeyboardInterrupt:
        print()
    finally:
        driver.stop()
    print("Saved. Start a goal with: python -m agent run \"<goal>\"")
    return 0


def _cmd_gui(args: argparse.Namespace) -> int:
    """Desktop app: starts/uses the sidecar, exposes run/learn/record UI."""
    try:
        from agent.gui import main as gui_main
    except Exception as exc:  # noqa: BLE001 - a missing tkinter must not vanish
        return _fatal_dialog(
            "The Onshape Agent window could not be loaded:\n\n"
            f"{type(exc).__name__}: {exc}\n\n"
            "Install Python 3.13 with the tkinter component, then try again.\n"
            "Fallback: run  python -m agent serve  then open http://127.0.0.1:8767"
        )
    return gui_main()


def _fatal_dialog(message: str) -> int:
    """Report a GUI startup failure where the user can actually see it."""
    print(message, file=sys.stderr)
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, message, "Onshape Agent", 0x10)
    except Exception:
        pass
    return 1


def _cmd_status(args: argparse.Namespace) -> int:
    import urllib.request

    url = "http://127.0.0.1:8767/status"
    try:
        with urllib.request.urlopen(url, timeout=3.0) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"sidecar not reachable at {url}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent", description="Onshape GUI agent (local, free, Ollama-powered)")
    sub = parser.add_subparsers(dest="command")

    p_serve = sub.add_parser("serve", help="run the localhost sidecar (default http://127.0.0.1:8767)")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8767)
    p_serve.add_argument("--dataset", default="")
    p_serve.set_defaults(func=_cmd_serve)

    p_run = sub.add_parser("run", help="run one goal to completion in the browser")
    p_run.add_argument("goal")
    p_run.add_argument("--model", default="")
    p_run.add_argument("--planner", default="auto")
    p_run.add_argument("--max-steps", type=int, default=160)
    p_run.add_argument("--no-plan", action="store_true", help="skip the LLM planner; pure vision steps")
    p_run.add_argument("--dataset", default="")
    p_run.set_defaults(func=_cmd_run)

    p_plan = sub.add_parser("plan", help="print the plan for a goal without running it")
    p_plan.add_argument("goal")
    p_plan.add_argument("--planner", default="auto")
    p_plan.add_argument("--dataset", default="")
    p_plan.set_defaults(func=_cmd_plan)

    p_learn = sub.add_parser("learn", help="learn Onshape techniques from a YouTube URL or local video")
    p_learn.add_argument("--url", default="")
    p_learn.add_argument("--path", default="", help="local video (must be inside the agent dataset)")
    p_learn.add_argument("--goal", default="")
    p_learn.add_argument("--model", default="")
    p_learn.add_argument("--minutes", type=float, default=6.0, help="max minutes of video to study (up to 1200)")
    p_learn.add_argument("--dataset", default="")
    p_learn.set_defaults(func=_cmd_learn)

    p_login = sub.add_parser("login", help="open the agent's browser to sign in to Onshape once")
    p_login.add_argument("--dataset", default="")
    p_login.set_defaults(func=_cmd_login)

    p_status = sub.add_parser("status", help="query a running sidecar")
    p_status.set_defaults(func=_cmd_status)

    p_gui = sub.add_parser("gui", help="open the desktop app (tkinter UI)")
    p_gui.set_defaults(func=_cmd_gui)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
