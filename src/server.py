"""Flask dashboard: the emulator's screen plus a live view of laya's reasoning.

Two separate streams, on purpose:

* the framebuffer goes out as **raw RGB bytes** (`/api/frame.raw`), which the page
  blits with a single `putImageData`. No JPEG encode on the Python side, no decode
  on the JS side, and the picture is pixel-exact instead of slightly wrong.
* the reasoning goes out as **JSON** (`/api/trace`), polled on a timer.

Together they answer the question the project exists to answer: what did laya decide,
on what input, and did the emulator do what it said.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file

if __package__ in (None, ""):  # allow `python src/server.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "src"

from .agent import GameRunner  # noqa: E402
from .config import (  # noqa: E402
    DEFAULT_CONFIDENCE_FLOOR,
    DEFAULT_DECISION_HZ,
    DEFAULT_MODEL,
    DEFAULT_PORT,
    DEFAULT_SPEED,
    ROM_DIR,
    WEB_DIR,
)
from .state import GameProfile, ProfileNotFound  # noqa: E402

app = Flask(__name__, static_folder=None)
runner: GameRunner | None = None


# ----------------------------------------------------------------------- pages
@app.get("/")
def index():
    return send_file(WEB_DIR / "index.html")


# ----------------------------------------------------------------------- video
@app.get("/api/frame.raw")
def frame_raw():
    if runner is None:
        return Response(b"", status=503, mimetype="application/octet-stream")
    payload = runner.frame_bytes
    if payload is None:
        return Response(b"", status=503, mimetype="application/octet-stream")
    response = Response(payload, mimetype="application/octet-stream")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    return response


@app.get("/api/snapshot.png")
def snapshot_png():
    """One still of the current frame. Needs Pillow; optional by design."""
    if runner is None:
        return Response(b"", status=503)
    import numpy as np

    try:
        from PIL import Image
    except ImportError:
        return jsonify({"error": "Pillow is not installed: pip install pillow"}), 501

    data = runner.frame_bytes
    if data is None:
        return jsonify({"error": "no frame yet"}), 503
    height, width = 240, 256
    array = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
    image = Image.fromarray(array).resize((width * 3, height * 3), Image.NEAREST)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return send_file(buffer, mimetype="image/png", download_name="nes-laya.png")


# ----------------------------------------------------------------------- state
@app.get("/api/snapshot")
def snapshot():
    if runner is None:
        return jsonify({"error": "no game running"}), 503
    return jsonify(runner.snapshot())


@app.get("/api/trace")
def trace():
    limit = int(request.args.get("limit", 30))
    if runner is None:
        return jsonify([])
    records = list(runner.trace)[-max(1, min(limit, 200)) :]
    return jsonify(records)


@app.get("/api/questions")
def questions():
    from .brain import QUESTIONS, questions_for

    profile_id = request.args.get("profile")
    if profile_id:
        try:
            return jsonify(questions_for(GameProfile.load(profile_id)))
        except ProfileNotFound as exc:
            return jsonify({"error": str(exc)}), 400
    return jsonify(QUESTIONS)


# --------------------------------------------------------------------- control
@app.post("/api/control")
def control():
    if runner is None:
        return jsonify({"error": "no game running"}), 503
    payload = request.get_json(silent=True) or {}
    action = payload.get("action")

    if action == "start":
        runner.start()
    elif action == "stop":
        runner.stop()
    elif action == "reset":
        runner.reset()
    elif action == "policy":
        policy = payload.get("policy")
        try:
            runner.set_policy(policy)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except RuntimeError as exc:
            # Almost always "laya is not installed" or "the checkpoint failed to
            # download". The dashboard must stay usable on the reflex policy, so this
            # is a 400 the UI can display, not a 500 traceback.
            return jsonify({"error": str(exc), "policy_unchanged": runner.policy}), 400
    elif action == "manual":
        runner.set_manual(payload.get("buttons"))
    elif action == "release":
        runner.set_manual(None)
    elif action == "floor":
        runner.set_confidence_floor(payload.get("value", 0.45))
    elif action == "speed":
        runner.set_speed(payload.get("value", 1.0))
    elif action == "warmup":
        if runner.brain is None:
            return jsonify({"error": "policy is reflex; no brain loaded"}), 400
        return jsonify(runner.brain.warmup())
    else:
        return jsonify({"error": "unknown action %r" % (action,)}), 400

    return jsonify({"ok": True, "action": action, "snapshot": runner.snapshot()})


# ------------------------------------------------------------------------ main
def available_profiles() -> list[str]:
    from .config import PROFILE_DIR

    return sorted(p.stem for p in Path(PROFILE_DIR).glob("*.json"))


def find_rom(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return path
        raise SystemExit("ROM not found: %s" % path)
    candidates = sorted(ROM_DIR.glob("*.nes"))
    if not candidates:
        raise SystemExit(
            "No .nes file in %s.\n"
            "Drop the ROM of the game you want to drive there and start again, or pass "
            "--rom explicitly.\n"
            "(A legal homebrew .nes can be generated with: python tools/make_test_rom.py)"
            % ROM_DIR
        )
    return candidates[0]


def main() -> None:
    global runner

    parser = argparse.ArgumentParser(description="Watch laya play an NES game.")
    parser.add_argument("--rom", default=None, help="path to a .nes file")
    parser.add_argument(
        "--profile",
        default="generic",
        help=(
            "game profile id from src/profiles/*.json (available: %s). "
            "'generic' needs no per-game tuning; 'smb1' additionally reads the "
            "community SMB1 RAM map for HUD numbers." % ", ".join(available_profiles())
        ),
    )
    parser.add_argument("--policy", default="hybrid", choices=("laya", "reflex", "hybrid"))
    parser.add_argument("--model", default=DEFAULT_MODEL, help="english | multilingual | typed-decisions")
    parser.add_argument("--hz", type=float, default=DEFAULT_DECISION_HZ)
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED)
    parser.add_argument("--confidence-floor", type=float, default=DEFAULT_CONFIDENCE_FLOOR)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--warmup", action="store_true", help="load the checkpoint before serving")
    args = parser.parse_args()

    profile = GameProfile.load(args.profile)
    rom = find_rom(args.rom)
    print("ROM      : %s" % rom)
    print("profile  : %s (%s)" % (profile.id, profile.name))
    print("policy   : %s" % args.policy)
    print("model    : %s" % args.model)
    print("rate     : %.1f decisions/sec target, %.2fx emulator speed" % (args.hz, args.speed))

    runner = GameRunner(
        rom_path=str(rom),
        policy=args.policy,
        model=args.model,
        decision_hz=args.hz,
        speed=args.speed,
        confidence_floor=args.confidence_floor,
        profile=profile,
    )
    runner.start()

    if args.warmup and runner.brain is not None:
        print("warming the checkpoint (first run also downloads the weights)...")
        runner.brain.warmup()
        print("warm.")

    print("dashboard: http://127.0.0.1:%d/" % args.port)
    app.run(host="127.0.0.1", port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
