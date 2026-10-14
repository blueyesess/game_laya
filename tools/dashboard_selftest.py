"""Exercise every dashboard route in-process, with no socket involved.

Starting a real server just to curl it is fragile: the child process dies with the
shell that launched it. Flask's test_client() exercises the same view functions, the
same JSON serialisation and the same numpy-to-bytes path, with none of that.

    python tools/dashboard_selftest.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent import GameRunner  # noqa: E402
from src.config import SCREEN_H, SCREEN_W  # noqa: E402

ROM = ROOT / "roms" / "inputtest.nes"
REPORT = ROOT / "logs" / "dashboard_selftest.json"

results = []


def check(name, ok, detail=""):
    results.append({"name": name, "ok": bool(ok), "detail": detail})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, (" -- " + detail) if detail else ""),
          file=sys.stderr)
    return bool(ok)


def main() -> int:
    from tools.make_test_rom import build_rom

    ROM.parent.mkdir(parents=True, exist_ok=True)
    ROM.write_bytes(build_rom())

    from src import server as server_module

    runner = GameRunner(rom_path=str(ROM), policy="reflex")
    runner.start()
    # Let the emulator thread produce a few hundred frames and the reflex brain
    # record some decisions, exactly as it would in a real run.
    time.sleep(2.0)
    server_module.runner = runner

    app = server_module.app
    app.config.update(TESTING=True)
    client = app.test_client()

    index = client.get("/")
    check("GET / serves the dashboard",
          index.status_code == 200 and b"laya" in index.data,
          "%d, %d bytes" % (index.status_code, len(index.data)))

    frame = client.get("/api/frame.raw")
    expected = SCREEN_W * SCREEN_H * 3
    check("GET /api/frame.raw returns one RGB frame",
          frame.status_code == 200 and len(frame.data) == expected,
          "%d bytes (expected %d)" % (len(frame.data), expected))
    check("frame response is not cacheable",
          "no-store" in frame.headers.get("Cache-Control", ""),
          frame.headers.get("Cache-Control", ""))

    snap = client.get("/api/snapshot")
    body = snap.get_json() or {}
    check("GET /api/snapshot returns state + observation + stats",
          snap.status_code == 200 and {"state", "observation", "stats", "buttons"} <= set(body),
          "keys=%s" % sorted(body))
    stats = body.get("stats", {})
    check("the emulator thread is advancing frames",
          stats.get("frames", 0) > 30, "%s frames in 2s" % stats.get("frames"))
    check("the reflex brain is producing decisions",
          stats.get("decisions", 0) > 0,
          "%s decisions, %s from reflex" % (stats.get("decisions"), stats.get("reflex_decisions")))

    # A decision thread that dies on its third iteration still leaves "decisions > 0"
    # behind, which is exactly how a missing stats key hid here. Assert it keeps
    # growing over a further interval instead.
    first = stats.get("decisions", 0)
    time.sleep(2.0)
    later = client.get("/api/snapshot").get_json()["stats"]
    check("the decision thread is still alive 2s later",
          later.get("decisions", 0) > first,
          "%d -> %d decisions" % (first, later.get("decisions", 0)))
    check("the measured decision rate is reported",
          later.get("decision_hz_actual", 0) > 0,
          "%.2f Hz" % later.get("decision_hz_actual", 0))

    trace = client.get("/api/trace?limit=5").get_json()
    check("GET /api/trace returns decision records",
          isinstance(trace, list) and len(trace) > 0 and "observation" in trace[0] and "decision" in trace[0],
          "%d records, latest frame %s" % (len(trace or []), (trace or [{}])[-1].get("frame")))
    latest = (trace or [{}])[-1]
    decision = latest.get("decision", {})
    check("each record carries the three typed answers",
          {"move", "jump", "risk", "confidence"} <= set(decision),
          "move=%s jump_p=%s risk=%s" % (
              decision.get("move", {}).get("label"),
              decision.get("jump", {}).get("p_true"),
              decision.get("risk", {}).get("label")))

    questions = client.get("/api/questions").get_json()
    check("GET /api/questions exposes the schema the model is asked",
          isinstance(questions, dict) and set(questions) == {"move", "jump", "risk"},
          ", ".join("%s:%s" % (k, v.get("type")) for k, v in (questions or {}).items()))

    png = client.get("/api/snapshot.png")
    check("GET /api/snapshot.png returns a PNG",
          png.status_code == 200 and png.data[:8] == b"\x89PNG\r\n\x1a\n",
          "%d bytes" % len(png.data))

    try:
        import laya  # noqa: F401

        have_laya = True
    except ImportError:
        have_laya = False

    if have_laya:
        control = client.post("/api/control", json={"action": "policy", "policy": "hybrid"})
        check("POST /api/control switches policy",
              control.status_code == 200 and control.get_json()["snapshot"]["stats"]["policy"] == "hybrid",
              "policy now %s" % control.get_json()["snapshot"]["stats"]["policy"])
        client.post("/api/control", json={"action": "policy", "policy": "reflex"})
    else:
        # With laya absent, asking for a model-backed policy must be a clean 400 the
        # dashboard can render, and must leave the runner on reflex rather than
        # half-switched. This is the path a user hits before `pip install laya`.
        control = client.post("/api/control", json={"action": "policy", "policy": "hybrid"})
        check("asking for a laya policy without laya installed returns a clean 400",
              control.status_code == 400 and "laya" in (control.get_json().get("error") or ""),
              "status %d, error=%r" % (control.status_code, control.get_json().get("error")))
        check("the failed policy switch leaves the runner on reflex",
              client.get("/api/snapshot").get_json()["stats"]["policy"] == "reflex",
              "policy=%s" % client.get("/api/snapshot").get_json()["stats"]["policy"])

    manual = client.post("/api/control", json={"action": "manual", "buttons": ["right", "A"]})
    snapshot = client.get("/api/snapshot").get_json()
    check("POST /api/control takes manual control",
          manual.status_code == 200 and sorted(snapshot["buttons"]) == ["A", "right"],
          "buttons=%s" % snapshot["buttons"])
    client.post("/api/control", json={"action": "release"})
    check("POST /api/control releases manual control",
          client.get("/api/snapshot").get_json()["manual"] is None)

    # START recovery has to pulse, and it has to keep pulsing forever. Two earlier
    # versions failed differently: `frame < 900` gave up permanently after a one-shot
    # budget, and holding START down produced a single edge that SMB1's title screen
    # ignored -- 2400 frames with the button down and Mario never appeared once.
    #
    # So sample the buttons over a window that spans several pulse periods and require
    # both phases to appear. A single snapshot cannot tell those two apart.
    seen_down = False
    seen_up = False
    for _ in range(40):
        buttons = client.get("/api/snapshot").get_json()["buttons"]
        if "start" in buttons:
            seen_down = True
        else:
            seen_up = True
        if seen_down and seen_up:
            break
        time.sleep(0.05)
    check("START recovery pulses instead of holding, and never stops",
          seen_down and seen_up,
          "pressed phase seen: %s, released phase seen: %s "
          "(this homebrew ROM never draws Mario)" % (seen_down, seen_up))

    bad = client.post("/api/control", json={"action": "policy", "policy": "nonsense"})
    check("an invalid policy is rejected, not silently accepted",
          bad.status_code == 400, "status %d" % bad.status_code)

    runner.stop()
    runner.runner.close()

    passed = sum(1 for r in results if r["ok"])
    REPORT.write_text(json.dumps({"passed": passed, "total": len(results), "results": results},
                                 indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n%d/%d passed -> %s" % (passed, len(results), REPORT), file=sys.stderr)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
