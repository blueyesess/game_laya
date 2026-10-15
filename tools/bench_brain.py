"""Measure what one laya decision actually costs on this machine.

The number that matters for the architecture is not the one in laya's README (145 ms
for three questions on the multilingual checkpoint, on the author's hardware). It is
the wall-clock cost here, next to a 60 fps emulator thread and a Flask server handing
184 KB frames to a browser, because that is the environment the brain has to keep up
with.

Reports, separately:

* cold build  -- loading the checkpoint, which also downloads it on a first run
* first call  -- includes lazy allocation inside the model
* steady state -- median and p90 over N further calls

The steady-state figure is what decides whether the emulator can free-run while the
brain holds its last action, or whether the decision rate has to be lowered further.

    python tools/bench_brain.py [--calls 6] [--model multilingual]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.brain import LayaBrain, questions_for  # noqa: E402
from src.state import GameProfile  # noqa: E402

REPORT = ROOT / "logs" / "brain_bench.json"

# A representative observation, in the same shape scene.build_state() produces: short,
# because laya encodes the state once per question.
STATE = {
    "scene": ("The player is on ground at x=140. An object close ahead, 24px to the "
              "right, 16px tall. The floor ahead is solid. 900px into the level. "
              "Coins 3, time 310, lives 3. Goal: Reach the right side of the screen "
              "and keep advancing without falling into a hole. Move by pressing right."),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--calls", type=int, default=6)
    parser.add_argument("--model", default="multilingual")
    parser.add_argument("--profile", default="generic", help="game profile id from src/profiles/*.json")
    args = parser.parse_args()

    profile = GameProfile.load(args.profile)
    questions = questions_for(profile)

    print("loading %s (first run also downloads the weights)..." % args.model, file=sys.stderr)
    t0 = time.perf_counter()
    brain = LayaBrain(model=args.model, profile=profile)
    cold_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    first = brain.decide(STATE)
    first_call = time.perf_counter() - t0

    timings = []
    samples = []
    for _ in range(max(1, args.calls)):
        t0 = time.perf_counter()
        out = brain.decide(STATE)
        timings.append((time.perf_counter() - t0) * 1000.0)
        samples.append(out)

    timings.sort()
    p90 = timings[min(len(timings) - 1, int(0.9 * len(timings)))]
    ok = [s for s in samples if s.get("ok")]

    lines = [
        "checkpoint            : %s" % args.model,
        "questions per call    : %d (%s)" % (len(questions), ", ".join(questions)),
        "input tokens          : %s" % (ok[0]["usage"]["input_tokens"] if ok else "?"),
        "truncated             : %s" % (ok[0]["usage"]["truncated"] if ok else "?"),
        "",
        "cold build            : %8.1f s" % cold_build,
        "first call            : %8.1f s" % first_call,
        "steady state median   : %8.0f ms" % statistics.median(timings),
        "steady state p90      : %8.0f ms" % p90,
        "steady state min/max  : %8.0f / %.0f ms" % (timings[0], timings[-1]),
        "",
        "implied decision rate : %8.2f Hz  (1 / median)" % (1000.0 / statistics.median(timings)),
        "NES frames per call   : %8.1f     (at %.4f fps)"
        % (statistics.median(timings) / 1000.0 * 60.0988, 60.0988),
    ]
    if ok:
        last = ok[-1]
        lines += [
            "",
            "answers from the last call:",
            "  move  : %s   probs=%s" % (last["move"]["label"],
                                        {k: round(v, 3) for k, v in last["move"]["probs"].items()}),
            "  jump  : P(true)=%.4f -> %s" % (last["jump"]["p_true"],
                                             "press the jump button" if last["jump"]["thresholded"] else "no jump"),
            "  risk  : %s  (expected %.2f)  legend=%s"
            % (last["risk"]["label"], last["risk"]["expected"], last["risk"]["legend"]),
            "  answer_confidence: %.3f" % last["confidence"],
            "  routed to: %s" % last["model"],
        ]
    if brain.last_error:
        lines += ["", "last error: %s" % brain.last_error]

    report = "\n".join(lines)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report + "\n", encoding="utf-8")
    (REPORT.with_suffix(".json")).write_text(
        json.dumps({"model": args.model, "cold_build_s": cold_build,
                    "first_call_s": first_call, "steady_ms": timings}, indent=2),
        encoding="utf-8")
    print("\n" + report, file=sys.stderr)
    print("\nreport -> %s" % REPORT, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
