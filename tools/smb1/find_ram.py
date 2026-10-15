"""Find the real SMB1 RAM addresses instead of trusting a table.

The community memory map has several variants that disagree by a byte or two, and the
addresses this project guessed at ($0028/$0029 for X, $0025 world, $0026 lives, $0027
status) turn out to be wrong for at least some SMB1 builds -- every one of them reads
zero while Mario is demonstrably running across the screen.

The way out is to stop trusting documentation and measure. `smb_state.find_mario()`
locates Mario from his two exact sprite colours, and that result is independent of RAM.
So this treats the *pixel-derived* screen position as ground truth and looks for the
bytes that agree with it:

* bytes that increase in step with Mario's screen X  -> world X, high and low byte
* bytes that fall steadily while in-level            -> the countdown timer
* bytes that hold a small constant in-level           -> world number, lives, status
* bytes that sit at zero everywhere                   -> ruled out

Every candidate address is printed with its correlation, so the table can be argued
with rather than trusted. Nothing here is SMB1-specific beyond the 6502 read of a
known-good ROM, which is the point.

    python tools/find_ram.py --rom path/to/smb.nes
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.emulator import NesRunner  # noqa: E402
from src.state import GameProfile  # noqa: E402
from src.vision import analyse, find_character  # noqa: E402

PROFILE = GameProfile.load("smb1")


def find_mario(screen):
    """The pixel-derived ground truth this tool correlates RAM against.

    Delegates to the generic geometric detector with the SMB1 profile. It is
    independent of RAM by construction, which is the whole reason it can serve
    as the arbiter when the RAM map disagrees with itself.
    """
    vis = analyse(screen, hud_rows=PROFILE.hud_rows)
    return find_character(screen, vis, PROFILE)

SCAN = 0x100  # scan the first page plus a bit; SMB1 keeps its whole state low
REPORT = ROOT / "logs" / "find_ram.json"


def pearson(xs, ys):
    n = len(xs)
    if n < 8:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rom", required=True)
    parser.add_argument("--run-frames", type=int, default=900)
    args = parser.parse_args()

    runner = NesRunner(args.rom)

    # Title, then into the level, then a long run to the right.
    phases = [
        ("title", (), 200),
        ("start", ("start",), 24),
        ("card", (), 110),
        ("run", ("right",), args.run_frames),
    ]

    samples = []  # (phase, frame, mario_x, ram[0:SCAN])
    frame = 0
    for name, buttons, count in phases:
        for _ in range(count):
            runner.step(buttons)
            frame += 1
            screen = runner.screen
            found = find_mario(screen)
            samples.append((name, frame, None if found is None else found["x"],
                            bytes(int(v) for v in runner.ram[:SCAN])))

    runner.close()

    # ---- restrict to the run phase, and to the prefix where Mario is advancing --
    # Once the camera catches up, mario_screen_x stops changing while world_x keeps
    # climbing, so correlation has to be measured before that happens.
    run = [s for s in samples if s[0] == "run" and s[2] is not None]
    prefix = []
    previous = None
    for sample in run:
        if previous is not None and sample[2] < previous:
            break  # camera locked or Mario changed direction
        prefix.append(sample)
        previous = sample[2]

    lines = []
    lines.append("samples: %d total, %d in run phase, %d in the advancing prefix"
                 % (len(samples), len(run), len(prefix)))
    lines.append("mario screen x over the prefix: %s"
                 % ([s[2] for s in prefix[:6]] + ["..."] + [s[2] for s in prefix[-3:]]
                    if len(prefix) > 9 else [s[2] for s in prefix]))
    lines.append("")

    if len(prefix) < 20:
        lines.append("Not enough forward motion to correlate against (%d frames)." % len(prefix))
        lines.append("Mario probably never got running; check the phase script.")
        REPORT.write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines), file=sys.stderr)
        return 1

    xs = [s[2] for s in prefix]

    lines.append("addresses that move in step with Mario's screen position")
    lines.append("  %-6s %-9s %-9s %s" % ("addr", "corr", "slope", "first / last values"))
    lines.append("  " + "-" * 64)
    movers = []
    constant = {}
    for addr in range(SCAN):
        series = [s[3][addr] for s in prefix]
        distinct = set(series)
        if len(distinct) == 1:
            constant[addr] = series[0]
            continue
        corr = pearson(xs, series)
        slope = (series[-1] - series[0]) / max(1, len(series) - 1)
        movers.append((abs(corr), corr, addr, series[0], series[-1], slope))

    movers.sort(reverse=True)
    for _, corr, addr, first, last, slope in movers[:14]:
        lines.append("  $%04X  %+.4f   %+8.4f   %3d -> %3d" % (addr, corr, slope, first, last))

    lines.append("")
    lines.append("constant bytes while running (candidates: world, lives, status, flags)")
    lines.append("  " + ", ".join("$%04X=%d" % (a, v) for a, v in sorted(constant.items())))

    # Differentiate the constants against the title screen, where world/lives differ.
    title = [s for s in samples if s[0] == "title"]
    if title:
        lines.append("")
        lines.append("bytes that DIFFER between the title screen and the run phase")
        lines.append("  %-6s %-8s %-8s %s" % ("addr", "title", "run", "note"))
        lines.append("  " + "-" * 58)
        tmap = {}
        for s in title[-40:]:
            for a in range(SCAN):
                tmap.setdefault(a, []).append(s[3][a])
        rmap = {a: [s[3][a] for s in prefix] for a in range(SCAN)}
        for addr in range(SCAN):
            tv = sorted(set(tmap.get(addr, [])))
            rv = sorted(set(rmap[addr]))
            if tv and rv and tv != rv:
                note = ""
                if rv == [1]:
                    note = "== 1 in level: world number 1-1?"
                elif len(rv) == 1 and rv[0] in (2, 3, 4, 5):
                    note = "constant %d in level: lives or status?" % rv[0]
                lines.append("  $%04X  %-8s %-8s %s" % (addr, tv[-3:], rv[-3:], note))

    lines.append("")
    lines.append("addresses that are zero throughout the run phase (ruled out)")
    lines.append("  " + ", ".join("$%04X" % a for a, v in sorted(constant.items()) if v == 0))

    report = "\n".join(lines)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")
    (ROOT / "logs" / "find_ram_samples.json").write_text(
        json.dumps([{"phase": p, "frame": f, "mario_x": x, "ram": list(r)}
                    for p, f, x, r in samples], allow_nan=True),
        encoding="utf-8")
    print(report, file=sys.stderr)
    print("\nreport -> logs/find_ram.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
