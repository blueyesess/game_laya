"""Pin the RAM map down with experiments instead of a documentation lookup.

`tools/find_ram.py` shows which bytes move with Mario. That is not enough to name
them, because several candidates look alike. This adds the two experiments that
settle it, using facts about SMB1 that do not depend on any address table:

* **the timer counts down.** In world 1-1 it starts near 400 and falls steadily, so
  the byte that decreases monotonically while in-level is the timer.
* **dying costs a life.** Holding RIGHT with no jumping walks Mario into the first
  pit, so the byte that reads 3 in-level and drops by exactly one across the death is
  the life counter -- and the byte that becomes 1 in-level but differs on the title
  screen is the world number.
* **X advances.** The byte whose value rises fastest while running is a world-X byte,
  and the one that rises with a fractional slope is the sub-pixel/fine offset.

The output is a short, argued list rather than a wall of hex, plus the frames at which
each conclusion was observed so it can be re-checked.

    python tools/verify_ram.py --rom path/to/smb.nes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.emulator import NesRunner  # noqa: E402
from src.state import GameProfile  # noqa: E402
from src.vision import analyse, find_character  # noqa: E402

SCAN = 0x100
PROFILE = GameProfile.load("smb1")


def find_mario(screen):
    """The pixel-derived ground truth, independent of any RAM address."""
    vis = analyse(screen, hud_rows=PROFILE.hud_rows)
    return find_character(screen, vis, PROFILE)


def _mario_x(runner) -> int | None:
    found = find_mario(runner.screen)
    return None if found is None else found["x"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rom", required=True)
    parser.add_argument("--max-frames", type=int, default=2600)
    args = parser.parse_args()

    runner = NesRunner(args.rom)

    # Hold RIGHT the whole time: past the title screen it needs START, then it walks
    # into the first pit unaided, which is the death we want.
    records = []
    frame = 0
    pressed_start = 0
    while frame < args.max_frames:
        if pressed_start < 2:
            buttons = ("start",)
            pressed_start += 1
        elif frame < 260:
            buttons = ("start",)
        else:
            buttons = ("right",)
        runner.step(buttons)
        frame += 1
        records.append((frame, _mario_x(runner),
                        bytes(int(v) for v in runner.ram[:SCAN])))

    runner.close()

    lines = []
    lines.append("frames run: %d" % len(records))

    # Split into segments by whether Mario is on screen: title/card vs in-level vs dead.
    in_level = [r for r in records if r[1] is not None]
    title = [r for r in records[:200]]
    lines.append("frames with Mario visible: %d (first at frame %s)"
                 % (len(in_level), in_level[0][0] if in_level else "never"))
    if len(in_level) < 60:
        lines.append("Mario never got on screen; cannot calibrate.")
        (ROOT / "logs").mkdir(parents=True, exist_ok=True)
        (ROOT / "logs" / "verify_ram.txt").write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines), file=sys.stderr)
        return 1

    # --- 1. the timer: strictly decreasing over a long in-level stretch ----------
    lines.append("")
    lines.append("1. bytes that fall steadily while in-level  ->  the countdown timer")
    best = []
    for addr in range(SCAN):
        series = [r[2][addr] for r in in_level]
        decreases = sum(1 for a, b in zip(series, series[1:]) if b < a)
        holds = sum(1 for a, b in zip(series, series[1:]) if b == a)
        if decreases < 30 or holds > decreases:
            continue
        span = series[0] - series[-1]
        best.append((decreases / max(1, holds), addr, series[0], series[-1], span, decreases))
    best.sort(reverse=True)
    if best:
        lines.append("  %-7s %-9s %-9s %-7s %s" % ("addr", "first", "last", "drops", "dec/hold"))
        for ratio, addr, first, last, span, drops in best[:8]:
            lines.append("  $%04X   %-9d %-9d %-7d %.2f" % (addr, first, last, drops, ratio))
    else:
        lines.append("  none found -- the timer may be two bytes, or held in an address above $00FF")

    # --- 2. the death: which byte loses exactly one --------------------------------
    lines.append("")
    lines.append("2. the death. Mario is held on screen, then lost, then is back: lives.")
    gaps = []
    for i in range(1, len(in_level) - 1):
        if in_level[i - 1][1] is not None and in_level[i][1] is None and in_level[i + 1][1] is not None:
            gaps.append(i)
    lines.append("  disappearance events inside the level: %d at frames %s"
                 % (len(gaps), [in_level[i][0] for i in gaps]))
    if gaps:
        i = gaps[0]
        before = in_level[max(0, i - 30)][2]
        after = None
        for j in range(i + 1, min(len(in_level), i + 120)):
            cand = in_level[j][2]
            if cand != before:
                after = cand
                lines.append("  RAM settled again at frame %d" % in_level[j][0])
                break
        if after:
            deltas = {a: (before[a], after[a]) for a in range(SCAN) if before[a] != after[a]}
            minus_one = sorted(a for a, (x, y) in deltas.items() if y == x - 1)
            lines.append("  bytes that dropped by exactly one: %s"
                         % (", ".join("$%04X" % a for a in minus_one) or "none"))
            for a in sorted(deltas)[:20]:
                lines.append("     $%04X  %3d -> %3d" % (a, deltas[a][0], deltas[a][1]))
    else:
        lines.append("  no disappearance observed; Mario may not have died in this window")

    # --- 3. world number: differs from the title screen, equals 1 in level --------
    lines.append("")
    lines.append("3. bytes that are 1 in-level but different on the title screen -> world")
    tmap = {}
    for r in title:
        for a in range(SCAN):
            tmap.setdefault(a, set()).add(r[2][a])
    worlds = []
    for addr in range(SCAN):
        vals = {r[2][addr] for r in in_level}
        if vals == {1} and tmap.get(addr, {1}) != {1}:
            worlds.append(addr)
    lines.append("  %s" % (", ".join("$%04X" % a for a in worlds) or "none"))

    # --- 4. X: fastest riser while running ---------------------------------------
    lines.append("")
    lines.append("4. bytes that rise fastest while holding RIGHT -> world X")
    movers = []
    for addr in range(SCAN):
        series = [r[2][addr] for r in in_level]
        span = series[-1] - series[0]
        if span > 8:
            movers.append((span, addr, series[0], series[-1]))
    movers.sort(reverse=True)
    for span, addr, first, last in movers[:8]:
        lines.append("  $%04X  %3d -> %3d   (+%d)" % (addr, first, last, span))

    report = "\n".join(lines)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    (ROOT / "logs" / "verify_ram.txt").write_text(report, encoding="utf-8")
    (ROOT / "logs" / "verify_ram_samples.json").write_text(
        json.dumps([{"frame": f, "mario_x": x, "ram": list(r)} for f, x, r in records]),
        encoding="utf-8")
    print(report, file=sys.stderr)
    print("\nreport -> logs/verify_ram.txt", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
