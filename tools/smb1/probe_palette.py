"""Print the actual colours of Mario's sprite instead of assuming them.

`src/vision.find_character` no longer looks for any colour at all -- it locates the
player geometrically, because palette matching failed on two of three real SMB1 dumps
(pure blue $0000FC is absent from the Virtual Console build entirely).

This tool remains useful for a different reason: it tells you what the character's
actual palette is, which is the thing you would need to know to build a
*colour-based* detector for a ROM where the geometric one struggles. It dumps every
colour in the sprite box and reports which classic thresholds each satisfies.

    python tools/smb1/probe_palette.py --rom path/to/smb.nes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.config import HUD_BOTTOM_ROW, SCREEN_H, SCREEN_W  # noqa: E402
from src.emulator import NesRunner  # noqa: E402
from src.vision import background_colour, ground_profile  # noqa: E402

# The thresholds find_mario currently uses.
TESTS = {
    "red  (r>190,g<60,b<60)": lambda r, g, b: (r > 190) & (g < 60) & (b < 60),
    "blue (b>200,r<90,g<90)": lambda r, g, b: (b > 200) & (r < 90) & (g < 90),
    "any red (r>140,r>g+40)": lambda r, g, b: (r > 140) & (r > g + 40) & (r > b + 40),
    "any blue (b>140,b>r+40)": lambda r, g, b: (b > 140) & (b > r + 40) & (b > g + 40),
    "dark (all<90)": lambda r, g, b: (r < 90) & (g < 90) & (b < 90),
}

SPRITE_HEIGHT = 24


def locate_mario_band(frame: np.ndarray):
    """A box just above the floor on the left, where Mario starts.

    Hand-picked coordinates were wrong twice: the first box held nothing but sky. The
    floor row comes from ground_profile(), so the box follows the actual level geometry
    instead of an assumption about it.
    """
    ground = ground_profile(frame, HUD_BOTTOM_ROW)
    rows = [g for g in ground if g < SCREEN_H]
    if not rows:
        return 0, HUD_BOTTOM_ROW, SCREEN_W, SCREEN_H
    floor = int(np.median(rows))
    return 16, max(HUD_BOTTOM_ROW, floor - SPRITE_HEIGHT), 120, floor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rom", required=True)
    parser.add_argument("--box", default=None,
                        help="override x0,y0,x1,y1; by default it is derived from the floor")
    args = parser.parse_args()

    runner = NesRunner(args.rom)
    # Into the level: START pulses, then a short run right.
    plan = [(("start",), 8), ((), 8)] * 6 + [((), 90), (("right",), 60), ((), 30)]
    for buttons, count in plan:
        for _ in range(count):
            runner.step(buttons)
    frame = np.array(runner.screen, copy=True)
    runner.close()

    OUT = ROOT / "logs" / "palette_probe.png"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        Image.fromarray(frame).resize((SCREEN_W * 3, SCREEN_H * 3), Image.NEAREST).save(OUT)
        print("wrote %s" % OUT, file=sys.stderr)
    except ImportError:
        pass

    if args.box:
        x0, y0, x1, y1 = (int(v) for v in args.box.split(","))
    else:
        x0, y0, x1, y1 = locate_mario_band(frame)

    sprite = frame[y0:y1, x0:x1, :].astype(int)
    r, g, b = sprite[:, :, 0], sprite[:, :, 1], sprite[:, :, 2]
    flat = sprite.reshape(-1, 3)
    colours, counts = np.unique(flat, axis=0, return_counts=True)
    order = np.argsort(-counts)

    lines = [
        "frame is %dx%d, HUD occupies the top %d rows" % (SCREEN_W, SCREEN_H, HUD_BOTTOM_ROW),
        "backdrop colour inferred from the frame: #%02X%02X%02X"
        % tuple(int(v) for v in background_colour(frame, HUD_BOTTOM_ROW)),
        "sprite box x=%d..%d y=%d..%d  (%d pixels)"
        % (x0, x1, y0, y1, flat.shape[0]),
        "",
        "colours in the box, most common first, with the tests find_mario applies:",
        "  %-18s %5s  %s" % ("colour", "count", "matches"),
    ]
    for i in order:
        cr, cg, cb = (int(v) for v in colours[i])
        one_r = np.full_like(r, cr)
        one_g = np.full_like(g, cg)
        one_b = np.full_like(b, cb)
        hits = [name for name, fn in TESTS.items() if bool(fn(one_r, one_g, one_b).any())]
        lines.append("  #%02X%02X%02X %9d  %s" % (cr, cg, cb, counts[i], ", ".join(hits) or "-"))

    lines.append("")
    lines.append("aggregate over the whole box:")
    for name, fn in TESTS.items():
        lines.append("  %-22s %5d px" % (name, int(fn(r, g, b).sum())))

    lines.append("")
    lines.append("pure blue #0000FC present: %s"
                 % bool(((r == 0) & (g == 0) & (b == 252)).any()))
    lines.append("mario red  #D82800 present: %s"
                 % bool(((r == 0xD8) & (g == 0x28) & (b == 0x00)).any()))

    report = "\n".join(lines)
    (ROOT / "logs" / "palette_probe.txt").write_text(report, encoding="utf-8")
    print("\n" + report, file=sys.stderr)
    print("\nreport -> logs/palette_probe.txt", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
