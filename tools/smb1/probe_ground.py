"""Directly exercise ground_profile() and background_colour() on a saved frame.

probe_palette reported "sprite box x=0..256 y=32..240", which is its fallback for
"ground_profile found no solid pixel in any column". That is either a real failure in
the function or a fallback bug, and guessing between them from the outside wastes more
time than printing the intermediate values.

Both functions now live in the generic vision layer (src/vision.py), so this tool works
on any ROM's saved frame, not just SMB1's.

    python tools/probe_ground.py logs/palette_probe.png
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.config import HUD_BOTTOM_ROW, SCREEN_H  # noqa: E402
from src.vision import background_colour, ground_profile  # noqa: E402


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "logs" / "palette_probe.png"
    if not path.exists():
        raise SystemExit("no such file: %s" % path)

    from PIL import Image

    frame = np.array(Image.open(path).convert("RGB").resize((256, 240), Image.NEAREST))
    # The saved PNG is 3x, so scale it back down before decoding.

    lines = []
    backdrop = background_colour(frame, HUD_BOTTOM_ROW)
    lines.append("backdrop colour  : #%02X%02X%02X" % tuple(int(v) for v in backdrop))
    lines.append("frame shape      : %s" % (frame.shape,))

    playfield = frame[HUD_BOTTOM_ROW:, :, :].astype(np.int16)
    dist = np.abs(playfield - backdrop.astype(np.int16)).sum(axis=2)
    lines.append("L1 distance to backdrop, per-row minimum:")
    for row in range(0, playfield.shape[0], 16):
        band = dist[row:row + 16]
        lines.append("  playfield row %3d (screen %3d): min=%3d  median=%3d"
                     % (row, row + HUD_BOTTOM_ROW, int(band.min()), int(np.median(band))))

    profile = ground_profile(frame, HUD_BOTTOM_ROW)
    solid_cols = profile < SCREEN_H
    lines.append("")
    lines.append("columns with a solid pixel : %d / %d" % (int(solid_cols.sum()), len(profile)))
    if solid_cols.any():
        rows = profile[solid_cols]
        lines.append("surface row  min/median/max : %d / %d / %d"
                     % (int(rows.min()), int(np.median(rows)), int(rows.max())))
        lines.append("profile[0:16]               : %s" % list(profile[:16]))
        lines.append("profile[120:136]            : %s" % list(profile[120:136]))
    else:
        lines.append("NO solid columns -> ground_profile cannot see the floor at all")
        uniq, counts = np.unique(frame[HUD_BOTTOM_ROW:, :, :].reshape(-1, 3), axis=0, return_counts=True)
        order = np.argsort(-counts)[:8]
        lines.append("most common playfield colours:")
        for i in order:
            lines.append("  #%02X%02X%02X  %d px" % (tuple(int(v) for v in uniq[i]) + (int(counts[i]),)))

    report = "\n".join(lines)
    (ROOT / "logs" / "ground_probe.txt").write_text(report, encoding="utf-8")
    print(report, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
