"""Run the agent headlessly and record what actually happens, frame by frame.

There is a specific question this answers: does the ROM play at all under the reflex
policy? Because if it does, then a hybrid run sitting on the title screen is not a ROM
problem -- it is the model answering `move: wait` and the player standing still.

So this drives a real GameRunner with no server, no browser and no model, samples the
frame periodically, and writes both a PNG per sample and a table saying whether the
player was visible and whether the floor ahead was solid. The PNGs are the point: a
table saying "player not visible" cannot distinguish a title screen from a detection
failure.

    python tools/run_headless.py --rom path --seconds 60 [--policy reflex] [--profile generic]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.agent import GameRunner  # noqa: E402
from src.config import SCREEN_H, SCREEN_W  # noqa: E402
from src.state import GameProfile  # noqa: E402

OUT = ROOT / "logs" / "headless"


def save(frame: np.ndarray, path: Path, scale: int = 3) -> None:
    from PIL import Image

    arr = np.ascontiguousarray(frame, dtype=np.uint8)
    img = Image.fromarray(arr)
    if scale != 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    img.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rom", required=True)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--policy", default="reflex", choices=("reflex", "laya", "hybrid"))
    parser.add_argument("--profile", default="generic", help="game profile id from src/profiles/*.json")
    parser.add_argument("--every", type=float, default=3.0, help="seconds between samples")
    parser.add_argument("--max-shots", type=int, default=20)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for stale in OUT.glob("*.png"):
        stale.unlink()

    runner = GameRunner(
        rom_path=args.rom,
        policy=args.policy,
        profile=GameProfile.load(args.profile),
    )
    runner.start()

    deadline = time.perf_counter() + args.seconds
    next_sample = time.perf_counter() + args.every
    shots = 0
    rows = []

    print("%6s %6s %8s %9s %6s %6s %8s %s"
          % ("t(s)", "frame", "player_x", "on_gnd", "pit", "decis", "src", "buttons"),
          file=sys.stderr)
    while time.perf_counter() < deadline:
        time.sleep(0.25)
        st = runner.state
        if st is None:
            continue
        stats = runner.stats
        latest = runner.trace[-1] if runner.trace else {}
        row = "%6.1f %6d %8s %9s %6s %6d %8s %s" % (
            args.seconds - (deadline - time.perf_counter()),
            st.frame,
            st.character_x if st.character_x is not None else "-",
            st.on_ground,
            st.pit_ahead,
            stats["decisions"],
            latest.get("source", "-"),
            ",".join(runner.snapshot()["buttons"]) or "-",
        )
        rows.append(row)
        print(row, file=sys.stderr)

        if time.perf_counter() >= next_sample and shots < args.max_shots:
            next_sample = time.perf_counter() + args.every
            # Decode the snapshot bytes rather than reading runner.runner.screen: that
            # is a live view into the emulator's buffer and this is a different thread.
            payload = runner.frame_bytes
            if payload:
                arr = np.frombuffer(payload, dtype=np.uint8).reshape(SCREEN_H, SCREEN_W, 3)
                save(arr, OUT / ("t%02d_f%06d.png" % (shots, st.frame)))
            shots += 1

    runner.stop()
    final = runner.stats
    print("\nframes=%d decisions=%d (laya=%d reflex=%d) errors=%d"
          % (final["frames"], final["decisions"], final["laya_decisions"],
             final["reflex_decisions"], final["errors"]), file=sys.stderr)
    print("shots -> %s" % OUT, file=sys.stderr)
    (OUT / "log.txt").write_text("\n".join(rows), encoding="utf-8")

    # A frame with the player on it and no title card is the whole ballgame.
    seen = [r for r in rows if r.split()[2] != "-"]
    print("samples with player visible: %d / %d" % (len(seen), len(rows)), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
