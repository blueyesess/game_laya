"""Decide which of nes_py's two frame accessors actually returns a coherent picture.

`NESEnv.screen` is a no-copy strided numpy view built in _native.pyx:

    dims    = (SCREEN_HEIGHT, SCREEN_WIDTH, 3)      # (240, 256, 3)
    strides = (SCREEN_WIDTH * 4, 4, -1)             # (1024, 4, -1)
    base    = buffer + 2                            # little-endian BGRA

That is correct *if* the native buffer is 32 bits per pixel. If it is 24 bits per
pixel, the true row stride is 768 and every row is read 256 bytes too far in, which
shears the image into progressive diagonal banding while leaving the HUD text, the
blocks and the ground all recognisably in the right places.

`NESEnv._env.copy_screen_rgb()` is the other accessor: it walks the buffer as
`NES_Pixel` (uint32) in C++ and writes a C-contiguous 24-bit array.

Rather than reason about which is right, render both and look. This saves a PNG per
candidate next to each other so the answer is visual and immediate.

    python tools/probe_screen.py [--rom path]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.config import SCREEN_H, SCREEN_W  # noqa: E402

OUT = ROOT / "logs"


def find_rom(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    hits = [p for p in ROOT.rglob("*.nes") if ".venv" not in str(p) and "probe_" not in p.name
            and "variant_" not in p.name]
    hits.sort(key=lambda p: ("inputtest" in p.name, len(str(p))))
    if not hits:
        raise SystemExit("no .nes found under %s" % ROOT)
    return hits[0]


def save(image: np.ndarray, path: Path, scale: int = 3) -> None:
    from PIL import Image

    arr = np.ascontiguousarray(image, dtype=np.uint8)
    out = Image.fromarray(arr)
    if scale != 1:
        out = out.resize((out.width * scale, out.height * scale), Image.NEAREST)
    out.save(path)
    print("  wrote %s  %dx%d" % (path.name, arr.shape[1], arr.shape[0]), file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rom", default=None)
    args = parser.parse_args()

    rom = find_rom(args.rom)
    print("rom: %s (%d bytes)" % (rom, rom.stat().st_size), file=sys.stderr)

    from nes_py import NESEnv

    env = NESEnv(str(rom))
    # NESEnv.__init__ leaves done=True, so step() is illegal until reset() is called.
    env.reset()

    # Get into the level: mash START past the title screen, then run right.
    # buttons_to_byte, not env.get_keys_to_action(): the latter is keyed by the
    # single characters it happens to bind to keyboard events ('d', 'a', ...).
    from src.config import buttons_to_byte

    plan = [(("start",), 30), ((), 90), (("start",), 20), ((), 60),
            (("right",), 200), ((), 60)]
    for buttons, count in plan:
        byte = buttons_to_byte(buttons)
        for _ in range(count):
            env.step(byte)

    candidates = {}

    view = env.screen
    candidates["A_env_screen_zerocopy"] = np.array(view, copy=True)

    native = env._env
    if hasattr(native, "copy_screen_rgb"):
        candidates["B_copy_screen_rgb"] = native.copy_screen_rgb().copy()
    if hasattr(native, "copy_screen_grayscale"):
        gray = native.copy_screen_grayscale().copy()
        candidates["C_grayscale"] = np.stack([gray] * 3, axis=-1)

    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "rom": str(rom),
        "env_screen_shape": list(view.shape),
        "env_screen_strides": list(view.strides),
        "env_screen_c_contiguous": bool(view.flags["C_CONTIGUOUS"]),
        "env_screen_dtype": str(view.dtype),
        "candidates": {},
    }

    for name, image in candidates.items():
        save(image, OUT / ("screen_%s.png" % name))
        flat = image.reshape(-1, 3).astype(int)
        # A coherent NES frame has few distinct colours (a fixed 64-entry palette);
        # a sheared one explodes into noise because every pixel samples the wrong
        # byte. This is a cheap quantitative stand-in for looking at the PNG.
        report["candidates"][name] = {
            "shape": list(image.shape),
            "unique_colours": int(len(np.unique(flat, axis=0))),
            "channel_means": [round(float(v), 1) for v in flat.mean(axis=0)],
        }
        print("  %-28s unique colours = %d" % (name, report["candidates"][name]["unique_colours"]),
              file=sys.stderr)

    if "A_env_screen_zerocopy" in candidates and "B_copy_screen_rgb" in candidates:
        same = np.array_equal(candidates["A_env_screen_zerocopy"], candidates["B_copy_screen_rgb"])
        report["A_equals_B"] = bool(same)
        print("\n  zero-copy view == C-contiguous copy: %s" % same, file=sys.stderr)

    # The server's contract with the browser: /api/frame.raw must be exactly the
    # 24-bit RGB stream, row-major, in the same order createImageData expects before
    # the page expands it. Pin it here so a future "optimisation" of the encoder
    # cannot quietly reorder or pad pixels.
    from src.emulator import frame_to_rgb_bytes

    served = np.frombuffer(frame_to_rgb_bytes(view), dtype=np.uint8).reshape(SCREEN_H, SCREEN_W, 3)
    faithful = np.array_equal(served, candidates["A_env_screen_zerocopy"])
    report["served_bytes_are_faithful_rgb"] = bool(faithful)
    check_line = "  /api/frame.raw payload == framebuffer (RGB, row-major): %s" % faithful
    print(check_line, file=sys.stderr)

    (OUT / "screen_probe.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    print("\n  report -> logs/screen_probe.json", file=sys.stderr)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
