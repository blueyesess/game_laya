"""End-to-end check of the emulator -> HTTP -> canvas pixel path.

`tools/check_dashboard.py` can prove the page does not memcpy RGB into RGBA. It cannot
prove the shipped `blit()` produces the *right* RGBA, because that needs to execute the
JavaScript. This does, in three steps:

1. capture a real frame from the emulator and write it as raw RGB
2. run tools/blit_check.js, which extracts the actual blit out of web/index.html and
   expands the real frame with it
3. compare the RGBA it produced against the canonical expansion computed here

Step 3 is what makes this worth running: it verifies the bytes the browser will
actually put on screen, not a paraphrase of them.

    python tools/check_blit.py [--rom path]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.config import SCREEN_H, SCREEN_W  # noqa: E402
from src.emulator import NesRunner, frame_to_rgb_bytes  # noqa: E402

OUT = ROOT / "logs"
REPORT = OUT / "blit_check.txt"

results = []


def check(name, ok, detail=""):
    results.append({"name": name, "ok": bool(ok), "detail": detail})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, (" -- " + detail) if detail else ""),
          file=sys.stderr)
    return bool(ok)


def find_rom(explicit):
    if explicit:
        return Path(explicit)
    hits = [p for p in ROOT.rglob("*.nes")
            if ".venv" not in str(p) and "probe_" not in p.name and "variant_" not in p.name]
    hits.sort(key=lambda p: ("inputtest" in p.name, len(str(p))))
    return hits[0] if hits else None


def canonical_rgba(rgb: np.ndarray) -> np.ndarray:
    """The RGBA layout createImageData expects: 4 bytes per pixel, alpha opaque."""
    n = rgb.shape[0] * rgb.shape[1]
    out = np.empty((n, 4), dtype=np.uint8)
    out[:, :3] = rgb.reshape(n, 3)
    out[:, 3] = 255
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rom", default=None)
    args = parser.parse_args()

    rom = find_rom(args.rom)
    if rom is None:
        raise SystemExit("no .nes found under %s" % ROOT)
    print("rom: %s" % rom, file=sys.stderr)

    runner = NesRunner(str(rom))
    # Title screen -> level card -> gameplay, so the frame has sky, blocks, ground
    # and a moving sprite rather than one flat colour.
    for buttons, count in ((("start",), 30), ((), 90), (("start",), 20), ((), 60),
                           (("right",), 200)):
        for _ in range(count):
            runner.step(buttons)

    rgb = np.frombuffer(frame_to_rgb_bytes(runner.screen), dtype=np.uint8)
    runner.close()

    check("served frame is W*H*3 bytes", rgb.size == SCREEN_W * SCREEN_H * 3,
          "%d bytes" % rgb.size)

    OUT.mkdir(parents=True, exist_ok=True)
    rgb_path = OUT / "frame.rgb"
    rgba_path = OUT / "frame_rgba.bin"
    rgb.tofile(rgb_path)

    proc = subprocess.run(
        ["node", str(ROOT / "tools" / "blit_check.js"),
         str(ROOT / "web" / "index.html"), str(rgb_path), str(rgba_path),
         str(SCREEN_W), str(SCREEN_H)],
        capture_output=True, text=True,
    )
    for line in (proc.stdout or "").splitlines():
        print("  " + line, file=sys.stderr)
    for line in (proc.stderr or "").splitlines():
        if line.strip():
            print("  " + line, file=sys.stderr)
    check("blit_check.js ran clean", proc.returncode == 0, "exit %d" % proc.returncode)

    if not rgba_path.exists():
        check("blit_check.js wrote the expanded frame", False, "no output file")
    else:
        produced = np.frombuffer(rgba_path.read_bytes(), dtype=np.uint8)
        expected = canonical_rgba(rgb.reshape(SCREEN_H, SCREEN_W, 3)).reshape(-1)
        check("expanded frame has W*H*4 bytes",
              produced.size == SCREEN_W * SCREEN_H * 4, "%d bytes" % produced.size)
        identical = produced.size == expected.size and np.array_equal(produced, expected)
        mismatches = int((produced[: min(produced.size, expected.size)]
                          != expected[: min(produced.size, expected.size)]).sum()) \
            if produced.size == expected.size else -1
        check("the bytes the browser will paint equal the canonical RGBA expansion",
              identical, "%d differing bytes" % mismatches if not identical else "exact")

    passed = sum(1 for r in results if r["ok"])
    report = ["%d/%d passed" % (passed, len(results))]
    report += ["  [%s] %s%s" % ("PASS" if r["ok"] else "FAIL", r["name"],
                                (" -- " + r["detail"]) if r["detail"] else "")
               for r in results]
    REPORT.write_text("\n".join(report), encoding="utf-8")
    print("\n%d/%d passed -> %s" % (passed, len(results), REPORT), file=sys.stderr)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
