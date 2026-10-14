"""End-to-end smoke test on the homebrew ROM.

This is the only part of the project that can be proven without a Super Mario Bros.
ROM, so it checks everything that is not SMB1-specific:

1. the test ROM assembles into a structurally valid iNES file
2. nes_py hands back a 240x256x3 framebuffer
3. **src/config.py's button bitmap agrees with nes_py's own table**, for all 256
   combinations -- checked against the emulator rather than against my memory
4. button presses reach the 6502 program, which reads them back into RAM at $0301/$0300
   with A in bit 0 ... RIGHT in bit 7
5. the frame actually changes colour with the controller, i.e. the video path works
6. smb_state.decode / scene.build_state survive a real framebuffer
7. the Flask app imports and exposes its routes

Results are written to logs/smoke_report.json as well as stderr, because stdout is
swallowed by some shells.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import BUTTON_BITS, buttons_to_byte  # noqa: E402
from src.smb_state import decode  # noqa: E402

ROM = ROOT / "roms" / "inputtest.nes"
REPORT = ROOT / "logs" / "smoke_report.json"

# Where the test ROM parks the button byte, and the per-vblank heartbeat.
RAM_BUTTONS = 0x0300
RAM_NMI_COUNT = 0x0302

results = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append({"name": name, "ok": bool(ok), "detail": detail})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, (" -- " + detail) if detail else ""), file=sys.stderr)
    return bool(ok)


def settle(runner, frames: int = 60):
    """Run frames with no buttons and return the last one."""
    frame = None
    for _ in range(frames):
        frame = runner.step(())
    return frame


def main() -> int:
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- 1. ROM
    from tools.make_test_rom import build_rom

    rom_bytes = build_rom()
    check(
        "test ROM is a valid iNES image",
        rom_bytes[:4] == b"NES\x1a" and len(rom_bytes) == 16 + 32768 + 8192,
        "%d bytes" % len(rom_bytes),
    )
    ROM.parent.mkdir(parents=True, exist_ok=True)
    ROM.write_bytes(rom_bytes)

    # ---------------------------------------------------------------- 2. emulator
    from src.emulator import NesRunner

    runner = NesRunner(str(ROM))
    screen = runner.screen
    check(
        "framebuffer is 240x256x3 uint8",
        screen.shape == (240, 256, 3) and screen.dtype.name == "uint8",
        "%s %s" % (screen.shape, screen.dtype),
    )
    ram = runner.ram
    check("RAM view is 2048 bytes", ram.shape[0] == 2048, str(ram.shape))

    # ---------------------------------------------------- 3. button bitmap agrees
    env = runner.env
    if hasattr(env, "get_keys_to_action"):
        nespy_keys = {
            ord("d"): "right", ord("a"): "left", ord("s"): "down", ord("w"): "up",
            ord("\r"): "start", ord(" "): "select", ord("p"): "B", ord("o"): "A",
        }
        table = env.get_keys_to_action()
        names = list(BUTTON_BITS)
        mismatches = []
        for mask in range(256):
            combo = tuple(names[i] for i in range(len(names)) if mask & (1 << i))
            theirs = table[tuple(sorted(nespy_keys_for(combo, nespy_keys)))]
            ours = buttons_to_byte(combo)
            if ours != theirs:
                mismatches.append("%s: ours=%d theirs=%d" % (combo, ours, theirs))
        check(
            "our button bitmap matches nes_py for all 256 combos",
            not mismatches,
            "; ".join(mismatches[:3]) if mismatches else "256/256 identical",
        )
    else:
        check("nes_py exposes get_keys_to_action()", False, "skipped: no such method")

    # ------------------------------------------------------------ 4. inputs land
    settle(runner)
    beats = int(runner.ram[RAM_NMI_COUNT]) & 0xFF
    check(
        "emulator delivers NMI (per-vblank heartbeat in $0302)",
        beats > 0,
        "heartbeat=%d after 60 frames%s"
        % (beats, " -- NMI never fired, so controller reads cannot be tested" if beats == 0 else ""),
    )

    idle_byte = int(runner.ram[RAM_BUTTONS]) & 0xFF
    check("idle buttons read back as 0x00", idle_byte == 0x00, "got 0x%02X" % idle_byte)

    for button, expected in (("right", 0x80), ("left", 0x40), ("A", 0x01), ("B", 0x02)):
        for _ in range(6):
            runner.step((button,))
        got = int(runner.ram[RAM_BUTTONS]) & 0xFF
        check(
            "pressing %s sets bit for %s" % (button, button),
            got == expected,
            "expected 0x%02X got 0x%02X" % (expected, got),
        )

    # --------------------------------------------------------- 5. video responds
    settle(runner)
    idle_frame = settle(runner, 4).copy()
    for _ in range(8):
        right_frame = runner.step(("right",))
    right_frame = right_frame.copy()
    check(
        "framebuffer is a single flat colour",
        _is_flat(idle_frame),
        "idle palette pixel %s" % (idle_frame[120, 128],),
    )
    diff = abs(int(idle_frame[120, 128].astype(int).sum()) - int(right_frame[120, 128].astype(int).sum()))
    check("holding RIGHT changes what the PPU draws", diff > 0, "channel sum delta %d" % diff)

    # ------------------------------------------------------- 6. decode survives
    try:
        state = decode(runner.ram, right_frame, runner.frame_index)
        obs_fields = state.to_dict()
        ok = "world_x" in obs_fields and "status" in obs_fields
        check("smb_state.decode() handles a real frame", ok, json.dumps(obs_fields)[:160])
    except Exception as exc:  # noqa: BLE001
        check("smb_state.decode() handles a real frame", False, repr(exc))
        traceback.print_exc(file=sys.stderr)

    from src.scene import build_state

    try:
        obs = build_state(state)
        # The observation is deliberately tiny -- laya re-encodes it once per question,
        # so every word costs three forward passes' worth of context. Check the shape,
        # not the richness.
        check(
            "scene.build_state() produces a laya state",
            isinstance(obs, dict) and isinstance(obs.get("scene"), str) and obs["scene"],
            obs.get("scene", "")[:120],
        )
        check(
            "an unverified RAM map suppresses the HUD numbers instead of inventing them",
            (not state.ram_plausible)
            and "hud" not in obs
            and "unknown" in obs["scene"].lower(),
            "ram_plausible=%s keys=%s" % (state.ram_plausible, sorted(obs)),
        )
    except Exception as exc:  # noqa: BLE001
        check("scene.build_state() produces a laya state", False, repr(exc))

    # ------------------------------------------------------------- 7. web routes
    try:
        from src.server import app

        rules = {rule.rule for rule in app.url_map.iter_rules()}
        needed = {"/", "/api/frame.raw", "/api/snapshot", "/api/trace", "/api/control"}
        missing = needed - rules
        check("Flask app exposes its routes", not missing, "missing: %s" % sorted(missing))
    except Exception as exc:  # noqa: BLE001
        check("Flask app exposes its routes", False, repr(exc))

    runner.close()

    passed = sum(1 for r in results if r["ok"])
    payload = {
        "passed": passed,
        "total": len(results),
        "results": results,
    }
    REPORT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n%d/%d passed -> %s" % (passed, len(results), REPORT), file=sys.stderr)
    return 0 if passed == len(results) else 1


def nespy_keys_for(combo, nespy_keys):
    """Translate our button names into the single-char keys nes_py uses."""
    inverse = {name: key for key, name in nespy_keys.items()}
    return [inverse[name] for name in combo]


def _is_flat(frame) -> bool:
    """True when a blank-tile screen shows nothing but the backdrop colour."""
    import numpy as np

    return len(np.unique(frame.reshape(-1, 3), axis=0)) <= 2


if __name__ == "__main__":
    raise SystemExit(main())
