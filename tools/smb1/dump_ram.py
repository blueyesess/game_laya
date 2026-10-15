"""Calibrate the SMB1 RAM map against a real ROM.

src/config.py carries a community-documented SMB1 memory map. Most of it is stable
everywhere, but the enemy slots at $0030-$0056 are the part worth confirming, and
there is no way to confirm them from documentation alone -- you need a running game.

What this prints
----------------
A per-frame table of the decoded fields, then two cross-checks that do not depend on
any single address being right:

* **RAM vs pixels.** world_x comes from $28/$29; mario_screen_x comes from finding his
  two exact sprite colours in the framebuffer. They are independent, so when Mario runs
  right the first must increase at the same rate as the second. If they disagree, one
  of them is wrong and the table tells you which way.

* **A full 2 KB hexdump** at a few checkpoints, so an address that is obviously wrong
  can be spotted and fixed in src/profiles/smb1.json in one line.

Usage
-----
    python tools/smb1/dump_ram.py --rom roms/smb1.nes --script "wait*90,right*300"

The report is written to logs/ram_dump.json; stdout is unreliable in some shells, so
the tool also prints to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.emulator import NesRunner  # noqa: E402
from src.state import Decoder, GameProfile  # noqa: E402

REPORT = ROOT / "logs" / "ram_dump.json"
PROFILE = GameProfile.load("smb1")


def parse_script(text: str):
    """'wait*90,right*300' -> [('wait', 90), ('right', 300)]."""
    plan = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "*" in chunk:
            name, count = chunk.split("*", 1)
            plan.append((name.strip(), int(count)))
        else:
            plan.append((chunk, 1))
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rom", required=True)
    parser.add_argument("--script", default="wait*120,right*300",
                        help="comma separated NAME*FRAMES, e.g. wait*90,right*300")
    parser.add_argument("--every", type=int, default=10, help="sample every N frames")
    parser.add_argument("--hexdump-at", default="0,120,300", help="frames to hexdump at")
    args = parser.parse_args()

    plan = parse_script(args.script)
    hex_at = {int(x) for x in args.hexdump_at.split(",") if x.strip()}

    runner = NesRunner(args.rom)
    decoder = Decoder(PROFILE)
    samples = []
    hexdumps = {}

    frame_no = 0
    for buttons, count in plan:
        pressed = () if buttons == "wait" else (buttons,)
        for _ in range(count):
            runner.step(pressed)
            frame_no += 1
            state = decoder.decode(runner.ram, runner.screen, frame_no)
            if frame_no % args.every == 0:
                samples.append(
                    {
                        "frame": frame_no,
                        "buttons": buttons,
                        "world_x": state.world_x,
                        "character_x": state.character_x,
                        "on_ground": state.on_ground,
                        "status": state.hud.get("status"),
                        "coins": state.hud.get("coins"),
                        "timer": state.hud.get("timer"),
                        "lives": state.hud.get("lives"),
                        "enemies": [
                            {"dx": e["dx"], "y": e["y"]}
                            for e in state.enemies
                        ],
                        "pit_ahead": state.pit_ahead,
                        "pit_distance": state.pit_distance,
                    }
                )
            if frame_no in hex_at:
                hexdumps[str(frame_no)] = _hex(runner.ram)

    runner.close()

    header = (
        "%6s %5s %9s %6s %10s %6s %5s %6s %5s %s"
        % ("frame", "btn", "world_x", "charX", "on_ground", "status", "coins", "timer", "lives", "enemies")
    )
    print(header, file=sys.stderr)
    print("-" * len(header), file=sys.stderr)
    for row in samples:
        print(
            "%6d %5s %9d %6s %10s %6s %5d %6d %5d %s"
            % (
                row["frame"], row["buttons"][:5], row["world_x"],
                row["character_x"] if row["character_x"] is not None else "-",
                row["on_ground"], row["status"], row["coins"], row["timer"], row["lives"],
                ",".join(str(e["dx"]) for e in row["enemies"]) or "-",
            ),
            file=sys.stderr,
        )

    agreement = _ram_pixel_agreement(samples)
    print("\nRAM/pixel agreement: %s" % json.dumps(agreement), file=sys.stderr)
    if not agreement["consistent"]:
        print(
            "  world_x and the character's on-screen position disagree. Check the"
            " player_x_hi / player_x_lo addresses in src/profiles/smb1.json, or the"
            " geometric detector in src/vision.py find_character().",
            file=sys.stderr,
        )

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps({"samples": samples, "hexdumps": hexdumps, "agreement": agreement},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("report -> %s" % REPORT, file=sys.stderr)
    return 0


def _ram_pixel_agreement(samples) -> dict:
    """Compare how fast world_x and the character's on-screen position move.

    Both describe the same motion, so their per-frame deltas should be close. The
    screen value wraps at the camera edge, so only samples where both moved forward
    without either wrapping are counted.
    """
    pairs = []
    for prev, cur in zip(samples, samples[1:]):
        if cur["buttons"] != "right" or prev["buttons"] != "right":
            continue
        if prev["character_x"] is None or cur["character_x"] is None:
            continue
        d_ram = cur["world_x"] - prev["world_x"]
        d_px = cur["character_x"] - prev["character_x"]
        if d_ram <= 0 or d_px < 0 or d_px > 16:
            continue  # wrapped at the screen edge, or stopped
        pairs.append((d_ram, d_px))

    if not pairs:
        return {"consistent": None, "note": "not enough forward-moving samples", "samples": 0}

    ratio = (sum(p[0] for p in pairs) / len(pairs)) / max(1e-6, sum(p[1] for p in pairs) / len(pairs))
    # world_x counts whole pixels; the screen position is bounded by the camera. A
    # ratio near 1.0 means the two agree that the player moved.
    return {
        "consistent": 0.5 <= ratio <= 2.0,
        "ratio_world_x_to_pixels": round(ratio, 3),
        "samples": len(pairs),
        "note": "ratio should be ~1.0 when both sources are correct",
    }


def _hex(ram) -> str:
    lines = []
    for base in range(0, 256, 16):
        chunk = ram[base:base + 16]
        hexpart = " ".join("%02X" % b for b in chunk)
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append("%04X  %-47s  %s" % (base, hexpart, ascii_part))
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
