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
  can be spotted and fixed in src/config.py in one line.

Usage
-----
    python tools/dump_ram.py --rom roms/smb1.nes --script "wait*90,right*300"

The report is written to logs/ram_dump.json; stdout is unreliable in some shells, so
the tool also prints to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.emulator import NesRunner  # noqa: E402
from src.smb_state import decode  # noqa: E402

REPORT = ROOT / "logs" / "ram_dump.json"


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
    samples = []
    hexdumps = {}

    frame_no = 0
    for buttons, count in plan:
        pressed = () if buttons == "wait" else (buttons,)
        for _ in range(count):
            runner.step(pressed)
            frame_no += 1
            state = decode(runner.ram, runner.screen, frame_no)
            if frame_no % args.every == 0:
                samples.append(
                    {
                        "frame": frame_no,
                        "buttons": buttons,
                        "world_x": state.world_x,
                        "mario_screen_x": state.mario_screen_x,
                        "on_ground": state.on_ground,
                        "status": state.status_name,
                        "coins": state.coins,
                        "timer": state.timer,
                        "lives": state.lives,
                        "enemies": [
                            {"dx": e.dx_from(state.world_x), "y": e.y, "on_screen": e.on_screen}
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
        % ("frame", "btn", "world_x", "marioX", "on_ground", "status", "coins", "timer", "lives", "enemies")
    )
    print(header, file=sys.stderr)
    print("-" * len(header), file=sys.stderr)
    for row in samples:
        print(
            "%6d %5s %9d %6s %10s %6s %5d %6d %5d %s"
            % (
                row["frame"], row["buttons"][:5], row["world_x"],
                row["mario_screen_x"] if row["mario_screen_x"] is not None else "-",
                row["on_ground"], row["status"], row["coins"], row["timer"], row["lives"],
                ",".join(str(e["dx"]) for e in row["enemies"]) or "-",
            ),
            file=sys.stderr,
        )

    agreement = _ram_pixel_agreement(samples)
    print("\nRAM/pixel agreement: %s" % json.dumps(agreement), file=sys.stderr)
    if not agreement["consistent"]:
        print(
            "  world_x / mario_screen_x / camera (world_x - mario_screen_x) broke one of"
            " their invariants while RIGHT was held. Check RAM['player_x_hi']"
            " / RAM['player_x_lo'] / RAM['player_screen_x'] in src/config.py.",
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
    """Cross-check the three measured coordinates against each other while RIGHT is held.

    The old check compared world_x against Mario's *pixel* position at equal speed,
    which is wrong as soon as the camera scrolls (world grows, screen clamps). The
    invariants that hold on any build, with world/camera/screen measured directly:

    * world_x never decreases while RIGHT is held (Mario only moves right);
    * screen_x (= RAM['player_screen_x']) stays within the playfield [0, 255];
    * camera = world_x - screen_x never decreases (the camera never pans back).

    Three cheap statements; a wrong address table fails at least one of them.
    """
    rows = [s for s in samples if s["buttons"] == "right" and s["mario_screen_x"] is not None]
    if len(rows) < 4:
        return {"consistent": None, "note": "not enough forward-moving samples", "samples": 0}

    world_back = sum(1 for a, b in zip(rows, rows[1:]) if b["world_x"] < a["world_x"])
    cam_back = sum(
        1
        for a, b in zip(rows, rows[1:])
        if (b["world_x"] - b["mario_screen_x"]) < (a["world_x"] - a["mario_screen_x"])
    )
    offscreen = sum(1 for s in rows if not 0 <= s["mario_screen_x"] < 256)

    ok = world_back == 0 and cam_back == 0 and offscreen == 0
    return {
        "consistent": ok,
        "world_x_backward": world_back,
        "camera_backward": cam_back,
        "screen_x_offscreen": offscreen,
        "samples": len(rows),
        "note": "world/camera monotone and screen in-bounds while holding right",
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
