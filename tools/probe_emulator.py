"""Bisect nes_py with deliberately tiny ROMs, one layer at a time.

smoke_test.py showed the framebuffer, RAM window and button bitmap are all healthy,
but that a ROM of ours did not appear to run. The failure turned out to be a bug in
*our* ROM, not the emulator: iNES byte 4 counts PRG in 16 KB units, so declaring a
32 KB PRG as "1" makes the core load only the first half, which hides the $FFFA
vectors and leaves the CPU resetting to $0000. This file is what caught that, and it
stays in the tree because the next layer down will fail for a similar reason.

Layers, each doing strictly less than the last:

  cpu    no PPU access: writes a constant, then free-runs a counter in X.
  ctrl   strobes $4016 and stores each of the eight reads separately at $0310-$0317,
         so a controller problem shows up as "which read returned what", not "0".
  ppu    writes the backdrop palette and enables rendering. No interrupts.
  nmi    as ppu, plus NMI enabled and a heartbeat at $0302, so a missing interrupt is
         distinguishable from a broken controller read.

    python tools/probe_emulator.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.emulator import NesRunner  # noqa: E402
from tools.make_test_rom import assemble  # noqa: E402

BASE = 0x8000

# --- cpu: no PPU access at all ----------------------------------------------------
PROG_CPU = [
    ("LABEL", "reset"),
    ("SEI", None), ("CLD", None),
    ("LDX", ("#", 0xFF)), ("TXS", None),
    ("LDA", ("#", 0x42)), ("STA", ("$", 0x0300)),
    ("LABEL", "spin"),
    ("INX", None), ("BNE", "spin"),
    ("INC", ("$", 0x0301)),      # not STX: X is 0 here, that is why BNE just fell through
    ("JMP", "spin"),
]

# --- ctrl: eight individual reads parked at $0310..$0317 --------------------------
PROG_CTRL = [
    ("LABEL", "reset"),
    ("SEI", None), ("CLD", None),
    ("LDX", ("#", 0xFF)), ("TXS", None),
    ("LDA", ("#", 0x42)), ("STA", ("$", 0x0300)),
    ("LABEL", "spin"),
    ("LDA", ("#", 0x01)), ("STA", ("$", 0x4016)),   # strobe high
    ("LDA", ("#", 0x00)), ("STA", ("$", 0x4016)),   # strobe low latches the pad
    ("LDX", ("#", 0x00)),
    ("LABEL", "rd"),
    ("LDA", ("$", 0x4016)),
    ("STA", ("$X", 0x0310)),                        # one byte per read
    ("INX", None),
    ("CPX", ("#", 0x08)),
    ("BNE", "rd"),
    ("JMP", "spin"),
]

# --- ppu: backdrop palette + rendering, no interrupts -----------------------------
PROG_PPU = [
    ("LABEL", "reset"),
    ("SEI", None), ("CLD", None),
    ("LDX", ("#", 0xFF)), ("TXS", None),
    ("LDX", ("#", 0x00)),
    ("STX", ("$", 0x2000)),        # NMI off
    ("STA", ("$", 0x4015)),
    ("LABEL", "vb1"), ("BIT", ("$", 0x2002)), ("BPL", "vb1"),
    ("LABEL", "vb2"), ("BIT", ("$", 0x2002)), ("BPL", "vb2"),
    ("LDA", ("#", 0x42)), ("STA", ("$", 0x0300)),
    ("LDA", ("#", 0x3F)), ("STA", ("$", 0x2006)),
    ("LDA", ("#", 0x00)), ("STA", ("$", 0x2006)),
    ("LDA", ("#", 0x16)), ("STA", ("$", 0x2007)),   # red backdrop
    ("LDA", ("#", 0x1E)), ("STA", ("$", 0x2001)),   # show background everywhere
    ("LABEL", "main"), ("JMP", "main"),
]

# --- nmi: as ppu, plus NMI enabled and a heartbeat -------------------------------
# Written out rather than concatenated onto PROG_PPU: that one ends with `JMP main`,
# so anything appended after it is unreachable dead code -- which is exactly how the
# first version of this probe managed to report "NMI never fires" about itself.
PROG_NMI = [
    ("LABEL", "reset"),
    ("SEI", None), ("CLD", None),
    ("LDX", ("#", 0xFF)), ("TXS", None),
    ("LDX", ("#", 0x00)),
    ("STX", ("$", 0x2000)),        # NMI off
    ("STA", ("$", 0x4015)),
    ("LABEL", "vb1"), ("BIT", ("$", 0x2002)), ("BPL", "vb1"),
    ("LABEL", "vb2"), ("BIT", ("$", 0x2002)), ("BPL", "vb2"),
    ("LDA", ("#", 0x42)), ("STA", ("$", 0x0300)),
    ("LDA", ("#", 0x3F)), ("STA", ("$", 0x2006)),
    ("LDA", ("#", 0x00)), ("STA", ("$", 0x2006)),
    ("LDA", ("#", 0x16)), ("STA", ("$", 0x2007)),   # red backdrop
    ("LDA", ("#", 0x1E)), ("STA", ("$", 0x2001)),   # show background everywhere
    ("LDA", ("#", 0x80)), ("STA", ("$", 0x2000)),   # enable NMI, before main
    ("LABEL", "main"), ("JMP", "main"),
    ("LABEL", "nmi"),
    ("INC", ("$", 0x0302)),
    ("RTI", None),
]


def build(program) -> bytes:
    code, labels, _ = assemble(program, BASE)
    prg = bytearray(32768)
    prg[: len(code)] = code
    for offset_key, label in (("nmi", "nmi"), ("reset", "reset"), ("irq", "reset")):
        if label not in labels:
            continue
        offset = {"nmi": 0x7FFA, "reset": 0x7FFC, "irq": 0x7FFE}[offset_key]
        value = labels[label]
        prg[offset] = value & 0xFF
        prg[offset + 1] = (value >> 8) & 0xFF
    # iNES byte 4 counts 16 KB PRG banks: 32 KB of PRG means 2, not 1.
    header = bytes([0x4E, 0x45, 0x53, 0x1A, 2, 1, 0, 0]) + bytes(8)
    return bytes(header) + bytes(prg) + bytes(8192)


def run(name: str, program, frames: int = 180, buttons=("right",)) -> dict:
    path = ROOT / "roms" / ("probe_%s.nes" % name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(build(program))

    runner = NesRunner(str(path))
    for _ in range(frames):
        runner.step(())
    idle = [int(v) for v in runner.ram[0x0300:0x0305]]
    reads_idle = [int(v) for v in runner.ram[0x0310:0x0318]]
    pixel_idle = runner.screen[120, 128].tolist()

    for _ in range(frames):
        runner.step(buttons)
    held = [int(v) for v in runner.ram[0x0300:0x0305]]
    reads_held = [int(v) for v in runner.ram[0x0310:0x0318]]
    pixel_held = runner.screen[120, 128].tolist()

    runner.close()
    return {
        "rom": str(path),
        "frames": frames,
        "ram_idle_0300_0304": idle,
        "ram_held_0300_0304": held,
        "controller_reads_idle": reads_idle,
        "controller_reads_held_right": reads_held,
        "buttons_held": list(buttons),
        "pixel_idle": pixel_idle,
        "pixel_held": pixel_held,
    }


def main() -> int:
    out = {}

    a = run("cpu", PROG_CPU)
    out["cpu"] = a
    cpu_alive = a["ram_idle_0300_0304"][0] == 0x42 and a["ram_idle_0300_0304"][1] > 0
    print("[%s] cpu      : $0300=%#04x $0301=%#04x  (want 0x42 and a running counter)"
          % ("PASS" if cpu_alive else "FAIL", a["ram_idle_0300_0304"][0], a["ram_idle_0300_0304"][1]),
          file=sys.stderr)

    c = run("ctrl", PROG_CTRL)
    out["ctrl"] = c
    ctrl_alive = c["ram_idle_0300_0304"][0] == 0x42
    print("[%s] ctrl cpu : $0300=%#04x" % ("PASS" if ctrl_alive else "FAIL", c["ram_idle_0300_0304"][0]),
          file=sys.stderr)
    print("      ctrl idle        : %s" % c["controller_reads_idle"], file=sys.stderr)
    print("      ctrl held RIGHT  : %s   (bit0=A bit1=B bit2=sel bit3=start "
          "bit4=up bit5=down bit6=left bit7=right)" % c["controller_reads_held_right"],
          file=sys.stderr)
    reads = c["controller_reads_held_right"]
    # The pressed button arrives in bit 0 of the returned byte; bit 6 is open bus.
    ctrl_ok = ctrl_alive and len(reads) == 8 and (reads[7] & 0x01) != 0
    print("[%s] ctrl read: 8th read=%#04x (want bit0 set for RIGHT; bit 6 is open bus)"
          % ("PASS" if ctrl_ok else "FAIL", reads[7]), file=sys.stderr)

    p = run("ppu", PROG_PPU)
    out["ppu"] = p
    ppu_ok = p["ram_idle_0300_0304"][0] == 0x42 and p["pixel_idle"][0] > p["pixel_idle"][2]
    print("[%s] ppu      : $0300=%#04x centre pixel %s (want red: R >> B)"
          % ("PASS" if ppu_ok else "FAIL", p["ram_idle_0300_0304"][0], p["pixel_idle"]), file=sys.stderr)

    n = run("nmi", PROG_NMI)
    out["nmi"] = n
    nmi_ok = n["ram_idle_0300_0304"][2] > 0
    print("[%s] nmi      : heartbeat $0302=%#04x after %d frames"
          % ("PASS" if nmi_ok else "FAIL", n["ram_idle_0300_0304"][2], n["frames"]), file=sys.stderr)

    verdict = []
    if not cpu_alive:
        verdict.append("FAIL cpu: the reset routine never ran. On a 32 KB PRG image this "
                       "is almost always the iNES bank count -- byte 4 is in 16 KB units.")
    else:
        verdict.append("cpu: reset routine runs.")
    if ctrl_alive and not ctrl_ok:
        verdict.append("FAIL ctrl: the emulator accepted the action byte but $4016 reads "
                       "never see it. Compare controller_reads_held_right against the bit "
                       "layout above.")
    elif ctrl_ok:
        verdict.append("ctrl: $4016 returns the pressed button in the documented bit order.")
    if not ppu_ok:
        verdict.append("FAIL ppu: palette/rendering writes are not taking effect.")
    else:
        verdict.append("ppu: palette writes and background rendering work.")
    if not nmi_ok:
        verdict.append("FAIL nmi: no vblank heartbeat, so the emulator is not delivering NMI.")
    else:
        verdict.append("nmi: vblank interrupts are delivered.")

    out["verdict"] = verdict
    print("\nVERDICT", file=sys.stderr)
    for line in verdict:
        print("  - %s" % line, file=sys.stderr)

    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    (ROOT / "logs" / "emulator_probe.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
