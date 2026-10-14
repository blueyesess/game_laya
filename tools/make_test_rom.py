"""Generate a legal homebrew .nes so the whole pipeline can be tested without SMB1.

Why bother: everything here except the SMB1 RAM map can be verified against a ROM we
write ourselves. The emulator wrapper, the 8-button bitmap, the framebuffer decode and
the dashboard are all game-agnostic, and a broken one of those would otherwise hide
behind "well, we don't have the ROM yet".

What the ROM does
-----------------
On every vblank it strobes controller port 1, reads all eight buttons into RAM at
$0301 (and mirrors to $0300), then writes the backdrop palette entry $3F00 to a
colour chosen by the button held: RIGHT red, LEFT green, A white, nothing blue.

Two things fall out of that, both of which are real assertions in smoke_test.py:

* $0300 ends up with **A in bit 0 ... RIGHT in bit 7** -- which is exactly nes_py's
  button bitmap, because the NES controller reports A,B,Select,Start,Up,Down,Left,
  Right in that order. So reading $0300 back from Python verifies src/config.py's
  BUTTON_BITS table against the emulator instead of against my memory of it.
* the screen turns a different colour per button, so the controller path is also
  visible with the naked eye in the dashboard.

The assembler below is deliberately tiny: 6502 opcodes, one pass to find labels and
one to emit. Nothing here depends on the 6502 program, only on bytes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "roms" / "inputtest.nes"

BASE = 0x8000

# Immediate (2 bytes), absolute (3), absolute,X (3), implied (1), branches (2).
IMM = {"LDA": 0xA9, "LDX": 0xA2, "LDY": 0xA0, "AND": 0x29, "ORA": 0x09, "CMP": 0xC9, "CPX": 0xE0, "CPY": 0xC0}
ABS = {"STA": 0x8D, "STX": 0x8E, "STY": 0x8C, "BIT": 0x2C, "ROL": 0x2E, "ROR": 0x6E, "ASL": 0x0E, "JMP": 0x4C, "LDA": 0xAD, "AND": 0x2D, "INC": 0xEE}
ABSX = {"STA": 0x9D, "LDA": 0xBD, "ROL": 0x3E}
IMP = {
    "SEI": 0x78, "CLD": 0xD8, "TXS": 0x9A, "TXA": 0x8A, "PHA": 0x48, "PLA": 0x68,
    "TAX": 0xAA, "INX": 0xE8, "DEX": 0xCA, "LSR": 0x4A, "RTI": 0x40, "NOP": 0xEA,
    "DEY": 0x88,
}
BRANCH = {"BNE": 0xD0, "BEQ": 0xF0, "BPL": 0x10, "BMI": 0x30, "BCC": 0x90, "BCS": 0xB0}

PROGRAM = [
    ("LABEL", "reset"),
    ("SEI", None), ("CLD", None),
    ("LDX", ("#", 0xFF)), ("TXS", None),
    ("LDX", ("#", 0x00)),
    ("STX", ("$", 0x2000)),   # NMI off
    ("STX", ("$", 0x4010)),   # DMC IRQ off
    ("STX", ("$", 0x4017)),   # APU frame IRQ off
    ("LDA", ("#", 0x00)), ("STA", ("$", 0x4015)),

    ("LABEL", "vb1"), ("BIT", ("$", 0x2002)), ("BPL", "vb1"),
    ("LABEL", "vb2"), ("BIT", ("$", 0x2002)), ("BPL", "vb2"),

    # Zero all 2 KB of RAM, eight pages at a time.
    ("LDA", ("#", 0x00)), ("LDX", ("#", 0x00)),
    ("LABEL", "clr"),
    ("STA", ("$X", 0x0000)), ("STA", ("$X", 0x0100)), ("STA", ("$X", 0x0200)),
    ("STA", ("$X", 0x0300)), ("STA", ("$X", 0x0400)), ("STA", ("$X", 0x0500)),
    ("STA", ("$X", 0x0600)), ("STA", ("$X", 0x0700)),
    ("INX", None), ("BNE", "clr"),

    # Clear the nametable so the picture does not depend on uninitialised VRAM.
    # Four passes of 256 cover the full $2000-$23FF window. X and Y are both set up
    # *before* the outer label: reloading Y inside the outer loop body would reset it
    # to 4 on every pass and the DEY could never reach zero.
    ("LDA", ("#", 0x20)), ("STA", ("$", 0x2006)),
    ("LDA", ("#", 0x00)), ("STA", ("$", 0x2006)),
    ("LDX", ("#", 0x00)), ("LDY", ("#", 0x04)),
    ("LABEL", "nt4"),
    ("LABEL", "nt"), ("STA", ("$", 0x2007)), ("INX", None), ("BNE", "nt"),
    ("DEY", None), ("BNE", "nt4"),

    # Paint the backdrop black from reset, not only from NMI. If the emulator ever
    # fails to deliver an interrupt, the screen stays black instead of showing
    # whatever uninitialised VRAM held -- which makes "NMI never fired" obvious
    # instead of turning it into a mystery grey.
    ("LDA", ("#", 0x3F)), ("STA", ("$", 0x2006)),
    ("LDA", ("#", 0x00)), ("STA", ("$", 0x2006)),
    ("LDA", ("#", 0x0F)), ("STA", ("$", 0x2007)),

    # NMI on; show the background everywhere (bits 1 and 3 defeat the leftmost-8 clip).
    ("LDA", ("#", 0x80)), ("STA", ("$", 0x2000)),
    ("LDA", ("#", 0x1E)), ("STA", ("$", 0x2001)),
    ("LABEL", "main"), ("JMP", "main"),

    # ------------------------------------------------------------------ NMI
    ("LABEL", "nmi"),
    ("PHA", None), ("TXA", None), ("PHA", None),
    # Heartbeat: one increment per vblank. If this stays 0 the emulator is not
    # delivering NMI at all, which is a completely different bug from "the
    # controller read is not working".
    ("INC", ("$", 0x0302)),

    ("LDA", ("#", 0x3F)), ("STA", ("$", 0x2006)),
    ("LDA", ("#", 0x00)), ("STA", ("$", 0x2006)),   # VRAM address = $3F00

    ("LDA", ("#", 0x01)), ("STA", ("$", 0x4016)),
    ("LDA", ("#", 0x00)), ("STA", ("$", 0x4016)),
    ("LDX", ("#", 0x08)),
    ("LABEL", "rd"),
    ("LDA", ("$", 0x4016)), ("LSR", None), ("ROL", ("$", 0x0301)),
    ("DEX", None), ("BNE", "rd"),

    # ROL shifts left, so read 0 (button A) has ended up in bit 7 and read 7 (button
    # RIGHT) in bit 0 -- the reverse of nes_py's action byte. Reverse it into $0300 so
    # that $0300 comes out byte-identical to the action byte (A bit0 .. RIGHT bit7),
    # which is what makes the round trip in smoke_test.py meaningful.
    ("LDA", ("#", 0x00)), ("STA", ("$", 0x0300)), ("LDX", ("#", 0x08)),
    ("LABEL", "rev"),
    ("ASL", ("$", 0x0301)), ("ROR", ("$", 0x0300)),
    ("DEX", None), ("BNE", "rev"),

    # Pick the colour from $0300, using nes_py's own bit layout (A bit0 .. RIGHT
    # bit7). Reading $0300 rather than $0301 matters: the bit reversal above shifts
    # $0301 down to zero on its last pass, so anything still testing $0301 afterwards
    # can never match and the screen stays one flat colour.
    ("LDA", ("$", 0x0300)), ("AND", ("#", 0x80)), ("BNE", "c_red"),     # RIGHT
    ("LDA", ("$", 0x0300)), ("AND", ("#", 0x40)), ("BNE", "c_green"),   # LEFT
    ("LDA", ("$", 0x0300)), ("AND", ("#", 0x01)), ("BNE", "c_white"),   # A
    ("LDA", ("#", 0x21)), ("BNE", "store"),                             # blue
    ("LABEL", "c_red"), ("LDA", ("#", 0x16)), ("BNE", "store"),
    ("LABEL", "c_green"), ("LDA", ("#", 0x2A)), ("BNE", "store"),
    ("LABEL", "c_white"), ("LDA", ("#", 0x30)),
    ("LABEL", "store"), ("STA", ("$", 0x2007)),

    ("PLA", None), ("TAX", None), ("PLA", None), ("RTI", None),
]


def _size(op: str, arg) -> int:
    if isinstance(arg, tuple):
        return 2 if arg[0] == "#" else 3
    if isinstance(arg, str):
        return 2 if op in BRANCH else 3
    return 1


def assemble(program, base: int = BASE):
    labels = {}
    pc = base
    for op, arg in program:
        if op == "LABEL":
            labels[arg] = pc
            continue
        pc += _size(op, arg)
    end = pc

    out = bytearray()
    pc = base
    for op, arg in program:
        if op == "LABEL":
            continue
        size = _size(op, arg)
        if isinstance(arg, tuple):
            mode, value = arg
            if mode == "#":
                out += bytes([IMM[op], value & 0xFF])
            elif mode == "$":
                out += bytes([ABS[op], value & 0xFF, (value >> 8) & 0xFF])
            else:
                out += bytes([ABSX[op], value & 0xFF, (value >> 8) & 0xFF])
        elif isinstance(arg, str):
            target = labels[arg]
            if op in BRANCH:
                rel = target - (pc + 2)
                if not -128 <= rel <= 127:
                    raise ValueError("branch out of range at $%04X: %d" % (pc, rel))
                out += bytes([BRANCH[op], rel & 0xFF])
            else:
                out += bytes([ABS[op], target & 0xFF, (target >> 8) & 0xFF])
        else:
            out += bytes([IMP[op]])
        pc += size

    return bytes(out), labels, end


def build_rom() -> bytes:
    code, labels, _ = assemble(PROGRAM)

    prg = bytearray(32768)
    prg[: len(code)] = code
    # iNES vectors live at the top of PRG: $FFFA NMI, $FFFC RESET, $FFFE IRQ.
    prg[0x7FFA] = labels["nmi"] & 0xFF
    prg[0x7FFB] = (labels["nmi"] >> 8) & 0xFF
    prg[0x7FFC] = labels["reset"] & 0xFF
    prg[0x7FFD] = (labels["reset"] >> 8) & 0xFF
    prg[0x7FFE] = labels["reset"] & 0xFF
    prg[0x7FFF] = (labels["reset"] >> 8) & 0xFF

    chr_data = bytes(8192)  # tile 0 blank -> the whole screen shows $3F00
    # iNES header. Byte 4 counts PRG in 16 KB units and byte 5 counts CHR in 8 KB
    # units, so a 32 KB PRG is declared as 2 -- not 1. Getting this wrong is silent
    # and nasty: the core loads only the first 16 KB, which puts the $FFFA-$FFFF
    # vectors at $3FFA where this image holds zeros, so the CPU resets to $0000 and
    # executes a stream of BRKs instead of the reset routine.
    header = bytes([0x4E, 0x45, 0x53, 0x1A, 2, 1, 0, 0]) + bytes(8)
    return bytes(header) + bytes(prg) + chr_data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    rom = build_rom()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(rom)
    # stdout is unreliable in some shells; write the summary where it can be read.
    print("wrote %s (%d bytes)" % (out, len(rom)), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
