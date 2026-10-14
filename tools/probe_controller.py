"""Pin down nes_py's exact $4016 semantics with one ROM, one button at a time.

Two earlier ROMs disagreed about which of the eight reads carried which button, and
comparing across ROMs cannot settle it: they run different numbers of loops at
different points in the frame. This one stores the eight raw reads at $0310-$0317 and
nothing else, then Python holds a single button for a while and reads all eight back.
Whatever permutation comes out is the emulator's real behaviour.

This matters beyond the test ROM. Super Mario Bros. contains its own, correct 6502
controller routine, so if $4016 did not behave like a real NES controller port, SMB1
would not play either.

    python tools/probe_controller.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import BUTTON_BITS  # noqa: E402
from src.emulator import NesRunner  # noqa: E402
from tools.make_test_rom import assemble  # noqa: E402

BASE = 0x8000

PROG = [
    ("LABEL", "reset"),
    ("SEI", None), ("CLD", None),
    ("LDX", ("#", 0xFF)), ("TXS", None),
    ("LDA", ("#", 0x42)), ("STA", ("$", 0x0300)),
    ("LABEL", "spin"),
    ("LDA", ("#", 0x01)), ("STA", ("$", 0x4016)),
    ("LDA", ("#", 0x00)), ("STA", ("$", 0x4016)),
    ("LDX", ("#", 0x00)),
    ("LABEL", "rd"),
    ("LDA", ("$", 0x4016)),
    ("STA", ("$X", 0x0310)),
    ("INX", None),
    ("CPX", ("#", 0x08)),
    ("BNE", "rd"),
    ("JMP", "spin"),
]


def build() -> bytes:
    code, labels, _ = assemble(PROG, BASE)
    prg = bytearray(32768)
    prg[: len(code)] = code
    value = labels["reset"]
    prg[0x7FFC] = value & 0xFF
    prg[0x7FFD] = (value >> 8) & 0xFF
    prg[0x7FFE] = value & 0xFF
    prg[0x7FFF] = (value >> 8) & 0xFF
    header = bytes([0x4E, 0x45, 0x53, 0x1A, 2, 1, 0, 0]) + bytes(8)
    return bytes(header) + bytes(prg) + bytes(8192)


def main() -> int:
    path = ROOT / "roms" / "probe_ctrl.nes"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(build())

    lines = []
    lines.append("nes_py button bitmap (from config.BUTTON_BITS, verified against "
                 "env.get_keys_to_action for all 256 combos):")
    lines.append("  " + ", ".join("%s=bit%d" % (n, b) for n, b in sorted(BUTTON_BITS.items(), key=lambda kv: kv[1])))
    lines.append("")
    lines.append("%-8s %-9s %s" % ("button", "action", "eight raw $4016 reads, in read order"))
    lines.append("-" * 78)

    results = {}
    for button in ("right", "left", "down", "up", "A", "B", "start", "select"):
        runner = NesRunner(str(path))
        for _ in range(150):
            runner.step(())
        reads_idle = [int(v) for v in runner.ram[0x0310:0x0318]]
        for _ in range(30):
            runner.step((button,))
        reads = [int(v) for v in runner.ram[0x0310:0x0318]]

        action_byte = 1 << BUTTON_BITS[button]
        # Which read index had bit0 set? That is the button arriving.
        carriers = [i for i, value in enumerate(reads) if value & 0x01]
        open_bus = sorted({value & ~0x01 for value in reads})

        results[button] = {
            "action_byte": action_byte,
            "reads_idle": reads_idle,
            "reads_held": reads,
            "carrier_read_index": carriers,
            "non_button_bits": open_bus,
        }
        lines.append(
            "%-8s 0x%02X      %s   carrier read index=%s"
            % (button, action_byte, reads, carriers)
        )
        runner.close()

    lines.append("")
    # Summarise the permutation: action bit -> read index that carries it.
    permutation = {}
    for button, data in results.items():
        carriers = data["carrier_read_index"]
        if len(carriers) == 1:
            permutation[BUTTON_BITS[button]] = carriers[0]
        else:
            permutation[BUTTON_BITS[button]] = carriers
    lines.append("permutation: nes_py action bit -> $4016 read index")
    lines.append("  " + json.dumps(permutation))
    ordered = [permutation.get(i) for i in range(8)]
    lines.append("  read order by button: " + ", ".join(
        "%s@read%d" % (n, permutation[i]) for n, i in sorted(BUTTON_BITS.items(), key=lambda kv: kv[1])
        if isinstance(permutation.get(i), int)))
    lines.append("  exactly the NES order A,B,Select,Start,Up,Down,Left,Right? %s"
                 % (ordered == list(range(8))))

    idle_bits = sorted({v & ~0x01 for r in results.values() for v in r["reads_idle"]})
    lines.append("  idle reads set these non-button bits: %s (0x40 is the standard "
                 "open-bus bit 6)" % [hex(b) for b in idle_bits])

    report = "\n".join(lines)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    (ROOT / "logs" / "controller_probe.txt").write_text(report, encoding="utf-8")
    (ROOT / "logs" / "controller_probe.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(report, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
