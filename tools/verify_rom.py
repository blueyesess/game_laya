"""Standalone check that the test ROM assembles into a structurally valid iNES image.

Kept separate from smoke_test.py so it can run before nes-py is installed, and it
only needs the standard library.

    python tools/verify_rom.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.make_test_rom import PROGRAM, assemble, build_rom  # noqa: E402


def main() -> int:
    lines = []
    failures = 0

    def check(name, ok, detail=""):
        nonlocal failures
        if not ok:
            failures += 1
        lines.append("[%s] %s%s" % ("PASS" if ok else "FAIL", name, (" -- " + detail) if detail else ""))

    rom = build_rom()
    code, labels, end = assemble(PROGRAM)

    check("iNES magic", rom[:4] == b"NES\x1a", repr(rom[:4]))
    check("total size is 16 + 32768 + 8192", len(rom) == 16 + 32768 + 8192, str(len(rom)))
    check("two 16KB PRG banks and one 8KB CHR bank", rom[4] == 2 and rom[5] == 1,
          "prg_banks=%d chr_banks=%d" % (rom[4], rom[5]))
    check("mapper 0 (NROM)", ((rom[6] >> 4) | (rom[7] & 0xF0)) == 0,
          "flags6=%02X flags7=%02X" % (rom[6], rom[7]))
    check("no trainer", not (rom[6] & 0x04))
    check("NTSC", not (rom[7] & 0x02), "flags7=%02X" % rom[7])

    reset = rom[16 + 0x7FFC] | (rom[16 + 0x7FFD] << 8)
    nmi = rom[16 + 0x7FFA] | (rom[16 + 0x7FFB] << 8)
    irq = rom[16 + 0x7FFE] | (rom[16 + 0x7FFF] << 8)
    check("RESET vector points at the reset label", reset == labels["reset"], "$%04X vs $%04X" % (reset, labels["reset"]))
    check("NMI vector points at the nmi label", nmi == labels["nmi"], "$%04X vs $%04X" % (nmi, labels["nmi"]))
    check("IRQ vector is non-zero", irq != 0, "$%04X" % irq)
    check("code starts at $8000", code[0] == 0x78, "first opcode %02X (SEI expected)" % code[0])
    check("code fits in PRG", len(code) < 32768, "%d bytes" % len(code))

    lines.append("")
    lines.append("code %d bytes, $8000..$%04X" % (len(code), end))
    lines.append("labels " + ", ".join("%s=$%04X" % (k, v) for k, v in sorted(labels.items(), key=lambda kv: kv[1])))
    lines.append("disassembly of the first 48 bytes:")
    lines.append("  " + " ".join("%02X" % b for b in code[:48]))

    report = "\n".join(lines)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    (ROOT / "logs" / "rom_verify.txt").write_text(report, encoding="utf-8")
    print(report, file=sys.stderr)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
