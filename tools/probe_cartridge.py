"""Ask the native core directly whether it accepts our test ROM.

The probe already established that the CPU never executes the reset routine. nes_py
ships two diagnostics that answer this without speculation: ``cartridge_error`` (what
the C++ Cartridge parser says) and ``cartridge_metadata`` (how it decoded the header).
The Python-side ``_rom.py`` parser and the C++ parser are separate implementations, so
a ROM that passes one can still be rejected by the other.

    python tools/probe_cartridge.py
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

lines = []

from nes_py import _native  # noqa: E402
from tools.make_test_rom import build_rom  # noqa: E402

lines.append("_native attrs: %s" % sorted(a for a in dir(_native) if not a.startswith("_")))

rom_path = ROOT / "roms" / "inputtest.nes"
rom_path.parent.mkdir(parents=True, exist_ok=True)
rom_bytes = build_rom()
rom_path.write_bytes(rom_bytes)

lines.append("")
lines.append("rom: %s (%d bytes)" % (rom_path, len(rom_bytes)))
lines.append("header bytes: %s" % " ".join("%02X" % b for b in rom_bytes[:16]))

# What we think the header says.
prg_banks = rom_bytes[4]
chr_banks = rom_bytes[5]
flags6, flags7 = rom_bytes[6], rom_bytes[7]
lines.append("python-side decode: prg_banks=%d chr_banks=%d flags6=%02X flags7=%02X mapper=%d"
             % (prg_banks, chr_banks, flags6, flags7, (flags6 >> 4) | (flags7 & 0xF0)))

for name in ("cartridge_error", "cartridge_metadata", "is_mapper_supported"):
    fn = getattr(_native, name, None)
    if fn is None:
        lines.append("%s: not present in this build" % name)
        continue
    lines.append("")
    lines.append("--- %s ---" % name)
    try:
        value = fn(str(rom_path)) if name != "is_mapper_supported" else fn(0)
        lines.append("  -> %r" % (value,))
    except Exception as exc:  # noqa: BLE001
        lines.append("  raised %s: %s" % (type(exc).__name__, exc))

# Try the variants a stricter parser might insist on. The most likely culprit is the
# 8 zero bytes after the 8-byte header: a parser that assumes a 12-byte header would
# start reading PRG four bytes late and execute garbage.
lines.append("")
lines.append("--- header variants ---")
variants = {
    "ines16_asis": rom_bytes,
    "ines12_trimmed": rom_bytes[:12] + rom_bytes[16:],
    "ines16_flags_prg16": None,  # filled below
}
# NROM-128 variant: 16 KB PRG mirrored into the 32 KB CPU space.
prg16 = rom_bytes[16:16 + 16384]
variants["ines16_flags_prg16"] = rom_bytes[:4] + bytes([1, 1, flags6, flags7]) + bytes(8) + prg16 * 2 + rom_bytes[16 + 32768:]

for label, data in variants.items():
    if data is None:
        continue
    path = ROOT / "roms" / ("variant_%s.nes" % label)
    path.write_bytes(data)
    entry = ["%s (%d bytes)" % (path.name, len(data))]
    for name in ("cartridge_error", "cartridge_metadata"):
        fn = getattr(_native, name, None)
        if fn is None:
            continue
        try:
            entry.append("%s=%r" % (name, fn(str(path))))
        except Exception as exc:  # noqa: BLE001
            entry.append("%s raised %s" % (name, exc))
    lines.append("  " + " | ".join(entry))

    # And, decisively: does the CPU actually run it?
    try:
        from src.emulator import NesRunner

        runner = NesRunner(str(path))
        for _ in range(120):
            runner.step(())
        lines.append("    -> RAM $0300 = 0x%02X (0x42 means the CPU ran)"
                     % int(runner.ram[0x0300]))
        runner.close()
    except Exception as exc:  # noqa: BLE001
        lines.append("    -> run failed: %s: %s" % (type(exc).__name__, exc))

lines.append("")
lines.append("--- is the iNES magic/offset maybe expected differently? ---")
lines.append("bytes 0..20: %s" % " ".join("%02X" % b for b in rom_bytes[:20]))
lines.append("prg[0..8]  : %s" % " ".join("%02X" % b for b in rom_bytes[16:24]))

report = "\n".join(lines)
(ROOT / "logs").mkdir(parents=True, exist_ok=True)
(ROOT / "logs" / "cartridge_probe.txt").write_text(report, encoding="utf-8")
print(report, file=sys.stderr)
