"""Dump the *installed* nes_py surface, not the one on GitHub master.

The CPU-not-executing result says the emulator is not running our ROM at all, and
the most likely reason is that the installed 9.0.1 API differs from the master-branch
source this project was written against. This prints what is actually importable.

    python tools/inspect_nespy.py
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

lines = []

try:
    import nes_py
except Exception as exc:  # noqa: BLE001
    print("import nes_py failed: %r" % (exc,), file=sys.stderr)
    raise SystemExit(1)

lines.append("nes_py file      : %s" % getattr(nes_py, "__file__", "?"))
lines.append("nes_py version   : %s" % getattr(nes_py, "__version__", "?"))
lines.append("nes_py dir       : %s" % [n for n in dir(nes_py) if not n.startswith("_")])

pkg_dir = Path(nes_py.__file__).parent
lines.append("")
lines.append("package contents :")
for path in sorted(pkg_dir.rglob("*")):
    if path.is_file():
        lines.append("   %-40s %8d" % (path.relative_to(pkg_dir), path.stat().st_size))

NESEnv = getattr(nes_py, "NESEnv", None)
lines.append("")
if NESEnv is None:
    lines.append("NO NESEnv in nes_py")
else:
    lines.append("NESEnv signature : %s" % inspect.signature(NESEnv.__init__))
    lines.append("NESEnv methods   : %s" % sorted(m for m in dir(NESEnv) if not m.startswith("__")))
    for name in ("__init__", "reset", "step", "observation", "_frame_advance", "close", "dump_state", "load_state", "ram_values"):
        fn = getattr(NESEnv, name, None)
        if fn is None:
            continue
        try:
            src = inspect.getsource(fn)
        except (OSError, TypeError):
            lines.append("  <no source for %s>" % name)
            continue
        lines.append("")
        lines.append("----- %s -----" % name)
        lines.append(src.rstrip())

report = "\n".join(lines)
(ROOT / "logs").mkdir(parents=True, exist_ok=True)
(ROOT / "logs" / "nespy_inspect.txt").write_text(report, encoding="utf-8")
print(report, file=sys.stderr)
