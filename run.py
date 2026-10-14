"""Entry point: start the dashboard and let laya play.

    python run.py --rom roms/smb1.nes --policy hybrid

With no --rom, any *.nes in roms/ is picked (preferring a name containing "smb" or
"mario"). If there is none, run tools/make_test_rom.py to get a legal homebrew ROM
so the plumbing can be exercised first.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.server import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
