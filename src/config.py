"""Shared configuration: paths, NES button bitmap, tunables.

Everything that another module might need to agree on lives here, so there is exactly
one place to look when a constant turns out to be wrong.

Note what is *not* here any more: the SMB1 RAM map, the SMB1 palette tables and the
SMB1 status codes. Those were game knowledge, and hard-coding game knowledge is what
made the previous version un-runnable on any other ROM. They live in
`src/profiles/smb1.json` now, loaded through `src/state.GameProfile`, so a new game is
a JSON file rather than an edit to this module.
"""

from __future__ import annotations

from pathlib import Path

# ----------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
ROM_DIR = ROOT / "roms"
WEB_DIR = ROOT / "web"
LOG_DIR = ROOT / "logs"
DECISION_LOG = LOG_DIR / "decisions.jsonl"
PROFILE_DIR = ROOT / "src" / "profiles"

# ----------------------------------------------------------------- video
SCREEN_W = 256
SCREEN_H = 240
NTSC_FPS = 60.0988  # exact NES frame rate; 21477272 / 3573685

# ------------------------------------------------- NES controller bitmap
# Bit order is taken from nes_py.NESEnv.get_keys_to_action, whose comment states
# "right = 7, left = 6, ..., B = 1, A = 0". The 8 buttons are therefore:
BUTTON_BITS = {
    "A": 0,
    "B": 1,
    "select": 2,
    "start": 3,
    "up": 4,
    "down": 5,
    "left": 6,
    "right": 7,
}
BUTTON_NAMES = tuple(BUTTON_BITS)


def buttons_to_byte(names) -> int:
    """Turn a collection of button names into the byte nes_py expects."""
    byte = 0
    for name in names:
        try:
            byte |= 1 << BUTTON_BITS[name]
        except KeyError:
            raise ValueError(
                "unknown NES button %r (valid: %s)" % (name, ", ".join(BUTTON_BITS))
            ) from None
    return byte


# Where the status-bar / HUD sits. The playfield below it is what we reason about.
# 32 is the NES's own attribute/HUD band height and both shipped profiles use it,
# but a profile can override it -- this is only the fallback for a frame decoder
# built without one.
HUD_BOTTOM_ROW = 32

# ----------------------------------------------------------------- agent tunables
# The whole design rests on this: laya (a 322M-421M encoder doing a full forward
# pass) decides far slower than the NES runs frames. So the emulator free-runs at
# 60.1 fps in one thread and the brain is a separate, slower loop that *holds* its
# last action between decisions -- the same cadence a human plays with.
DEFAULT_DECISION_HZ = 4.0  # target decisions/sec; the real rate is latency-bound
DEFAULT_SPEED = 1.0  # emulator frame pacing multiplier (1.0 = real time)
JUMP_HOLD_FRAMES = 7  # how long A stays down after a jump decision

# Below this answer_confidence the hybrid policy hands control to the reflex layer
# instead of trusting a decision the model itself flagged as unsure.
#
# 0.40 is measured, not chosen. On this machine the multilingual checkpoint answers a
# three-question call in ~3.5 s with answer_confidence around 0.44 for the shipped
# schema (tools/bench_brain.py). A floor of 0.45 would sit just above the model's
# typical confidence and hand almost every decision to the rules, which would make
# "hybrid" indistinguishable from "reflex".
DEFAULT_CONFIDENCE_FLOOR = 0.40

DEFAULT_MODEL = "multilingual"  # 322M; "english" is 421M, "typed-decisions" is the fine-tune

DEFAULT_PORT = 8770
