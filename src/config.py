"""Shared configuration: paths, NES button bitmap, SMB1 RAM map, tunables.

Everything that another module might need to agree on lives here, so there is exactly
one place to look when a constant turns out to be wrong.
"""

from __future__ import annotations

from pathlib import Path

# ----------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
ROM_DIR = ROOT / "roms"
WEB_DIR = ROOT / "web"
LOG_DIR = ROOT / "logs"
DECISION_LOG = LOG_DIR / "decisions.jsonl"

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


# ----------------------------------------------------------------- SMB1 RAM map
# Community-documented addresses, CORRECTED by direct measurement on
# "Super Mario Bros. (Japan, USA)" (mapper 0) using OAM sprite position and
# background pixel scroll as ground truth:
#
# * $06D:$086 world X - $086 wraps 255->0 on exactly the frame $06D steps +1
#   (observed twice), and at level start ($06D:$086) == Mario's screen x with
#   the camera at 0. World X = camera + screen, as a level coordinate should be.
# * $005 screen X - equals Mario's OAM sprite x on every measured frame:
#   camera scroll, edge clamping, and jumps included. This is the byte the game
#   itself uses to position Mario's sprite.
# * $001:$002 world Y - equals the OAM sprite y plus a constant on grounded
#   frames exactly (airborne within the 8px sprite-pose jitter).
#
# The old community table ($0028/$0029 for X) reads ZERO for a whole level on
# this build while Mario visibly runs, so it was replaced, not trusted.
#
# HONESTY NOTE: $0021-$002A (timer, world, lives, status) and the enemy slots
# at $0030-$0056 still read zero on this build and remain the community values
# -- plausibility-guarded by ram_is_plausible(), not verified. tools/dump_ram.py
# re-verifies the coordinate bytes against pixel motion in one minute.
RAM = {
    # --- player -------------------------------------------------------------
    "player_x_hi": 0x6D,  # bits 8..15 of the absolute world X (measured)
    "player_x_lo": 0x86,  # bits 0..7 of the absolute world X (measured)
    "player_y_page": 0x01,  # bits 8..15 of the absolute world Y (measured)
    "player_y_subpx": 0x02,  # bits 0..7 of the absolute world Y (measured)
    "player_screen_x": 0x3AD,  # Mario's screen X, as the game draws him.
    #                                Measured: equals OAM sprite x on every frame
    #                                (the first candidate, $005, matched only 52%
    #                                -- it flickers between valid values and 0/146
    #                                on some frames; $3AD never does. $755 holds
    #                                the same value, likely a second copy.)
    "scroll_x": 0x03,  # community "camera low byte"; measured constant here,
    #                   -- real camera = world_x - player_screen_x, computed
    "scroll_y": 0x05,  # community value; note $005 is player_screen_x (above)
    "player_spr_x": 0x06,  # fine sub-tile offsets (community)
    "player_spr_y": 0x07,
    "coins": 0x1F,  # BCD, 00-99 (community; reads 0 on this build)
    "timer_high": 0x21,
    "timer_low": 0x22,  # combined with 0x21 -> 8-bit countdown, ticks at ~24/s
    "player_y_px": 0x24,
    "world": 0x25,
    "lives": 0x26,
    "status": 0x27,
    # --- four enemy slots, 9 bytes each starting at 0x30 --------------------
    "enemy_base": 0x30,
}

# $27 player status codes
STATUS_SMALL = 0x00
STATUS_GROWING = 0x01
STATUS_BIG = 0x02
STATUS_SHRINKING = 0x03
STATUS_DEAD = 0x04
STATUS_CLEAR = 0x05
STATUS_GAME_OVER = 0x06

STATUS_NAMES = {
    STATUS_SMALL: "small",
    STATUS_GROWING: "growing",
    STATUS_BIG: "big",
    STATUS_SHRINKING: "shrinking",
    STATUS_DEAD: "dead",
    STATUS_CLEAR: "level_clear",
    STATUS_GAME_OVER: "game_over",
}

# Enemy slot layout, relative to RAM["enemy_base"]. Offsets are the community
# values; dump_ram.py is how you confirm them for your ROM.
ENEMY_STRIDE = 9
ENEMY_FIELDS = {
    "page_col": 0,
    "x_hi": 1,
    "x_lo": 2,
    "y_hi": 3,
    "y_lo": 4,
    "spr_x": 5,
    "spr_y": 6,
}

# An enemy slot is "empty" when its page column is 0 and the coordinates are 0.
ENEMY_EMPTY = (0x00, 0x00, 0x00)

# ----------------------------------------------------------------- SMB1 palette
# The handful of colours we classify pixels against when reading the framebuffer.
# Mario is red/blue, Goombas and bricks are orange-brown, pipes and Koopas green.
C_SKY = (0x5C, 0x94, 0xFC)
C_MARIO_RED = (0xD8, 0x28, 0x00)
C_MARIO_BLUE = (0x00, 0x00, 0xFC)
C_BRICK = (0xC8, 0x4C, 0x0C)
C_GROUND = (0xE4, 0x5C, 0x10)
C_PIPE = (0x00, 0xA8, 0x00)
C_GOOMBA = (0x9C, 0x4A, 0x00)
C_UNDERGROUND = (0x00, 0x00, 0x00)

# Where the status-bar / HUD sits. The playfield below it is what we reason about.
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
