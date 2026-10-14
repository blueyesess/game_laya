"""Turn a decoded game state into the text laya actually reads.

laya accepts a dict as state and serialises it to JSON itself, so we hand it
structure *and* a prose line. Both matter: the model was trained on text, and the
examples in its README are explicit that rewording the same facts moves the
answers. The prose line is written the way a person would describe the screen.

Keep it short. `usage["truncated"]` tells you when the head ran out of context, and
everything past the cut is simply not read.

The important rule here: **never assert a number we cannot back up.** The RAM map in
config.py is a community table and it does not fit every SMB1 build -- on the
"25th Anniversary (Promo, Virtual Console)" dump every HUD field reads zero for a
whole level. Serialising that into the state would tell the model "world 0-1, 0
lives, 0 time left", which is worse than telling it nothing, because the model will
reason from the fiction. When `SmbState.ram_plausible` is false the observation falls
back to what the framebuffer can actually prove and says the rest is unknown.
"""

from __future__ import annotations

from .smb_state import SmbState

# How far ahead is "close"? In SMB1 one tile is 16px and Mario runs at up to
# ~2.5 px/frame, so 32px is roughly the reaction window at full sprint.
NEAR_PX = 32
MID_PX = 64


def build_state(st: SmbState) -> dict:
    """Compose the observation handed to laya.

    Length is a performance concern, not a style one. laya encodes the state once per
    question, so a three-question call pays for it three times. This observation was
    originally a long prose paragraph *plus* a structured dict restating the same
    facts, and it measured 926 input tokens at ~5.1 s per decision on CPU. The prose
    alone carries everything, so the duplicate structure is gone; tools/bench_brain.py
    is what to re-run after changing this.
    """
    usable = st.ram_plausible
    parts = []

    if st.mario_screen_x is None:
        parts.append("Mario is off screen.")
    else:
        footing = "ground" if st.on_ground else ("air" if st.on_ground is False else "footing ?")
        parts.append("Mario %s at x=%d." % (footing, st.mario_screen_x))

    if usable:
        nearest = st.nearest_enemy(ahead=True)
        if nearest is None:
            parts.append("No enemy visible.")
        else:
            dx = nearest.dx_from(st.world_x)
            band = "close" if abs(dx) <= NEAR_PX else ("near" if abs(dx) <= MID_PX else "far")
            parts.append("Enemy %s, %dpx %s." % (band, abs(dx), "ahead" if dx >= 0 else "behind"))
    else:
        parts.append("Enemy positions unknown.")

    if st.pit_ahead:
        parts.append("Floor missing %dpx ahead." % (st.pit_distance or 0))
    elif st.mario_screen_x is not None:
        parts.append("Floor ahead solid.")

    if usable:
        parts.append(
            "%dpx in, %s. Coins %d, time %d, lives %d."
            % (st.world_x, "big" if st.big else "small", st.coins, st.timer, st.lives)
        )
    else:
        # Say it once so the model treats these as missing rather than as zero.
        parts.append("Progress/coins/time/lives unknown: RAM map does not match this ROM.")

    observation = {"scene": " ".join(parts)}
    if usable:
        observation["level"] = "%d-1" % st.world
        observation["hud"] = [st.coins, st.timer, st.lives]
    return observation


def build_state_text(st: SmbState) -> str:
    """A plain-text variant, for logging and for eyeballing the input."""
    return build_state(st)["scene"]
