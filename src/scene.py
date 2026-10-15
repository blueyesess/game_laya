"""Turn a decoded game state into the text laya actually reads.

laya accepts a dict as state and serialises it to JSON itself, so we hand it
structure *and* a prose line. Both matter: the model was trained on text, and the
examples in its README are explicit that rewording the same facts moves the
answers. The prose line is written the way a person would describe the screen.

Everything here is profile-driven: the goal, the move vocabulary and the
"what counts as close" bands come from the game profile, so this module needs no
editing to drive a different ROM.

Keep it short. laya re-encodes the state once *per question*, so a long
observation is paid for three times over. Measured on the previous SMB1 version:
926 input tokens cost 5068 ms per decision, and dropping a structured dict that
merely restated the prose took that to 368 tokens and 2693 ms. `usage["truncated"]`
tells you when the head ran out of context, and everything past the cut is
simply not read.

The rule that matters most: **never assert a number we cannot back up.** A RAM
map that does not match the dump does not read as zero, it reads as plausible
fiction -- the model is told "world 0-1, 0 lives, 0 time left" and reasons from
it. When `GameState.ram_plausible` is false the observation falls back to what
the framebuffer can actually prove and says the rest is unknown.
"""

from __future__ import annotations

from .state import GameProfile, GameState

# How far ahead is "close"? In a NES side-scroller one tile is 16px and the
# player runs at up to ~2.5 px/frame, so 32px is roughly the reaction window at
# full sprint.
NEAR_PX = 32
MID_PX = 64


def build_state(st: GameState, profile: GameProfile) -> dict:
    """Compose the observation handed to laya."""
    parts = []
    progress = " or ".join(profile.progress_buttons)

    if not st.character_visible:
        parts.append("The player is not on screen.")
    else:
        footing = (
            "on ground" if st.on_ground else ("in the air" if st.on_ground is False else "footing unknown")
        )
        parts.append("The player is %s at x=%d." % (footing, st.character_x))
        if st.character_uncertain:
            # The geometric detector falls back to "the only thing on screen is
            # probably the player" before anything else has entered the frame. Say
            # so, because a model that reasons from a mis-identified position will
            # be confidently wrong in a way no confidence threshold can catch.
            parts.append("(Position uncertain: the player is the only thing detected.)")

    if st.pit_ahead:
        parts.append("The floor is missing %dpx ahead." % (st.pit_distance or 0))
    elif st.wall_ahead:
        parts.append(
            "A wall or block rises %dpx %dpx ahead." % (st.wall_height, st.wall_distance or 0)
        )
    elif st.character_visible:
        parts.append("The floor ahead is solid.")

    if st.climbable_ahead:
        parts.append("There is something to climb just ahead.")

    nearest = st.nearest_blob_ahead()
    if nearest is not None:
        dx = nearest.x - (st.character_x or 0)
        band = "close" if dx <= NEAR_PX else ("near" if dx <= MID_PX else "far")
        parts.append(
            "An object %s ahead, %dpx to the right, %dpx tall." % (band, dx, nearest.height)
        )

    if st.ram_plausible and st.hud:
        parts.append(
            "%dpx into the level. Coins %s, time %s, lives %s."
            % (
                st.world_x or 0,
                st.hud.get("coins"),
                st.hud.get("timer"),
                st.hud.get("lives"),
            )
        )
    else:
        # Say it once so the model treats these as missing rather than as zero.
        parts.append("Progress/coins/time/lives unknown: no verified RAM map for this ROM.")

    parts.append("Goal: %s Move by pressing %s." % (profile.goal, progress))

    observation = {"scene": " ".join(parts)}
    if st.ram_plausible and st.hud:
        observation["hud"] = [st.hud.get("coins"), st.hud.get("timer"), st.hud.get("lives")]
    return observation


def build_state_text(st: GameState, profile: GameProfile) -> str:
    """A plain-text variant, for logging and for eyeballing the input."""
    return build_state(st, profile)["scene"]


__all__ = ["MID_PX", "NEAR_PX", "build_state", "build_state_text"]
