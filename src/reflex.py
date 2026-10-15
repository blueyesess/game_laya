"""A deterministic policy, used when laya is unsure or unavailable.

This is not decoration. laya decides a few times a second; the NES runs sixty
times a second. In between decisions somebody has to be holding the buttons, and
when `answer_confidence` falls below the floor the model has effectively said
"I don't know". Rather than execute a coin flip, control goes here for a few
frames.

The rules read only what the vision layer can *prove* from pixels -- which is
the point of the generic refactor. There is no RAM map here, no palette, no
per-game address table; every trigger is geometry, so the same three rules work
on any side-scroller:

* hold the progress direction, because standing still never finishes a level
* jump when the floor ahead is missing, or when a wall or object rises close
  ahead and the player is on the ground
* press up when the profile says climbing is possible and something climbable
  is just ahead

The dashboard labels every action with which layer produced it, so a hybrid run
never pretends the model did something the reflex layer did.
"""

from __future__ import annotations

from .state import GameProfile

# One tile is typically 16px. These are distances at which the rules act, in
# screen pixels: a jump covers about 4 tiles at a run, so anything inside
# JUMP_REACH_PX must already be handled.
JUMP_REACH_PX = 48
# Jump earlier for a wall (it does not move toward you) than for a pit, and
# earlier still for anything the vision layer flagged as close.
WALL_TRIGGER_PX = 26
PIT_TRIGGER_PX = 34
OBJECT_TRIGGER_PX = 30


class ReflexPolicy:
    def __init__(self, profile: GameProfile, jump_hold: int | None = None):
        self.profile = profile
        self.jump_hold = jump_hold or profile.jump_hold_frames
        self._jump_until_frame = -1

    def act(self, st) -> dict:
        profile = self.profile
        buttons = list(profile.progress_buttons)
        reasons = ["hold %s" % "/".join(profile.progress_buttons)]

        jump_now = False
        climb_now = False

        if st.on_ground:
            if st.pit_ahead and (st.pit_distance or 99) <= PIT_TRIGGER_PX:
                jump_now = True
                reasons.append("pit %dpx ahead" % (st.pit_distance or 0))
            elif st.wall_ahead and (st.wall_distance or 99) <= WALL_TRIGGER_PX:
                jump_now = True
                reasons.append("wall %dpx ahead" % (st.wall_distance or 0))
            else:
                nearest = st.nearest_blob_ahead()
                if nearest is not None and st.character_x is not None:
                    dx = nearest.x - st.character_x
                    if dx <= OBJECT_TRIGGER_PX:
                        jump_now = True
                        reasons.append("object %dpx ahead" % dx)

            if (
                not jump_now
                and st.climbable_ahead
                and "up" in profile.actions
            ):
                up_buttons = tuple(profile.actions["up"])
                if up_buttons:
                    climb_now = True
                    reasons.append("something to climb ahead")

        if jump_now:
            self._jump_until_frame = st.frame + self.jump_hold
            reasons.append("jump")

        for name in profile.jump_buttons:
            if self._jump_until_frame > st.frame and name not in buttons:
                buttons.append(name)
                reasons.append(
                    "holding %s for %d more frame(s)" % (name, self._jump_until_frame - st.frame)
                )

        if climb_now:
            for name in profile.actions.get("up", []):
                if name not in buttons:
                    buttons.append(name)

        return {
            "ok": True,
            "source": "reflex",
            "buttons": tuple(buttons),
            "reason": ", ".join(reasons),
            "move": {
                "label": "progress",
                "buttons": tuple(profile.progress_buttons),
                "probs": {},
                "confidence": 1.0,
            },
            "jump": {
                "p_true": 1.0 if jump_now else 0.0,
                "thresholded": jump_now,
                "confidence": 1.0,
            },
            "risk": {
                "label": "hazard_close" if jump_now else "safe",
                "index": 1 if jump_now else 0,
                "expected": float(jump_now),
                "legend": {},
                "probs": {},
                "confidence": 1.0,
            },
            "confidence": 1.0,
            "latency_ms": 0.0,
            "model": "rules",
            "usage": {},
        }


__all__ = ["JUMP_REACH_PX", "PIT_TRIGGER_PX", "ReflexPolicy", "WALL_TRIGGER_PX"]
