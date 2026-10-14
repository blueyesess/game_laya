"""A deterministic policy, used when laya is unsure or unavailable.

This is not decoration. laya decides a few times a second; the NES runs sixty times
a second. In between decisions somebody has to be holding the buttons, and when
`answer_confidence` falls below the floor the model has effectively said "I don't
know". Rather than execute a coin flip, control goes here for a few frames.

The rules are intentionally boring and exact, because they read positions that the
RAM map gives exactly:

* always hold RIGHT, because standing still never finishes a level
* jump when there is a hole in the floor within one jump's reach
* jump when an enemy is within jumping distance in front

The dashboard labels every action with which of the two produced it, so a hybrid run
never pretends the model did something the reflex layer did.
"""

from __future__ import annotations

# One SMB1 tile is 16 px. Mario's jump covers about 4 tiles at a run.
JUMP_REACH_PX = 48
# Jump earlier for enemies (they move toward you) than for static floor holes.
ENEMY_TRIGGER_PX = 30
PIT_TRIGGER_PX = 34


class ReflexPolicy:
    def __init__(self, jump_hold: int = 7):
        self.jump_hold = jump_hold
        self._jump_until_frame = -1

    def act(self, st) -> dict:
        buttons = ["right"]
        reasons = ["hold right"]

        jump_now = False
        if st.on_ground:
            if st.pit_ahead and (st.pit_distance or 99) <= PIT_TRIGGER_PX:
                jump_now = True
                reasons.append("pit %dpx ahead" % (st.pit_distance or 0))
            else:
                nearest = st.nearest_enemy(ahead=True)
                if nearest is not None:
                    dist = abs(nearest.dx_from(st.world_x))
                    if dist <= ENEMY_TRIGGER_PX:
                        jump_now = True
                        reasons.append("enemy %dpx ahead" % dist)

        if jump_now:
            self._jump_until_frame = st.frame + self.jump_hold
            reasons.append("jump")

        if self._jump_until_frame > st.frame:
            if "A" not in buttons:
                buttons.append("A")
            reasons.append("holding A for %d more frame(s)" % (self._jump_until_frame - st.frame))

        return {
            "ok": True,
            "source": "reflex",
            "buttons": tuple(buttons),
            "reason": ", ".join(reasons),
            "move": {"label": "right", "buttons": ("right",), "probs": {}, "confidence": 1.0},
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
