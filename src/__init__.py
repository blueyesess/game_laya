"""laya as an NES player's reflex layer -- a playable, inspectable experiment.

The generic pipeline is: pixels -> `vision` -> `state` (profile-driven) -> `scene`
-> laya -> buttons. `src/profiles/*.json` is where a new game is described.
"""

__all__ = [
    "agent",
    "brain",
    "config",
    "emulator",
    "reflex",
    "scene",
    "server",
    "state",
    "vision",
]
