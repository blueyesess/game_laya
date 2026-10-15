"""Generic game state: profile-driven decoding of one frame of pixels (+ optional RAM).

The split between this module and `vision.py` is the split that makes the
project generic:

* `vision.py` knows pixels and nothing else. It answers "what is on screen",
  for any ROM, with no per-game tables.
* this module knows a *profile* -- a JSON file describing one game: where its
  HUD is, what its buttons mean, what its RAM map looks like if it has one --
  and turns the vision output into a `GameState` that the observation layer,
  the reflex policy and laya all read.

Adding a new NES game is therefore a JSON file, not a Python module. The
profile format is documented inline in `GameProfile`; the two shipped ones are
`src/profiles/generic.json` (any side-scroller) and `src/profiles/smb1.json`.

Two things worth stating plainly, because the previous version got both wrong:

* **a character is located by geometry, not by colour.** Colour thresholds
  worked on one dump and silently failed on two others (see vision.find_character).
* **RAM numbers are only serialised when they pass the game's own
  invariants.** A wrong RAM map does not read as zero, it reads as *plausible
  fiction*: the model is told "world 0-1, 0 lives, 0 time left" and reasons
  from it. Better to say unknown.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from .config import HUD_BOTTOM_ROW, PROFILE_DIR, SCREEN_H, SCREEN_W
from . import vision

# A RAM address outside this range is a profile bug, not a game state.
RAM_SIZE = 2048


class ProfileNotFound(LookupError):
    """Raised when `--profile` names a JSON file that does not exist.

    A LookupError rather than SystemExit so that a Flask request handler can
    return it as a 400 instead of taking the server down with it.
    """


# ------------------------------------------------------------------- profile
@dataclass
class RamMap:
    """Optional per-game RAM layout, read only when its invariants hold."""

    addresses: dict
    enemy_stride: int = 0
    enemy_fields: dict = field(default_factory=dict)
    enemy_empty: tuple = (0, 0, 0)
    invariants: dict = field(default_factory=dict)
    # Status byte values that mean "this run is over" (dead / level clear / game
    # over). Per-game by definition, so a profile declares them or omits them.
    terminated_statuses: tuple = ()

    @property
    def enemy_base(self) -> Optional[int]:
        return self.addresses.get("enemy_base")

    def plausible(self, fields: dict) -> bool:
        """Do the decoded values satisfy the game's own invariants?

        These are properties of the game, not of any address table: world
        numbers start at 1, you begin with lives, coins are 0-99, the timer
        starts at 400. A community map that does not match the dump trips
        every one of them at once.
        """
        if not self.invariants:
            return False
        for key, (lo, hi) in self.invariants.items():
            value = fields.get(key)
            if value is None or not (lo <= value <= hi):
                return False
        return True


@dataclass
class GameProfile:
    """Everything that varies between games, loaded from JSON."""

    id: str
    name: str
    description: str = ""
    hud_rows: int = HUD_BOTTOM_ROW
    sprite_h: int = 16
    max_fall_px: int = 40
    lookahead_px: int = 40
    climb_lookahead_px: int = 24
    max_wall_rise: int = 12
    goal: str = "Reach the right side of the screen."
    move: dict = field(default_factory=lambda: {"progress": ["right"], "back": ["left"]})
    actions: dict = field(default_factory=lambda: {"jump": ["A"], "run": ["B"]})
    jump_hold_frames: int = 7
    ram: Optional[RamMap] = None

    @property
    def progress_buttons(self) -> tuple:
        return tuple(self.move.get("progress") or ("right",))

    @property
    def back_buttons(self) -> tuple:
        return tuple(self.move.get("back") or ("left",))

    @property
    def jump_buttons(self) -> tuple:
        return tuple(self.actions.get("jump") or ("A",))

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, profile_id: str) -> "GameProfile":
        path = Path(PROFILE_DIR) / ("%s.json" % profile_id)
        if not path.is_file():
            available = ", ".join(sorted(p.stem for p in Path(PROFILE_DIR).glob("*.json")))
            raise ProfileNotFound(
                "unknown profile %r (available: %s)\n"
                "To drive a new NES game, copy src/profiles/generic.json and edit it."
                % (profile_id, available)
            )
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, data: dict) -> "GameProfile":
        ram = data.get("ram")
        ram_map = None
        if ram:
            ram_map = RamMap(
                addresses={k: int(v) for k, v in ram.items() if isinstance(v, int)},
                enemy_stride=int(ram.get("enemy_stride", 0)),
                enemy_fields={k: int(v) for k, v in (ram.get("enemy_fields") or {}).items()},
                enemy_empty=tuple(ram.get("enemy_empty") or (0, 0, 0)),
                invariants={
                    k: (int(v[0]), int(v[1]))
                    for k, v in (data.get("invariants") or {}).items()
                },
                terminated_statuses=tuple(int(v) for v in (ram.get("terminated_statuses") or ())),
            )
        return cls(
            id=data["id"],
            name=data.get("name", data["id"]),
            description=data.get("description", ""),
            hud_rows=int(data.get("hud_rows", HUD_BOTTOM_ROW)),
            sprite_h=int(data.get("sprite_h", 16)),
            max_fall_px=int(data.get("max_fall_px", 40)),
            lookahead_px=int(data.get("lookahead_px", 40)),
            climb_lookahead_px=int(data.get("climb_lookahead_px", 24)),
            max_wall_rise=int(data.get("max_wall_rise", 12)),
            goal=data.get("goal", "Reach the right side of the screen."),
            move=data.get("move") or {"progress": ["right"], "back": ["left"]},
            actions=data.get("actions") or {"jump": ["A"], "run": ["B"]},
            jump_hold_frames=int(data.get("jump_hold_frames", 7)),
            ram=ram_map,
        )


# ------------------------------------------------------------------- state
@dataclass
class Blob:
    """A generic hazard the player should jump over: a wall, pipe or block."""

    x: int  # screen x
    top: int  # topmost row of the obstacle
    height: int


@dataclass
class GameState:
    """What one frame proves about the game, for any ROM."""

    frame: int = 0
    # --- from the framebuffer (always present) -----------------------------
    character_x: Optional[int] = None
    character_y: Optional[int] = None
    character_bottom: Optional[int] = None
    character_visible: bool = False
    character_uncertain: bool = False
    # Full bounding box, for the dashboard overlay only (x/y already carry the
    # values the policies read).
    character_box: Optional[dict] = None
    on_ground: Optional[bool] = None
    pit_ahead: bool = False
    pit_distance: Optional[int] = None
    wall_ahead: bool = False
    wall_distance: Optional[int] = None
    wall_height: int = 0
    climbable_ahead: bool = False
    floor_row: int = SCREEN_H
    backdrop: str = ""
    floor_colour: str = ""
    # Blobs that are not the character: for the dashboard overlay and for the
    # reflex layer's obstacle logic.
    blobs: List[Blob] = field(default_factory=list)
    # --- from RAM, only when the profile's invariants hold -----------------
    ram_plausible: bool = False
    hud: dict = field(default_factory=dict)
    world_x: Optional[int] = None
    enemies: List[dict] = field(default_factory=list)
    # --- derived -----------------------------------------------------------
    terminated: bool = False

    def nearest_enemy(self, ahead: bool = True) -> Optional[dict]:
        pool = list(self.enemies)
        if ahead:
            pool = [e for e in pool if e.get("dx", 0) > -8]
        if not pool:
            return None
        return min(pool, key=lambda e: abs(e.get("dx", 0)))

    def nearest_blob_ahead(self) -> Optional[Blob]:
        if self.character_x is None:
            return None
        ahead = [b for b in self.blobs if b.x > self.character_x + 4]
        if not ahead:
            return None
        return min(ahead, key=lambda b: b.x - self.character_x)

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "character_x": self.character_x,
            "character_visible": self.character_visible,
            "character_uncertain": self.character_uncertain,
            "character_box": self.character_box,
            "on_ground": self.on_ground,
            "pit_ahead": self.pit_ahead,
            "pit_distance": self.pit_distance,
            "wall_ahead": self.wall_ahead,
            "wall_distance": self.wall_distance,
            "wall_height": self.wall_height,
            "climbable_ahead": self.climbable_ahead,
            "floor_row": self.floor_row,
            "backdrop": self.backdrop,
            "floor_colour": self.floor_colour,
            "blobs": [{"x": b.x, "top": b.top, "height": b.height} for b in self.blobs],
            "ram_plausible": self.ram_plausible,
            "world_x": self.world_x,
            "hud": dict(self.hud),
            "enemies": list(self.enemies),
            "terminated": self.terminated,
        }


# ------------------------------------------------------------------- RAM
def _u8(ram: Optional[np.ndarray], addr: int) -> int:
    if ram is None or addr < 0 or addr >= RAM_SIZE:
        return 0
    return int(ram[addr]) & 0xFF


def _u16(ram: Optional[np.ndarray], hi_addr: int, lo_addr: int) -> int:
    return (_u8(ram, hi_addr) << 8) | _u8(ram, lo_addr)


def _bcd(value: int) -> int:
    """Some games store counters as BCD: $05 means 5, $42 means 42."""
    return ((value >> 4) & 0x0F) * 10 + (value & 0x0F)


# ------------------------------------------------------------------- decode
class Decoder:
    """Profile-driven decoder: framebuffer (+ optional RAM) -> GameState.

    Holds a one-frame character box as a tracking hint. That hint is what
    stops the geometric search from handing the role to a Goomba the moment it
    wanders further left than the player -- the exact failure mode of the
    previous "leftmost blob on the floor" rule.
    """

    def __init__(self, profile: GameProfile):
        self.profile = profile
        self._hint: Optional[tuple] = None

    def reset(self) -> None:
        self._hint = None

    def decode(self, ram: Optional[np.ndarray], frame: np.ndarray, frame_index: int = 0) -> GameState:
        profile = self.profile
        st = GameState(frame=frame_index)

        vis = vision.analyse(frame, hud_rows=profile.hud_rows)
        st.floor_row = vis.floor_row
        st.backdrop = "#%02X%02X%02X" % tuple(int(v) for v in vis.backdrop_colour)
        st.floor_colour = "#%02X%02X%02X" % tuple(int(v) for v in vis.floor_colour)

        blobs = vision.objects(frame, vis, hud_rows=profile.hud_rows)
        character = vision.find_character(frame, vis, profile, blobs, hint=self._hint)

        if character:
            st.character_x = character["x"]
            st.character_y = character["y_min"]
            st.character_bottom = character["bottom"]
            st.character_visible = True
            st.character_uncertain = bool(character.get("uncertain"))
            st.character_box = {
                "x_min": character["x_min"],
                "x_max": character["x_max"],
                "y_min": character["y_min"],
                "bottom": character["bottom"],
            }
            self._hint = (
                character["x_min"],
                character["x_max"],
                character["y_min"],
                character["bottom"],
            )
        else:
            self._hint = None

        if st.character_x is not None and st.character_bottom is not None:
            col = int(np.clip(st.character_x, 0, SCREEN_W - 1))
            surface = int(vis.ground[col])
            st.on_ground = (surface - st.character_bottom) <= 8 and surface < SCREEN_H

            hazard = vision.ground_hazard_ahead(
                vis.ground, col, profile.lookahead_px, profile.max_wall_rise
            )
            if hazard:
                if hazard["kind"] == "pit":
                    st.pit_ahead = True
                    st.pit_distance = hazard["distance"]
                else:
                    st.wall_ahead = True
                    st.wall_distance = hazard["distance"]
                    st.wall_height = hazard.get("height", 0)
            st.climbable_ahead = vision.climbable_ahead(
                vis, col, profile.climb_lookahead_px, profile.max_wall_rise
            )

        # Non-character blobs, kept for the overlay and for obstacle logic.
        for blob in blobs:
            if character and blob.x0 <= character["x_max"] and blob.x1 >= character["x_min"]:
                continue  # part of the character
            if blob.y1 >= SCREEN_H - 4 and blob.x1 - blob.x0 > 40:
                continue  # the floor itself
            st.blobs.append(
                Blob(x=blob.x, top=blob.y0, height=blob.y1 - blob.y0)
            )

        # --- optional RAM -------------------------------------------------
        if profile.ram is not None and ram is not None:
            self._decode_ram(profile.ram, ram, st)
        # `terminated` only comes from a status byte the profile actually mapped;
        # with no RAM map we simply never declare the run over.
        st.terminated = st.hud.get("terminated", False)
        return st

    @staticmethod
    def _decode_ram(ram_map: RamMap, ram: np.ndarray, st: GameState) -> None:
        addr = ram_map.addresses
        fields = {
            "world": _u8(ram, addr["world"]) if "world" in addr else None,
            "lives": _u8(ram, addr["lives"]) if "lives" in addr else None,
            "coins": _bcd(_u8(ram, addr["coins"])) if "coins" in addr else None,
            "timer": (
                _u16(ram, addr["timer_high"], addr["timer_low"])
                if "timer_high" in addr and "timer_low" in addr
                else None
            ),
            "status": _u8(ram, addr["status"]) if "status" in addr else None,
        }
        if not ram_map.plausible(fields):
            # A map that does not match this dump does not read as zero, it
            # reads as plausible fiction. Drop it and say unknown.
            st.ram_plausible = False
            return
        st.ram_plausible = True
        st.hud = {
            "coins": fields["coins"],
            "timer": fields["timer"],
            "lives": fields["lives"],
            "world": fields["world"],
            "status": fields["status"],
            "terminated": fields["status"] in ram_map.terminated_statuses,
        }
        if "player_x_hi" in addr and "player_x_lo" in addr:
            st.world_x = _u16(ram, addr["player_x_hi"], addr["player_x_lo"])

        base = ram_map.enemy_base
        if base is None or st.world_x is None:
            return
        fields_def = ram_map.enemy_fields
        for i in range(4):
            slot = base + i * ram_map.enemy_stride
            page = _u8(ram, slot + fields_def["page_col"])
            x_hi = _u8(ram, slot + fields_def["x_hi"])
            x_lo = _u8(ram, slot + fields_def["x_lo"])
            y_hi = _u8(ram, slot + fields_def["y_hi"])
            y_lo = _u8(ram, slot + fields_def["y_lo"])
            if (page, x_hi, x_lo) == tuple(ram_map.enemy_empty):
                continue
            x = (page << 8) | (x_hi << 8) | x_lo
            dx = x - st.world_x
            if abs(dx) > SCREEN_W:
                continue
            st.enemies.append(
                {
                    "x": x,
                    "y": (y_hi << 8) | y_lo,
                    "dx": dx,
                    "screen_x": int(np.clip(dx + SCREEN_W // 2, 0, SCREEN_W - 1)),
                }
            )


def decode(ram: Optional[np.ndarray], frame: np.ndarray, frame_index: int = 0,
           profile: GameProfile | None = None) -> GameState:
    """One-shot convenience wrapper; long runs should hold a `Decoder`."""
    return Decoder(profile or GameProfile.load("generic")).decode(ram, frame, frame_index)


__all__ = [
    "Blob",
    "Decoder",
    "GameProfile",
    "GameState",
    "ProfileNotFound",
    "RamMap",
    "decode",
]
