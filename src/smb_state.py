"""Read Super Mario Bros.'s state out of the emulator.

Two independent sources, each used where it is strongest:

* **RAM** gives the logical state: how far along the level Mario is, the timer,
  coins, lives, growth status, and the four enemy slots the game keeps live.
* **The framebuffer** gives the physical state that RAM makes awkward: exactly
  where Mario's sprite is on screen, where the ground surface is, and whether the
  floor ahead of him is missing.

Keeping them separate matters because they fail differently. If an enemy-slot
offset is wrong you lose enemy range but Mario still moves correctly; if the
palette match drifts you lose ground detection but RAM is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .config import (
    ENEMY_EMPTY,
    ENEMY_FIELDS,
    ENEMY_STRIDE,
    HUD_BOTTOM_ROW,
    RAM,
    SCREEN_H,
    SCREEN_W,
    STATUS_CLEAR,
    STATUS_DEAD,
    STATUS_GAME_OVER,
    STATUS_NAMES,
)


@dataclass
class Enemy:
    page: int
    x: int  # world pixels, same scale as SmbState.world_x
    y: int
    on_screen: bool
    screen_x: Optional[int] = None

    def dx_from(self, mario_x: int) -> int:
        return self.x - mario_x


@dataclass
class SmbState:
    frame: int = 0
    # --- from RAM -----------------------------------------------------------
    world_x: int = 0
    world_y: int = 0
    scroll_x: int = 0
    coins: int = 0
    timer: int = 0
    world: int = 1
    lives: int = 3
    status: int = 0
    enemies: List[Enemy] = field(default_factory=list)
    # --- from the framebuffer ---------------------------------------------
    mario_screen_x: Optional[int] = None
    mario_bottom: Optional[int] = None
    on_ground: Optional[bool] = None
    pit_ahead: bool = False
    pit_distance: Optional[int] = None
    ground_ahead_ok: bool = True
    # --- derived ------------------------------------------------------------
    terminated: bool = False
    ram_plausible: bool = True

    @property
    def status_name(self) -> str:
        return STATUS_NAMES.get(self.status, "unknown(%d)" % self.status)

    @property
    def big(self) -> bool:
        return self.status in (0x01, 0x02)

    def nearest_enemy(self, ahead: bool = True) -> Optional[Enemy]:
        """Closest live enemy, optionally only ones still in front of Mario."""
        pool = [e for e in self.enemies if e.on_screen]
        if ahead:
            pool = [e for e in pool if e.dx_from(self.world_x) > -8]
        return min(pool, key=lambda e: abs(e.dx_from(self.world_x)), default=None)

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "world_x": self.world_x,
            "world_y": self.world_y,
            "coins": self.coins,
            "timer": self.timer,
            "world": self.world,
            "lives": self.lives,
            "status": self.status_name,
            "mario_screen_x": self.mario_screen_x,
            "on_ground": self.on_ground,
            "pit_ahead": self.pit_ahead,
            "pit_distance": self.pit_distance,
            "terminated": self.terminated,
            "ram_plausible": self.ram_plausible,
        }


def _u8(ram: np.ndarray, addr: int) -> int:
    return int(ram[addr]) & 0xFF


def _u16(ram: np.ndarray, hi_addr: int, lo_addr: int) -> int:
    return (_u8(ram, hi_addr) << 8) | _u8(ram, lo_addr)


def _bcd(value: int) -> int:
    """SMB1 stores coins as BCD: $05 means 5, $42 means 42."""
    return ((value >> 4) & 0x0F) * 10 + (value & 0x0F)


def ram_is_plausible(world: int, lives: int, coins: int, timer: int, status: int) -> bool:
    """Does the decoded RAM look like a real SMB1 in-level state?

    The community memory map in config.py does not hold for every SMB1 build. On the
    "25th Anniversary (Promo, Virtual Console)" dump every one of these fields reads
    zero for the whole level -- world 0, no lives, no timer -- while Mario is plainly
    running across the screen.

    That matters more than it looks. These numbers are what gets serialised into the
    text laya reads, so a wrong map does not merely show wrong values on the
    dashboard, it actively tells the model "world 0-1, 0 lives, 0 time". Better to
    admit the numbers are unavailable and let the observation fall back to what the
    framebuffer can prove.

    The invariants below are properties of the game, not of any address table: world
    numbers start at 1, you begin with lives, coins are 0-99, the timer is 400, and
    the status byte is one of seven known values.
    """
    return (
        1 <= world <= 4
        and 1 <= lives <= 99
        and 0 <= coins <= 99
        and 1 <= timer <= 400
        and status in STATUS_NAMES
    )


# --------------------------------------------------------------------- framebuffer
def find_mario(frame: np.ndarray, ground: np.ndarray | None = None, near: int | None = None):
    """Locate Mario's sprite blob without assuming a palette.

    Two earlier attempts keyed on absolute colours and both were wrong on real ROMs:

    * red alone ($D82800, Mario's hat) also matches the copyright screen's
      "(c)1985 NINTENDO", so it reported Mario on the title screen at frame 34 and
      quietly invalidated every measurement built on top of it;
    * adding pure blue ($0000FC, his overalls) fixed that, but the "25th Anniversary
      (Promo, Virtual Console)" build has neither colour -- its Mario red is #F83800
      and there is no pure blue anywhere in the frame, so the detector found nothing
      while world 1-1 was plainly running.

    So nothing here names a colour. What is stable across SMB1 builds is the geometry:
    the backdrop is whatever dominates the top of the playfield, the floor is whatever
    dominates the bottom, and Mario is a small blob standing on the floor. Find the
    floor, subtract both backdrop and floor colour, and take the leftmost compact
    cluster touching the floor. That works on a palette it has never seen, and when it
    guesses wrong it fails by finding some other object rather than by finding nothing.

    The weakness, stated plainly: "leftmost thing standing on the floor" is Mario only
    at the start of a level. Once a Goomba is further left than he is, or he is under a
    block, this returns the wrong object -- on this ROM it locked onto the hill at
    x=11 for a whole level and reported that as Mario's position.

    `near` contains that flaw: when RAM gives the column the game actually draws Mario
    in (RAM["player_screen_x"], measured exact), the search starts there instead of at
    x=0, so only a blob at *his* column can match. The scan still returns None rather
    than a wrong object when nothing stands at that column.
    """
    if ground is None:
        ground = ground_profile(frame)

    backdrop = background_colour(frame).astype(np.int16)
    floor_rows = ground[ground < SCREEN_H]
    if floor_rows.size == 0:
        return None
    floor_y = int(np.median(floor_rows))

    # Floor colour: the modal colour of a thin band just above the surface.
    band = frame[max(HUD_BOTTOM_ROW, floor_y - 2):floor_y + 1, :, :].reshape(-1, 3)
    colours, counts = np.unique(band, axis=0, return_counts=True)
    floor_colour = colours[int(np.argmax(counts))].astype(np.int16)

    playfield = frame[HUD_BOTTOM_ROW:, :, :].astype(np.int16)
    away_from_backdrop = np.abs(playfield - backdrop).sum(axis=2) > 24
    away_from_floor = np.abs(playfield - floor_colour).sum(axis=2) > 24
    objects = away_from_backdrop & away_from_floor
    # Black sprite outlines belong to the object, not the backdrop, so keep them.
    objects |= playfield.max(axis=2) <= 8

    # Column -> lowest object pixel, ignoring anything at or below that column's floor.
    rows_index = np.arange(objects.shape[0])[:, None]
    above_floor = rows_index < (ground[None, :] - HUD_BOTTOM_ROW)
    standing = objects & above_floor

    start = int(np.clip(near - 8, 0, SCREEN_W - 1)) if near is not None else 0
    for column in range(start, SCREEN_W):
        if near is not None and column > near + 16:
            break  # nothing at Mario's column; do not wander off after other objects
        ys = np.nonzero(standing[:, column])[0]
        if ys.size == 0:
            continue
        lowest = int(ys.max())
        # A blob sitting on the floor: its bottom must be within a sprite height of it.
        if (ground[column] - HUD_BOTTOM_ROW) - lowest > 26:
            continue
        y0 = max(0, lowest - 32)
        window = objects[y0:lowest + 1, max(0, column - 8):column + 24]
        ys2, xs2 = np.nonzero(window)
        if ys2.size < 8:  # too small to be a sprite; probably mortar or a pebble
            continue
        x_off = max(0, column - 8)
        return {
            "x": int(xs2.mean()) + x_off,
            "x_min": int(xs2.min()) + x_off,
            "x_max": int(xs2.max()) + x_off,
            "y_min": int(ys2.min()) + y0 + HUD_BOTTOM_ROW,
            "bottom": int(ys2.max()) + y0 + HUD_BOTTOM_ROW,
            "pixels": int(window.sum()),
        }
    return None


def background_colour(frame: np.ndarray) -> np.ndarray:
    """The colour of the playfield's backdrop, inferred from the frame itself.

    Hard-coding $5C94FC works on an original SMB1 dump and silently fails on
    everything else: the "25th Anniversary (Promo, Virtual Console)" build has a
    #6888FC sky, which is 12 away on two channels, so a tolerance-8 test misses it.
    That makes every sky pixel count as solid ground, which makes ground detection,
    `on_ground` and `pit_ahead` meaningless -- quietly, since they still return values.

    The backdrop is instead whatever dominates the top of the playfield: sky above
    ground in an outdoor level, flat black underground. Taking the modal colour of a
    band near the top of the playfield gets both without naming either.
    """
    band = frame[HUD_BOTTOM_ROW:HUD_BOTTOM_ROW + 8, :, :].reshape(-1, 3)
    colours, counts = np.unique(band, axis=0, return_counts=True)
    return colours[int(np.argmax(counts))]


def ground_profile(frame: np.ndarray) -> np.ndarray:
    """For each of the 256 columns, the row of the lowest non-background pixel.

    SCREEN_H (240) means the column is empty all the way down, i.e. a pit.
    """
    playfield = frame[HUD_BOTTOM_ROW:, :, :].astype(np.int16)
    backdrop = background_colour(frame).astype(np.int16)
    # Distance to the two nearest corners of the RGB cube would be exact; an L1
    # threshold is enough because NES art never lands near the backdrop by accident.
    is_bg = np.abs(playfield - backdrop).sum(axis=2) <= 24
    # Black is always background, even outdoors: the letterbox and sprite outlines use
    # it, and a column of it inside a pit must not read as floor.
    is_bg |= playfield.max(axis=2) <= 8
    solid = ~is_bg
    any_solid = solid.any(axis=0)
    # The row count comes from the cropped array, not from SCREEN_HEIGHT. Using
    # SCREEN_HEIGHT - 1 here shifted every surface down by exactly HUD_BOTTOM_ROW, so
    # a real floor row landed at 240+ -- which is the sentinel for "pit". Every column
    # therefore read as a hole, quietly, while still returning plausible numbers.
    rows = playfield.shape[0]
    lowest = rows - 1 - np.argmax(solid[::-1], axis=0)
    return np.where(any_solid, lowest + HUD_BOTTOM_ROW, SCREEN_H).astype(np.int32)


# --------------------------------------------------------------------- combined
def decode(ram: np.ndarray, frame: np.ndarray, frame_index: int = 0) -> SmbState:
    """Build an SmbState from one frame of RAM plus one framebuffer.

    Coordinates come from RAM, measured on this ROM (see config.py RAM map):
    world X = $06D:$086, screen X = $005, world Y = $001:$002. The framebuffer
    is then used to *anchor* the sprite search to that column -- not to guess
    the position, which the pixel-only method got wrong (it locked onto a hill
    at x=11 for a whole level).
    """
    st = SmbState(frame=frame_index)

    st.world_x = _u16(ram, RAM["player_x_hi"], RAM["player_x_lo"])
    st.world_y = _u16(ram, RAM["player_y_page"], RAM["player_y_subpx"])
    # Measured exact: the byte the game itself positions Mario's sprite with.
    ram_screen_x = _u8(ram, RAM["player_screen_x"])
    # Outside a level (title, level card, death) world X reads 0 while $005 keeps
    # a stale value (133 on this build), which would assert a position the screen
    # contradicts. World >= screen is exactly "camera >= 0" -- true in-level,
    # false for the stale residue, so it doubles as an in-level check.
    coords_live = st.world_x >= ram_screen_x
    st.mario_screen_x = ram_screen_x if coords_live else None
    # Camera = world - screen (identity; $003 stayed constant on this build,
    # so the community scroll byte cannot be it).
    st.scroll_x = st.world_x - ram_screen_x if coords_live else 0
    st.coins = _bcd(_u8(ram, RAM["coins"]))
    st.timer = _u16(ram, RAM["timer_high"], RAM["timer_low"])
    st.world = _u8(ram, RAM["world"])
    st.lives = _u8(ram, RAM["lives"])
    st.status = _u8(ram, RAM["status"])
    st.terminated = st.status in (STATUS_DEAD, STATUS_GAME_OVER, STATUS_CLEAR)
    st.ram_plausible = ram_is_plausible(st.world, st.lives, st.coins, st.timer, st.status)

    # Enemy slots: keep only ones the game has actually populated.
    for i in range(4):
        base = RAM["enemy_base"] + i * ENEMY_STRIDE
        page = _u8(ram, base + ENEMY_FIELDS["page_col"])
        x_hi = _u8(ram, base + ENEMY_FIELDS["x_hi"])
        x_lo = _u8(ram, base + ENEMY_FIELDS["x_lo"])
        y_hi = _u8(ram, base + ENEMY_FIELDS["y_hi"])
        y_lo = _u8(ram, base + ENEMY_FIELDS["y_lo"])
        if (page, x_hi, x_lo) == ENEMY_EMPTY:
            continue
        enemy = Enemy(
            page=page,
            x=(page << 8) | (x_hi << 8) | x_lo,
            y=(y_hi << 8) | y_lo,
            on_screen=False,
        )
        # RAM coordinates are world-space; only enemies near the camera matter.
        enemy.on_screen = abs(enemy.dx_from(st.world_x)) <= SCREEN_W
        if enemy.on_screen:
            enemy.screen_x = int(
                np.clip(enemy.dx_from(st.world_x) + SCREEN_W // 2, 0, SCREEN_W - 1)
            )
        st.enemies.append(enemy)

    # A wrong map does not merely report zero enemies: the slots hold whatever the
    # real game happens to keep there, and computing distances against a world_x of 0
    # invents enemies at arbitrary offsets. Drop them rather than act on fiction.
    if not st.ram_plausible:
        st.enemies = []

    # The floor is needed by both the sprite search and the pit logic, and computing
    # it twice on a 240x256 frame is pure waste at 60 fps.
    ground = ground_profile(frame)

    # Anchor the pixel search at the column RAM says Mario is drawn in. The search
    # can then only confirm footing (and his sprite bottom), never re-position him.
    mario = find_mario(frame, ground=ground, near=st.mario_screen_x)
    if mario:
        st.mario_bottom = mario["bottom"]

    if st.mario_screen_x is not None:
        col = int(np.clip(st.mario_screen_x, 0, SCREEN_W - 1))
        surface = int(ground[col])
        if st.mario_bottom is not None:
            # Sprite rows overlap the surface row when standing on it.
            st.on_ground = (surface - st.mario_bottom) <= 4 and surface < SCREEN_H

        # Look for a hole in the floor just ahead of him.
        lookahead_end = min(SCREEN_W - 1, col + 40)
        window = ground[col + 8 : lookahead_end + 1]
        holes = np.nonzero(window >= SCREEN_H - 1)[0]
        if holes.size:
            st.pit_ahead = True
            st.pit_distance = int(holes[0]) + 8
        else:
            st.ground_ahead_ok = True

    return st
