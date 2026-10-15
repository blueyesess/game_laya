"""Palette-agnostic frame analysis: the generic "eyes" of the project.

The previous decoder was a Super Mario Bros. decoder wearing a generic
dataclass. Every game-specific assumption lived inline -- HUD rows, RAM maps,
palette tables, "the leftmost thing standing on the floor is Mario". This
module keeps the part that is *actually* game-agnostic and states its
assumptions up front, so a new NES ROM can be driven by dropping in a profile
JSON (src/profiles/*.json) rather than by editing Python.

What is assumed here, and why it is a reasonable floor for NES side-scrollers:

* the camera follows the player, so the player tends to sit in the middle third
  of the screen once anything else enters the frame;
* the backdrop (sky, cave wall) dominates the top of the playfield, the floor
  dominates the bottom, and everything else is an object;
* the HUD, if any, is a thin band at the very top (overridable per profile);
* platforms are roughly horizontal, so "column is empty down to the bottom" is
  a hole;
* staircases are dense diagonal edges, which is what ladders and stair blocks
  look like before any per-game knowledge.

What is deliberately *not* assumed: any colour, any RAM address, any sprite
name. The three SMB1 palettes that used to be hard-coded in smb_state.py were
wrong on two of three real dumps; everything here is inferred per frame.

Two layers of failure, kept separate on purpose:

* vision failures (a mis-detected blob) only cost you a wrong character box;
* profile failures (a wrong RAM map) only cost you HUD numbers.
They fail independently, so `state.ram_plausible` and `state.character_visible`
degrade on their own instead of poisoning each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .config import HUD_BOTTOM_ROW, SCREEN_H, SCREEN_W

# L1 distance in RGB below which two pixels count as the same colour. NES art
# never lands near the backdrop by accident, so this is generous on purpose.
BG_TOLERANCE = 24

# Connected components are 8-connected: sprite outlines diagonally touch
# their fill, and a Goomba's feet are 1px wide.
_N8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


@dataclass
class Blob:
    """One connected object in the playfield."""

    x0: int
    x1: int
    y0: int
    y1: int
    pixels: int

    @property
    def x(self) -> int:  # horizontal centre
        return (self.x0 + self.x1) // 2

    @property
    def x_right(self) -> int:
        return self.x1

    @property
    def bottom(self) -> int:
        return self.y1

    def overlaps_x(self, x: int, slack: int = 4) -> bool:
        return self.x0 - slack <= x <= self.x1 + slack


@dataclass
class VisionResult:
    """Everything one frame of pixels can prove, without any game knowledge."""

    ground: np.ndarray  # per-column surface row; SCREEN_H == no surface (hole)
    # Same, but with a sprite's own pixels smoothed past. Use this wherever the
    # question is "what is *under* this thing" -- a character standing on the floor
    # makes the raw profile jump to the top of his head in that column.
    surface: np.ndarray
    floor_row: int  # the dominant surface row; None-ish -> SCREEN_H
    backdrop_colour: np.ndarray
    floor_colour: np.ndarray
    background: np.ndarray = field(default=None)  # alias of `ground`, kept for callers
    # Blobs whose vertical position could not be resolved (no floor, or a
    # floating platform); useful for overlays, not for logic.
    loose_blobs: List[Blob] = field(default_factory=list)


# ------------------------------------------------------------------ colour maths
def background_colour(frame: np.ndarray, hud_rows: int) -> np.ndarray:
    """The backdrop is whatever dominates the top of the playfield.

    Hard-coding a sky colour is what broke this twice on real dumps: an
    original SMB1 has #5C94FC and the Virtual Console re-release has #6888FC,
    12 away on two channels, so a tolerance-8 test reads the re-release sky as
    solid ground and every pit goes undetected. Inferred per frame, both work,
    and an underground level's black backdrop works too.
    """
    band = frame[hud_rows : hud_rows + 8, :, :].reshape(-1, 3)
    colours, counts = np.unique(band, axis=0, return_counts=True)
    return colours[int(np.argmax(counts))]


def ground_profile(frame: np.ndarray, hud_rows: int) -> np.ndarray:
    """For each of the 256 columns, the row of its *standing surface*: the top of the
    lowest contiguous run of non-background pixels.

    SCREEN_H (240) means the column is empty all the way down, i.e. a hole.

    Three subtleties, each of which was a silent bug before:

    * it must be the *surface*, not the lowest solid pixel. A floor that reaches the
      bottom of the screen is solid at row 239 in every column, so "lowest pixel"
      returns 239 everywhere -- and then a character standing on it measures 44px off
      the floor and is rejected as airborne, while a 40px lookahead finds "no surface"
      anywhere and reports a pit on perfectly flat ground.
    * the row count comes from the *cropped* array, not SCREEN_H. Using SCREEN_H - 1
      here shifts every surface down by exactly hud_rows, so a real floor row lands at
      240+ -- which is the sentinel for "hole", so every column reads as a pit while
      still returning plausible numbers.
    * black is background even outdoors. The letterbox and the sprite outlines use
      it, and a column of it inside a pit must not read as floor.
    """
    playfield = frame[hud_rows:, :, :].astype(np.int16)
    backdrop = background_colour(frame, hud_rows).astype(np.int16)
    is_bg = np.abs(playfield - backdrop).sum(axis=2) <= BG_TOLERANCE
    is_bg |= playfield.max(axis=2) <= 8
    solid = ~is_bg
    rows = solid.shape[0]

    # Row of the last non-solid pixel in each column, or -1 when the column is solid
    # from the very top. One below that is the top of the lowest run -- where feet
    # land. A column with no solid pixel at all is a hole.
    gaps = np.where(solid, -1, np.arange(rows, dtype=np.int32)[:, None])
    last_gap = gaps.max(axis=0)
    return np.where(last_gap == rows - 1, SCREEN_H, last_gap + 1 + hud_rows).astype(np.int32)


def dominant_colour(pixels: np.ndarray) -> np.ndarray:
    colours, counts = np.unique(pixels.reshape(-1, 3), axis=0, return_counts=True)
    return colours[int(np.argmax(counts))]


# --------------------------------------------------------------- connected blobs
def _runs_of_row(row: np.ndarray):
    """[(x0, x1_inclusive), ...] for the True stretches of one row."""
    if not row.any():
        return []
    padded = np.concatenate(([False], row, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


class _Union:
    """Union-find over run ids. Flat array + path halving, no recursion."""

    __slots__ = ("parent",)

    def __init__(self) -> None:
        self.parent: List[int] = []

    def add(self) -> int:
        self.parent.append(len(self.parent))
        return len(self.parent) - 1

    def find(self, a: int) -> int:
        parent = self.parent
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:  # path halving
            parent[a], a = root, parent[a]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _blobs_from(mask: np.ndarray, hud_rows: int, min_pixels: int) -> List[Blob]:
    """Connected components of `mask`, as Blobs in playfield coordinates.

    This runs on every frame at 60 fps, so it is run-length based: instead of
    touching all ~53k pixels individually, it scans each row for stretches of
    True (a handful of runs per row) and unions runs that touch across the
    row boundary. 8-connectivity means a run on row y+1 overlaps a run on row
    y if their x ranges intersect at all, widened by one for the diagonal.
    """
    if not mask.any():
        return []

    union = _Union()
    row_runs: List[List[tuple]] = []
    for y in range(mask.shape[0]):
        runs = _runs_of_row(mask[y])
        ids: List[int] = []
        for x0, x1 in runs:
            rid = union.add()
            if row_runs:
                prev_runs, prev_ids = row_runs[-1]
                for (px0, px1), pid in zip(prev_runs, prev_ids):
                    if px1 + 1 >= x0 and px0 - 1 <= x1:
                        union.union(rid, pid)
            ids.append(rid)
        row_runs.append((runs, ids))

    # Reduce per component with numpy rather than a second pass over the pixels.
    components: dict = {}
    for y, (runs, ids) in enumerate(row_runs):
        for (x0, x1), rid in zip(runs, ids):
            root = union.find(rid)
            entry = components.get(root)
            count = x1 - x0 + 1
            if entry is None:
                components[root] = [x0, x1, y, y, count]
            else:
                entry[0] = min(entry[0], x0)
                entry[1] = max(entry[1], x1)
                entry[2] = min(entry[2], y)
                entry[3] = max(entry[3], y)
                entry[4] += count

    out: List[Blob] = []
    for x0, x1, y0, y1, count in components.values():
        if count < min_pixels:
            continue
        out.append(Blob(x0=x0, x1=x1, y0=y0 + hud_rows, y1=y1 + hud_rows, pixels=int(count)))
    out.sort(key=lambda blob: blob.x0)
    return out


def _running_max(a: np.ndarray, half_width: int) -> np.ndarray:
    """max over each window [i-half_width, i+half_width], in O(n) vectorised passes."""
    left = a.copy()
    right = a.copy()
    for k in range(1, half_width + 1):
        left[k:] = np.maximum(left[k:], a[:-k])
        right[:-k] = np.maximum(right[:-k], a[k:])
    return np.maximum(left, right)


# Half-width of the window used to look past a sprite. A character is at most ~32px
# wide, so 20 columns either side reaches the real floor under its feet while still
# staying inside a 40px-wide wall.
SURFACE_SMOOTH_HALF = 20

# One NES tile is 16px. Something is "climbable" when the surface rises by two tiles
# over a run no wider than one -- a staircase or a ladder. Wider and it is a wall.
CLIMB_MAX_WIDTH_PX = 16
CLIMB_MIN_RISE_PX = 32


# --------------------------------------------------------------------- analysis
def _floor_colour(frame: np.ndarray, floor_row: int, hud_rows: int) -> np.ndarray:
    """The colour of the floor material, sampled just *below* the standing surface.

    Sampling above the surface is the trap: the surface is where feet land, so the
    rows above it are backdrop and whatever is standing on it. On a test frame that
    meant the "floor colour" came back as sky, which then subtracted sky from the
    whole playfield and left nothing but a 256px-wide slab for object detection.
    """
    height = frame.shape[0]
    if floor_row >= SCREEN_H:
        band = frame[max(hud_rows, height - 4):, :, :]
    else:
        y0 = min(height - 1, floor_row + 1)
        band = frame[y0:min(height, y0 + 4), :, :]
    if band.size == 0:
        return frame[0, 0].astype(np.uint8)
    return dominant_colour(band)


def analyse(
    frame: np.ndarray,
    hud_rows: int = HUD_BOTTOM_ROW,
    min_object_pixels: int = 6,
) -> VisionResult:
    """Everything a single frame proves, before any game knowledge is applied."""
    ground = ground_profile(frame, hud_rows)
    backdrop = background_colour(frame, hud_rows)
    floor_rows = ground[ground < SCREEN_H]
    floor_row = int(np.median(floor_rows)) if floor_rows.size else SCREEN_H
    floor_colour = _floor_colour(frame, floor_row, hud_rows)

    # A sprite is contiguous with the floor it stands on, so the raw profile jumps up
    # to the top of the character in that column. The smoothed profile looks past it
    # and gives the real floor under his feet.
    holes_filled = np.where(ground < SCREEN_H, ground, floor_row).astype(np.int32)
    surface = _running_max(holes_filled, SURFACE_SMOOTH_HALF)

    return VisionResult(
        background=ground,
        ground=ground,
        surface=surface,
        floor_row=floor_row,
        backdrop_colour=backdrop,
        floor_colour=floor_colour,
    )


def objects(
    frame: np.ndarray,
    vis: VisionResult,
    hud_rows: int = HUD_BOTTOM_ROW,
    min_pixels: int = 6,
    max_gap: int = 20,
) -> List[Blob]:
    """Every non-backdrop, non-floor blob, merged across small gaps.

    Merging matters more than it looks: a character's hat, face and overalls are
    three separate colours, and a per-colour detector sees three blobs where
    there is one character. Threshold-free subtraction plus run merging gives
    back the union the eye sees.
    """
    playfield = frame[hud_rows:, :, :].astype(np.int16)
    away_bg = np.abs(playfield - vis.backdrop_colour.astype(np.int16)).sum(axis=2) > BG_TOLERANCE
    away_floor = np.abs(playfield - vis.floor_colour.astype(np.int16)).sum(axis=2) > BG_TOLERANCE
    mask = away_bg & away_floor
    # Sprite outlines are black; they belong to the object, not the backdrop.
    mask |= playfield.max(axis=2) <= 8

    # Drop everything at or below each column's floor. Without this a character
    # standing on the floor is 8-connected to the floor itself, and the "object"
    # blob becomes a 256px-wide slab spanning the whole screen -- which both kills
    # character detection and makes every observation claim an object is ahead.
    surface = vis.surface - hud_rows
    rows_index = np.arange(mask.shape[0], dtype=np.int32)[:, None]
    mask &= rows_index < surface[None, :]

    raw = _blobs_from(mask, hud_rows, min_pixels)
    merged: List[Blob] = []
    for blob in raw:
        if merged and blob.x0 - merged[-1].x1 <= max_gap and abs(blob.y0 - merged[-1].y0) <= 8:
            prev = merged[-1]
            merged[-1] = Blob(
                x0=prev.x0,
                x1=blob.x1,
                y0=min(prev.y0, blob.y0),
                y1=max(prev.y1, blob.y1),
                pixels=prev.pixels + blob.pixels,
            )
        else:
            merged.append(blob)
    return merged


def find_character(
    frame: np.ndarray,
    vis: VisionResult,
    profile,
    blobs: Optional[List[Blob]] = None,
    hint: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[dict]:
    """Locate the player, without naming a colour.

    A palette match was tried twice and failed on real dumps both times: pure
    red also matches a copyright screen's wordmark (so the player "appeared" on
    the title screen and quietly poisoned every measurement built on it), and a
    re-release that recolours the sprite has neither canonical hue, so the
    detector returned nothing while the game was plainly running.

    So the search is geometric, using three invariants that hold for any NES
    side-scroller the camera follows:

    1. the player stands on the floor -- its bottom edge is within one sprite
       height of the surface;
    2. the player is the thing nearest the middle third of the screen once
       anything else has entered the frame;
    3. the player is the *player*: no other blob has the whole screen to
       itself, so a level-opening heuristic only claims confidence once a
       second blob exists to vote against.
    """
    if blobs is None:
        blobs = objects(frame, vis, hud_rows=profile.hud_rows)

    if vis.floor_row >= SCREEN_H:
        return None

    h = SCREEN_H - profile.hud_rows
    window = (SCREEN_W // 3, 2 * SCREEN_W // 3)
    # A blob half off the right edge is a fresh spawn, not the player.
    max_width = 64

    def height(blob: Blob) -> int:
        return blob.y1 - blob.y0

    def surface(blob: Blob) -> int:
        """The floor row under *this* blob, not the screen-wide median.

        The median is right for a flat level and wrong the moment the player climbs
        a staircase or stands on a block: their local surface is 48px above the
        median, so a median-based test reads them as permanently airborne.
        """
        return int(vis.surface[int(np.clip(blob.x, 0, SCREEN_W - 1))])

    standing: List[Blob] = []
    for blob in blobs:
        local = surface(blob)
        if local >= SCREEN_H:
            continue  # over a pit: nothing to stand on
        # Within one sprite height of its own surface = standing on it.
        if local - blob.y1 > profile.sprite_h:
            continue
        if height(blob) > profile.sprite_h * 3:
            continue  # a wall, not a character
        if blob.pixels > 2048 or (blob.x1 - blob.x0) > max_width:
            continue  # terrain or a scrolling band, not a character
        standing.append(blob)

    if not standing:
        return None

    def box(blob: Blob) -> dict:
        return {
            "x": int(blob.x),
            "x_min": int(blob.x0),
            "x_max": int(blob.x1),
            "y_min": int(blob.y0),
            "bottom": int(blob.y1),
            "pixels": int(blob.pixels),
        }

    # A remembered box is a far better prior than geometry: it survives a
    # Goomba running past on the left, which is exactly where the previous
    # "leftmost blob" heuristic went wrong.
    if hint is not None:
        hx0, hx1, hy0, hy1 = hint
        hcx = (hx0 + hx1) / 2
        hcy = (hy0 + hy1) / 2
        hheight = hy1 - hy0
        for blob in standing:
            cx = blob.x
            cy = (blob.y0 + blob.y1) / 2
            if abs(cx - hcx) > 28:
                continue
            if abs(cy - hcy) > 24:
                continue
            if hheight and not (0.4 <= height(blob) / hheight <= 2.5):
                continue
            return box(blob)

    if len(standing) > 1:
        # The middle third of the screen is where the camera keeps the player.
        def middle_score(blob: Blob) -> float:
            distance = min(
                abs(blob.x - window[0]),
                abs(blob.x - window[1]),
                abs(blob.x - (window[0] + window[1]) // 2),
            )
            return (distance, abs(height(blob) - profile.sprite_h), blob.x0)

        best = min(standing, key=middle_score)
        return box(best)

    # Only one candidate: the level-opening case. It is usually the player,
    # but say so rather than pretend.
    only = standing[0]
    result = box(only)
    result["uncertain"] = True
    return result


def ground_hazard_ahead(
    ground: np.ndarray,
    col: int,
    lookahead: int,
    max_rise: int,
) -> Optional[dict]:
    """Is there a hole in the floor or a wall within `lookahead` px of `col`?

    A hole is a column with no surface at all. A wall is the surface jumping
    up by more than `max_rise` px in one column, which is what a pipe, a brick
    stack or a staircase looks like from the side. Both are things the player
    must jump over, and neither needs any game knowledge to detect.
    """
    end = min(SCREEN_W - 1, col + lookahead)
    if col + 1 > end:
        return None
    window = ground[col + 1 : end + 1]

    holes = np.nonzero(window >= SCREEN_H - 1)[0]
    if holes.size:
        return {"kind": "pit", "distance": int(holes[0]) + 1}

    # An elevated surface is a *smaller* row number, so a wall is a downward step in
    # this array. Require a run of elevated columns, not a single noisy pixel.
    for i in range(len(window) - 1):
        rise = int(window[i]) - int(window[i + 1])
        if rise > max_rise and int(window[i + 1]) < SCREEN_H:
            run = 0
            j = i + 1
            while j < len(window) and int(window[j]) < int(window[i]):
                run += 1
                j += 1
            if run >= 2:
                return {"kind": "wall", "distance": i + 1, "height": rise}
    return None


def climbable_ahead(vis: VisionResult, col: int, lookahead: int, max_rise: int = 12) -> bool:
    """Is there a narrow, steep rise in the surface just ahead of `col`?

    A ladder or a staircase reads the same from the side: the standing surface jumps
    up by at least two tiles over a run no wider than one tile. A wall or a pipe is
    the same rise over a *wide* run, and pressing up is never the answer against one.

    Honest limitation: this is a heuristic on shape, not on meaning. A single tall
    narrow pillar that the game intends you to jump will be reported as climbable.
    Nothing in the pixels distinguishes intent, and the profile's `climb_lookahead_px`
    and this tile budget are the knobs to tune per game.
    """
    start = min(SCREEN_W - 1, col + 4)
    end = min(SCREEN_W - 1, col + lookahead)
    if start >= end:
        return False
    window = vis.ground[start : end + 1]

    for i in range(len(window) - 1):
        rise = int(window[i]) - int(window[i + 1])
        if rise <= max_rise or int(window[i + 1]) >= SCREEN_H:
            continue
        # How many consecutive columns stay above where we started?
        run = 0
        j = i + 1
        while j < len(window) and int(window[j]) < int(window[i]) - max_rise:
            run += 1
            j += 1
        # Narrow (<= one 16px tile) and tall (>= two tiles above the take-off point).
        if 2 <= run <= CLIMB_MAX_WIDTH_PX and rise >= CLIMB_MIN_RISE_PX:
            return True
    return False


__all__ = [
    "BG_TOLERANCE",
    "Blob",
    "VisionResult",
    "analyse",
    "background_colour",
    "climbable_ahead",
    "dominant_colour",
    "find_character",
    "ground_hazard_ahead",
    "ground_profile",
    "objects",
]
