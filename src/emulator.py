"""Thin, defensive wrapper around nes_py.NESEnv.

nes_py exposes everything we need but through raw buffers, so this module turns it
into an ordinary object with a `press(buttons)` call and a `frame` property, and it
degrades with clear errors instead of AttributeErrors when a buffer is missing.
"""

from __future__ import annotations

import time
from typing import Iterable

import numpy as np

from .config import NTSC_FPS, SCREEN_H, SCREEN_W, buttons_to_byte


class RomNotFound(FileNotFoundError):
    pass


class NesRunner:
    """One NES instance. Not thread-safe: exactly one thread owns it."""

    def __init__(self, rom_path: str):
        self.rom_path = str(rom_path)
        try:
            from nes_py import NESEnv
        except ImportError as exc:  # pragma: no cover - environment problem
            raise RuntimeError(
                "nes_py is not installed. Run: pip install nes-py"
            ) from exc

        try:
            self.env = NESEnv(self.rom_path)
        except FileNotFoundError as exc:
            raise RomNotFound(str(exc)) from exc

        self.frame_index = 0
        self._last_buttons = 0
        self.reset()

    # ------------------------------------------------------------------ basics
    @property
    def screen(self) -> np.ndarray:
        """(240, 256, 3) uint8 RGB, a live view into the emulator's buffer."""
        buf = getattr(self.env, "screen", None)
        if buf is None:
            raise RuntimeError("nes_py did not expose env.screen")
        return buf

    @property
    def ram(self) -> np.ndarray:
        """2048 uint8, the CPU's address space mirrored."""
        buf = getattr(self.env, "ram", None)
        if buf is None:
            raise RuntimeError("nes_py did not expose env.ram")
        return buf

    def reset(self) -> np.ndarray:
        result = self.env.reset()
        # gymnasium reset() -> (observation, info); older builds return just obs.
        obs = result[0] if isinstance(result, tuple) else result
        self.frame_index = 0
        self._last_buttons = 0
        return obs

    def step(self, buttons: Iterable[str] = ()) -> np.ndarray:
        """Hold `buttons` down for exactly one frame."""
        byte = buttons_to_byte(buttons)
        self._last_buttons = byte
        result = self.env.step(byte)
        obs = result[0] if isinstance(result, tuple) else result
        self.frame_index += 1
        return obs

    @property
    def buttons(self) -> int:
        """The bitmap byte most recently passed to the emulator."""
        return self._last_buttons

    # ------------------------------------------------------------------ states
    def dump_state(self):
        if hasattr(self.env, "dump_state"):
            return self.env.dump_state()
        raise RuntimeError("this nes_py build has no dump_state()")

    def load_state(self, snapshot) -> None:
        if hasattr(self.env, "load_state"):
            self.env.load_state(snapshot)
        else:
            raise RuntimeError("this nes_py build has no load_state()")

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:  # noqa: BLE001 - closing must never raise
            pass


class FramePacer:
    """Keeps the emulator at a target frame rate without busy-waiting the whole core."""

    def __init__(self, fps: float = NTSC_FPS, speed: float = 1.0):
        self.frame_interval = 1.0 / (fps * max(speed, 1e-6))
        self._next = time.perf_counter()

    def reset(self) -> None:
        self._next = time.perf_counter()

    def wait(self) -> None:
        self._next += self.frame_interval
        now = time.perf_counter()
        if self._next < now:
            # We fell behind (e.g. laya hogged a core). Resync instead of
            # accumulating debt, which would otherwise spiral into fast-forward.
            self._next = now
            return
        time.sleep(self._next - now)


def frame_to_rgb_bytes(frame: np.ndarray) -> bytes:
    """Flatten a framebuffer into bytes for a zero-copy canvas putImageData."""
    if frame.shape != (SCREEN_H, SCREEN_W, 3):
        raise ValueError("unexpected frame shape %r" % (frame.shape,))
    return np.ascontiguousarray(frame, dtype=np.uint8).tobytes()
