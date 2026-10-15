"""The game loop: an emulator thread and a brain thread, deliberately decoupled.

The central constraint is a latency mismatch, and the whole architecture follows from
it. The NES produces 60.0988 frames a second. A laya forward pass over three
questions costs on the order of 145 ms. So:

    emulator thread   free-runs at full frame rate, holds whatever buttons the
                      current action says, decodes state, publishes a snapshot
    brain thread      wakes at whatever rate laya actually manages, decides, and
                      *holds* that decision until it is replaced

Nobody ever waits on anybody. If the brain stalls for 400 ms the emulator still runs
at full speed and simply repeats the last action -- which is exactly how held
controller input behaves, and NES games are built to be played that way.

A third source of buttons sits on top of both: manual override from the dashboard,
so you can grab the controller mid-run.

Everything game-specific is injected through a `GameProfile`, so this loop drives
any NES ROM that a profile can describe -- it holds no palette, no RAM map and no
button assumptions of its own.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from typing import Optional

import numpy as np

from .brain import LayaBrain
from .config import (
    DECISION_LOG,
    DEFAULT_CONFIDENCE_FLOOR,
    DEFAULT_DECISION_HZ,
    DEFAULT_MODEL,
    DEFAULT_SPEED,
    LOG_DIR,
    SCREEN_H,
    SCREEN_W,
)
from .emulator import FramePacer, NesRunner, frame_to_rgb_bytes
from .reflex import ReflexPolicy
from .scene import build_state
from .state import Decoder, GameProfile, GameState

# How long the player may be off screen before we assume the game is waiting for a
# START press, and how that press is delivered.
#
# Two bugs lived here, both found by running the real ROM rather than by reading code.
#
# 1. This used to be a one-shot budget, `frame < 900`. It works right up until it
#    doesn't: once the counter passes, START is never pressed again, so a run that
#    later lands on a title card stays there forever. Observed at frame 8000 with the
#    world-1-1 HUD, an enemy and the ground all on screen and the title card drawn
#    over the top of them.
#
# 2. Holding START continuously is not a press. Title screens use edge detection,
#    and the title -> "1 PLAYER GAME" -> world-card sequence needs two separate
#    presses. Holding the button gave exactly one edge: measured over 45 seconds
#    and 2400 frames with START down the whole time, the player never appeared
#    once. It has to be pulsed.
#
# "The player has not been visible for N frames" is the actual condition, and it
# holds at boot, after a death and after a game over without any special-casing.
START_MASH_AFTER_FRAMES = 24
START_PULSE_PERIOD = 24   # frames between rising edges
START_PULSE_ON = 6         # frames the button is actually down


class GameRunner:
    def __init__(
        self,
        rom_path: str,
        policy: str = "hybrid",
        model: str = DEFAULT_MODEL,
        decision_hz: float = DEFAULT_DECISION_HZ,
        speed: float = DEFAULT_SPEED,
        confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR,
        jump_hold: int | None = None,
        brain: Optional[LayaBrain] = None,
        profile: GameProfile | None = None,
    ):
        if policy not in ("laya", "reflex", "hybrid"):
            raise ValueError("policy must be laya, reflex or hybrid")
        self.profile = profile or GameProfile.load("generic")
        self.policy = policy
        self.model = model
        self.decision_hz = max(decision_hz, 0.05)
        self.speed = speed
        self.confidence_floor = confidence_floor
        self.jump_hold = jump_hold or self.profile.jump_hold_frames

        self.runner = NesRunner(rom_path)
        self.decoder = Decoder(self.profile)
        self.reflex = ReflexPolicy(self.profile, jump_hold=self.jump_hold)
        self.brain = brain

        self._lock = threading.Lock()
        self._state: Optional[GameState] = None
        self._frame_bytes: Optional[bytes] = None
        self._base_buttons: tuple = ()
        self._jump_until_frame = -1
        self._manual: Optional[tuple] = None
        self._last_player_frame = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._brain_thread: Optional[threading.Thread] = None
        # Owned by the emulator thread but swapped out by set_speed(), which the
        # emulator thread adopts on its next lap.
        self._pacer = FramePacer(speed=speed)

        self.trace: deque = deque(maxlen=200)
        self.stats = {
            "frames": 0,
            "decisions": 0,
            "laya_decisions": 0,
            "reflex_decisions": 0,
            "low_confidence": 0,
            "errors": 0,
            "decision_hz_actual": 0.0,
            "started_at": None,
            "policy": policy,
            "model": model,
            "profile": self.profile.id,
            "rom": str(rom_path),
        }

        # Prime a frame so the dashboard has something to show immediately.
        frame = self.runner.screen
        self._frame_bytes = frame_to_rgb_bytes(frame)
        self._state = self.decoder.decode(self.runner.ram, frame, 0)
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self.stats["started_at"] = time.time()
        self._thread = threading.Thread(target=self._emulator_loop, name="nes", daemon=True)
        self._thread.start()
        # The decision thread runs for every policy, not just the model-backed ones:
        # under `reflex` it simply calls rules and skips laya entirely. Starting it
        # only for laya/hybrid would leave the reflex policy with nobody holding the
        # buttons, and the game would sit still while the emulator happily advanced.
        self._ensure_brain_thread()

    def _ensure_brain_thread(self) -> None:
        if self.policy in ("laya", "hybrid") and self.brain is None:
            self.brain = LayaBrain(model=self.model, profile=self.profile)
        if self._brain_thread is not None:
            return
        self._brain_thread = threading.Thread(target=self._brain_loop, name="brain", daemon=True)
        self._brain_thread.start()

    def stop(self) -> None:
        self._running = False
        for thread in (self._thread, self._brain_thread):
            if thread is not None:
                thread.join(timeout=3.0)
        self._thread = None
        self._brain_thread = None

    def set_policy(self, policy: str) -> None:
        if policy not in ("laya", "reflex", "hybrid"):
            raise ValueError("policy must be laya, reflex or hybrid")
        if policy in ("laya", "hybrid"):
            # Build the brain *before* committing the policy change, so a missing laya
            # install leaves the runner on its current policy instead of half-switched.
            if self.brain is None:
                self.brain = LayaBrain(model=self.model, profile=self.profile)
        self.policy = policy
        self.stats["policy"] = policy
        if self._running:
            self._ensure_brain_thread()

    def set_manual(self, buttons) -> None:
        """Pass None to hand control back to the agent."""
        with self._lock:
            self._manual = None if buttons is None else tuple(buttons)

    def set_confidence_floor(self, value: float) -> None:
        self.confidence_floor = max(0.0, min(1.0, float(value)))

    def set_speed(self, value: float) -> None:
        """Change emulator pacing live. The pacer is owned by the emulator thread,
        so this only records the target; that thread picks it up on its next lap."""
        with self._lock:
            self.speed = max(0.05, float(value))
            self._pacer = FramePacer(speed=self.speed)

    def reset(self) -> None:
        with self._lock:
            self._manual = None
            self._base_buttons = ()
            self._jump_until_frame = -1
            self._last_player_frame = 0
            self.reflex = ReflexPolicy(self.profile, jump_hold=self.jump_hold)
        self.decoder.reset()
        self.runner.reset()
        self.trace.clear()

    # ------------------------------------------------------------------ reads
    @property
    def state(self) -> Optional[GameState]:
        with self._lock:
            return self._state

    @property
    def frame_bytes(self) -> Optional[bytes]:
        with self._lock:
            return self._frame_bytes

    def snapshot(self) -> dict:
        with self._lock:
            st = self._state
            return {
                "state": st.to_dict() if st else None,
                "observation": build_state(st, self.profile) if st else None,
                "buttons": sorted(self._current_buttons_locked(st)),
                "manual": sorted(self._manual) if self._manual else None,
                "stats": dict(self.stats),
                "profile": {
                    "id": self.profile.id,
                    "name": self.profile.name,
                    "goal": self.profile.goal,
                    "actions": self.profile.actions,
                    "jump_buttons": list(self.profile.jump_buttons),
                    "has_ram": self.profile.ram is not None,
                },
            }

    def _current_buttons_locked(self, st: Optional[GameState]) -> tuple:
        if self._manual is not None:
            return self._manual
        if st is not None and self._needs_start(st):
            # Pulsed, not held: see START_PULSE_PERIOD. A held button produces one
            # edge and the game sits there waiting for the next one forever.
            if (st.frame % START_PULSE_PERIOD) < START_PULSE_ON:
                return ("start",)
            return ()
        buttons = list(self._base_buttons)
        if st is not None and self._jump_until_frame > st.frame:
            for name in self.profile.jump_buttons:
                if name not in buttons:
                    buttons.append(name)
        return tuple(buttons)

    def _needs_start(self, st: GameState) -> bool:
        """Is the game sitting on a screen that only a START press will leave?

        Title screens, level cards, death animations and game-over screens all
        share one property: the player sprite is absent from the playfield. So
        "no player visible" is the trigger, and it needs no per-game knowledge.
        START does nothing during ordinary play, so mashing it when this is
        wrongly true is harmless -- the right trade against getting stuck.
        """
        if st.character_visible:
            self._last_player_frame = st.frame
            return False
        return (st.frame - self._last_player_frame) > START_MASH_AFTER_FRAMES

    # ------------------------------------------------------------------ threads
    def _emulator_loop(self) -> None:
        pacer = self._pacer
        pacer.reset()
        while self._running:
            with self._lock:
                if pacer is not self._pacer:
                    # set_speed() swapped in a new pacer; adopt it and resync.
                    pacer = self._pacer
                    pacer.reset()
                st = self._state
                buttons = self._current_buttons_locked(st)
            try:
                self.runner.step(buttons)
            except Exception:  # noqa: BLE001 - a bad frame must not kill the thread
                self.stats["errors"] += 1
                time.sleep(0.1)
                continue

            frame = self.runner.screen
            index = self.runner.frame_index
            new_state = self.decoder.decode(self.runner.ram, frame, index)
            payload = frame_to_rgb_bytes(frame)

            with self._lock:
                self._state = new_state
                self._frame_bytes = payload
                self.stats["frames"] = index
            pacer.wait()

    def _brain_loop(self) -> None:
        interval = 1.0 / self.decision_hz
        last_decision_at = None
        while self._running:
            started = time.perf_counter()
            with self._lock:
                st = self._state
            if st is None:
                time.sleep(0.05)
                continue

            observation = build_state(st, self.profile)
            decision, source = self._decide(st, observation)

            if decision.get("ok") and decision.get("move", {}).get("buttons") is not None:
                with self._lock:
                    self._base_buttons = tuple(decision["move"]["buttons"])
                    if decision.get("jump", {}).get("thresholded"):
                        self._jump_until_frame = st.frame + self.jump_hold
            self._record(st, observation, decision, source)

            now = time.perf_counter()
            if last_decision_at is not None:
                gap = now - last_decision_at
                if gap > 0:
                    # Exponential moving average: the number worth watching, because
                    # it is what tells you whether the brain can keep up at all.
                    rate = 1.0 / gap
                    prev = self.stats["decision_hz_actual"]
                    self.stats["decision_hz_actual"] = rate if prev <= 0 else 0.7 * prev + 0.3 * rate
            last_decision_at = now

            elapsed = time.perf_counter() - started
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def _decide(self, st: GameState, observation: dict):
        """Returns (decision, source) where source is 'laya' or 'reflex'."""
        if self.policy == "reflex":
            return self.reflex.act(st), "reflex"

        decision = self.brain.decide(observation)

        # Low confidence is laya's own signal that it is guessing, so we do not.
        if not decision.get("ok") or decision.get("confidence", 0.0) < self.confidence_floor:
            if self.policy == "hybrid":
                reflex = self.reflex.act(st)
                reflex["deferred_reason"] = (
                    "answer_confidence %.3f < floor %.3f"
                    % (decision.get("confidence", 0.0), self.confidence_floor)
                    if decision.get("ok")
                    else decision.get("error", "decision failed")
                )
                return reflex, "reflex"
            return decision, "laya"
        return decision, "laya"

    # ------------------------------------------------------------------ logging
    def _record(self, st: SmbState, observation: dict, decision: dict, source: str) -> None:
        self.stats["decisions"] += 1
        if source == "laya":
            self.stats["laya_decisions"] += 1
        else:
            self.stats["reflex_decisions"] += 1
        if not decision.get("ok"):
            self.stats["errors"] += 1

        record = {
            "t": round(time.time(), 3),
            "frame": st.frame,
            "source": source,
            "observation": observation,
            "decision": _jsonable(decision),
        }
        self.trace.append(record)
        try:
            with DECISION_LOG.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass  # logging must never break the game


def _jsonable(obj):
    """Coerce numpy scalars and tuples so json.dumps does not choke."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


__all__ = ["GameRunner", "SCREEN_W", "SCREEN_H"]
