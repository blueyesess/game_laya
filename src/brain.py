"""laya as the player's reflex layer, for any NES game described by a profile.

The schema is the interesting part. Three typed questions in one forward pass:

* ``move``  -- `choice`, a closed set. The whole point of a non-
  autoregressive head is that scoring a handful of options costs nothing extra,
  so the alternatives are enumerated rather than described.
* ``jump``  -- `noul`, a calibrated P(true) that we threshold ourselves. Better
  than a choice here because "should I jump" is genuinely a continuum: P=0.55 and
  P=0.95 mean the same answer but very different confidence.
* ``risk``  -- `score`, an ordinal rubric. Its own README calls `score` the
  weakest primitive, so this one is advisory: it modulates the jump gate rather
  than driving the controls directly.

The question wording is generated from the game profile, so this module needs no
editing to drive a different ROM: `move`'s options and their criteria come from
`profile.move`, and the goal sentence comes from `profile.goal`.

Two honest caveats about running this at 60 fps:

1. A 322M encoder forward pass costs ~145 ms for three questions, so laya decides at
   single-digit Hz, not 60. The emulator free-runs and the brain *holds* its last
   action between decisions. That matches how a person plays, and NES games are
   forgiving of held input.
2. `answer_confidence` is the field to gate on (laya's own example 18). When it
   drops below the floor the hybrid policy defers to `reflex.py` instead of
   trusting a decision the model itself flagged as unsure.
"""

from __future__ import annotations

import time
from typing import Any, Dict

from .state import GameProfile

# Two competing pressures, both measured rather than guessed:
#
# * laya encodes the state once *per question*, so a long observation is paid for
#   three times. Trimming the state from 926 input tokens to 368 took the call from
#   5068 ms to 2693 ms on CPU. That trim is free -- the prose was restating a struct.
# * laya's own examples say the criteria text *is* the prompt. Cutting it to bare
#   phrases was not free: `move` came back near-uniform (0.38 / 0.31 / 0.31 instead of
#   0.88 / 0.06 / 0.07), because the model no longer had a basis for telling the
#   options apart. So the criteria are kept descriptive and only the state was trimmed.
#
# tools/bench_brain.py measures latency, input tokens and answer_confidence together.
# Re-run it after editing either side of this trade.

# Criteria wording per move option. The profile supplies the option *names*
# (which are also the button names); this supplies the reasoning behind each.
MOVE_CRITERIA = {
    "progress": "the way ahead is clear or only lightly blocked; advancing makes progress toward the goal",
    "back": "advancing now would walk straight into a hazard or off the edge of a platform; back away",
    "wait": "standing still is safer than moving in either direction at this instant",
}


def questions_for(profile: GameProfile) -> Dict[str, dict]:
    """Build the laya question set for one game profile.

    `move`'s options are the profile's own move vocabulary, so a game whose goal
    is vertical (climb up) can declare `{"up": ..., "down": ..., "wait": ...}`
    and get a schema that matches its controls instead of a hard-coded
    left/right/wait that would be nonsense.
    """
    move_options = list(profile.move.keys()) + ["wait"]
    criteria = {}
    for option in move_options:
        criteria[option] = MOVE_CRITERIA.get(
            option, "pressing %s serves the goal: %s" % (option, profile.goal)
        )
    # "wait" is universal: every game has a moment where doing nothing is safest.
    criteria.setdefault("wait", MOVE_CRITERIA["wait"])

    progress_phrase = " or ".join(profile.progress_buttons)
    return {
        "move": {
            "type": "choice",
            "instructions": (
                "Which way should the player move right now, to survive and make progress "
                "toward this goal: %s" % profile.goal
            ),
            "criteria": criteria,
        },
        "jump": {
            "type": "noul",
            "instructions": (
                "Does the player need to jump on this frame? Answer true if a hazard is "
                "close in front, the floor just ahead is missing, or a block must be "
                "cleared. Answer false if the way ahead is open ground."
            ),
        },
        "risk": {
            "type": "score",
            "instructions": "How dangerous is the player's immediate situation?",
            "criteria": [
                "safe: nothing threatening is within reach",
                "a hazard is close and one mistake would be costly",
                "a hazard is unavoidable right now and the player is about to be hit or to fall",
            ],
        },
    }


# The generic profile's schema, kept as a module-level constant so tools that
# only want *a* question set (bench_brain.py, /api/questions) do not have to
# build a profile first.
QUESTIONS: Dict[str, dict] = questions_for(GameProfile.load("generic"))


def move_to_buttons(label: str | None, profile: GameProfile) -> tuple:
    """Map a move label back onto the profile's actual NES buttons."""
    if not label:
        return ()
    if label == "wait":
        return ()
    buttons = profile.move.get(label)
    if buttons is None:
        return ()
    return tuple(buttons)


RISK_LABELS = {0: "safe", 1: "hazard_close", 2: "unavoidable"}


class LayaBrain:
    """One loaded checkpoint behind a Router, so hot-swapping models is a config change."""

    def __init__(self, model: str = "multilingual", device: str | None = None,
                 profile: GameProfile | None = None):
        try:
            from laya import Router
        except ImportError as exc:  # pragma: no cover - environment problem
            raise RuntimeError(
                "the laya package is not installed. Install the brain extra:\n"
                '    .venv\\Scripts\\python -m pip install --proxy http://127.0.0.1:57417 -e ".[brain]"\n'
                "torch must already come from the CPU index (PyPI's Windows build is CUDA, "
                "about 2.5 GB); see the header of pyproject.toml."
            ) from exc

        self.model = model
        self.device = device
        self.profile = profile or GameProfile.load("generic")
        self.questions = questions_for(self.profile)
        # Router builds checkpoints lazily, so constructing one is free. The first
        # decide() call pays the cold build (~30-60 s on CPU) and the first run
        # downloads the weights into HF_HOME.
        self.router = Router()
        self.calls = 0
        self.last_error: str | None = None

    # ------------------------------------------------------------------ warmup
    def warmup(self, questions: dict | None = None) -> dict:
        """Load the checkpoint and run one throwaway decision.

        Worth doing explicitly: otherwise the first few decisions are spent
        downloading weights instead of playing.
        """
        return self.decide({"scene": "warmup"}, questions)

    # ------------------------------------------------------------------ decide
    def decide(self, state: Any, questions: dict | None = None) -> dict:
        questions = questions or self.questions
        t0 = time.perf_counter()
        try:
            result = self.router.predict(state, questions, model=self.model)
        except TypeError:
            # Older/simpler builds may not accept model= on predict(); the Router
            # then routes on its own. Fall back rather than fail the whole run.
            result = self.router.predict(state, questions)
        except Exception as exc:  # noqa: BLE001 - a decision failure must not kill the game
            self.last_error = "%s: %s" % (type(exc).__name__, exc)
            return self._failed(self.last_error)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.calls += 1
        self.last_error = None
        return self._normalise(result, latency_ms, self.profile)

    # ------------------------------------------------------------------ parsing
    @staticmethod
    def _answers(result: dict) -> dict:
        return result.get("answers", {}) or {}

    @classmethod
    def _normalise(cls, result: dict, latency_ms: float, profile: GameProfile) -> dict:
        answers = cls._answers(result)

        move = answers.get("move") or {}
        move_label = move.get("choice")
        move_probs = dict(move.get("probabilities") or {})

        jump = answers.get("jump") or {}
        jump_p = float(jump.get("noul", 0.0)) if "noul" in jump else 0.0

        risk = answers.get("risk") or {}
        legend = risk.get("legend") or {}
        probs = risk.get("probabilities") or {}
        risk_score = float(risk.get("score", 0.0)) if "score" in risk else 0.0
        # The argmax bucket is a better label than the expected value, which can
        # land between two levels on an ordinal rubric.
        risk_index = -1
        risk_p = 0.0
        for key, value in probs.items():
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if v > risk_p:
                risk_p = v
                try:
                    risk_index = int(key)
                except (TypeError, ValueError):
                    risk_index = -1
        risk_label = RISK_LABELS.get(risk_index, str(legend.get(str(risk_index), "?")))

        confidences = [
            float(a.get("answer_confidence", 0.0))
            for a in (move, jump, risk)
            if isinstance(a, dict) and "answer_confidence" in a
        ]
        confidence = min(confidences) if confidences else 0.0

        routing = result.get("routing") or {}
        usage = result.get("usage") or {}

        return {
            "ok": True,
            "move": {
                "label": move_label,
                "buttons": move_to_buttons(move_label, profile),
                "probs": move_probs,
                "confidence": float(move.get("answer_confidence", 0.0)),
            },
            "jump": {
                "p_true": jump_p,
                "thresholded": jump_p > 0.5,
                "confidence": float(jump.get("answer_confidence", 0.0)),
            },
            "risk": {
                "label": risk_label,
                "index": risk_index,
                "expected": risk_score,
                "legend": {str(k): v for k, v in legend.items()},
                "probs": {str(k): float(v) for k, v in probs.items()},
                "confidence": float(risk.get("answer_confidence", 0.0)),
            },
            "confidence": confidence,
            "latency_ms": latency_ms,
            "model": routing.get("model") or result.get("model") or "unknown",
            "usage": {
                "input_tokens": usage.get("input_tokens"),
                "truncated": usage.get("truncated"),
                "truncated_questions": usage.get("truncated_questions"),
            },
        }

    @staticmethod
    def _failed(message: str) -> dict:
        return {
            "ok": False,
            "error": message,
            "move": {"label": None, "buttons": (), "probs": {}, "confidence": 0.0},
            "jump": {"p_true": 0.0, "thresholded": False, "confidence": 0.0},
            "risk": {
                "label": "unknown",
                "index": -1,
                "expected": 0.0,
                "legend": {},
                "probs": {},
                "confidence": 0.0,
            },
            "confidence": 0.0,
            "latency_ms": 0.0,
            "model": "error",
            "usage": {},
        }


def decision_to_buttons(decision: dict, jump_hold: int, profile: GameProfile) -> tuple:
    """Fold a decision into an NES button set.

    `jump_hold` is how many frames the jump button stays down. Most NES jump
    arcs are fixed-height, so a single-frame tap barely leaves the ground; the
    button must be held. `profile.jump_buttons` names which button that is (A
    for most games).
    """
    buttons = list(decision.get("move", {}).get("buttons") or ())
    if decision.get("jump", {}).get("thresholded"):
        for name in profile.jump_buttons:
            if name not in buttons:
                buttons.append(name)
    return tuple(buttons)
