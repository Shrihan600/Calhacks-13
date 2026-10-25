"""Settings for gesture detection and gesture → action bindings (``gestures:`` in app config)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from opengaze.gestures.keys import parse_key_combo
from opengaze.gestures.types import Gesture


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BlinkSettings(_Model):
    long_blink_ms: float = Field(default=400.0, gt=0.0)
    # A "short" blink counts toward a double/triple blink. Natural blinks are ~100–150 ms.
    short_blink_min_ms: float = Field(default=40.0, ge=0.0)
    short_blink_max_ms: float = Field(default=300.0, gt=0.0)
    # After a short blink ends, wait this long for another one before deciding the count.
    multi_blink_gap_ms: float = Field(default=450.0, gt=0.0)
    cooldown_ms: float = Field(default=800.0, ge=0.0)

    @model_validator(mode="after")
    def _ordered(self) -> BlinkSettings:
        if not self.short_blink_min_ms < self.short_blink_max_ms < self.long_blink_ms:
            raise ValueError("need short_blink_min_ms < short_blink_max_ms < long_blink_ms")
        return self


class WinkSettings(_Model):
    min_wink_ms: float = Field(default=200.0, ge=0.0)
    long_wink_ms: float = Field(default=800.0, gt=0.0)
    # Winks of the same eye count as one run (double, triple) when the eye closes again
    # within this long of reopening. The cooldown doesn't apply inside a run.
    multi_wink_gap_ms: float = Field(default=500.0, gt=0.0)
    # A held wink presses (hold_wink_* "start") once the eye has been closed this long, so a
    # twitch never presses the button.
    hold_press_ms: float = Field(default=300.0, gt=0.0)
    # During a hold the other eye may close this long (a natural blink) without ending it.
    hold_blink_tolerance_ms: float = Field(default=350.0, ge=0.0)
    cooldown_ms: float = Field(default=600.0, ge=0.0)

    @model_validator(mode="after")
    def _ordered(self) -> WinkSettings:
        if not self.min_wink_ms < self.long_wink_ms:
            raise ValueError("need min_wink_ms < long_wink_ms")
        return self


class GlanceSettings(_Model):
    # How far past the screen edge gaze must go, in normalised screen units (0 = the edge).
    edge_margin: float = Field(default=0.02, ge=-0.2, le=1.0)
    # Gaze must come back inside by this much before the zone is left (debounce).
    hysteresis: float = Field(default=0.03, ge=0.0, le=0.5)
    min_glance_ms: float = Field(default=60.0, ge=0.0)
    max_glance_ms: float = Field(default=700.0, gt=0.0)
    hold_ms: float = Field(default=900.0, gt=0.0)
    # Repeat a hold every this many ms while held (e.g. for scrolling); null = fire once.
    hold_repeat_ms: float | None = Field(default=350.0, gt=0.0)
    cooldown_ms: float = Field(default=400.0, ge=0.0)

    @model_validator(mode="after")
    def _ordered(self) -> GlanceSettings:
        if not self.min_glance_ms < self.max_glance_ms < self.hold_ms:
            raise ValueError("need min_glance_ms < max_glance_ms < hold_ms")
        return self


MouseAction = Literal[
    "click",           # repeated winks: double/triple click (see GestureEvent.count)
    "hold",            # held gestures: left button down at "start", up at "end" (drag)
    "right_click",
    "double_click",
    "middle_click",
    "scroll_up",
    "scroll_down",
    "scroll_left",
    "scroll_right",
]
AppAction = Literal[
    "pause", "toggle_cursor", "recalibrate", "recenter", "keyboard", "practice", "quit"
]


class Binding(_Model):
    """Exactly one of ``key`` (system keystroke, e.g. ``cmd+shift+tab``), ``mouse`` (system mouse
    event at the gaze position) or ``app`` (an OpenGaze action)."""

    key: str | None = None
    mouse: MouseAction | None = None
    app: AppAction | None = None

    @field_validator("key")
    @classmethod
    def _valid_key(cls, value: str | None) -> str | None:
        if value is not None:
            parse_key_combo(value)  # raises ValueError naming the bad part
        return value

    @model_validator(mode="after")
    def _exactly_one(self) -> Binding:
        if sum(v is not None for v in (self.key, self.mouse, self.app)) != 1:
            raise ValueError("a binding needs exactly one of key, mouse or app")
        return self

    def describe(self) -> str:
        if self.key is not None:
            return f"key {self.key}"
        if self.mouse is not None:
            return f"mouse {self.mouse}"
        return f"app {self.app}"


class GestureSettings(_Model):
    enabled: bool = True
    blink: BlinkSettings = BlinkSettings()
    wink: WinkSettings = WinkSettings()
    glance: GlanceSettings = GlanceSettings()
    # Unbound gestures are still detected (and shown in the UI) but do nothing.
    bindings: dict[Gesture, Binding] = Field(
        default_factory=lambda: {Gesture.DOUBLE_BLINK: Binding(mouse="click")}
    )

    @field_validator("bindings")
    @classmethod
    def _hold_needs_a_held_gesture(cls, value: dict[Gesture, Binding]) -> dict[Gesture, Binding]:
        # Only held gestures send the "end" that releases the button.
        held = {Gesture.HOLD_WINK_LEFT, Gesture.HOLD_WINK_RIGHT}
        for gesture, binding in value.items():
            if binding.mouse == "hold" and gesture not in held:
                raise ValueError(
                    f"{gesture.value}: 'mouse: hold' only works on hold_wink_left / "
                    "hold_wink_right (other gestures never release the button)"
                )
        return value
