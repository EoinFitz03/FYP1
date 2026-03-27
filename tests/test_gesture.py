"""
tests/test_gesture.py

Unit tests for gesture classification and debounce logic.
No camera or MediaPipe runtime required — landmarks are mocked directly.
"""

import sys
import os
import time
import pytest
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Path setup — allow imports from HandGestures/ without installing the package
# ---------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "HandGestures"))

from gestures_live import (
    Gesture,
    DebounceState,
    classify_gesture,
    _is_extended_y,
    _thumb_up_basic,
)


# ---------------------------------------------------------------------------
# Helpers — build fake MediaPipe landmark objects
# ---------------------------------------------------------------------------

def _lm(x=0.5, y=0.5, z=0.0):
    """Create a single fake landmark with x, y, z attributes."""
    lm = MagicMock()
    lm.x = x
    lm.y = y
    lm.z = z
    return lm


def _make_landmarks(overrides: dict) -> list:
    """
    Build a 21-landmark list (all defaulting to (0.5, 0.5, 0.0)).
    Pass a dict of {index: (x, y, z)} to override specific landmarks.
    """
    lms = [_lm() for _ in range(21)]
    for idx, (x, y, z) in overrides.items():
        lms[idx].x = x
        lms[idx].y = y
        lms[idx].z = z
    return lms


def _make_hand(overrides: dict):
    """Wrap a landmark list in a mock hand_landmarks object."""
    hand = MagicMock()
    hand.landmark = _make_landmarks(overrides)
    return hand


# ---------------------------------------------------------------------------
# MediaPipe landmark index constants (mirrors gestures_live.py)
# ---------------------------------------------------------------------------
WRIST        = 0
THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
INDEX_TIP,  INDEX_PIP,  INDEX_MCP  = 8,  6,  5
MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP = 12, 10, 9
RING_TIP,   RING_PIP,   RING_MCP   = 16, 14, 13
PINKY_TIP,  PINKY_PIP,  PINKY_MCP  = 20, 18, 17


# ===========================================================================
# _is_extended_y
# ===========================================================================

class TestIsExtendedY:

    def test_finger_extended_when_tip_above_pip(self):
        # tip y=0.2 (higher up screen), pip y=0.5 → extended
        lms = _make_landmarks({INDEX_TIP: (0.5, 0.2, 0.0), INDEX_PIP: (0.5, 0.5, 0.0)})
        assert _is_extended_y(lms, INDEX_TIP, INDEX_PIP) is True

    def test_finger_folded_when_tip_below_pip(self):
        # tip y=0.8 (lower on screen), pip y=0.5 → folded
        lms = _make_landmarks({INDEX_TIP: (0.5, 0.8, 0.0), INDEX_PIP: (0.5, 0.5, 0.0)})
        assert _is_extended_y(lms, INDEX_TIP, INDEX_PIP) is False

    def test_finger_folded_when_tip_equal_pip(self):
        lms = _make_landmarks({INDEX_TIP: (0.5, 0.5, 0.0), INDEX_PIP: (0.5, 0.5, 0.0)})
        assert _is_extended_y(lms, INDEX_TIP, INDEX_PIP) is False


# ===========================================================================
# classify_gesture — OPEN_PALM
# ===========================================================================

class TestClassifyGestureOpenPalm:

    def _open_palm_hand(self):
        """All 4 fingers extended (tips well above pips)."""
        return _make_hand({
            INDEX_TIP:  (0.4, 0.1, 0.0), INDEX_PIP:  (0.4, 0.5, 0.0),
            MIDDLE_TIP: (0.5, 0.1, 0.0), MIDDLE_PIP: (0.5, 0.5, 0.0),
            RING_TIP:   (0.6, 0.1, 0.0), RING_PIP:   (0.6, 0.5, 0.0),
            PINKY_TIP:  (0.7, 0.1, 0.0), PINKY_PIP:  (0.7, 0.5, 0.0),
        })

    def test_open_palm_returns_correct_gesture(self):
        assert classify_gesture(self._open_palm_hand()) == Gesture.OPEN_PALM

    def test_open_palm_is_not_unknown(self):
        assert classify_gesture(self._open_palm_hand()) != Gesture.UNKNOWN

    def test_open_palm_is_not_thumbs_up(self):
        assert classify_gesture(self._open_palm_hand()) != Gesture.THUMBS_UP


# ===========================================================================
# classify_gesture — FIST
# ===========================================================================

class TestClassifyGestureFist:

    def _fist_hand(self):
        """
        All 4 fingers folded (tips below pips).
        Thumb also NOT pointing up (tip below its own joints).
        Hand size set so thumb_extended check fails.
        """
        return _make_hand({
            # Wrist low on screen
            WRIST:      (0.5, 0.9, 0.0),
            # Thumb tip below its MCP (not extended, not pointing up)
            THUMB_TIP:  (0.3, 0.8, 0.0),
            THUMB_IP:   (0.3, 0.7, 0.0),
            THUMB_MCP:  (0.3, 0.6, 0.0),
            # Middle MCP for hand-size reference
            MIDDLE_MCP: (0.5, 0.7, 0.0),
            # All finger tips below their pips
            INDEX_TIP:  (0.4, 0.75, 0.0), INDEX_PIP:  (0.4, 0.6, 0.0),
            MIDDLE_TIP: (0.5, 0.75, 0.0), MIDDLE_PIP: (0.5, 0.6, 0.0),
            RING_TIP:   (0.6, 0.75, 0.0), RING_PIP:   (0.6, 0.6, 0.0),
            PINKY_TIP:  (0.7, 0.75, 0.0), PINKY_PIP:  (0.7, 0.6, 0.0),
        })

    def test_fist_returns_correct_gesture(self):
        assert classify_gesture(self._fist_hand()) == Gesture.FIST

    def test_fist_is_not_open_palm(self):
        assert classify_gesture(self._fist_hand()) != Gesture.OPEN_PALM


# ===========================================================================
# classify_gesture — THUMBS_UP
# ===========================================================================

class TestClassifyGestureThumbsUp:

    def _thumbs_up_hand(self):
        """
        Thumb extended and pointing clearly upward.
        Other fingers folded. Hand size large enough for threshold.
        """
        return _make_hand({
            WRIST:      (0.5, 0.9, 0.0),
            MIDDLE_MCP: (0.5, 0.7, 0.0),   # hand_size ref ≈ 0.2 units
            # Thumb — tip well above its joints
            THUMB_TIP:  (0.3, 0.3, 0.0),   # high on screen (small y)
            THUMB_IP:   (0.3, 0.5, 0.0),
            THUMB_MCP:  (0.3, 0.65, 0.0),
            # All other fingers folded (tips below pips)
            INDEX_TIP:  (0.45, 0.85, 0.0), INDEX_PIP:  (0.45, 0.72, 0.0),
            MIDDLE_TIP: (0.50, 0.85, 0.0), MIDDLE_PIP: (0.50, 0.72, 0.0),
            RING_TIP:   (0.55, 0.85, 0.0), RING_PIP:   (0.55, 0.72, 0.0),
            PINKY_TIP:  (0.60, 0.85, 0.0), PINKY_PIP:  (0.60, 0.72, 0.0),
            # MCP refs so folded_by_dist check passes
            INDEX_MCP:  (0.45, 0.65, 0.0),
            MIDDLE_MCP: (0.50, 0.65, 0.0),
            RING_MCP:   (0.55, 0.65, 0.0),
            PINKY_MCP:  (0.60, 0.65, 0.0),
        })

    def test_thumbs_up_returns_correct_gesture(self):
        assert classify_gesture(self._thumbs_up_hand()) == Gesture.THUMBS_UP

    def test_thumbs_up_is_not_open_palm(self):
        assert classify_gesture(self._thumbs_up_hand()) != Gesture.OPEN_PALM

    def test_thumbs_up_is_not_fist(self):
        assert classify_gesture(self._thumbs_up_hand()) != Gesture.FIST


# ===========================================================================
# DebounceState
# ===========================================================================

class TestDebounceState:

    def test_unknown_before_window_fills(self):
        state = DebounceState(window_size=8, cooldown_seconds=0.0)
        for _ in range(7):
            state.update(Gesture.OPEN_PALM)
        assert state.stable_gesture() == Gesture.UNKNOWN

    def test_stable_after_window_fills_with_same_gesture(self):
        state = DebounceState(window_size=8, cooldown_seconds=0.0)
        for _ in range(8):
            state.update(Gesture.OPEN_PALM)
        assert state.stable_gesture() == Gesture.OPEN_PALM

    def test_unknown_when_window_contains_mixed_gestures(self):
        state = DebounceState(window_size=8, cooldown_seconds=0.0)
        for _ in range(4):
            state.update(Gesture.OPEN_PALM)
        for _ in range(4):
            state.update(Gesture.THUMBS_UP)
        assert state.stable_gesture() == Gesture.UNKNOWN

    def test_unknown_gesture_in_window_returns_unknown(self):
        state = DebounceState(window_size=8, cooldown_seconds=0.0)
        for _ in range(8):
            state.update(Gesture.UNKNOWN)
        assert state.stable_gesture() == Gesture.UNKNOWN

    def test_can_fire_initially(self):
        state = DebounceState(window_size=8, cooldown_seconds=1.0)
        assert state.can_fire() is True

    def test_cannot_fire_immediately_after_mark_fired(self):
        state = DebounceState(window_size=8, cooldown_seconds=1.0)
        state.mark_fired()
        assert state.can_fire() is False

    def test_can_fire_after_cooldown_expires(self):
        state = DebounceState(window_size=8, cooldown_seconds=0.05)
        state.mark_fired()
        time.sleep(0.1)
        assert state.can_fire() is True

    def test_window_is_rolling_old_values_drop_off(self):
        """Filling window with UNKNOWN then overwriting with OPEN_PALM should stabilise."""
        state = DebounceState(window_size=4, cooldown_seconds=0.0)
        for _ in range(4):
            state.update(Gesture.UNKNOWN)
        # Now overwrite entire window with OPEN_PALM
        for _ in range(4):
            state.update(Gesture.OPEN_PALM)
        assert state.stable_gesture() == Gesture.OPEN_PALM

    def test_stable_gesture_for_thumbs_up(self):
        state = DebounceState(window_size=8, cooldown_seconds=0.0)
        for _ in range(8):
            state.update(Gesture.THUMBS_UP)
        assert state.stable_gesture() == Gesture.THUMBS_UP

    def test_stable_gesture_for_fist(self):
        state = DebounceState(window_size=8, cooldown_seconds=0.0)
        for _ in range(8):
            state.update(Gesture.FIST)
        assert state.stable_gesture() == Gesture.FIST