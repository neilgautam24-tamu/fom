"""Real-time playback helpers for Nymeria body motion (~240 Hz source)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

# Nymeria / Xsens body stream nominal rate.
SOURCE_BODY_FPS = 240.0


@dataclass(frozen=True)
class MotionTiming:
    rel_seconds: np.ndarray
    dt_s: float
    loaded_fps: float
    duration_s: float


def motion_timing_from_timestamps(timestamps_us: np.ndarray) -> MotionTiming:
    """Derive per-frame spacing from device timestamps (microseconds)."""
    ts = np.asarray(timestamps_us, dtype=np.int64)
    rel = (ts - int(ts[0])) * 1e-6
    if rel.shape[0] > 1:
        dt = float(np.median(np.diff(rel)))
    else:
        dt = 1.0 / SOURCE_BODY_FPS
    loaded_fps = 1.0 / dt if dt > 0 else SOURCE_BODY_FPS
    return MotionTiming(
        rel_seconds=rel.astype(np.float64),
        dt_s=dt,
        loaded_fps=loaded_fps,
        duration_s=float(rel[-1]) if rel.shape[0] else 0.0,
    )


class SequentialPlayback:
    """Advance one loaded frame at a time using recorded inter-frame timing."""

    def __init__(self, rel_seconds: np.ndarray, *, default_dt_s: float) -> None:
        self.rel_seconds = np.asarray(rel_seconds, dtype=np.float64)
        self.default_dt_s = float(default_dt_s)
        self.speed = 1.0
        self._next_advance_wall: float | None = None

    @property
    def num_frames(self) -> int:
        return int(self.rel_seconds.shape[0])

    def frame_duration_s(self, idx: int) -> float:
        speed = max(1e-3, float(self.speed))
        if idx < self.num_frames - 1:
            dt = float(self.rel_seconds[idx + 1] - self.rel_seconds[idx])
            if dt > 0:
                return dt / speed
        return self.default_dt_s / speed

    def reset_clock(self, *, after_idx: int | None = None) -> None:
        idx = 0 if after_idx is None else int(after_idx)
        self._next_advance_wall = time.time() + self.frame_duration_s(idx)

    def sync_to_index(self, idx: int) -> None:
        self.reset_clock(after_idx=idx)

    def should_advance(self) -> bool:
        if self._next_advance_wall is None:
            self.reset_clock()
        return time.time() >= self._next_advance_wall

    def mark_advanced(self, idx: int) -> None:
        self._next_advance_wall = time.time() + self.frame_duration_s(idx)

    def sleep_until_next(self) -> None:
        if self._next_advance_wall is None:
            time.sleep(0.01)
            return
        time.sleep(max(0.0, self._next_advance_wall - time.time()))


def sync_playback_anchor(
    playback: SequentialPlayback,
    gui_speed,
    current_idx: int,
) -> None:
    playback.speed = float(gui_speed.value)
    playback.sync_to_index(current_idx)


class FrameGate:
    """Drop overlapping frame draws so Viser does not queue stale mesh updates."""

    def __init__(self) -> None:
        self._busy = threading.Lock()

    def try_begin(self) -> bool:
        return self._busy.acquire(blocking=False)

    def end(self) -> None:
        self._busy.release()


def run_playback_loop(
    gui_play,
    gui_frame,
    playback: SequentialPlayback,
    show_frame: Callable[[int], None],
    *,
    get_current_idx: Callable[[], int],
    set_current_idx: Callable[[int], None],
    playback_lock: threading.Lock,
    slider_guard: dict[str, bool] | None = None,
    frame_gate: FrameGate | None = None,
) -> None:
    """Single-threaded playback driver.

    Viser ``on_update`` handlers run in a worker thread, so playback must not
    rely on ``gui_frame.value`` as state. Use ``get_current_idx`` / ``set_current_idx``
    under ``playback_lock`` instead.
    """
    guard = slider_guard if slider_guard is not None else {"programmatic": False}
    n = playback.num_frames

    while True:
        if gui_play.value and n > 1:
            if frame_gate is not None and not frame_gate.try_begin():
                playback.sleep_until_next()
                continue
            try:
                with playback_lock:
                    if playback.should_advance():
                        idx = int(get_current_idx())
                        nxt = 0 if idx >= n - 1 else idx + 1
                        show_frame(nxt)
                        set_current_idx(nxt)
                        playback.mark_advanced(idx)
                        guard["programmatic"] = True
                        try:
                            gui_frame.value = nxt
                        finally:
                            guard["programmatic"] = False
            finally:
                if frame_gate is not None:
                    frame_gate.end()
            playback.sleep_until_next()
        else:
            playback._next_advance_wall = None
            time.sleep(0.02)
