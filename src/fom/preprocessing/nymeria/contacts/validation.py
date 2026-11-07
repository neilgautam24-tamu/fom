"""Validate hand-object contact events against articulated hand geometry."""

from __future__ import annotations

import numpy as np

from fom.preprocessing.nymeria.body.smplx_hands import NUM_HAND_JOINT_SAMPLES
from fom.preprocessing.nymeria.contacts.detection import ContactEvent

# Finger joints must be within this distance of the wrist to count as real contact.
MAX_WRIST_REACH_M = 0.18


def is_joint_contact(ev: ContactEvent) -> bool:
    """True when the contact was triggered by a wrist/finger joint, not a mesh vertex."""
    idx = int(ev.sample_idx)
    return 0 <= idx < NUM_HAND_JOINT_SAMPLES


def is_plausible_contact(
    ev: ContactEvent,
    hand_samples: np.ndarray,
    *,
    require_joint: bool = True,
    max_wrist_reach_m: float = MAX_WRIST_REACH_M,
) -> bool:
    """Reject arm/torso mesh vertices that sit near an object OBB but are not the hand."""
    sample_idx = int(ev.sample_idx)
    if sample_idx < 0 or sample_idx >= hand_samples.shape[0]:
        return False
    if require_joint and not is_joint_contact(ev):
        return False
    wrist = np.asarray(hand_samples[0], dtype=np.float64)
    sample = np.asarray(hand_samples[sample_idx], dtype=np.float64)
    return float(np.linalg.norm(sample - wrist)) <= float(max_wrist_reach_m)


def validated_contact_event(
    events: list[ContactEvent],
    hand: str,
    hand_samples: np.ndarray,
) -> ContactEvent | None:
    """Best joint-based contact for one hand on a frame, or None."""
    matches = [e for e in events if e.hand == hand and is_plausible_contact(e, hand_samples)]
    if not matches:
        return None
    return min(matches, key=lambda e: e.distance_m)
