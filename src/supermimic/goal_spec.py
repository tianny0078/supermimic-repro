"""Goal feedback dimensions, shared without importing the simulator or PyTorch."""
from __future__ import annotations


def feedback_dim(anchor_obs: bool = False, anchor_orientation: bool = False) -> int:
    if anchor_obs and anchor_orientation:
        raise ValueError("Choose orientation-only or full anchor pose feedback, not both")
    return 9 if anchor_obs else 6 if anchor_orientation else 0


def goal_dim(future_frames: int, anchor_obs: bool = False, anchor_orientation: bool = False) -> int:
    if future_frames < 1:
        raise ValueError("future_frames must be positive")
    return 68 * future_frames + feedback_dim(anchor_obs, anchor_orientation)
