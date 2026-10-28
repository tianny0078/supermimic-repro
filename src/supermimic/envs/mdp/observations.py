from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from supermimic.envs.mdp.commands import MultiClipMotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def goal_future_frames(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """(N, K*68) future reference-frame features (plan D5)."""
    command = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return command.goal


def goal_anchor_orientation(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Current relative torso orientation, using the same analytic head as imagination."""
    from supermimic.world_model.heads import anchor_orientation_obs

    command = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return anchor_orientation_obs(command.robot_anchor_quat_w, command.anchor_quat_w)
