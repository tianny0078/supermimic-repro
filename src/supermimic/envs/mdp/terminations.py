"""Paper terminations: root (pelvis) height deviation and tilt; clip end as time-out."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from supermimic.envs.mdp.commands import MultiClipMotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def synchronize_state(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Lifecycle bridge: refresh final-step FK before reward/termination evaluation.

    Registered first; always false. MuJoCo Warp's step leaves derived quantities
    one substep behind qpos/qvel, and relative reward targets also need the new
    robot anchor. This keeps logged transitions and analytic heads synchronized.
    """
    env.sim.forward()
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    cmd.update_relative_body_poses()
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def bad_root_height(env: ManagerBasedRlEnv, command_name: str, threshold: float = 0.25) -> torch.Tensor:
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return (cmd.root_pos_w[:, 2] - cmd.robot_root_pos_w[:, 2]).abs() > threshold


def bad_root_tilt(env: ManagerBasedRlEnv, command_name: str, threshold: float = 0.8) -> torch.Tensor:
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    g = cmd.robot.data.gravity_vec_w
    ref = quat_apply_inverse(cmd.root_quat_w, g)[:, 2]
    rob = quat_apply_inverse(cmd.robot_root_quat_w, g)[:, 2]
    return (ref - rob).abs() > threshold


def clip_end(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    # cmd.frame is the target reached by this just-completed control interval.
    # Waiting for frame > last would execute the final target a second time.
    return cmd.frame >= cmd.library.last_frame_of(cmd.frame)
