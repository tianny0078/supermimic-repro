"""A torso error must not be substituted for the paper's pelvis root error."""
import pytest
import torch

from supermimic.envs.mdp.rewards import TrackingRewardCfg, tracking_reward_terms


def test_global_root_reward_uses_pelvis_with_torso_body_anchor():
    reference_pos = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.5]]])
    robot_pos = reference_pos.clone()
    robot_pos[:, 1, 0] = 0.3  # Torso displaced, pelvis unchanged.
    quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])
    zero_vel = torch.zeros_like(robot_pos)
    zero_joint = torch.zeros(1, 29)
    limits = torch.tensor([[-1.0, 1.0]]).repeat(29, 1)
    terms = tracking_reward_terms(
        robot_body_pos_w=robot_pos, robot_body_quat_w=quat,
        robot_body_lin_vel_w=zero_vel, robot_body_ang_vel_w=zero_vel,
        ref_body_pos_w=reference_pos, ref_body_quat_w=quat,
        ref_body_lin_vel_w=zero_vel, ref_body_ang_vel_w=zero_vel,
        anchor_index=1, root_index=0, action=zero_joint, prev_action=zero_joint,
        joint_pos=zero_joint, soft_limits=limits, cfg=TrackingRewardCfg())
    assert terms["motion_global_root_pos"].item() == pytest.approx(0.01)
