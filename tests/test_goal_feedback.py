"""Goal coordinates must preserve yaw feedback without adding robot position."""
import math

import pytest
import torch

from supermimic.goal_spec import feedback_dim, goal_dim
from supermimic.world_model.heads import anchor_orientation_obs


def test_orientation_is_reference_rotation_in_robot_frame_with_live_gradient():
    yaw = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    zero = torch.zeros_like(yaw)
    robot = torch.stack((torch.cos(yaw / 2), zero, zero, torch.sin(yaw / 2)))
    reference = torch.tensor([math.cos(0.5 / 2), 0, 0, math.sin(0.5 / 2)], dtype=torch.float64)
    actual = anchor_orientation_obs(robot, reference)
    expected = torch.tensor([math.cos(0.3), -math.sin(0.3), math.sin(0.3), math.cos(0.3), 0, 0], dtype=torch.float64)
    torch.testing.assert_close(actual, expected)
    derivative, = torch.autograd.grad(actual[0], yaw)
    assert derivative.item() == pytest.approx(math.sin(0.3))
    assert goal_dim(10, anchor_orientation=True) == 686
    assert goal_dim(10) == 680
    assert goal_dim(10, anchor_obs=True) == 689
    with pytest.raises(ValueError, match="not both"):
        feedback_dim(True, True)


@pytest.mark.parametrize("orientation", [False, True])
def test_experiment_changes_only_goal_feedback(orientation):
    from dataclasses import asdict
    from supermimic.experiment import Experiment

    baseline = asdict(Experiment().config())
    variant = asdict(Experiment(orientation_goal=orientation).config())
    assert variant["env"].pop("anchor_orientation") == orientation
    baseline["env"].pop("anchor_orientation")
    assert variant["policy"].pop("goal_dim") == 680 + 6 * orientation
    baseline["policy"].pop("goal_dim")
    assert baseline == variant


def test_conflicting_experiment_feedback_fails_before_startup():
    from supermimic.experiment import Experiment

    with pytest.raises(ValueError, match="not both"):
        Experiment(anchor_goal=True, orientation_goal=True).config()
