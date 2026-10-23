"""Independent gradient and terminal-boundary checks for the paper objective."""
from types import SimpleNamespace

import pytest
import torch

from supermimic.imagination.objective import gae_targets, policy_objective


def test_bptt_matches_finite_difference_with_terminal_value():
    # Scalar dynamics x'=0.8x+a, a=theta*x. The model coefficient is
    # frozen; actor and terminal-value state gradients remain live.
    model = torch.nn.Parameter(torch.tensor(0.8, dtype=torch.float64), requires_grad=False)

    def evaluate(theta):
        state = torch.tensor([0.7], dtype=torch.float64)
        rewards = []
        for _ in range(4):
            state = model * state + theta * state
            rewards.append(-(state - 0.2).square())
        rollout = SimpleNamespace(rewards=torch.stack(rewards), survival=torch.ones(5, 1, dtype=torch.float64))
        return policy_objective(rollout, -2 * state.square(), gamma=0.9)

    theta = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    gradient, = torch.autograd.grad(evaluate(theta), theta)
    epsilon = 1e-6
    difference = (evaluate(theta.detach() + epsilon) - evaluate(theta.detach() - epsilon)) / (2 * epsilon)
    assert gradient.item() == pytest.approx(difference.item(), abs=1e-8)
    assert gradient.abs() > 0
    assert model.grad is None


def test_termination_removes_gae_tail_even_with_large_value():
    rewards = torch.tensor([[1.0], [2.0]])
    terminated = torch.tensor([[1.0], [0.0]])
    values = torch.tensor([[0.0], [1e6], [1e6]])
    targets = gae_targets(rewards, terminated, values, 0.99, 0.95)
    assert targets[0].item() == 1.0
