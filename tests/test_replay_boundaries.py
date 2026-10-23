"""Replay windows must preserve chronology after wrapping and episode resets."""
import torch

from supermimic.data.replay_buffer import ReplayBuffer


def add(buffer, value, *, terminal=False, start=False):
    buffer.add(torch.tensor([0]), s=torch.tensor([[float(value)]]),
               a=torch.tensor([[float(value)]]), a_exec=torch.tensor([[float(value) + 0.1]]),
               xi=torch.zeros(1, 1), r=torch.ones(1), done=torch.tensor([float(terminal)]),
               frame=torch.tensor([value]), clip=torch.zeros(1, dtype=torch.long),
               terminal=torch.tensor([terminal]), ep_start=torch.tensor([start]))


def test_same_episode_ring_wrap_does_not_join_newest_to_oldest():
    buffer = ReplayBuffer(1, 8, 1, 1, 1, "cpu")
    for value in range(12):
        add(buffer, value, start=value == 0)
    window = buffer.sample_windows(200, 3)
    assert torch.equal(window.states[1:] - window.states[:-1], torch.ones_like(window.states[1:]))
    assert window.states.min() >= 4
    assert window.states.max() <= 11


def test_terminal_target_allowed_but_reset_boundary_never_crossed():
    buffer = ReplayBuffer(1, 16, 1, 1, 1, "cpu")
    for value in range(100, 104):
        add(buffer, value, terminal=value == 103, start=value == 100)
    buffer.end_episode(torch.tensor([0]))
    for value in range(200, 204):
        add(buffer, value, start=value == 200)
    window = buffer.sample_windows(500, 2)
    assert torch.equal(window.states[1:] - window.states[:-1], torch.ones_like(window.states[1:]))
    assert torch.any(window.done == 1)
    starts = buffer.sample_starts(200, 3)
    assert torch.all(starts.hist_states // 100 == starts.state[None] // 100)
    first = starts.state[:, 0] == 200
    assert torch.all(starts.hist_actions[:, first] == 0)
