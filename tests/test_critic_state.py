from types import SimpleNamespace

import torch

from supermimic.imagination.policy_trainer import PolicyTrainer
from supermimic.policy.nets import Critic, CriticCfg, PolicyCfg
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg


def test_full_state_critic_keeps_translation_information():
    torch.manual_seed(0)
    state = torch.zeros(2, 1, 71, requires_grad=True)
    with torch.no_grad():
        state[..., 2] = 1.0
        state[..., 3] = 1.0
    rollout = SimpleNamespace(states=state, goals=torch.zeros(2, 1, 680), e_root=torch.zeros(2, 1, 4))
    norms = {}
    for mode in ("full", "canonical"):
        trainer = object.__new__(PolicyTrainer)
        trainer.wm = SimpleNamespace(spec=StateSpec(StateSpecCfg(), "cpu"))
        trainer.critic = Critic(CriticCfg(hidden=(16,), state_features=mode), pcfg=PolicyCfg())
        features, goals, errors = trainer._critic_inputs(rollout)
        value = trainer.critic(features, goals, torch.zeros(2, 1, 33), errors).sum()
        gradient, = torch.autograd.grad(value, state)
        norms[mode] = float(gradient[..., :2].abs().sum())
    assert norms["full"] > 1e-8
    assert norms["canonical"] == 0.0
