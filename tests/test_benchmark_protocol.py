"""Full-library metrics must penalize failure and retain the reference length."""
from types import SimpleNamespace

import pytest
import torch

from supermimic.eval.benchmark import ClipResult, summarize


def test_early_failure_does_not_shorten_reference_denominator():
    # Both clips have perfect per-step tracking until termination. The long clip
    # fails after ten of its 100 reference intervals, so it must score 0.1.
    failed = ClipResult("long", frames=101, steps=10, norm_return=0.1, failed=True, fail_step=10)
    complete = ClipResult("short", frames=11, steps=10, norm_return=1.0, failed=False, fail_step=-1)
    report = summarize([failed, complete])
    assert report["mean_norm_return"] == pytest.approx(0.55)
    assert report["reference_weighted_norm_return"] == pytest.approx(20 / 110)
    assert report["completion"] == pytest.approx(20 / 110)
    assert report["failed"] == 1


def test_evaluator_uses_full_clip_after_early_failure(monkeypatch):
    from supermimic.eval import benchmark

    lib = SimpleNamespace(num_clips=1, clip_start=torch.tensor([0]),
                          clips=[SimpleNamespace(name="early", length=101)])
    cmd = SimpleNamespace(library=lib, forced_frame=torch.tensor([-1]))
    obs = {"policy": torch.zeros(1, 1), "goal": torch.zeros(1, 1)}

    class FakeEnv:
        command_manager = SimpleNamespace(get_term=lambda _: cmd)
        reward_manager = SimpleNamespace(active_terms=["tracking"], term_rewards=torch.tensor([[0.1]]))
        step_count = 0
        closed = False

        def reset(self):
            return obs, {}

        def step(self, action):
            self.step_count += 1
            return obs, torch.tensor([0.1]), torch.tensor([self.step_count == 10]), torch.tensor([False]), {}

        def close(self):
            self.closed = True

    env = FakeEnv()
    monkeypatch.setattr(benchmark, "make_env", lambda *a, **k: env)
    results = benchmark.run_benchmark(lambda *a: torch.zeros(1, 1), ["dummy.npz"], device="cpu")
    assert results[0].norm_return == pytest.approx(0.1)
    assert results[0].failed
    assert results[0].steps == 10
    assert env.closed
