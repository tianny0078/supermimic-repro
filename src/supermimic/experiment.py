"""Named paper settings with a separately labelled short pilot workload."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro

from supermimic.artifacts import write_json
from supermimic.config import TrainConfig


@dataclass
class Experiment:
    name: str = "lafan1-paper-seed0"
    mode: Literal["paper", "pilot"] = "paper"
    data: Path = Path("data/npz/lafan1")
    output: Path = Path("logs")
    seed: int = 0
    max_hours: float | None = None
    resume: Path | None = None
    # Alternative interpretation of underspecified goal encoding. Explicitly
    # labelled in configuration; never switched on to silently improve a score.
    anchor_goal: bool = False

    def config(self) -> TrainConfig:
        cfg = TrainConfig(run_name=self.name, log_root=self.output, seed=self.seed, resume=self.resume)
        cfg.data.npz_dir = self.data
        cfg.env.anchor_obs = self.anchor_goal
        cfg.env.reset_noise = False  # Initial-state perturbation ranges not specified by paper.
        cfg.critic.root_error_features = False
        cfg.schedule.max_hours = self.max_hours
        if self.mode == "pilot":
            # Preserve all model sizes, batch sizes and H=32; only shorten runtime.
            cfg.schedule.total_iterations = 240
            cfg.schedule.checkpoint_interval = 240
            # Do not compress the paper's 250k-step LR decay into the short pilot.
            cfg.schedule.lr_decay_fraction = 250_000 / 240
        return cfg


def main() -> None:
    experiment = tyro.cli(Experiment)
    from supermimic.training.loop import Trainer

    cfg = experiment.config()
    if len(list(cfg.data.npz_dir.glob("*.npz"))) != 40:
        raise ValueError("LAFAN1 paper/pilot profiles require all 40 converted clips")
    run_dir = cfg.log_root / cfg.run_name
    if run_dir.exists() and cfg.resume is None:
        raise FileExistsError(f"Use a new run name or --resume: {run_dir}")
    trainer = None
    try:
        trainer = Trainer(cfg)
        trainer.run()
    except BaseException as error:
        write_json(run_dir / "status.json", {"state": "failed", "error": type(error).__name__, "message": str(error),
                                             "iteration": trainer.iteration if trainer else None})
        raise
    finally:
        if trainer is not None:
            trainer.env.close()


if __name__ == "__main__":
    main()
