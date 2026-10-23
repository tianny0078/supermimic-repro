"""Train SuperMimic.  Example:

  uv run sm-train --run-name e1_walk --data.clips walk1_subject1 --env.num-envs 1024
"""

from __future__ import annotations

import os

# torchrun may only partition devices already exposed by the allocation.
if int(os.environ.get("LOCAL_WORLD_SIZE", "1")) > 1:
    visible = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d]
    local_world = int(os.environ["LOCAL_WORLD_SIZE"])
    if len(visible) < local_world:
        raise RuntimeError("torchrun needs at least one allocation-visible GPU per local rank")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible[int(os.environ["LOCAL_RANK"])]
    os.environ["SM_GPU_PINNED"] = "1"

import tyro

from supermimic.config import TrainConfig


def main() -> None:
    cfg = tyro.cli(TrainConfig)
    from supermimic.training.loop import Trainer

    Trainer(cfg).run()


if __name__ == "__main__":
    main()
