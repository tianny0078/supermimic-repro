import json
from pathlib import Path

import pytest

from supermimic.artifacts import write_json
from supermimic.eval.benchmark import ClipResult, save_results


def test_evaluation_round_trip_with_checkpoint_path_config(tmp_path):
    output = tmp_path / "nested" / "benchmark.json"
    result = ClipResult("walk", frames=101, steps=100, norm_return=0.8, failed=False, fail_step=-1)
    save_results([result], output, {"train_config": {"data": {"npz_dir": Path("data/npz/lafan1")}}})
    document = json.loads(output.read_text())
    assert document["metadata"]["train_config"]["data"]["npz_dir"] == "data/npz/lafan1"
    assert document["summary"]["reference_weighted_norm_return"] == pytest.approx(0.8)
    assert document["clips"][0]["steps"] == 100


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), object()])
def test_failed_write_preserves_previous_valid_artifact(tmp_path, invalid):
    output = tmp_path / "report.json"
    write_json(output, {"valid": True})
    before = output.read_bytes()
    with pytest.raises((ValueError, TypeError)):
        write_json(output, {"invalid": invalid})
    assert output.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))
