import importlib.util
import subprocess
from pathlib import Path

import pytest


def test_queued_source_stays_fixed_and_data_is_shared(tmp_path):
    module_path = Path(__file__).resolve().parents[1] / "scripts/submit_experiment.py"
    spec = importlib.util.spec_from_file_location("submit_experiment", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo)

    git("init", "-q")
    (repo / "model.py").write_text("revision_one = True\n")
    git("add", "model.py")
    git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "initial")
    output = tmp_path / "output"
    (output / "data").mkdir(parents=True)
    (output / "data" / "large_dataset").write_text("shared data")
    snapshot, revision = module.freeze_source(repo, output)
    (repo / "model.py").write_text("revision_two = True\n")
    assert (snapshot / "model.py").read_text() == "revision_one = True\n"
    assert (snapshot / "SOURCE_REVISION").read_text().strip() == revision
    assert (snapshot / "data").is_symlink()
    assert (snapshot / "data" / "large_dataset").read_text() == "shared data"
    with pytest.raises(ValueError, match="Commit local changes"):
        module.freeze_source(repo, output)
