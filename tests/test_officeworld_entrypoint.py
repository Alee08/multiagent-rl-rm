import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_officeworld_entrypoint_runs_as_module(tmp_path):
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "multiagent_rlrm.environments.office_world.office_main",
            "--help",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Run experiments with selected map" in result.stdout
