import shutil
import subprocess
import pytest


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not installed")
def test_compose_config_is_valid():
    result = subprocess.run(["docker", "compose", "config"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "redpanda" in result.stdout
    assert "worker-aggregate" in result.stdout
