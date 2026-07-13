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


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not installed")
def test_compose_has_triton_service():
    result = subprocess.run(["docker", "compose", "--profile", "triton", "config"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "triton" in result.stdout
    assert "tritonserver" in result.stdout
