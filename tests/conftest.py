"""Pytest fixtures for CI: mock llama + zero-delay overrides."""

import os
import sys
import subprocess
import time
import pytest
import httpx
import re

MOCK_LLAMA_PORT = int(os.environ.get("MOCK_LLAMA_PORT", "18080"))

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
import yaml
with open(CONFIG_PATH, "r") as _f:
    _CFG = yaml.safe_load(_f)
MAIN_PORT = _CFG["server"]["port"]



# ── Session-scoped server fixtures ──────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def mock_llama_server():
    proc = subprocess.Popen(
        [sys.executable, "-m", "tests.mock_llama"],
        env={**os.environ, "MOCK_LLAMA_PORT": str(MOCK_LLAMA_PORT)},
        stdout=sys.stdout.buffer,
        stderr=sys.stderr.buffer,
    )
    url = f"http://127.0.0.1:{MOCK_LLAMA_PORT}/v1/completions"
    for _ in range(40):
        try:
            if httpx.post(url, json={"prompt": "ping", "stream": False}, timeout=2).status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.25)
    else:
        proc.kill()
        raise RuntimeError("Mock llama server failed to start")

    yield proc
    proc.terminate()
    proc.wait(timeout=5)


@pytest.fixture(scope="session", autouse=True)
def main_app_server(mock_llama_server):
    orig_cfg = open(CONFIG_PATH).read()
    patched = orig_cfg.replace(
        _CFG["llama_server"]["url"],
        f"http://127.0.0.1:{MOCK_LLAMA_PORT}"
    )
    # Patch delays for CI speed
    patched = re.sub(r"(batch_delay:\s*)\d+(\.\d+)?", r"\g<1>0", patched)
    patched = re.sub(r"(retry_delay:\s*)\d+(\.\d+)?", r"\g<1>0", patched)
    patched = re.sub(r"(max_retries:\s*)\d+(\.\d+)?", r"\g<1>1", patched)

    with open(CONFIG_PATH, "w") as f:
        f.write(patched)

    proc = subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=os.path.join(os.path.dirname(__file__), ".."),
        stdout=subprocess.PIPE,
        stderr=sys.stderr.buffer,
    )
    url = f"http://127.0.0.1:{MAIN_PORT}/api/config"
    for _ in range(60):
        try:
            if httpx.get(url, timeout=2).status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.3)
    else:
        proc.kill()
        with open(CONFIG_PATH, "w") as f:
            f.write(orig_cfg)
        raise RuntimeError("Main app failed to start")

    yield proc
    proc.terminate()
    proc.wait(timeout=5)
    with open(CONFIG_PATH, "w") as f:
        f.write(orig_cfg)

