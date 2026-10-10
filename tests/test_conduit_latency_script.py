"""MS5 design P7: the latency script's token contract and its ROI read."""

import ast
import importlib.util
import io
import os
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "conduit_latency.py"
spec = importlib.util.spec_from_file_location("conduit_latency", SCRIPT)
conduit_latency = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conduit_latency)


def _run(*args):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "AMENT_PREFIX_PATH")}
    out = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    return out.returncode, out.stdout.strip().splitlines()


def test_the_script_honours_the_token_contract_in_its_text():
    text = SCRIPT.read_text()
    assert "LATENCY_OK " in text and "LATENCY_FAIL " in text and "os._exit(main())" in text
    body = ast.parse(text).body
    top = {a.name for n in body if isinstance(n, ast.Import) for a in n.names}
    top |= {n.module for n in body if isinstance(n, ast.ImportFrom) and n.module}
    assert "maker_arm_lab.real.ros" not in top and "PIL" not in top


def test_a_usage_error_prints_the_fail_token():
    code, lines = _run("photon", "--on_cmd", "true")
    assert code == 2 and lines[-1].startswith("LATENCY_FAIL ") and "usage" in lines[-1]


def test_without_ros_the_stamp_mode_fails_with_a_token():
    code, lines = _run("stamp", "--seconds", "0.1")
    assert code == 1 and lines[-1].startswith("LATENCY_FAIL ")


def test_an_interrupt_prints_one_interrupted_token(monkeypatch, capsys):
    def interrupted(args):
        raise KeyboardInterrupt

    monkeypatch.setattr(conduit_latency, "run_stamp", interrupted)
    assert conduit_latency.main(["stamp"]) == 1
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines == ['LATENCY_FAIL {"reason": "interrupted"}']


def test_a_ctrl_c_in_the_stamp_wait_prints_one_token_and_exits_through_os_exit():
    """MS5 final review I3: a real SIGINT during the 300 s wait, in a child that runs the
    script's own ``__main__`` line with the ROS subscriber replaced by a sleep."""
    child = (
        "import importlib.util, os, sys, time\n"
        f"spec = importlib.util.spec_from_file_location('m', {str(SCRIPT)!r})\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "def wait(args):\n"
        "    print('READY', flush=True)\n"
        "    time.sleep(args.seconds)\n"
        "m.run_stamp = wait\n"
        "os._exit(m.main(['stamp', '--seconds', '60']))\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "AMENT_PREFIX_PATH")}
    proc = subprocess.Popen(
        [sys.executable, "-c", child], stdout=subprocess.PIPE, text=True, env=env
    )
    assert proc.stdout.readline().strip() == "READY"
    proc.send_signal(signal.SIGINT)
    out, _ = proc.communicate(timeout=30)
    assert proc.returncode == 1
    tokens = [line for line in out.splitlines() if line.startswith("LATENCY_")]
    assert tokens == ['LATENCY_FAIL {"reason": "interrupted"}']


def test_roi_brightness_reads_only_the_roi():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[100:200, 300:400] = 255
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="JPEG", quality=95)
    data = buf.getvalue()
    assert conduit_latency.roi_brightness(data, (320, 120, 380, 180)) > 240
    assert conduit_latency.roi_brightness(data, (0, 0, 100, 100)) < 15
