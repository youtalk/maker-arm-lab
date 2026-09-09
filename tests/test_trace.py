"""CPU tests for core.trace -- the per-step state/action dump behind --trace_dir."""

import numpy as np
import pytest

from maker_arm_lab.core.trace import RolloutTrace


def _step(value: float, num_envs: int = 2):
    state = np.full((num_envs, 8), value, dtype=np.float32)
    raw = np.full((num_envs, 7), value + 1, dtype=np.float32)
    clamped = np.full((num_envs, 7), value + 2, dtype=np.float32)
    return state, raw, clamped


def test_save_writes_one_npz_per_env(tmp_path):
    trace = RolloutTrace(num_envs=2)
    trace.append(*_step(0.0), active=np.array([True, True]))

    paths = trace.save(tmp_path, "eval_002")

    assert [p.name for p in paths] == ["eval_002_env0_trace.npz", "eval_002_env1_trace.npz"]
    assert all(p.exists() for p in paths)


def test_each_env_keeps_only_its_own_rows(tmp_path):
    trace = RolloutTrace(num_envs=2)
    trace.append(*_step(1.0), active=np.array([True, True]))
    trace.append(*_step(2.0), active=np.array([True, True]))

    paths = trace.save(tmp_path, "eval_000")

    with np.load(paths[0]) as d:
        assert d["state"].shape == (2, 8)
        assert d["action_raw"].shape == (2, 7)
        assert d["action_clamped"].shape == (2, 7)
        np.testing.assert_allclose(d["state"][:, 0], [1.0, 2.0])
        np.testing.assert_allclose(d["action_raw"][:, 0], [2.0, 3.0])


def test_inactive_envs_stop_accumulating(tmp_path):
    """A finished env auto-resets into an unrelated episode; its trace must end."""
    trace = RolloutTrace(num_envs=2)
    trace.append(*_step(1.0), active=np.array([True, True]))
    trace.append(*_step(2.0), active=np.array([False, True]))

    paths = trace.save(tmp_path, "eval_000")

    with np.load(paths[0]) as d:
        assert d["state"].shape[0] == 1
    with np.load(paths[1]) as d:
        assert d["state"].shape[0] == 2


def test_step_indices_are_recorded(tmp_path):
    trace = RolloutTrace(num_envs=1)
    trace.append(*_step(0.0, 1), active=np.array([True]))
    trace.append(*_step(0.0, 1), active=np.array([False]))
    trace.append(*_step(0.0, 1), active=np.array([True]))

    paths = trace.save(tmp_path, "eval_000")

    with np.load(paths[0]) as d:
        assert d["step"].tolist() == [0, 2]


def test_an_env_that_never_ran_still_produces_a_readable_file(tmp_path):
    trace = RolloutTrace(num_envs=1)

    paths = trace.save(tmp_path, "eval_000")

    with np.load(paths[0]) as d:
        assert d["state"].shape == (0, 8)


def test_mismatched_batch_size_is_rejected():
    trace = RolloutTrace(num_envs=2)
    with pytest.raises(ValueError):
        trace.append(*_step(0.0, 3), active=np.array([True, True, True]))
