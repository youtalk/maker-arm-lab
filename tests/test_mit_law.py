"""Tests for maker_arm_lab.core.mit_law (CPU-only)."""

import torch

from maker_arm_lab.core.mit_law import mit_mode_feedforward, mit_mode_torque


def test_pd_plus_feedforward():
    tau = mit_mode_torque(
        q=torch.tensor([0.1]),
        qd=torch.tensor([0.2]),
        q_des=torch.tensor([0.5]),
        qd_des=torch.tensor([0.0]),
        kp=torch.tensor([100.0]),
        kd=torch.tensor([10.0]),
        tau_ff=torch.tensor([1.5]),
        effort_limit=torch.tensor([100.0]),
    )
    # 100*(0.5-0.1) + 10*(0.0-0.2) + 1.5 = 40 - 2 + 1.5
    assert torch.allclose(tau, torch.tensor([39.5]))


def test_effort_clamp_both_signs():
    kwargs = {
        "qd": torch.zeros(2),
        "qd_des": torch.zeros(2),
        "kp": torch.full((2,), 1000.0),
        "kd": torch.zeros(2),
        "tau_ff": torch.zeros(2),
        "effort_limit": torch.full((2,), 10.0),
    }
    tau = mit_mode_torque(q=torch.tensor([0.0, 1.0]), q_des=torch.tensor([1.0, 0.0]), **kwargs)
    assert torch.allclose(tau, torch.tensor([10.0, -10.0]))


def test_batched_shapes():
    n_envs, n_joints = 4, 6
    tau = mit_mode_torque(
        q=torch.zeros(n_envs, n_joints),
        qd=torch.zeros(n_envs, n_joints),
        q_des=torch.ones(n_envs, n_joints),
        qd_des=torch.zeros(n_envs, n_joints),
        kp=torch.full((n_joints,), 50.0),
        kd=torch.full((n_joints,), 5.0),
        tau_ff=torch.zeros(n_joints),
        effort_limit=torch.full((n_joints,), 10.0),
    )
    assert tau.shape == (n_envs, n_joints)
    assert torch.allclose(tau, torch.full((n_envs, n_joints), 10.0))


def test_feedforward_without_gravity_is_passthrough():
    tau_cmd = torch.tensor([1.0, -2.0, 3.0])
    assert mit_mode_feedforward(tau_cmd, None) is tau_cmd


def test_feedforward_adds_gravity():
    tau = mit_mode_feedforward(torch.tensor([1.0, -2.0]), torch.tensor([0.5, 0.25]))
    assert torch.allclose(tau, torch.tensor([1.5, -1.75]))


def test_gravity_feedforward_sign_holds_a_static_joint():
    """kp=kd=0, no user feed-forward: the law must reproduce G(q) exactly.

    This pins the sign convention that the empirical GPU check verifies on the
    real articulation: G(q) is the torque that must be APPLIED to hold the
    arm, so it enters tau_ff positively. Flipping it here would command
    -G(q), i.e. twice the gravity torque in the falling direction.
    """
    tau_gravity = torch.tensor([5.3, -1.2])
    tau = mit_mode_torque(
        q=torch.tensor([1.57, 1.31]),
        qd=torch.zeros(2),
        q_des=torch.tensor([1.57, 1.31]),
        qd_des=torch.zeros(2),
        kp=torch.zeros(2),
        kd=torch.zeros(2),
        tau_ff=mit_mode_feedforward(torch.zeros(2), tau_gravity),
        effort_limit=torch.full((2,), 28.0),
    )
    assert torch.allclose(tau, tau_gravity)
