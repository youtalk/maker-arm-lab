"""MS5 design P2: the perceived loop reaches its perception through one backend, the env's own
when it carries one (the real rig), else the simulator's, which changes nothing."""

from maker_arm_lab.core import loop
from test_perceived_loop import MS7D_ON, _rollout, _teacher_scene

MEMBERS = {"perceiver", "marker", "class_views", "wrist", "overhead", "wrist_look", "lift_class"}


class _Recording:
    """The simulator's backend, counting every call by member."""

    def __init__(self):
        self.inner = loop.SimPerception()
        self.calls: dict[str, int] = {}

    def __getattr__(self, name):
        member = getattr(self.inner, name)

        def call(*args, **kwargs):
            self.calls[name] = self.calls.get(name, 0) + 1
            return member(*args, **kwargs)

        return call


class _Bare:
    pass


def test_an_env_without_a_backend_gets_the_simulators():
    assert isinstance(loop.backend_of(_Bare()), loop.SimPerception)
    env = _Bare()
    env.perception_backend = None
    assert isinstance(loop.backend_of(env), loop.SimPerception)


def test_an_env_backend_is_used_as_it_is():
    env = _Bare()
    env.perception_backend = backend = _Recording()
    assert loop.backend_of(env) is backend


def test_the_sim_backend_has_every_member():
    assert MEMBERS <= set(dir(loop.SimPerception))


def test_every_perceived_call_goes_through_the_backend_and_the_sim_one_changes_nothing(
    monkeypatch,
):
    env, rig, policy, budget, switches = _teacher_scene(monkeypatch)
    plain = _rollout(env, rig, policy, budget=budget, perceived=True, **switches, **MS7D_ON)
    env.close()
    env, rig, policy, budget, switches = _teacher_scene(monkeypatch)
    env.perception_backend = recording = _Recording()
    routed = _rollout(env, rig, policy, budget=budget, perceived=True, **switches, **MS7D_ON)
    env.close()
    assert set(recording.calls) == MEMBERS
    for key in ("cycle_reasons", "cycle_targets", "cycle_steps", "done_reason"):
        assert plain["episodes"][0][key] == routed["episodes"][0][key], key
    assert plain["perception"] == routed["perception"]
