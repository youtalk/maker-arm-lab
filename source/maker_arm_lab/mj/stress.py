"""MS7j stress block: the real-arm conditions the readiness suite turns on, all off by default.

``Stress.block()`` names only the knobs off their default, so a run with every knob off writes no
``stress`` key and stays byte-identical to main. ``scripts/run_gate_eval.py`` keeps a stdlib copy
of ``STRESS_KNOBS`` and ``stress_block``, and a test pins it to this module."""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class Stress:
    sparse_gap: float = 0.0  # m, 0 = the normal heap drop
    obs_delay: int = 0  # control steps of camera delay
    overhead_offset: tuple[float, ...] = ()  # (rx, ry, rz deg, dx, dy, dz m), () = off
    wrist_offset: tuple[float, ...] = ()
    overhead_fov_delta: float = 0.0  # deg added to the rendered fovy
    wrist_fov_delta: float = 0.0
    warp_back: bool = False
    jpeg_quality: int = 0  # 0 = off, else 1..100
    blur_sigma: float = 0.0  # px at 224
    exposure_stops: float = 0.0
    hard_light: bool = False

    def block(self) -> dict:
        """The knobs off their default as JSON values, ``{}`` when all are off."""
        out = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value != f.default and value != ():
                out[f.name] = [float(v) for v in value] if isinstance(value, tuple) else value
        return out

    def overhead_camera_on(self) -> bool:
        return bool(self.overhead_offset) or self.overhead_fov_delta != 0.0


STRESS_KNOBS: tuple[str, ...] = tuple(f.name for f in fields(Stress))


def add_stress_arguments(parser) -> None:
    parser.add_argument("--sparse_gap", type=float, default=0.0, help="MS7j: sparse heap, in m")
    parser.add_argument("--obs_delay", type=int, default=0, help="MS7j: camera delay, in steps")
    for name in ("overhead_offset", "wrist_offset"):
        parser.add_argument(
            f"--{name}", type=str, default="", help="MS7j: rx,ry,rz (deg),dx,dy,dz (m)"
        )
    parser.add_argument("--overhead_fov_delta", type=float, default=0.0, help="MS7j: deg")
    parser.add_argument("--wrist_fov_delta", type=float, default=0.0, help="MS7j: deg")
    parser.add_argument("--warp_back", action="store_true", help="MS7j: warp to the sim view")
    parser.add_argument("--jpeg_quality", type=int, default=0, help="MS7j: 0 off, else 1..100")
    parser.add_argument("--blur_sigma", type=float, default=0.0, help="MS7j: px at 224")
    parser.add_argument("--exposure_stops", type=float, default=0.0, help="MS7j: exposure, stops")
    parser.add_argument("--hard_light", action="store_true", help="MS7j: hard light rig")


def _offset(text: str, name: str) -> tuple[float, ...]:
    if not text:
        return ()
    values = tuple(float(v) for v in text.split(","))
    if len(values) != 6:
        raise ValueError(f"--{name} needs exactly six numbers, got {len(values)}")
    return values


def stress_from_args(args) -> Stress:
    """The parsed ``Stress``. Raises ``ValueError`` on a bad combination."""
    stress = Stress(
        **{
            name: getattr(args, name, default)
            for name, default in (
                ("sparse_gap", 0.0),
                ("obs_delay", 0),
                ("overhead_fov_delta", 0.0),
                ("wrist_fov_delta", 0.0),
                ("warp_back", False),
                ("jpeg_quality", 0),
                ("blur_sigma", 0.0),
                ("exposure_stops", 0.0),
                ("hard_light", False),
            )
        },
        overhead_offset=_offset(getattr(args, "overhead_offset", ""), "overhead_offset"),
        wrist_offset=_offset(getattr(args, "wrist_offset", ""), "wrist_offset"),
    )
    fov = stress.overhead_fov_delta != 0.0 or stress.wrist_fov_delta != 0.0
    if fov and not stress.warp_back:
        raise ValueError(
            "a fov delta needs --warp_back: the real path always warps to the sim view"
        )
    if stress.warp_back and not (stress.overhead_offset or stress.wrist_offset or fov):
        raise ValueError("--warp_back needs a camera offset or a fov delta")
    if not 0 <= stress.jpeg_quality <= 100:
        raise ValueError("--jpeg_quality must lie in 0 to 100")
    if stress.blur_sigma < 0 or stress.obs_delay < 0:
        raise ValueError("--blur_sigma and --obs_delay must not be negative")
    if stress.overhead_camera_on() and getattr(args, "frame_jitter", False):
        raise ValueError("overhead camera knobs refuse --frame_jitter, which rewrites the pose")
    if stress.sparse_gap:
        if not getattr(args, "tray", False):
            raise ValueError("--sparse_gap needs --tray")
        if getattr(args, "tray_refill", False):
            raise ValueError("--sparse_gap refuses --tray_refill")
    return stress
