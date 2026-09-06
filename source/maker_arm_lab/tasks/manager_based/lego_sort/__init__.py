"""LegoSort task registration for the Maker Arm (spec section 2.3: five ids)."""

import gymnasium as gym


def _register() -> None:
    for env_id, module, cfg_name in (
        ("MakerArm-LegoSort-v0", "lego_sort_env_cfg", "LegoSortEnvCfg"),
        ("MakerArm-LegoSort-IK-Rel-v0", "variants", "LegoSortIKRelEnvCfg"),
        ("MakerArm-LegoSort-IK-Rel-Vision-v0", "vision", "LegoSortIKRelVisionEnvCfg"),
        ("MakerArm-LegoSort-Datagen-v0", "datagen", "LegoSortDatagenEnvCfg"),
        ("MakerArm-LegoSort-Eval-v0", "eval_cfg", "LegoSortEvalEnvCfg"),
    ):
        gym.register(
            id=env_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={"env_cfg_entry_point": f"{__name__}.{module}:{cfg_name}"},
        )


_register()
del _register
