import gymnasium as gym
from . import rma_cfg, rma_ppo_cfg, rma2_cfg, rma2_bc_cfg

# Go1 Baseline
gym.register(
    id="RMA-Spot-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rma_cfg.SpotRmaCfg,
        "rma_cfg_entry_point": rma_ppo_cfg.SpotFlatPPORunnerCfg,
    },
)

gym.register(
    id="RMA-Play-Spot-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rma_cfg.SpotRmaCfg_PLAY,
        "rma_cfg_entry_point": rma_ppo_cfg.SpotFlatPPORunnerCfg,
    },
)

gym.register(
    id="RMA-Phase2-Spot-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rma2_cfg.SpotRma2Cfg,
        "rma_cfg_entry_point": rma2_bc_cfg.SpotRoughBcRunnerCfg,
    },
)