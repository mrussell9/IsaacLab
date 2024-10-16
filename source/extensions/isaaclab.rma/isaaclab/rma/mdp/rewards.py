from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import ManagerTermBase, SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    from omni.isaac.lab.managers import RewardTermCfg

def foot_impact_penalty(env: ManagerBasedRLEnvCfg, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, cutoff: float) -> torch.Tensor:
    """Penalize foot impact when coming into contact with the ground"""
    asset: Articulation = env.scene[asset_cfg.name]
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # get contact state
    is_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    # get velocity at contact
    foot_down_velocity = torch.clamp(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2], min=-abs(cutoff), max=0.0)
    # penalty is the velocity at contact squared when in contact
    reward = is_contact * torch.square(foot_down_velocity)
    return torch.sum(reward, dim=1)