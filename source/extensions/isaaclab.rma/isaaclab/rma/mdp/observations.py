"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`omni.isaac.lab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg, ManagerTermBase, ObservationTermCfg
from omni.isaac.lab.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv

def contact_sensor(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    return sensor.data.net_forces_w[:,sensor_cfg.body_ids,:].flatten(start_dim=1)

class contact_friction(ManagerTermBase):
    def __init__(self, env: ManagerBasedRLEnvCfg, cfg: ObservationTermCfg) -> None:
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        assert cfg.params["asset_cfg"].body_ids != slice(None), "Error: must provide body_ids to the 'contact_friction' ObservationTermCfg"
        num_shapes_per_body = []
        for link_path in asset.root_physx_view.link_paths[0]:
            link_physx_view = asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
            num_shapes_per_body.append(link_physx_view.max_shapes)

        materials = asset.root_physx_view.get_material_properties().to(env.device)
        friction = torch.zeros(env.num_envs, 2, device=env.device)
        for body_id in cfg.params["asset_cfg"].body_ids:
            # start index of shape
            start_idx = sum(num_shapes_per_body[:body_id])
            # end index of shape
            end_idx = start_idx + num_shapes_per_body[body_id]
            friction += materials[:, start_idx:end_idx, :2].mean(dim=1)
        self.friction = friction / len(cfg.params["asset_cfg"].body_ids)
    
    def __call__(self, env: ManagerBasedRLEnvCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        return self.friction.clone()