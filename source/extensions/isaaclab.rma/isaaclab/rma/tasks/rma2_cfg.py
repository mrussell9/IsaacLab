import math
from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.terrains as terrain_gen
from omni.isaac.lab.envs import ViewerCfg
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab.rma.mdp as rma_mdp

from .rma_cfg import SpotRmaCfg

##
# Pre-defined configs
##
from omni.isaac.lab_assets.spot import SPOT_CFG  # isort: skip

@configclass
class ObservationsPhase2Cfg:
    """Observation specifications for the MDP."""

    @configclass
    class TeacherCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        foot_force = ObsTerm(
            func=rma_mdp.contact_sensor,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
            # noise=Unoise(n_min=-5, n_max=5),
        )
        ground_friction = ObsTerm(
            func=rma_mdp.contact_friction,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")},
            noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))

        actions = ObsTerm(func=mdp.last_action)
        

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=rma_mdp.HistoryObsWrapper, params={
            "func": mdp.base_lin_vel, "history_len": 50,
            "func_params": {}}, 
            noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(func=rma_mdp.HistoryObsWrapper, params={
            "func":mdp.base_ang_vel, "history_len": 50,
            "func_params": {}},
            noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        projected_gravity = ObsTerm(func=rma_mdp.HistoryObsWrapper, params={
            "func":mdp.projected_gravity, "history_len": 50,
            "func_params": {}},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=rma_mdp.HistoryObsWrapper, params={
            "func":mdp.generated_commands, "history_len": 50,
            "func_params": {"command_name": "base_velocity"}},
        )
        joint_pos = ObsTerm(func=rma_mdp.HistoryObsWrapper, params={
            "func":mdp.joint_pos_rel, "history_len": 50,
            "func_params": {}},
            noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(func=rma_mdp.HistoryObsWrapper, params={
            "func":mdp.joint_vel_rel, "history_len": 50,
            "func_params": {}},
            noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        actions = ObsTerm(func=rma_mdp.HistoryObsWrapper, params={
            "func":mdp.last_action, "history_len": 50,
            "func_params": {}},
        )

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    teacher: TeacherCfg = TeacherCfg()

@configclass
class SpotRma2Cfg(SpotRmaCfg):
    observations: ObservationsPhase2Cfg = ObservationsPhase2Cfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4 # 50 Hz
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
