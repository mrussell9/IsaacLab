from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from isaaclab.rma.frameworks.robot_rl.utils import RmaActorCriticsCfg, BcRunnerCfg, BcAlgorithmCfg


@configclass
class SpotRoughBcRunnerCfg(BcRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 500
    experiment_name = "spot_rma"
    empirical_normalization = False
    store_code_state = False
    logger = "wandb"
    wandb_project = "Spot_RMA"
    policy = RmaActorCriticsCfg(
        class_name="RMA2",
        init_noise_std=1.0,
        env_size=2400,
        prev_step_size=48,
        z_size=8,
        actor_hidden_dims=[512, 256, 128],
        encoder_hidden_dims=[128, 32],
        activation="elu",
    )
    algorithm = BcAlgorithmCfg(
        learning_rate=0.0005,
        num_learning_epochs=1,
        num_mini_batches=1,
    )
    teacher = RmaActorCriticsCfg(
        class_name="RMA1",
        init_noise_std=1.0,
        env_size=201,
        prev_step_size=48,
        z_size=8,
        actor_hidden_dims=[512, 256, 128],
        encoder_hidden_dims=[256, 128, 8],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )