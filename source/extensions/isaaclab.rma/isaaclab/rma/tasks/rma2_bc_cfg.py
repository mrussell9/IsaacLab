from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from isaaclab.rma.frameworks.robot_rl.utils import RmaActorCriticsCfg, RmaAdaptionModuleCfg, BcRunnerCfg, BcAlgorithmCfg


@configclass
class SpotRoughBcRunnerCfg(BcRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 500
    experiment_name = "spot_rma"
    empirical_normalization = False
    store_code_state = False
    logger = "wandb"
    wandb_project = "Spot_RMA_Phase2"
    policy = RmaAdaptionModuleCfg(
        class_name="RMA2",
        env_size=48,
        prev_step_size=48,
        z_size=30,
        encoder_hidden_dims=[512, 256, 128, 32],
        conv_params=[[32, 32, 8, 4], [32, 32, 5, 1], [32, 32, 5, 1]],
        activation="elu",
        history_length=50,
    )
    algorithm = BcAlgorithmCfg(
        learning_rate=0.00005,
        num_learning_epochs=1,
        num_mini_batches=4,
    )
    teacher = RmaActorCriticsCfg(
        class_name="RMA1",
        init_noise_std=1.0,
        env_size=201,
        prev_step_size=48,
        z_size=30,
        actor_hidden_dims=[512, 256, 128],
        encoder_hidden_dims=[256, 128, 30],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )