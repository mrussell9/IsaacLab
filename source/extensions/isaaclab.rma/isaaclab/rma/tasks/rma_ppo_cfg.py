from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from isaaclab.rma.frameworks.robot_rl.utils import RmaActorCriticsCfg


@configclass
class SpotFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 500
    experiment_name = "spot_rma"
    empirical_normalization = False
    store_code_state = False
    logger = "wandb"
    wandb_project = "Spot_RMA"
    policy = RmaActorCriticsCfg(
        class_name="RMA1",
        init_noise_std=1.0,
        env_size=201,
        prev_step_size=48,
        z_size=128,
        actor_hidden_dims=[512, 256, 128],
        encoder_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0025,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
