from __future__ import annotations

import wandb
import argparse
import torch
import torch.nn as nn
from torchinfo import summary
import isaaclab.rma.frameworks.robot_rl as robot_rl
from isaaclab.rma.frameworks.robot_rl.algorithms import PPO
from isaaclab.rma.frameworks.robot_rl.env import VecEnv
from isaaclab.rma.frameworks.robot_rl.modules import (
    ActorCritic,
    ActorCriticRecurrent,
    EmpiricalNormalization,
    RMA1,
)

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


class RMA2(nn.Module):
    def __init__(self,
                 model,
                 encoder_hidden_dims=[128],
                 encoder_out=32,
                 activation="elu",):
        super().__init__()

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

        policy = wandb.restore("model_49999.pt", run_path="Spot_RMA/runs/ll3q83hn/")
        loaded_dict = torch.load(policy.name)
        # print(policy)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]
    
    def encode_states(self, prev_states, activation: str, encoder_hidden_dims: list, encoder_out: int):
        prev_state_obs = prev_states.flatten(dim=1)
        # Encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(prev_state_obs, encoder_hidden_dims[0]))
        encoder_layers.append(activation)
        for layer_index in range(len(encoder_hidden_dims)):
            if layer_index == len(encoder_hidden_dims) - 1:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[layer_index], encoder_out))
            else:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[layer_index], encoder_hidden_dims[layer_index + 1]))
                encoder_layers.append(activation)
        self.encoder = nn.Sequential(*encoder_layers)

def arg_parser():
    # add argparse arguments
    parser = argparse.ArgumentParser(description="Train adaption module for RMA.")
    parser.add_argument("--model", type=str, default=None, help="Policy Model.")
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = arg_parser()

    # cfg = adaption_module_cfg

    RMA2(args.model)