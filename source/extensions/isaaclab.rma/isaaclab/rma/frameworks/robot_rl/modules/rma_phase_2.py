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
                 activation="elu",
                 **kwargs,
    ):
        super().__init__()
        if kwargs:
            print(
                "RMA1.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        # Spot_RMA/runs/ll3q83hn/
        activation = get_activation(activation)

        # Phase2 Prev States and Actions Encoder, phi
        phi = []
        phi.append(nn.Linear(self.num_env_obs, encoder_hidden_dims[0]))
        phi.append(activation)
        for layer_index in range(len(phi_hidden_dims)):
            if layer_index == len(phi_hidden_dims) - 1:
                phi.append(nn.Linear(phi_hidden_dims[layer_index], phi_hidden_dims[-1]))
            else:
                phiappend(nn.Linear(phi_hidden_dims[layer_index], phi_hidden_dims[layer_index + 1]))
                phiappend(activation)
        self.phi = nn.Sequential(*phi)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    def update_distribution(self, observations):
        obs_e = observations[:, :self.num_env_obs]
        obs_actor = observations[:, self.num_env_obs:]
        z = self.get_latent(obs_e)
        actor_input = torch.cat([z, obs_actor], dim=-1)
        mean = self.actor(actor_input)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
    
    def get_latent(self, encoder_observations):
        return self.encoder(encoder_observations)

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
