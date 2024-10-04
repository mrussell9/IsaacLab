#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


class RMA1(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_policy_obs, # latent size
        num_encoder_obs, #17 in paper
        num_critic_obs,
        num_actions=8 + 12 + 30, # latent, prev action, state
        num_latent=8,
        policy_hidden_dims=[128, 128],
        encoder_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "RMA1.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = get_activation(activation)

        # Policy
        policy_layers = []
        policy_layers.append(nn.Linear(num_policy_obs, policy_hidden_dims[0]))
        policy_layers.append(activation)
        for layer_index in range(len(policy_hidden_dims)):
            if layer_index == len(policy_hidden_dims) - 1:
                policy_layers.append(nn.Linear(policy_hidden_dims[layer_index], num_actions))
            else:
                policy_layers.append(nn.Linear(policy_hidden_dims[layer_index], policy_hidden_dims[layer_index + 1]))
                policy_layers.append(activation)
        self.policy = nn.Sequential(*policy_layers)

        # Env Factor Encoder
        encoder = []
        encoder.append(nn.Linear(num_encoder_obs, encoder_hidden_dims[0]))
        encoder.append(activation)
        for layer_index in range(len(encoder_hidden_dims)):
            if layer_index == len(encoder_hidden_dims) - 1:
                encoder.append(nn.Linear(encoder_hidden_dims[layer_index], num_latent))
            else:
                encoder.append(nn.Linear(encoder_hidden_dims[layer_index], encoder_hidden_dims[layer_index + 1]))
                encoder.append(activation)
        self.encoder = nn.Sequential(*encoder)

         # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

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

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
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
