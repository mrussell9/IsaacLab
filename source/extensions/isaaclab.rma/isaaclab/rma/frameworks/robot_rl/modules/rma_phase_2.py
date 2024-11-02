from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from isaaclab.rma.frameworks.robot_rl.modules import ActorCritic


class RMA2(nn.Module):
    def __init__(
        self,
        num_actions,
        env_size,
        prev_step_size,
        z_size,
        encoder_hidden_dims=[48, 48, 32],
        conv_params=[[32, 32, 8, 4], [32, 32, 5, 1], [32, 32, 5, 1]],
        activation="elu",
        history_length=50,
        **kwargs,
    ):
        if kwargs:
            print(
                "RMA2.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        torch.nn.Module.__init__(self)
        # Spot_RMA/runs/ll3q83hn/
        activation = get_activation(activation)
        num_policy_obs = prev_step_size + z_size # HARDCODED
        self.num_env_obs = env_size
        self.history_length = history_length
        self.all_env_obs = self.num_env_obs * self.history_length
        
        # Fix Hard Coding Later
        
        # Phase2 Prev States and Actions Encoder
        encoder = []
        encoder.append(nn.Linear(self.num_env_obs, encoder_hidden_dims[0]))
        encoder.append(activation)
        for layer_index in range(len(encoder_hidden_dims)):
            if layer_index == len(encoder_hidden_dims) - 1:
                encoder.append(nn.Linear(encoder_hidden_dims[layer_index], encoder_hidden_dims[-1]))
            else:
                encoder.append(nn.Linear(encoder_hidden_dims[layer_index], encoder_hidden_dims[layer_index + 1]))
                encoder.append(activation)
        self.encoder = nn.Sequential(*encoder)

        conv_net = []
        for conv_layer in conv_params:
            conv_net.append(nn.Conv1d(in_channels=conv_layer[0], out_channels=conv_layer[1],
                                     kernel_size=conv_layer[2], stride=conv_layer[3], groups=32))
        conv_net.append(nn.Flatten(start_dim=1))
        conv_net.append(nn.Linear(96, z_size))
        self.conv_net = nn.Sequential(*conv_net)
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self, observations):
        ######### Need to transform single list of obs into matrix ############
        obs = observations.view(observations.shape[0], self.num_env_obs, self.history_length)
        obs = obs.transpose(2,1)
        x = (self.encoder(obs))
        x = x.transpose(2,1)
        x = (self.conv_net(x))
        return x

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
