#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from isaaclab.rma.frameworks.robot_rl.modules import ActorCritic, RMA1, RMA2
from isaaclab.rma.frameworks.robot_rl.storage import BcRolloutStorage


class BC:
    adaption_module: RMA2
    teacher: RMA1

    def __init__(
        self,
        adaption_module,
        teacher,
        gamma=0.998,
        learning_rate=0.00005,
        num_learning_epochs=1,
        num_mini_batches=4,
        device="cpu",
    ):
        self.device = device
        self.gamma = gamma
        self.learning_rate = learning_rate

        # PPO components
        self.adaption_module = adaption_module
        self.adaption_module.to(self.device)
        self.teacher = teacher
        self.teacher.to(self.device)
        self.storage = None  # initialized later
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.adaption_module.parameters(), lr=learning_rate)
        self.transition = BcRolloutStorage.Transition()

        # BC parameters
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, teacher_obs_shape, action_shape):
        self.storage = BcRolloutStorage(
            num_envs, num_transitions_per_env, actor_obs_shape, teacher_obs_shape, action_shape, self.device
        )

    def test_mode(self):
        self.adaption_module.test()
        self.teacher.test()

    def train_mode(self):
        self.adaption_module.train()
        self.teacher.test()

    def act(self, obs, teacher_obs):
        # Compute the actions and values
        ##########################################################
        # This is ugly and should be fixed but for now... it works...
        # Position of most recent observations for each term
        obs_dict = {
            "lin_vel": [0,3],
            "ang_vel": [150,153],
            "proj_g": [300,303],
            "vel_com": [450,453],
            "joint_pos": [600,612],
            "joint_vel": [1200,1212],
            "actions": [1800, 1812]
        }
        # Most recent observations
        obs_actor = torch.cat(
            [obs[:, obs_dict['lin_vel'][0]:obs_dict['lin_vel'][1]],
            obs[:, obs_dict['ang_vel'][0]:obs_dict['ang_vel'][1]],
            obs[:, obs_dict['proj_g'][0]:obs_dict['proj_g'][1]],
            obs[:, obs_dict['vel_com'][0]:obs_dict['vel_com'][1]],
            obs[:, obs_dict['joint_pos'][0]:obs_dict['joint_pos'][1]],
            obs[:, obs_dict['joint_vel'][0]:obs_dict['joint_vel'][1]],
            obs[:, obs_dict['actions'][0]:obs_dict['actions'][1]]],
            dim=-1
        )
        # Obs are currently stored as [x,y,z,x,y,z... * history] for each term,
        # For our conv net we want them to be [x,x,x * history, y,y,y * history, z,z,z * history, etc]
        obs_ordered = torch.cat(
            [obs[:,0:150:3],
            obs[:,1:150:3],
            obs[:,2:150:3],
            obs[:,150:300:3],
            obs[:,151:300:3],
            obs[:,152:300:3],
            obs[:,300:450:3],
            obs[:,301:450:3],
            obs[:,302:450:3],
            obs[:,450:600:3],
            obs[:,451:600:3],
            obs[:,452:600:3],
            obs[:,600:1200:12],
            obs[:,601:1200:12],
            obs[:,602:1200:12],
            obs[:,603:1200:12],
            obs[:,604:1200:12],
            obs[:,605:1200:12],
            obs[:,606:1200:12],
            obs[:,607:1200:12],
            obs[:,608:1200:12],
            obs[:,609:1200:12],
            obs[:,610:1200:12],
            obs[:,611:1200:12],
            obs[:,1200:1800:12],
            obs[:,1201:1800:12],
            obs[:,1202:1800:12],
            obs[:,1203:1800:12],
            obs[:,1204:1800:12],
            obs[:,1205:1800:12],
            obs[:,1206:1800:12],
            obs[:,1207:1800:12],
            obs[:,1208:1800:12],
            obs[:,1209:1800:12],
            obs[:,1210:1800:12],
            obs[:,1211:1800:12],
            obs[:,1800:2400:12],
            obs[:,1801:2400:12],
            obs[:,1802:2400:12],
            obs[:,1803:2400:12],
            obs[:,1804:2400:12],
            obs[:,1805:2400:12],
            obs[:,1806:2400:12],
            obs[:,1807:2400:12],
            obs[:,1808:2400:12],
            obs[:,1809:2400:12],
            obs[:,1810:2400:12],
            obs[:,1811:2400:12]], dim=-1
        )
        ##########################################################
        z_hat = self.adaption_module(obs_ordered)
        self.transition.actions = self.teacher.act_inference(obs_actor, z=z_hat).detach()
        self.transition.observations = obs
        self.transition.teacher_observations = teacher_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.dones = dones

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.adaption_module.reset(dones)

    def update(self):
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            teacher_obs_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            z_hat =self.adaption_module(obs_batch)
            z = self.teacher.get_latent(teacher_obs_batch)
            # self.teacher.act_inference(obs_batch, z=z_hat)

            loss = self.loss_fn(z_hat, z)

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.storage.clear()

        return loss

    def act_inference(self, observations):
        obs_actor = observations[:, self.num_env_obs:]
        z = self.get_latent(observations)
        actor_input = torch.cat([z, obs_actor], dim=-1)
        actions_mean = self.actor(actor_input)
        return actions_mean