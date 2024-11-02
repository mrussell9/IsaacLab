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
        z_hat = self.adaption_module(obs)
        self.transition.actions = self.teacher.act_inference(teacher_obs, z=z_hat).detach()
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
            self.teacher.act_inference(teacher_obs_batch, z=z_hat)

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