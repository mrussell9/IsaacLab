#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from isaaclab.rma.frameworks.robot_rl.modules import ActorCritic, RMA1, RMA2
from isaaclab.rma.frameworks.robot_rl.storage import BcRolloutStorage


class BC:
    actor_critic: RMA2
    teacher: RMA1

    def __init__(
        self,
        actor_critic,
        teacher,
        learning_rate=0.0005,
        num_learning_epochs=1,
        num_mini_batches=1,
        device="cpu",
    ):
        self.device = device

        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.teacher = teacher
        self.teacher.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = BcRolloutStorage.Transition()

        # BC parameters
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, teacher_obs_shape, action_shape):
        self.storage = BcRolloutStorage(
            num_envs, num_transitions_per_env, actor_obs_shape, teacher_obs_shape, action_shape, self.device
        )

    def test_mode(self):
        self.actor_critic.test()
        self.teacher.test()

    def train_mode(self):
        self.actor_critic.train()
        self.teacher.test()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        # self.transition.actions = self.actor_critic.act(obs).detach()
        # self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        # self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        # self.transition.action_mean = self.actor_critic.action_mean.detach()
        # self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.transition.teacher_observations = teacher_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def update(self):
        mse_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            teacher_obs_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            z_hat =self.actor_critic.get_latents(obs_batch)
            z = self.teacher.get_latents(teacher_obs_batch)

            mean_squared_error = (z - z_hat).pow(2).mean()
            loss = mean_squared_error

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            mse_loss += mean_squared_error.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mse_loss /= num_updates
        self.storage.clear()

        return mse_loss
