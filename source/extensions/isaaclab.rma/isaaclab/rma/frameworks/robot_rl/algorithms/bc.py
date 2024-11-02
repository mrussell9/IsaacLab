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
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.teacher = teacher
        self.teacher.to(self.device)
        self.storage = None  # initialized later
        self.loss_fn = nn.MSELoss()
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

    def act(self, obs, teacher_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        z_hat = self.actor_critic.get_latent(obs)
        self.transition.actions = self.teacher.act_inference(z_hat, teacher_obs).detach()
        # self.transition.values = self.actor_critic.evaluate(teacher_obs).detach()
        # self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        # self.transition.action_mean = self.actor_critic.action_mean.detach()
        # self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.teacher_observations = teacher_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        # self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def update(self):
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
            z_hat =self.actor_critic.get_latent(obs_batch)
            z = self.teacher.get_latent(teacher_obs_batch)
            self.teacher.act_inference(z_hat, teacher_obs_batch)
            # self.teacher.act_inference(z_hat, teacher_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])

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