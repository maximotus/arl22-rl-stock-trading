import gym
import numpy as np
import torch
import torch.nn as nn

from typing import Tuple

from rltrading.learner.replayMemory import ReplayMemory, Transition

class Agent:
	def __init__(self, replay_memory: ReplayMemory, neural_net: nn.Module, env: gym.Env, gamma: float, epsilon: float, batch_size: int = 128):
		self.replay_memory = replay_memory
		self.neural_net = neural_net

		# gym environment
		self.env = env
		self.state = self.env.reset()

		# RL parameters
		self.gamma = gamma
		self.epsilon = epsilon

		self.batch_size = batch_size

	def get_action(self) -> int:
		"""Choses an action to carry out, with an epsilon-greedy policy

		Returns
		-------

		action	
			An action dependent on the action space	
		"""
		if np.random.random() < self.epsilon:
			action = self.env.action_space.sample()
		else:
			state = torch.tensor([self.state])
			actions = self.neural_net.forward(state)
			action = torch.argmax(actions).item()

		return action

	def step(self) -> Tuple[float, bool]:
		action = self.get_action()

		new_state, reward, done, info = self.env.step(action)
		transition = Transition(self.state, action, new_state, reward)

		self.replay_memory.push(transition)

		self.state = new_state
		if done:
			self.env.reset()
		return reward, done

	def learn(self):
		if len(self.replay_memory) < self.batch_size:
			return	
		transitions: list[Transition] = self.replay_memory.sample(self.batch_size)

		batch = Transition(*zip(*transitions))

		# Todo -> maybe send to other device here
		state_batch = torch.cat(batch.state)
		action_batch = torch.cat(batch.action)
		next_state_batch = torch.cat(batch.next_state)
		reward_batch = torch.cat(batch.reward)

		state_action_values = self.neural_net.forward(state_batch)
		




