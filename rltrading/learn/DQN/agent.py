import gym
import numpy as np
import torch

from typing import Tuple
from rltrading.learn.DQN.dqn import DQN

from rltrading.learn.DQN.replayMemory import ReplayMemory, Transition


class Agent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        gamma: float,
        epsilon: float,
        input_dims: int,
        fc1_dims: int,
        fc2_dims: int,
        batch_size: int = 128,
        replay_memory_size: int = 10000,
    ) -> None:

        self.replay_memory = ReplayMemory(replay_memory_size)

        # gym environment
        self.env = env
        self.state = self.env.reset()
        n_actions = self.env.action_space.n

        self.neural_net = DQN(learning_rate, input_dims, fc1_dims, fc2_dims, n_actions)
        self.target_nn = DQN(learning_rate, input_dims, fc1_dims, fc2_dims, n_actions)

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

        state_action_values = self.neural_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size)

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
