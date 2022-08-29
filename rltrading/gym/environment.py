import gym
import numpy as np

from numpy import inf
from gym.vector.utils import spaces

from rltrading.data.data import Data


class Environment(gym.Env):
    BUY_ACTION = 2
    SELL_ACTION = 1
    HOLD_ACTION = 0

    """
    Specify how many shares of a given stock and how much money the agent has
    """

    def __init__(self: "Environment", shares: int, money: float, data: Data):
        self.amount = 1
        self.active_position = False
        self.shares = 0
        self.money = money
        self.balance = money
        self.last_price = 0

        self.data = data
        self.time = 0
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-inf, high=inf, shape=(1,), dtype=np.float32
        )

    def _get_obs(self):
        curr_observation = self.data.item(self.time)
        curr_close = curr_observation.value("close")
        return np.array([curr_close])

    def reset(self):
        self.time = 0
        self.shares = 0
        self.balance = self.money
        self.active_position = False
        self.last_price = 0
        return self._get_obs()

    def step(self, action: int):
        curr_observation = self.data.item(self.time)
        curr_close = curr_observation.value("close")
        reward = 0

        if self.active_position == False:
            if action == self.BUY_ACTION:
                self.balance -= curr_close * self.amount
                self.shares += self.amount
                self.last_price = curr_close
                self.active_position = True

            if action == self.SELL_ACTION:
                self.balance -= curr_close * self.amount
                self.shares -= self.amount
                self.last_price = curr_close
                self.active_position = True
        else:
            if action == self.BUY_ACTION and self.shares < 0:
                reward = (-self.shares) * self.last_price - curr_close * self.shares
                self.shares = 0
                self.active_position = False
            if action == self.SELL_ACTION and self.shares > 0:
                reward = curr_close * self.shares - self.shares * self.last_price  
                self.shares = 0
                self.active_position = False

                
        self.time += 1
        info = {}
        done = self.data.has_next(self.time)
        return self._get_obs(), reward, done, info

    def render(self):
        pass

