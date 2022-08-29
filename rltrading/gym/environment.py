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

    def __init__(self: "Environment", shares: int, money: float, data: Data, lookback: int):
        self.data = data
        self.amount = 1
        self.money = money
        self.lookback = lookback

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-inf, high=inf, shape=(lookback,), dtype=np.float32
        )
        self.reset()

    def reset(self):
        self.time = self.lookback - 1
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
        done = not self.data.has_next(self.time)
        return self._get_obs(), reward, done, self._get_info()

    def render(self):
        pass

    def _get_obs(self):
        obs = []
        for i in range(self.lookback):
            curr_observation = self.data.item(self.time - (self.lookback - 1 + i))
            curr_close = curr_observation.value("close")
            obs.append(curr_close)

        return np.array(obs)

    def _get_info(self):
        return {}


