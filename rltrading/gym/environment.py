from enum import Enum

import gym
import numpy as np

from numpy import inf
from gym.vector.utils import spaces

from rltrading.data.data import Data

class Positions(Enum):
    Short = 0
    Long = 1

class Actions(Enum):
    Sell = 0
    Buy = 1

class Environment(gym.Env):

    """
    Specify how many shares of a given stock and how much money the agent has
    """

    def __init__(
        self: "Environment", shares: int, money: float, data: Data, window_size: int
    ):
        self.data = data
        self.amount = 1
        self.money = money
        self.window_size = window_size

        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(
            low=-inf, high=inf, shape=(window_size,), dtype=np.float32
        )
        self.reset()

    def reset(self):
        self.time = self.window_size - 1
        self.shares = 0
        self.balance = self.money
        self.active_position = Positions.Long
        self._total_profit = 1.0
        self.last_trade_price = self.data.item(self.time).value("close")
        self.done = False
        return self._get_obs()

    def step(self, action: int):
        curr_observation = self.data.item(self.time)
        curr_close = curr_observation.value("close")
        reward = 0

        if self.active_position == Positions.Long:
            if action == Actions.Sell.value:
                reward = self.last_trade_price - curr_close
                self.active_position = Positions.Short
                self.last_trade_price = curr_close
                quantity = self._total_profit / self.last_trade_price
                self._total_profit = quantity * curr_close

        if self.active_position == Positions.Short:
            if action == Actions.Buy.value:
                reward = curr_close - self.last_trade_price
                self.active_position = Positions.Long
                self.last_trade_price = curr_close
                quantity = self._total_profit * self.last_trade_price
                self._total_profit = quantity / curr_close

        # else:
        #     if action == self.BUY_ACTION and self.shares < 0:

        #     if action == self.SELL_ACTION and self.shares > 0:
        #         reward = curr_close * self.shares - self.shares * self.last_price
        #         self.shares = 0
        #         self.active_position = False

        self.time += 1
        done = not self.data.has_next(self.time)
        return self._get_obs(), reward, done, self._get_info()

    def render(self):
        pass

    def _get_obs(self):
        obs = []
        for i in range(self.window_size):
            curr_observation = self.data.item(self.time - (self.window_size - 1 + i))
            curr_close = curr_observation.value("close")
            obs.append(curr_close)
        return np.array(obs)

    def _get_info(self):
        return {}

