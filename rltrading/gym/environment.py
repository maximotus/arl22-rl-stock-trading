from enum import Enum

import logging

import gym
import numpy as np

from numpy import inf
from gym.vector.utils import spaces

from rltrading.data.data import Data

logger = logging.getLogger("root")


class Positions(Enum):
    Short = 0
    Long = 1


class Actions(Enum):
    Sell = 0
    Buy = 1
    Hold = 2


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
            low=-inf,
            high=inf,
            shape=(window_size, self.data.shape[1] + 2),
            dtype=np.float32,
        )
        self.reset()

    def reset(self):
        # the initial time will be the window_size-th index plus one (so the window already fits; time starts with 1)
        self.time = self.window_size

        self.shares = 0
        self.balance = self.money
        self.active_position = Positions.Long
        self._total_profit = 1.0
        self._total_reward = 0.0
        self.last_trade_price = self.data.item(self.time).value("close")
        self.done = False
        return self._get_obs()

    def step(self, action: int):
        curr_observation = self.data.item(self.time)
        curr_close = curr_observation.value("close")
        step_reward = 0.0
        logger.debug(f"Action choosen: {action}")
        if self.active_position == Positions.Long:
            if action == Actions.Sell.value:
                step_reward = self.last_trade_price - curr_close
                self.active_position = Positions.Short
                self.last_trade_price = curr_close
                quantity = self._total_profit / self.last_trade_price
                self._total_profit = quantity * curr_close
            elif action == Actions.Hold.value:
                step_reward = self.last_trade_price - curr_close
                quantity = self._total_profit / self.last_trade_price
                self._total_profit = quantity * curr_close

        if self.active_position == Positions.Short:
            if action == Actions.Buy.value:
                step_reward = curr_close - self.last_trade_price
                self.active_position = Positions.Long
                self.last_trade_price = curr_close
                quantity = self._total_profit * self.last_trade_price
                self._total_profit = quantity / curr_close
            elif action == Actions.Hold.value:
                step_reward = self.last_trade_price - curr_close
                quantity = self._total_profit * self.last_trade_price
                self._total_profit = quantity / curr_close

        self._total_reward += step_reward
        logger.debug(f"Step Reward: {step_reward}")
        logger.debug(f"Total Reward: {self._total_reward}")
        logger.debug(f"Total Profit: {self._total_profit}")

        # else:
        #     if action == self.BUY_ACTION and self.shares < 0:

        #     if action == self.SELL_ACTION and self.shares > 0:
        #         reward = curr_close * self.shares - self.shares * self.last_price
        #         self.shares = 0
        #         self.active_position = False
        self.time += 1
        done = not self.data.has_next(self.time)

        return self._get_obs(), step_reward, done, self._get_info()

    def render(self, **kwargs):
        pass

    def _get_obs(self):
        obs = []
        # apply window on the data to get the observation
        for i in range(self.window_size):
            # apply window from the very left to the very right relative to the current time using i
            window_index = self.time - (self.window_size + i)

            # add fix data to observation
            curr_observation = self.data.item(window_index)

            # add dynamic data to observation
            curr_observation = curr_observation.all()
            curr_observation.extend([self._total_profit, self._total_reward])

            obs.append(curr_observation)
        obs = np.array(obs)
        return obs

    def _get_info(self):
        return dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            position=self.active_position,
        )
