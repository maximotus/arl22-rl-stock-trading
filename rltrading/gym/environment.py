import gym
import logging
import matplotlib.pyplot as plt
import numpy as np

from enum import IntEnum
from gym.vector.utils import spaces
from numpy import inf
from rltrading.data.data import Data

logger = logging.getLogger("root")


class Positions(IntEnum):
    Short = 0
    Long = 1


class Actions(IntEnum):
    Sell = 0
    Buy = 1


class Environment(gym.Env):
    """
    Specify how many shares of a given stock and how much money the agent has
    """

    def __init__(
        self: "Environment",
        data: Data,
        window_size: int,
        norm_min: float = 130.01,
        norm_max: float = 182.84,
        enable_render: bool = False,
        scale_reward: int = 10000,
        use_time: bool = True,
    ):
        self.data = data

        self.enable_render = enable_render
        self.window_size = window_size
        self.scale_reward = scale_reward
        self._use_time = use_time

        self.action_space = spaces.Discrete(len(Actions))

        # Either way, time will be appended to data for plotting purposes.
        # However, if one explicitly specifies time as learnable parameter,
        # it does not have to be removed and thus there is no need to subtract
        # one from the observation space shape in dimension 1.
        # Moreover, since the active position is always added to the observation,
        # one has to be added in every case.
        dim_1 = (
            (self.data.shape[1]) if self._use_time else (self.data.shape[1] - 1)
        )
        dim_0 = window_size

        self.observation_space = spaces.Box(
            low=-inf,
            high=inf,
            shape=(dim_0, dim_1),
            dtype=np.float32,
        )
        logger.info(f"Using action space: {self.action_space}")
        logger.info(f"Using observation space of shape: {self.observation_space.shape}")

        self.norm_min = norm_min
        self.norm_max = norm_max

        self.reset()
    
    def __rescale(self, val: float) -> float:
        return val * (self.norm_max - self.norm_min) + self.norm_min

    def reset(self):
        # the initial time will be the window_size-th index plus one (so the window already fits; time starts with 1)
        self.time = self.window_size
        self.active_position = Positions.Short
        self._total_profit = 1.0
        self._total_reward = 0.0
        self.close_prices = dict(date=[], price=[])
        # avoid division by 0 if data is normalized
        self.last_trade_price = self.data.item(self.time - 1).value("close")
        self.done = False
        self._rendering = False
        self.current_info = None
        return self._get_obs()

    def step(self, action: int):
        self.time += 1
        curr_observation = self.data.item(self.time)

        step_reward = 0.0
        # logger.debug(f"Action choosen: {action}")
        if (self.active_position == Positions.Long) and (action == Actions.Sell.value):
            # avoid division by 0 if data is normalized
            curr_close = curr_observation.value("close")
            step_reward = (curr_close - self.last_trade_price) * self.scale_reward
            self.active_position = Positions.Short            
            self._total_profit *= self.__rescale(curr_close) / self.__rescale(self.last_trade_price)
            self.last_trade_price = curr_close

        if (self.active_position == Positions.Short) and (action == Actions.Buy.value):
            # avoid division by 0 if data is normalized
            curr_close = curr_observation.value("close")
            step_reward = (self.last_trade_price - curr_close) * self.scale_reward
            self.active_position = Positions.Long            
            self._total_profit *= self.__rescale(self.last_trade_price) / self.__rescale(curr_close)
            self.last_trade_price = curr_close

        self._total_reward += step_reward
        # logger.debug(f"Step Reward: {step_reward}")
        # logger.debug(f"Total Reward: {self._total_reward}")
        # logger.debug(f"Total Profit: {self._total_profit}")

        done = not self.data.has_next(self.time)

        self.current_info = self._get_info(curr_observation.to_dict())
        return (
            self._get_obs(),
            step_reward,
            done,
            self.current_info,
        )

    def render(self, mode='human', **kwargs):
        if not self.enable_render:
            return

        if not self._rendering:
            plt.ion()
            self._fig = plt.figure(figsize=(17, 7))
            self._fig.suptitle(f"Applying learned policy on data...")
            self._ax = self._fig.add_subplot()
            self._ax.set_title("Evolution of the close price...")
            x = np.linspace(0, len(self.data), len(self.data))
            y = np.zeros(len(self.data))
            (self._line1,) = self._ax.plot(x, y)
            self._rendering = True

        # rendering the evolution of the close price
        # TODO also render decisions as points (e.g. sell = red point, buy = green point)
        new_y = np.zeros(len(self.data))
        new_y[new_y == 0] = np.nan
        new_y[0 : len(self.close_prices["price"])] = self.close_prices["price"]
        self._ax.set_ylim(
            [
                min(self.close_prices["price"]) - 2,
                max(self.close_prices["price"]) + 2,
            ]
        )

        labels = [item.get_text() for item in self._ax.get_xticklabels()]
        labels[0 : len(self.close_prices["date"])] = self.close_prices["date"]
        self._ax.set_xticklabels(labels)

        self._line1.set_ydata(new_y)
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

        # TODO render the evolution of the step_reward
        # TODO render the evolution of the _total_reward
        # TODO render the evolution of the _total_profit

    def _get_obs(self):
        # apply window on the data to get the observation
        obs = self.data.batch(self.time - self.window_size + 1, self.time + 1)
        # remove timesteps from observations
        if not self._use_time:
            obs = obs.drop('time', axis=1)

        return np.array(obs)

    def _get_info(self, observation: dict) -> dict:
        return {
            "total_reward": self._total_reward,
            "total_profit": self._total_profit,
            "position": self.active_position,
            "observation": observation,
        }
