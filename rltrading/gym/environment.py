import gym
import numpy as np

from rltrading.data.data import Data


class Environment(gym.Env):
    BUY_ACTION = 1
    SELL_ACTION = -1
    HOLD_ACTION = 0

    """
    Specify how many shares of a given stock and how much money the agent has
    """

    def __init__(self: "Environment", shares: int, money: float, data: Data):
        self.data = data
        self.state = [0, 0, 0, 0, 0, 0, 0]  # TODO
        self.money = np.asarray([money], dtype=float)
        self.action = np.asarray([], dtype=float)
        self.shares = shares
        self.portfolio_value = 0
        self.time = 0

    def reset(self):
        self.time = 0
        self.money = np.asarray([0])
        self.shares = 0
        self.portfolio_value = 0

    def step(self, action: float):
        curr_money = self.money[self.time]
        curr_observation = self.data.item(self.time)
        curr_close = curr_observation.value("close")

        # Buy
        if action > self.HOLD_ACTION:
            amount_of_shares = int(action * curr_money / curr_close)
            costs = curr_close * amount_of_shares
            self.money = np.append(self.money, (curr_money - costs))
            self.shares += amount_of_shares
            self.action = np.append(self.action, self.BUY_ACTION)

        # Sell
        if action < self.HOLD_ACTION:
            amount_of_shares = int(action * self.portfolio_value / curr_close) * -1
            gain = curr_close * amount_of_shares
            if amount_of_shares <= self.shares:
                self.money.loc[len(self.money.index)] = curr_money + gain
                self.money = np.append(self.money, (curr_money + gain))
                self.shares -= amount_of_shares
                self.action = np.append(self.action, self.SELL_ACTION)

        # Hold
        if action == self.HOLD_ACTION:
            self.action = np.append(self.action, self.HOLD_ACTION)

        self.time += 1
        next_money = self.money[self.time]
        next_observation = self.data.item(self.time)

        self.portfolio_value = self.shares * next_observation.value("close")

        next_money = self.money[self.time]
        next_observation = self.data.item(self.time)
        self.state = [
            *next_observation.all(),
            self.shares,
            self.portfolio_value,
            next_money,
            self.action,
        ]

        info = {"Round: ", self.time}
        self.reward = (self.money[self.time] - self.money[self.time - 1]) / self.money[
            self.time - 1
        ]

        done = self.data.has_next(self.time)
        return self.state, self.reward, done, info

    def render(self):
        pass

    def getObsStateSize(self):
        return len(self.state)
