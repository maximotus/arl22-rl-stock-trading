from typing import Optional, Union
from rltrading.data.data import Data

import gym
from gym.core import ObsType, ActType
from gym.spaces import Tuple

import pandas as pd


class Environment(gym.Env):

    """
    Specify how many shares of a given stock and how much money the agent has
    """
    def __init__(self, shares, money):

        Data = Data()
        Data.fetch()
        self.data = pd.read_csv("../Data/Data.csv")

        self.state = [0, 0, 0, 0, 0, 0, 0] #TODO
        self.money = pd.DataFrame({"Money": [money]})
        self.action = pd.DataFrame({"Action": []})
        self.shares = shares
        self.portfolio_value = 0
        self.time = 0

    def reset(self):
        self.time = 0
        self.money = pd.DataFrame({"Money": [0]})
        self.shares = 0
        self.portfolio_value = 0


    def step(self, action: float):
        #Buy
        if action > 0:
            amount_of_shares = int(action * self.money["Money"][self.time] / self.data["close"][self.time])
            costs = self.data["close"][self.time] * amount_of_shares
            self.money.loc[len(self.money.index)] = self.money["Money"][self.time] - costs
            self.shares += amount_of_shares
            self.action.loc[len(self.action.index)] = 1

        #Sell
        if action < 0:
            amount_of_shares = int(action * self.portfolio_value / self.data["close"][self.time]) * -1
            gain = self.data["close"][self.time] * amount_of_shares
            if amount_of_shares <= self.shares:
                self.money.loc[len(self.money.index)] = self.money["Money"][self.time] + gain
                self.shares -= amount_of_shares
                self.action.loc[len(self.action.index)] = -1

        #Hold
        if action == 0:
            self.action.loc[len(self.action.index)] = 0


        self.time += 1

        self.portfolio_value = self.shares * self.data["close"][self.time]

        self.state = [self.data["open"][self.time],
                    self.data["close"][self.time],
                    self.data["high"][self.time],
                    self.shares,
                    self.portfolio_value,
                    self.money["Money"][self.time],
                    self.action]

        if self.time == self.data.shape[0]-1:
            done = True
        else:
            done = False

        info = {"Round: ", self.time}

        self.reward = (self.money["Money"][self.time] - self.money["Money"][self.time - 1]) / self.money["Money"][self.time - 1]

        return self.state, self.reward, done, info


    def getObsStateSize(self):
        return len(self.state)