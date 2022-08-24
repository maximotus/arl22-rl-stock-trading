import os
from rltrading import Data, Environment, Agent

#  Assuming already generated AAPL data.
curr_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(curr_dir, "data", "example")

aapl_data = Data()
aapl_data.load(symbol="AAPL", dir_path=data_dir)
aapl_data.reduce_attributes(["time", "open", "close", "low", "high"])

gym = Environment(shares=2, money=100000, data=aapl_data)
agent = Agent(gym, "DQN", "MlpPolicy", 0, 10, 5, "./result_dqn_trading")
agent.learn()
agent.apply()
