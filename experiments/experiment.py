from cgitb import enable
import logging
import os
import gym
import gym_anytrading
import pandas as pd

from datetime import timedelta
from dateutil.parser import parse
from rltrading import Data, Environment, Agent
from rltrading.data.data import Config

logger = logging.getLogger("root")


class TrainExperiment:
    """
    Represents a training experiment with its main components, namely a gym environment and agent.

    An instance of this class reads the necessary fields of a given configuration (dict) and initializes
    the gym environment, the data and the machine learning agent that should learn a policy in the environment.
    The corresponding configurable attributes are described in ./config/template-*.yaml.
    """

    def __init__(self, config: dict):
        logger.info("Initializing training experiment...")

        gym_environment_config = config.get("gym_environment")
        window_size = gym_environment_config.get("window_size")
        enable_render = gym_environment_config.get("enable_render")

        data_config_raw = gym_environment_config.get("data")
        symbol = data_config_raw.get("symbol")
        data_path = data_config_raw.get("path")
        training_data_path = data_path + "\\training"
        testing_data_path = data_path + "\\testing"
        attributes = data_config_raw.get("attributes")
        n_peers = data_config_raw.get("n_peers")
        social_lookback = timedelta(days=data_config_raw.get("social_lookback"))
        training_start = parse(data_config_raw.get("training_start"))
        training_end = parse(data_config_raw.get("training_end"))
        testing_start = parse(data_config_raw.get("testing_start"))
        testing_end = parse(data_config_raw.get("testing_end"))
 
        finnhub_api_key = config.get("finnhub_api_key")

        # TODO maybe allow the option to specify a timeframe for ohlcv candles like d1, h1, m30, ... not only m1
        training_data_config = Config(
            symbol=symbol,
            from_=training_start,
            to=training_end,
            lookback=social_lookback,
            finnhub_api_key=finnhub_api_key,
        )

        testing_data_config = Config(
            symbol=symbol,
            from_=testing_start,
            to=testing_end,
            lookback=social_lookback,
            finnhub_api_key=finnhub_api_key,
        )

        # fetch data if not already exists
        # TODO @jflxb if this will already be done in data.py, it can be removed here
        # TODO use n_peers properly
        training_data = Data()
        if not os.path.exists(os.path.join(training_data_path, f"{symbol}.csv")):
            logger.info("Training data does not exist. Fetching data...")
            training_data.fetch(config=training_data_config, dir_path=training_data_path)
        else:
            logger.info("Training data already exists. Loading data...")
            training_data.load(symbol=symbol, dir_path=training_data_path)

        if attributes:
            training_data.reduce_attributes(attributes)

        testing_data = Data()
        if not os.path.exists(os.path.join(testing_data_path, f"{symbol}.csv")):
            logger.info("Testing data does not exist. Fetching data...")
            testing_data.fetch(config=testing_data_config, dir_path=testing_data_path)
        else:
            logger.info("Testing data already exists. Loading data...")
            testing_data.load(symbol=symbol, dir_path=testing_data_path)

        if attributes:
            testing_data.reduce_attributes(attributes)


        logger.info(
            f"Using training data of symbol {symbol} with length={len(training_data)} and shape={training_data.shape}," +
            f"and using testing data of symbol {symbol} with length={len(testing_data)} and shape={testing_data.shape}"
        )

        training_gym = gym.make('forex-v0', df=training_data.getDataframe(), frame_bound=(window_size, training_data.shape[0]))
        testing_gym = gym.make('forex-v0', df=testing_data.getDataframe(), frame_bound=(window_size, testing_data.shape[0]))
        #training_gym = Environment(data=training_data, window_size=window_size, enable_render=enable_render)
        #testing_gym = Environment(data=testing_data, window_size=window_size, enable_render=enable_render)


        logger.info(
            f"Using gym environment with #windowsize={window_size} and #enable_render={enable_render}"
        )

        agent_config = config.get("agent")
        timesteps = agent_config.get("episodes") * len(training_data)
        log_interval = agent_config.get("log_interval")
        sb_logger = agent_config.get("sb_logger")
        save_path = config.get("experiment_path")

        model_config = agent_config.get("model")

        agent = Agent(
            training_gym_env=training_gym,
            testing_gym_env=testing_gym,
            timesteps=timesteps,
            log_interval=log_interval,
            sb_logger=sb_logger,
            save_path=save_path,
            model_config=model_config,
        )

        self.training_gym = training_gym
        self.testing_gym = testing_gym
        self.agent = agent

        logger.info("Successfully initialized training experiment")

    def conduct(self):
        logger.info("Starting training experiment...")
        logger.info("Learning the agent...")
        self.agent.learn()
        logger.info("Finished learning the agent")
        logger.info("Applying learned policy (endless)...")
        self.agent.apply()
