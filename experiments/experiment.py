import logging
import os

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
        shares = gym_environment_config.get("shares")
        money = gym_environment_config.get("money")

        data_config_raw = gym_environment_config.get("data")
        symbol = data_config_raw.get("symbol")
        data_path = data_config_raw.get("path")
        attributes = data_config_raw.get("attributes")
        n_peers = data_config_raw.get("n_peers")
        social_lookback = timedelta(days=data_config_raw.get("social_lookback"))
        start = parse(data_config_raw.get("start"))
        end = parse(data_config_raw.get("end"))
        finnhub_api_key = config.get("finnhub_api_key")

        data_config = Config(
            symbol=symbol,
            from_=start,
            to=end,
            lookback=social_lookback,
            finnhub_api_key=finnhub_api_key,
        )

        # fetch data if not already exists
        # TODO @jflxb if this will already be done in data.py, it can be removed here
        # TODO use n_peers properly
        data = Data()
        if not os.path.exists(os.path.join(data_path, f"{symbol}.csv")):
            logger.info("Data does not exist. Fetching data...")
            data.fetch(config=data_config, dir_path=data_path)
        else:
            logger.info("Data already exists. Loading data...")
            data.load(symbol=symbol, dir_path=data_path)
        data.reduce_attributes(attributes)

        logger.info(
            f"Using data of symbol {symbol} with length={len(data)} and shape={data.shape}"
        )

        gym = Environment(shares=shares, money=money, data=data, lookback=6)

        logger.info(f"Using gym environment with #shares={shares} and #money={money}")

        agent_config = config.get("agent")
        epochs = agent_config.get("epochs")
        log_interval = agent_config.get("log_interval")
        sb_logger = agent_config.get("sb_logger")
        save_path = config.get("experiment_path")

        model_config = agent_config.get("model")

        agent = Agent(
            gym_env=gym,
            epochs=epochs,
            log_interval=log_interval,
            sb_logger=sb_logger,
            save_path=save_path,
            model_config=model_config,
        )

        self.gym = gym
        self.agent = agent

        logger.info("Successfully initialized training experiment")

    def conduct(self):
        logger.info("Starting training experiment...")
        logger.info("Learning the agent...")
        self.agent.learn()
        logger.info("Finished learning the agent")
        logger.info("Applying learned policy (endless)...")
        self.agent.apply()
