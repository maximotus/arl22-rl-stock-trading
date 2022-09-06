import logging
import os

from rltrading import Data, Environment, Agent

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
        training_data_path = data_config_raw.get("train_path")
        testing_data_path = data_config_raw.get("test_path")
        attributes = data_config_raw.get("attributes")

        # assuming that the data already exists
        if not os.path.exists(os.path.join(training_data_path)):
            logger.error(
                f"The specified train_path does not exist: {training_data_path}"
            )
        if not os.path.exists(os.path.join(testing_data_path)):
            logger.error(f"The specified test_path does not exist: {testing_data_path}")

        training_data = Data()
        training_data.load(training_data_path)
        training_data.reduce_attributes(attributes)

        testing_data = Data()
        testing_data.load(testing_data_path)
        testing_data.reduce_attributes(attributes)

        logger.info(
            f"Using training data located in {training_data_path} with length={len(training_data)} and shape={training_data.shape},"
            + f" and using testing data located in {testing_data_path} with length={len(testing_data)} and shape={testing_data.shape}"
        )

        training_gym = Environment(data=training_data, window_size=window_size)
        testing_gym = Environment(
            data=testing_data, window_size=window_size, enable_render=enable_render
        )

        logger.info(
            f"Using gym environment with window_size={window_size} and enable_render={enable_render}"
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

        self.test_policy = config.get("test_policy")
        self.training_gym = training_gym
        self.testing_gym = testing_gym
        self.agent = agent

        logger.info("Successfully initialized training experiment")

    def conduct(self):
        logger.info("Starting training experiment...")
        logger.info("Learning the agent...")
        self.agent.learn()
        logger.info("Finished learning the agent")

        if self.test_policy:
            logger.info("Testing learned policy on test data...")
            self.agent.test()
            logger.info("Finished testing the learned policy on test data")
