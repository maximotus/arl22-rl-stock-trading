import logging
import os

from rltrading import Data, Environment, Agent

logger = logging.getLogger("root")


class Experiment:
    """
    Represents an experiment with its main components, namely a gym environment
    for training and testing each and a reinforcement learning agent.

    An instance of this class (and all inheriting classes) read(s) the necessary
    fields of a given configuration (dict) and initializes the both gym environments
    and the required data. The corresponding configurable attributes are described
    in ./config/template-*.yaml.
    """

    def __init__(self, config: dict):
        # access overall experiment parameters from configuration
        # access gym related parameters
        gym_environment_config = config.get("gym_environment")
        window_size = gym_environment_config.get("window_size")
        enable_render = gym_environment_config.get("enable_render")
        scale_reward = gym_environment_config.get("scale_reward")

        # access data related parameters
        data_config_raw = gym_environment_config.get("data")
        training_data_path = data_config_raw.get("train_path")
        testing_data_path = data_config_raw.get("test_path")
        attributes = data_config_raw.get("attributes")

        logger.info(
            f"Using the following features as learnable parameters: {attributes}"
        )

        # remember if config specifies time as learnable attribute and
        # append it, so it can be used for plotting
        time_key = "time"
        use_time = time_key in attributes
        if not use_time:
            attributes.append(time_key)
        logger.info(f"Using time as learnable parameter: {use_time}")

        # assuming that the data already exists
        if not os.path.exists(os.path.join(training_data_path)):
            logger.error(
                f"The specified train_path does not exist: {training_data_path}"
            )
        if not os.path.exists(os.path.join(testing_data_path)):
            logger.error(f"The specified test_path does not exist: {testing_data_path}")

        # initialize training data
        self.training_data = Data()
        self.training_data.load(training_data_path)
        self.training_data.reduce_attributes(attributes)

        # initialize testing data
        self.testing_data = Data()
        self.testing_data.load(testing_data_path)
        self.testing_data.reduce_attributes(attributes)

        logger.info(
            f"Using training data located in {training_data_path} with length={len(self.training_data)} and shape={self.training_data.shape},"
            + f" and using testing data located in {testing_data_path} with length={len(self.testing_data)} and shape={self.testing_data.shape}"
        )

        # initialize training gym environment
        self.training_gym = Environment(
            data=self.training_data,
            window_size=window_size,
            scale_reward=scale_reward,
            use_time=use_time,
        )

        # initialize testing gym environment
        self.testing_gym = Environment(
            data=self.testing_data,
            window_size=window_size,
            enable_render=enable_render,
            use_time=use_time,
        )

        # initialize overall experiment related parameters
        self.model_config = config.get("agent").get("model")
        self.save_path = config.get("experiment_path")

        logger.info(
            f"Using gym environment with window_size={window_size}, scale_reward={scale_reward} and enable_render={enable_render}"
        )

    def conduct(self):
        raise NotImplementedError


class TrainExperiment(Experiment):
    """
    Explicitly represents a training experiment using a reinforcement learning agent
    that is trained to learn a policy based on the given gym environment and data
    (from base class).

    An instance of this class reads the necessary fields of a given configuration (dict)
    and initializes the reinforcement learning agent that should learn a policy in the
    environment. The corresponding configurable attributes are described
    in ./config/template-*.yaml.
    """

    def __init__(self, config: dict):
        logger.info("Initializing training experiment...")
        super().__init__(config)

        agent_config = config.get("agent")
        episodes = agent_config.get("episodes")
        timesteps = episodes * len(self.training_data)
        log_interval = agent_config.get("log_interval")
        sb_logger = agent_config.get("sb_logger")

        agent = Agent(
            save_path=self.save_path,
            training_gym_env=self.training_gym,
            testing_gym_env=self.testing_gym,
            episodes=episodes,
            timesteps=timesteps,
            log_interval=log_interval,
            sb_logger=sb_logger,
            model_config=self.model_config,
        )

        self.test_policy = config.get("test_policy")
        self.agent = agent

        logger.info("Successfully initialized training experiment")

    def conduct(self):
        logger.info("Starting training experiment...")
        logger.info("Learning the agent...")
        self.agent.learn()
        logger.info("Finished learning the agent")

        if self.test_policy:
            logger.info("Testing learned policy on test and train data...")
            self.agent.test(["test", "train"])
            logger.info("Finished testing the learned policy on test and train data")


class EvalExperiment(Experiment):
    """
    Explicitly represents an evaluation experiment using an already trained
    reinforcement learning agent that applies its policy on the given
    gym environment and data (from base class).

    An instance of this class reads the necessary fields of a given configuration (dict)
    and initializes the reinforcement learning agent should apply a policy in the environment
    and collect some evaluation data. The corresponding configurable attributes are described
    in ./config/template-*.yaml.
    """

    def __init__(self, config: dict):
        logger.info("Initializing evaluation experiment...")
        super().__init__(config)

        self.agent = Agent(
            save_path=self.save_path,
            training_gym_env=self.training_gym,
            testing_gym_env=self.testing_gym,
            model_config=self.model_config,
        )

    def conduct(self):
        logger.info("Evaluating learned policy...")
        self.agent.eval()
        logger.info("Finished evaluating the learned policy")
