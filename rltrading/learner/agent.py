import gym
import logging
import torch

from functools import partial
from stable_baselines3 import DQN, PPO, A2C

logger = logging.getLogger("root")


class Agent:
    def __init__(
        self: "Agent",
        gym_env: gym.Env,
        rl_model_id: str,
        policy_id: str,
        verbose: int,
        epochs: int,
        log_interval: int,
        save_path: str,
        device_name: str,
        specific_parameters: dict,
    ):
        logger.info("Initializing agent...")

        self.gym_env = gym_env
        self.epochs = epochs
        self.log_interval = log_interval
        self.save_path = save_path

        # initialize device if it is known
        devices = ["cpu", "cuda", "auto"]
        if device_name not in devices:
            msg = f"Unknown device name: {device_name}"
            logger.error(msg)
            raise ValueError(msg)
        device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device_name == "auto"
            else torch.device(device_name)
        )
        logger.info(f"Using device {device}")

        # initialize model if policy_id and model_id are known
        # one could improve the rl model aliasing by differentiating between
        # stable_baselines3.common.off_policy_algorithm.OffPolicyAlgorithm (DQN) and
        # from stable_baselines3.common.on_policy_algorithm.OnPolicyAlgorithm (PPO and A2C)
        # and only then regard the real model-specific parameters
        policy_ids = ["MlpPolicy", "CnnPolicy", "MultiInputPolicy"]
        if policy_id not in policy_ids:
            msg = f"Unknown policy id: {policy_id}"
            logger.error(msg)
            raise ValueError(msg)
        logger.info(f"Using policy {policy_id}")

        rl_model_aliases = {
            "PPO": partial(
                PPO
                # TODO add model specific parameters
            ),
            "A2C": partial(
                A2C
                # TODO add model specific parameters
            ),
            "DQN": partial(
                DQN,
                buffer_size=specific_parameters.get("buffer_size"),
                learning_starts=specific_parameters.get("learning_starts"),
                batch_size=specific_parameters.get("batch_size"),
                tau=specific_parameters.get("tau"),
                train_freq=specific_parameters.get("train_freq"),
                gradient_steps=specific_parameters.get("gradient_steps"),
            ),
        }

        if rl_model_id not in rl_model_aliases.keys():
            msg = f"Unknown RL-Model: {rl_model_id}"
            logger.error(msg)
            raise ValueError(msg)

        # initialize model with shared parameters of all models
        # the model-specific parameters are taken from the dictionary above using partial
        self.model = rl_model_aliases[rl_model_id](
            policy=policy_id,
            env=self.gym_env,
            verbose=verbose,
            device=device,
            learning_rate=specific_parameters.get("learning_rate"),
            gamma=specific_parameters.get("gamma"),
        )
        logger.info(f"Using model {rl_model_id}")

        logger.info("Successfully initialized agent")

    def learn(self):
        self.model.learn(total_timesteps=self.epochs, log_interval=self.log_interval)
        self.model.save(self.save_path)
        logger.info(f"Saved the model at {self.save_path}")

    def apply(self):
        obs = self.gym_env.reset()
        while True:
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.gym_env.step(action)
            self.gym_env.render()
            if done:
                obs = self.gym_env.reset()


# proving that the above should work (can be removed if the env is debugged and everything works)
# if __name__ == "__main__":
#     env = gym.make("CartPole-v0")
#
#     rl_models = {
#         "PPO": partial(
#             DQN,
#             policy="MlpPolicy",
#             env=env,
#             verbose=0,
#             device=torch.device("cpu")
#         ),
#         "A2C": partial(
#             DQN,
#             policy="MlpPolicy",
#             env=env,
#             verbose=0,
#             device=torch.device("cpu")
#         ),
#         "DQN": partial(
#             DQN,
#             learning_rate=0.0005,
#             buffer_size=50000,
#             learning_starts=50000,
#             batch_size=32,
#             tau=1.3,
#             gamma=0.99,
#             train_freq=4,
#             gradient_steps=1
#         )
#     }
#     model = rl_models["DQN"](policy="MlpPolicy",
#                              env=env,
#                              verbose=2,
#                              device=torch.device("cpu"))
#     print(model.tau)
#     print(model.verbose)
