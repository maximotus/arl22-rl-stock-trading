from typing import List
import gym
import logging
import torch
import os

from functools import partial
from collections import namedtuple
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.logger import configure

from rltrading.learn.resultPlotting import plotResults

logger = logging.getLogger("root")

ResultMemory = namedtuple(
    "ResultMemory", field_names=["observation", "action", "state", "reward"]
)


class Agent:
    def __init__(
        self: "Agent",
        gym_env: gym.Env,
        epochs: int,
        log_interval: int,
        sb_logger: List[str],
        save_path: str,
        model_config: dict,
    ):
        logger.info("Initializing agent...")

        self.gym_env = gym_env
        self.epochs = epochs
        self.log_interval = log_interval
        self.model_save_path = os.path.join(save_path, "model")
        self.stats_save_path = os.path.join(save_path, "stats")
        self.sb_logger = configure(self.stats_save_path, sb_logger)

        # initialize device if it is known
        device_name = model_config.get("device")
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
        policy_id = model_config.get("policy")
        if policy_id not in policy_ids:
            msg = f"Unknown policy id: {policy_id}"
            logger.error(msg)
            raise ValueError(msg)
        logger.info(f"Using policy {policy_id}")

        rl_model_aliases = {
            "PPO": partial(
                PPO,
                n_steps=model_config.get("n_steps"),
                batch_size=model_config.get("batch_size"),
                n_epochs=model_config.get("n_epochs"),
                gae_lambda=model_config.get("gae_lambda"),
                clip_range=model_config.get("clip_range"),
                clip_range_vf=model_config.get("clip_range_vf"),
                normalize_advantage=model_config.get("normalize_advantage "),
                ent_coef=model_config.get("ent_coef"),
                vf_coef=model_config.get("vf_coef"),
                target_kl=model_config.get("target_kl"),
            ),
            "A2C": partial(
                A2C,
                n_steps=model_config.get("n_steps"),
                gae_lambda=model_config.get("gae_lambda"),
                ent_coef=model_config.get("ent_coef"),
                vf_coef=model_config.get("vf_coef"),
                rms_prop_eps=model_config.get("rms_prop_eps "),
                use_rms_prop=model_config.get("use_rms_prop"),
                normalize_advantage=model_config.get("normalize_advantage "),
            ),
            "DQN": partial(
                DQN,
                buffer_size=model_config.get("buffer_size"),
                learning_starts=model_config.get("learning_starts"),
                batch_size=model_config.get("batch_size"),
                tau=model_config.get("tau"),
                train_freq=model_config.get("train_freq"),
                gradient_steps=model_config.get("gradient_steps"),
                exploration_fraction=model_config.get("exploration_fraction"),
                exploration_initial_eps=model_config.get("exploration_initial_eps"),
                exploration_final_eps=model_config.get("exploration_final_eps"),
            ),
        }

        rl_model_id = model_config.get("name")
        if rl_model_id not in rl_model_aliases.keys():
            msg = f"Unknown RL-Model: {rl_model_id}"
            logger.error(msg)
            raise ValueError(msg)

        # initialize model with shared parameters of all models
        # the model-specific parameters are taken from the dictionary above using partial
        self.model = rl_model_aliases[rl_model_id](
            policy=policy_id,
            env=self.gym_env,
            device=device,
            verbose=model_config.get("verbose"),
            learning_rate=model_config.get("learning_rate"),
            gamma=model_config.get("gamma"),
            seed=model_config.get("seed"),
        )
        logger.info(f"Using model {rl_model_id}")

        logger.info("Successfully initialized agent")

    def learn(self):
        self.model.set_logger(self.sb_logger)
        self.model.learn(total_timesteps=self.epochs, log_interval=self.log_interval)
        self.model.save(self.model_save_path)
        logger.info(f"Saved the model at {self.model_save_path}")

    def apply(self):
        memory = []
        obs = self.gym_env.reset()
        while True:
            action, _states = self.model.predict(obs, deterministic=False)

            obs, reward, done, info = self.gym_env.step(action)
            memory.append(ResultMemory(obs, action, _states, reward))
            self.gym_env.render()
            if done:
                obs = self.gym_env.reset()
                break
        plotResults(result_memory=memory)


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
