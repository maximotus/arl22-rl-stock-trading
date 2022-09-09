from typing import List
import gym
import logging
import torch
import os

from functools import partial
from collections import namedtuple
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback

from rltrading.learn.result_handler import plot_result, save_result

logger = logging.getLogger("root")

ResultMemory = namedtuple(
    "ResultMemory", field_names=["observation", "action", "state", "reward", "info"]
)


class Agent:
    def __init__(
        self: "Agent",
        save_path: str,
        training_gym_env: gym.Env,
        testing_gym_env: gym.Env,
        save_model_interval: int = -1,
        episodes: int = -1,
        timesteps: int = -1,
        log_interval: int = 5,
        sb_logger: List[str] = None,
        model_config: dict = None,
    ):
        logger.info("Initializing agent...")

        self.training_gym_env = training_gym_env
        self.testing_gym_env = testing_gym_env
        self.episodes = episodes
        self.timesteps = timesteps
        self.log_interval = log_interval
        self.save_model_interval = save_model_interval
        self.predict_deterministic = model_config.get("predict_deterministic")
        self.model_save_path = os.path.join(save_path, "model")
        self.stats_save_path = os.path.join(save_path, "stats")
        self.sb_logger = (
            configure(self.stats_save_path, sb_logger) if sb_logger else None
        )
        best_save_path = os.path.join(self.model_save_path, "best")
        self.eval_callback = EvalCallback(
            self.testing_gym_env,
            best_model_save_path=best_save_path,
            log_path=self.model_save_path,
            eval_freq=(self.timesteps / self.episodes) * self.save_model_interval,
            deterministic=self.predict_deterministic,
            render=False,
        )

        # initialize device if it is known
        device_name = model_config.get("device")
        devices = ["cpu", "cuda", "auto"]
        if device_name not in devices:
            msg = f"Unknown device name: {device_name}"
            logger.error(msg)
            raise ValueError(msg)
        device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device_name in ["auto", "cuda"]
            else torch.device(device_name)
        )
        if device_name == "cuda" and not torch.cuda.is_available():
            logger.warning(
                f"Specified device cuda but cuda is not available! "
                f"torch.cuda.is_available()=={torch.cuda.is_available()}"
            )
        logger.info(f"Using device {device}")

        # check compatibility of specified rl_model_id
        rl_model_id = model_config.get("name")
        rl_models = ["PPO", "DQN", "A2C"]
        if rl_model_id not in rl_models:
            msg = f"Unknown RL-Model: {rl_model_id}"
            logger.error(msg)
            raise ValueError(msg)

        # initialize model depending on whether there is a pretrained one specified
        model_path = model_config.get("pretrained_path")
        self.model = None
        if model_path is not None:
            self._init_pretrained_model(rl_model_id, model_path, device)
        else:
            self._init_new_model(rl_model_id, model_config, device)
        assert self.model
        logger.info(f"Using model {rl_model_id}")

        logger.info("Successfully initialized agent")

    def _init_pretrained_model(self, rl_model_id, model_path, device):
        rl_model_aliases = {"PPO": PPO, "DQN": DQN, "A2C": A2C}

        self.model = rl_model_aliases[rl_model_id].load(path=model_path, device=device)

    def _init_new_model(self, rl_model_id, model_config, device):
        # initialize model if policy_id is known
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
                rms_prop_eps=model_config.get("rms_prop_eps"),
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

        # initialize model with shared parameters of all models
        # the model-specific parameters are taken from the dictionary above using partial
        self.model = rl_model_aliases[rl_model_id](
            policy=policy_id,
            env=self.training_gym_env,
            device=device,
            verbose=model_config.get("verbose"),
            learning_rate=model_config.get("learning_rate"),
            gamma=model_config.get("gamma"),
            seed=model_config.get("seed"),
        )

    def learn(self):
        self.model.set_logger(self.sb_logger)
        self.model.learn(total_timesteps=self.timesteps, log_interval=self.log_interval, callback=self.eval_callback)
        #self.model.save(self.model_save_path)
        logger.info(f"Saved the models at {self.model_save_path}")

    def test(self, envs: List[str]):
        env_aliases = {"test": self.testing_gym_env, "train": self.training_gym_env}

        for env in envs:
            logger.info(f"Testing on {env} gym environment")
            memory = []
            obs = env_aliases[env].reset()
            while True:
                action, _states = self.model.predict(
                    obs, deterministic=self.predict_deterministic
                )
                obs, reward, done, info = env_aliases[env].step(action)
                memory.append(ResultMemory(obs.tolist(), action, _states, reward, info))
                env_aliases[env].render()
                if done:
                    _ = env_aliases[env].reset()
                    break
            save_path = os.path.join(self.stats_save_path, f"{env}-env")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                print("Created directory", save_path)
            plot_result(result_memory=memory, save_path=save_path)
            save_result(memory, save_path)
            logger.info(f"Saved the results at {save_path}")

    def eval(self):
        self.test(envs=["test", "train"])
        # TODO do some more evaluation
        # e.g. compare with heuristics
