import gym
import logging

from stable_baselines import PPO1, PPO2, DQN, ACKTR

logger = logging.getLogger("root")
rl_models = {"PPO1": PPO1, "PPO2": PPO2, "DQN": DQN, "ACKTR": ACKTR}


class Agent:
    def __int__(
        self, gym_id, rl_model_id, policy_id, verbose, epochs, log_interval, save_path
    ):
        self.env = gym.make(gym_id)
        self.model = rl_models[rl_model_id](policy_id, self.env, verbose=verbose)

        # use a model pretrained for #epochs timesteps
        self.model.learn(total_timesteps=epochs, log_interval=log_interval)

    def learn(self):
        logger.info("Start learning using model " + self.model)

        # TODO the following is just copied from https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
        # as an example how to use this function
        obs = self.env.reset()
        while True:
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            self.env.render()
            if done:
                obs = self.env.reset()
