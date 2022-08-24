import gym
import logging

from stable_baselines3 import DQN, PPO, A2C

logger = logging.getLogger("root")
rl_models = {"PPO": PPO, "DQN": DQN, "A2C": A2C}
policy_ids = ["MlpPolicy", "CnnPolicy", "MultiInputPolicy"]


class Agent:
    def __init__(
        self: "Agent",
        gym_env: gym.Env,
        rl_model_id,
        policy_id,
        verbose,
        epochs,
        log_interval,
        save_path,
    ):
        # validity checking
        if rl_model_id not in rl_models.keys():
            raise ValueError(f"Unknown RL-Model: {rl_model_id}")
        if policy_id not in policy_ids:
            raise ValueError(f"Unknown policy id: {policy_id}")

        # initialization
        self.gym_env = gym_env
        self.model = rl_models[rl_model_id](policy_id, self.gym_env, verbose=verbose)
        self.epochs = epochs
        self.log_interval = log_interval
        self.save_path = save_path

    def learn(self):
        self.model.learn(total_timesteps=self.epochs, log_interval=self.log_interval)
        self.model.save(self.save_path)

    def apply(self):
        obs = self.gym_env.reset()
        while True:
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.gym_env.step(action)
            self.gym_env.render()
            if done:
                obs = self.gym_env.reset()


# for testing
if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    a = Agent(env, "DQN", "MlpPolicy", 0, 1000, 100, "./result_dqn_cartpole")
    a.learn()
    a.apply()
