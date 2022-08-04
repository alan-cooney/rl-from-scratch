import gym
import numpy as np
from gym import Env
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.ppo.policies import MlpPolicy
from xvfbwrapper import Xvfb


def create_model(env: Env) -> BaseAlgorithm:
    """Create the model"""
    model = PPO(MlpPolicy, env, verbose=0)
    return model


def evaluate(model: BaseAlgorithm, num_episodes=100):
    """Evaluate a trained model

    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    env = model.get_env()
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)

    return mean_episode_reward


def record_video(env_id: str, model: BaseAlgorithm, video_length=500, prefix='', video_folder='.videos/'):
    """Record a video of the model in action"""

    # Start a fake graphics server, for rendering
    vdisplay = Xvfb()
    vdisplay.start()

    eval_env = DummyVecEnv([lambda: gym.make(env_id)])
    # Start the video at step=0 and record 500 steps
    eval_env = VecVideoRecorder(eval_env, video_folder=video_folder,
                                record_video_trigger=lambda step: step == 0, video_length=video_length,
                                name_prefix=prefix)

    obs = eval_env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs)
        obs, _, _, _ = eval_env.step(action)

    # Close the video recorder
    eval_env.close()

    # Stope the fake graphics server
    vdisplay.stop()


def main():
    """Train a model and record a video of it in action

    We train Cartpole, which obtains a reward of 1 for every timestep it stays
    up. Cartpole-v1 has an episode length of 500 (i.e. 500 is the max score).

    https://www.gymlibrary.ml/environments/classic_control/cart_pole/"""

    # Setup
    env = gym.make('CartPole-v1')
    model = create_model(env)
    total_training_timesteps = 25000

    # Evaluate at the start (acting randomly)
    mean_reward_start = evaluate(model)
    print("The mean reward at the start (acting randomly) is", mean_reward_start)
    print("Starting training for", total_training_timesteps,
          "total training timesteps")

    # Train
    model.learn(total_timesteps=total_training_timesteps)

    # Evaluate
    mean_reward_end = evaluate(model)
    print("The mean reward at the end of training is", mean_reward_end)
    record_video('CartPole-v1', model, video_length=500,
                 prefix='ppo2-cartpole')


if __name__ == "__main__":
    main()
