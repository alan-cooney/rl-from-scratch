import gym
import numpy as np
import torch
import torch.nn as nn

"""REINFORCE implementation"""

EPISODES = 1


class ReinforceModel(nn.Module):
    """REINFORCE model

    Based on the approach in 'Simple Statistical Gradient-Following Algorithms
    for Connectionist Reinforcement Learning'. Note that the paper doesn't
    specify specific model parameters, so a fairly standard MLP is used.

    https://link.springer.com/content/pdf/10.1007/BF00992696
    """

    def __init__(self, num_action: int, num_input: int):
        """Initialise the model

        Args:
            num_action (int): Number of actions in the action space
            num_input (int): Number of inputs in the observation space
        """
        super(ReinforceModel, self).__init__()

        self.linear_stack = nn.Sequential(
            nn.Linear(num_input, 64),
            nn.ReLU(),
            nn.Linear(64, num_action),
            nn.Softmax(1)
        )

    def forward(self, observations_numpy: np.ndarray) -> tuple[int, torch.Tensor]:
        """Model forward pass

        Args:
            observations_numpy (np.ndarray): Numpy array of observations

        Returns:
            tuple[int, torch.Tensor]: Tuple containing the action index and the
            log probability of that action.
        """

        # Convert the gym observations (Numpy array) to a Pytorch Tensor
        # We also unsqueeze as Pytorch is designed to work with batches
        # (equivalent to saying we have a batch of size 1)
        observations = torch.tensor(observations_numpy).unsqueeze(0)

        # Action probabiltiies are a batch size * possible actions space (1*2) matrix
        action_probabilities = self.linear_stack(observations)

        # In pratice we're using a batch size of 1, so just return a single
        # action and it's probability
        signle_batch_action_probabilities = action_probabilities.squeeze(0)
        action = signle_batch_action_probabilities.multinomial(1).item()
        log_prob_action = torch.log(action_probabilities.squeeze(0))[action]
        return action, log_prob_action


def main():
    """Train a model"""

    # Setup
    env = gym.make("CartPole-v0")
    model = ReinforceModel(2, 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    all_episode_rewards = []

    # Loop for each episode
    for i in range(EPISODES):
        episode_number = i + 1

        done = False
        observation = env.reset()
        log_probabilities = []
        episode_rewards = []

        # Loop for each step, until done
        while not done:
            action, log_prob = model(observation)
            observation, reward, done, _info = env.step(action)
            log_probabilities.append(log_prob)
            episode_rewards.append(reward)

            if done:
                all_episode_rewards.append(np.sum(episode_rewards))
                if episode_number % 100 == 0:
                    print(
                        f"Episode {episode_number}: Score {np.sum(episode_rewards)}")

                break

        discounted_rewards = []
        for t in range(len(episode_rewards)):
            Gt = 0
            for reward in episode_rewards[t:]:
                Gt = Gt + reward
            discounted_rewards.append(Gt)

        discounted_rewards = torch.tensor(discounted_rewards)

        discounted_rewards = (
            discounted_rewards - torch.mean(discounted_rewards)) / (torch.std(discounted_rewards))

        print(log_probabilities)
        log_prob = torch.stack(log_probabilities)
        print(log_prob)

        policy_gradient = -log_prob*discounted_rewards

        model.zero_grad()
        policy_gradient.sum().backward()

        optimizer.step()


if __name__ == "__main__":
    main()
