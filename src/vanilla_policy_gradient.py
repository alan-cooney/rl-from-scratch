import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam, Optimizer
import numpy as np
import gym


def create_model(number_observation_features: int, number_actions: int) -> nn.Module:
    """Create the MLP model

    Args:
        number_observation_features (int): Number of features in the (flat)
        observation tensor
        number_actions (int): Number of actions

    Returns:
        nn.Module: Simple MLP model
    """
    hidden_layer_features = 32

    return nn.Sequential(
        nn.Linear(in_features=number_observation_features,
                  out_features=hidden_layer_features),
        nn.ReLU(),
        nn.Linear(in_features=hidden_layer_features,
                  out_features=number_actions),
    )


def get_policy(model: nn.Module, observation: np.ndarray) -> Categorical:
    """Get the policy from the model, for a specific observation

    Args:
        model (nn.Module): MLP model
        observation (np.ndarray): Environment observation

    Returns:
        Categorical: Multinomial distribution paramtised by model logits
    """
    observation_tensor = torch.as_tensor(observation, dtype=torch.float32)
    logits = model(observation_tensor)

    # Categorical will also normalise the logits for us
    return Categorical(logits=logits)


def get_action(policy: Categorical) -> tuple[int, float]:
    """Sample an action from the policy

    Args:
        policy (Categorical): Policy

    Returns:
        tuple[int, float]: Tuple of the action and it's log probability
    """
    action = policy.sample()  # Unit tensor

    # Converts to an int, as this is what Gym environments require
    action_int = action.item()

    # Calculate the log probability of the action, which is required for
    # calculating the loss later
    log_probability_action = policy.log_prob(action)

    return action_int, log_probability_action


def calculate_loss(epoch_log_probability_actions: torch.Tensor, epoch_action_rewards: torch.Tensor) -> float:
    """Calculate the 'loss' required to get the policy gradient

    Formula for gradient at
    https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#deriving-the-simplest-policy-gradient

    Note that this isn't really loss - it's just the sum of the log probability
    of each action times the episode return. We calculate this so we can
    back-propogate to get the policy gradient.

    Args:
        epoch_log_probability_actions (torch.Tensor): Log probabilities of the
        actions taken
        epoch_action_rewards (torch.Tensor): Rewards for each of these actions

    Returns:
        float: Pseudo-loss
    """
    return -(epoch_log_probability_actions * epoch_action_rewards).mean()


def train_one_epoch(env: gym.Env, model: nn.Module, optimizer: Optimizer, episodes=100, max_timesteps=200) -> float:

    # Returns from each episode (to keep track of progress)
    epoch_returns: list[int] = []

    # Actions, log probabilities and rewards per step (for calculating loss)
    epoch_log_probability_actions = []
    epoch_action_rewards = []

    # Loop through episodes
    for _ in range(episodes):

        # Running total of this episode's rewards
        episode_reward: int = 0

        # Reset the environment and get a fresh observation
        observation = env.reset()

        # Loop through timesteps until the epsiode is done (or the max is hit)
        for timestep in range(max_timesteps):
            # Get the policy and act
            policy = get_policy(model, observation)
            action, log_probability_action = get_action(policy)
            observation, reward, done, _ = env.step(action)

            # Increment the episode rewards
            episode_reward += reward

            # Add epoch action log probabilities
            epoch_log_probability_actions.append(log_probability_action)

            # Finish the action loop if this episode is done
            if done == True:
                # Add one reward per timestep
                epoch_action_rewards.append(episode_reward)
                for _ in range(timestep):
                    epoch_action_rewards.append(episode_reward)

                break

        # Increment the epoch returns
        epoch_returns.append(episode_reward)

    # Calculate the policy gradient, and use it to step the weights & biases
    epoch_loss = calculate_loss(torch.stack(
        epoch_log_probability_actions),
        torch.as_tensor(
        epoch_action_rewards, dtype=torch.float32)
    )

    epoch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return np.mean(epoch_returns)


def train(epochs=10):
    # Create the Gym Environment
    env = gym.make('CartPole-v0')

    # Use random seeds (to make experiments deterministic)
    torch.manual_seed(0)
    env.seed(0)

    # Create the MLP model
    number_observation_features = env.observation_space.shape[0]
    number_actions = env.action_space.n
    model = create_model(number_observation_features, number_actions)

    # Create the optimizer
    optimizer = Adam(model.parameters(), 1e-2)

    # Loop for each epoch
    for epoch in range(epochs):
        average_return = train_one_epoch(env, model, optimizer)
        print('epoch: %3d \t return: %.3f' % (epoch, average_return))


if __name__ == '__main__':
    train()
