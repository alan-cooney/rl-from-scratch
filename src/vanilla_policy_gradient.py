import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
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
        Categorical: Nultinomial distribution paramtised by model logits
    """
    observation_tensor = torch.as_tensor(observation, dtype=torch.float32)
    logits = model(observation_tensor)

    # Categorical will also normalise the logits for us
    return Categorical(logits=logits)


def get_action(policy: Categorical) -> int:
    action = policy.sample().item()
    return action


def compute_loss(policy: Categorical, action: torch.Tensor, weights: torch.Tensor):
    logp = policy.log_prob(action)
    return -(logp * weights).mean()


def train(epochs=30, batch_size=5000):
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

    # for training policy
    def train_one_epoch() -> float:
        epoch_timesteps_elapsed = 0

        # make some empty lists for logging.
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns

        # reset episode-specific variables
        observation = env.reset()       # first obs comes from starting distribution
        ep_rews = []            # list for rewards accrued throughout ep

        # Loop through timesteps
        while True:
            # Increment the timesteps
            epoch_timesteps_elapsed += 1

            # act in the environment
            policy = get_policy(model, observation)
            action = get_action(policy)
            observation, reward, done, _ = env.step(action)

            # save action, reward
            batch_acts.append(action)
            ep_rews.append(reward)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                observation = env.reset()
                ep_rews = []

                # end experience loop if we have enough of it
                if epoch_timesteps_elapsed > batch_size:
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(policy,
                                  torch.as_tensor(
                                      batch_acts, dtype=torch.int32),
                                  torch.as_tensor(
                                      batch_weights, dtype=torch.float32)
                                  )
        batch_loss.backward()
        optimizer.step()

        return np.mean(batch_rets)

    # training loop
    for epoch in range(epochs):
        average_return = train_one_epoch()
        print('epoch: %3d \t return: %.3f' % (epoch, average_return))


if __name__ == '__main__':
    train()
