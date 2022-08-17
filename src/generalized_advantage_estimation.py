import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam, Optimizer
import numpy as np
import gym  # type: ignore


def create_model(number_observation_features: int, number_outputs: int) -> nn.Module:
    """Create a simple MLP model

    Args:
        number_observation_features (int): Number of features in the (flat)
        observation tensor
        number_actions (int): Number of outputs (actions in the case of the
        policy function, or 1 in the case of the value function)

    Returns:
        nn.Module: Simple MLP model
    """
    hidden_layer_features = 32

    return nn.Sequential(
        nn.Linear(in_features=number_observation_features,
                  out_features=hidden_layer_features),
        nn.ReLU(),
        nn.Linear(in_features=hidden_layer_features,
                  out_features=number_outputs),
    )


def get_policy(model: nn.Module, observation: np.ndarray) -> Categorical:
    """Get the policy from the model, for a specific observation

    Args:
        model (nn.Module): MLP model
        observation (np.ndarray): Environment observation

    Returns:
        Categorical: Multinomial distribution parameterized by model logits
    """
    observation_tensor = torch.as_tensor(observation)
    logits = model(observation_tensor)

    # Categorical will also normalize the logits for us
    return Categorical(logits=logits)


def get_action(policy: Categorical) -> tuple[int, torch.Tensor]:
    """Sample an action from the policy

    Args:
        policy (Categorical): Policy

    Returns:
        tuple[int, torch.Tensor]: Tuple of the action and it's log probability
    """
    action = policy.sample()  # Unit tensor

    # Converts to an int, as this is what Gym environments require
    action_int = int(action.item())

    # Calculate the log probability of the action, which is required for
    # calculating the loss later
    log_probability_action = policy.log_prob(action)

    return action_int, log_probability_action


def calculate_policy_fn_loss(
        epoch_log_probability_actions: torch.Tensor,
        epoch_action_advantage_estimators: torch.Tensor) -> torch.Tensor:
    """Calculate the policy function (actor) 'loss' required to get the policy gradient

    Formula for gradient at
    https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#deriving-the-simplest-policy-gradient

    Note that this isn't really loss - it's just the sum of the log probability
    of each action times the episode return. We calculate this so we can
    back-propagate to get the policy gradient.

    Args:
        epoch_log_probability_actions (torch.Tensor): Log probabilities of the
            actions taken
        epoch_action_advantage_estimators (torch.Tensor): Advantage estimators for each of these actions

    Returns:
        torch.Tensor: Pseudo-loss
    """
    return -(epoch_log_probability_actions * epoch_action_advantage_estimators).mean()


def calculate_value_fn_loss(
        epoch_action_returns: list[float],
        epoch_state_value_estimates: list[torch.Tensor]) -> torch.Tensor:
    """Calculate the value function (critic) loss

    Uses mean squared error

    Args:
        epoch_action_returns (list[float]): State-action returns
        epoch_state_value_estimates (list[torch.Tensor]): State value estimates

    Returns:
        float: Mean square error (MSE)
    """
    value_estimates = torch.stack(epoch_state_value_estimates)
    returns = torch.as_tensor(epoch_action_returns)
    return ((value_estimates - returns)**2).mean()


def generalized_advantage_estimates(
    rewards: list[float],
    state_value_estimates: list[torch.Tensor],
    gamma=0.99,
    lam=0.95
) -> list[float]:
    """Generalized advantage estimates for one episode

    Follows the approach in 'High-Dimensional Continuous Control Using
    Generalized Advantage Estimation' at https://arxiv.org/abs/1506.02438 .

    Args:
        rewards (list[int]): Episode rewards
        state_value_estimates (list[torch.Tensor]): State value estimates from
        the critic model
        gamma (float, optional): Gamma hyperparameter. Defaults to 0.99.
        lam (float, optional): Lambda hyperparameter. Defaults to 0.95.

    Returns:
        list[float]: GAEs (one per timestep of the episode)
    """

    episode_length = len(rewards)

    # Calculate the delta terms (page 4)
    # delta^V_t = r_t + gamma * V_(S_t+1) - V_(S_t)
    deltas: list[float] = []

    for timestep in range(episode_length):
        # The next state value estimate is zero for the terminal timestep
        next_value = state_value_estimates[timestep + 1].item() if len(
            state_value_estimates) > timestep + 1 else 0.

        delta = rewards[timestep] \
            + gamma * next_value \
            - state_value_estimates[timestep].item()

        deltas.append(delta)

    # Calculate the gaes (in reverse order)
    gaes: list[float] = []
    for timestep in reversed(range(episode_length)):
        prev_gae = gaes[0] if gaes else 0
        gae = prev_gae * gamma * lam + deltas[timestep]
        gaes.insert(0, gae)

    return gaes


def run_one_episode(
    env: gym.Env,
    policy_function_model: nn.Module,
    value_function_model: nn.Module
) -> tuple[list[float], list[torch.Tensor], list[torch.Tensor], list[float], float]:
    """Run one episode

    Args:
        env (gym.Env): Gym environment
        policy_function_model (nn.Module): Policy function model (actor)
        value_function_model (nn.Module): Value function model (critic)
        optimizer (Optimizer): Optimizer

    Returns:
        tuple[list[float], list[torch.Tensor], list[torch.Tensor], list[float],
        float]: Tuple of action returns, log probabilities of those actions, state
        value estimates, gaes, and the total episode return.
    """
    # Keep track of metrics
    rewards: list[float] = []
    state_value_estimates: list[torch.Tensor] = []
    log_probability_actions: list[torch.Tensor] = []

    # Reset the environment and get a fresh observation
    observation = env.reset()

    # Loop through timesteps (until done)
    while True:
        # Estimate the value of the state
        state_value_estimate = value_function_model(
            torch.as_tensor(observation))
        state_value_estimates.append(state_value_estimate)

        # Get the policy and act
        policy = get_policy(policy_function_model, observation)
        action, log_probability_action = get_action(policy)
        observation, reward, done, _ = env.step(action)

        # Update the metrics
        rewards.append(reward)
        log_probability_actions.append(log_probability_action)

        # Finish the action loop if this episode is done
        if done is True:
            break

    # Calculate the generalized advantage estimates (GAEs)
    gaes = generalized_advantage_estimates(rewards, state_value_estimates)

    # Calculate the action returns
    episode_return: float = sum(rewards)
    total_timesteps = len(rewards)
    action_returns: list[float] = [episode_return] * total_timesteps

    return action_returns, log_probability_actions, state_value_estimates, gaes, episode_return


def train_one_epoch(
    env: gym.Env,
    policy_function_model: nn.Module,
    policy_function_optimizer: Optimizer,
    value_function_model: nn.Module,
    value_function_optimizer: Optimizer
) -> float:
    """Train one epoch

    Args:
        env (gym.Env): Gym environment
        policy_function_model (nn.Module): Policy function model (actor)
        policy_function_optimizer (Optimizer): Policy function optimizer (actor)

    Returns:
        float: Average return from the epoch
    """
    # Keep track of metrics
    epoch_action_returns: list[float] = []
    epoch_advantage_estimators: list[float] = []
    epoch_log_probability_actions: list[torch.Tensor] = []
    epoch_state_value_estimates: list[torch.Tensor] = []
    epoch_episode_returns: list[float] = []

    # Run a batch of episodes
    for _ in range(200):
        action_returns, log_probability_actions, state_value_estimates, gaes, episode_return = run_one_episode(
            env,
            policy_function_model,
            value_function_model)
        epoch_action_returns += action_returns
        epoch_advantage_estimators += gaes
        epoch_log_probability_actions += log_probability_actions
        epoch_state_value_estimates += state_value_estimates
        epoch_episode_returns.append(episode_return)

    # Calculate the policy function (actor) loss
    policy_function_loss = calculate_policy_fn_loss(
        torch.stack(epoch_log_probability_actions),
        torch.as_tensor(epoch_advantage_estimators)
    )

    # Calculate the value function (critic) loss
    value_function_loss = calculate_value_fn_loss(
        epoch_action_returns, epoch_state_value_estimates)

    # Step the weights and biases
    value_function_loss.backward()
    value_function_optimizer.step()

    policy_function_loss.backward()
    policy_function_optimizer.step()

    value_function_optimizer.zero_grad()
    policy_function_optimizer.zero_grad()

    return float(np.mean(epoch_episode_returns))


def train(epochs=30) -> None:
    """Train a Vanilla Policy Gradient model on CartPole

    Args:
        episodes (int, optional): The number of episodes to run for. Defaults to 2000.
    """

    # Create the Gym Environment
    env = gym.make('CartPole-v0')

    # Use random seeds (to make experiments deterministic)
    torch.manual_seed(0)
    env.seed(0)

    # Create the MLP models
    # Note we create 2 separate models - one for the policy function and one for
    # the value function (i.e. actor and critic). It is common to instead create
    # 2 models with some shared layers for this purpose (as the lower layers are
    # learning similar things), but for simplicity here they're kept entirely
    # separate.
    number_observation_features = env.observation_space.shape[0]
    number_actions = env.action_space.n
    policy_function_model = create_model(
        number_observation_features, number_actions)
    value_function_model = create_model(number_observation_features, 1)

    # Create the optimizers
    policy_function_optimizer = Adam(policy_function_model.parameters(), 1e-2)
    value_function_optimizer = Adam(value_function_model.parameters(), 1e-2)

    # Loop for each epoch
    for epoch in range(epochs):
        average_return = train_one_epoch(
            env,
            policy_function_model,
            policy_function_optimizer,
            value_function_model,
            value_function_optimizer)

        print('epoch: %3d \t return: %.3f' % (epoch, average_return))


if __name__ == '__main__':
    train()
