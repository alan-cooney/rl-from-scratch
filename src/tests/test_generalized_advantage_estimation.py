from ..generalized_advantage_estimation import generalized_advantage_estimates
import torch


class TestGeneralizedAdvantageEstimates:
    def test_correct_fixed_advantage(self):
        rewards = [3.]
        state_value_estimates = [torch.tensor(1.)]
        res = generalized_advantage_estimates(rewards, state_value_estimates)
        assert res == [2.]

    def test_two_timesteps(self):
        rewards = [3., 2.]
        state_value_estimates = [torch.tensor(2.), torch.tensor(1.)]
        gamma = 0.99
        lam = 0.95
        res = generalized_advantage_estimates(
            rewards,
            state_value_estimates,
            gamma,
            lam)

        # Manual calcs for the first state
        delta_0 = rewards[0] + gamma * \
            state_value_estimates[1] - state_value_estimates[0]
        delta_1 = rewards[1] + gamma * 0 - state_value_estimates[1]
        gae_0 = 1 * delta_0 + gamma * lam * delta_1
        gae_1 = delta_1

        assert res[0] == gae_0
        assert res[1] == gae_1
