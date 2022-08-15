# RL Algorithms from Scratch

Implementations of various RL algorithms from scratch.

## Included Algorithms

### Vanilla Policy Gradient (VPG)

The simplest form of a policy gradient algorithm, based on the [OpenAI Spinning
Up](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
documentation. A write-up of my approach here is included on [my
blog](https://medium.com/@alancooney/vanilla-policy-gradient-from-scratch-3c9ebb4de441).

### Generalized Advantage Estimation (GAE)

Implements the generalised advantage estimation from 
[High-Dimensional Continuous Control Using Generalized Advantage
Estimation](https://arxiv.org/abs/1506.02438) by Schulman et al. To do this I
have converted the VPG model into an actor-critic model with a separately
trained value function. 

Note that this differs from the original paper, which
applied GAE to a [Trust Region Policy
Optimization](https://arxiv.org/abs/1502.05477) (TRPO) algorithm instead. The
choice was made to do this as TRPO adds unnecessary complexity here and has
fallen out of favour with the advent of A2C and PPO.

## Benchmark

This repository also includes a benchmark [Proximal Policy
Optimization](https://arxiv.org/abs/1707.06347) (PPO) algorithm
from [Stable Baselines
3](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html). This is
useful for comparing performance.
