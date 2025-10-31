# Finite and Infinite-horizon Policies for Mountain Car and Pendulum

This project implements dynamic programming algorithms for reinforcement learning on continuous environments (MountainCarContinuous-v0 and Pendulum-v1) using discretized state spaces.

## Mountain Car: finite-horizon Dynamic Programming

### Overview

The real environments have continuous states (e.g., position ∈ [-1.2, 0.6], velocity ∈ [-0.07, 0.07]) and continuous actions (e.g., thrust ∈ [-1, 1]).So we have to approximate the continuous action space by picking a few representative actions. Dynamic programming only works if both states and actions are discrete. So we must approximate:In MountainCarContinuous-v0, the environment’s state is a 2-D vector:

In MountainCarContinuous-v0, the state is a 2-D vector: `s = [position, velocity]`. Both are continuous with ranges:

$$ position \in [−1.2, 0.6]; \quad velocity \in [−0.07, 0.07] $$

To use dynamic programming, you must approximate them with discrete bins.

```python

MDP_state_bounds = {        
    "vel_name": "velocity"
    "pos_bounds": (pos_min, pos_max),    }
    "pos_bins": 20,  # number of bins for position axis
    "vel_bounds": (vel_min, vel_max),
    "vel_bins": 24,  # number of bins for velocity axis
    "pos_label": "position",
    "vel_label": "velocity"

}
The position axis and velocity axis is split into 20 and 24 bins respectively.
```

This creates a grid of **20×24 = 480 discrete states**. Each cell represents all continuous states within those position–velocity ranges.

### Mountain Car Action Discretization

The continuous action space [-1, 1] is approximated with 9 representative actions.

```python
action_space = [-1.0, -0.5, -0.25, -0.22, 0.0, 0.22, 0.25, 0.5, 1.0]
```

### Algorithm: Finite-Horizon Dynamic Programmingand the optimal policy

The function ```def dynamic_programming_finite_horizon(MDP, action_space_env, gamma, max_iterations)``` implements finite-horizon dynamic programming (DP). It finds the optimal, time-dependent policy $\pi^*_t(s)$ and value function $V^*_t(s)$ for a given horizon $T$. The optimal policy is:

$$\pi_t(s) = \argmax_a\bigg[R(s,a) + \gamma \sum_{s'}P(s'|s,a)V_{t+1}(s') \bigg]$$

The value function is defined recursively backward in time as:
$$V_t(s) = \max_a \bigg[R(s,a) + \gamma \sum_{s'} P(s'|s,a)V_{t+1}(s')\bigg], \quad t=T-1,T-2,...,0$$

### How to Run Mountain Car

```python
# Build MDP, compute optimal policy, and evaluate
python mountain_car_dynamic_programming.py

# Or reuse a saved pickle
python mountain_car_dynamic_programming.py mc_mdp.pkl
```

**Note:** Building the MDP from scratch takes several minutes as it samples 100 transitions per state-action pair (480 states × 9 actions × 100 samples = 432,000 samples).

### Pendulum: Infinite-Horizon Value Iteration

In Pendulum-v1, the state is a 3-D observation vector: `obs = [cos(θ), sin(θ), θ_dot]`. We convert this to a 2-D state representation: `s = [θ, θ_dot]`
$$ \theta (angle) \in [-\pi, \pi]; \quad \theta_{dot} (\text{angular velocity}) \in [-8, 8] $$

#### State Discretization

```python
MDP_state_bounds = {
    "pos_bounds": (theta_min, theta_max),
    "pos_bins": 31,  # number of bins for angle
    "vel_bounds": (vel_min, vel_max),
    "vel_bins": 31,  # number of bins for angular velocity
    "pos_label": "angle",
    "vel_label": "angular_velocity"
}
```

This creates a grid of **31×31 = 961 discrete states**.

#### Pendulem Action Discretization

```python
action_space = [-2, -1.33, -0.67, 0, 0.67, 1.33, 2]
```

The continuous action space [-2, 2] is approximated with 7 representative actions.

#### Algorithm: Value Iteration

The function `value_iteration(MDP, action_space_env, gamma, eps, max_iterations)` implements infinite-horizon value iteration. It finds the stationary optimal policy $\pi^*(s)$ and value function $V^*(s)$.

The Bellman optimality equation:

$$V(s) = \max_a \bigg[R(s,a) + \gamma \sum_{s'} P(s'|s,a)V(s')\bigg]$$

Value iteration iteratively updates:

$$V_{k+1}(s) = \max_a \bigg[R(s,a) + \gamma \sum_{s'} P(s'|s,a)V_k(s')\bigg]$$

until convergence: $\max_s |V_{k+1}(s) - V_k(s)| < \epsilon$

The optimal policy is:

$$\pi^*(s) = \argmax_a\bigg[R(s,a) + \gamma \sum_{s'}P(s'|s,a)V^*(s') \bigg]$$

The optimal action depends only on state not time.

#### Alternative: Policy Iteration

The code also includes `policy_iteration(MDP, action_space_env, gamma, max_iterations)`, which alternates between:

1. **Policy Evaluation**: Compute $V^\pi$ for current policy $\pi$
2. **Policy Improvement**: Update policy to be greedy with respect to $V^\pi$

Policy iteration often converges in fewer iterations than value iteration but requires solving a linear system in each policy evaluation step.

#### How to Run Pendulum

```bash
# Build MDP, compute optimal policy with value iteration, and evaluate
python pendulum_dynamic_programming.py

# Or reuse a saved MDP pickle
python pendulum_dynamic_programming.py pendulum_mdp.pkl
```

**Note:** Building the MDP from scratch takes several minutes as it samples 100 transitions per state-action pair (961 states × 7 actions × 100 samples = 672,700 samples).

#### MDP Construction: `get_MDP()`

For each discrete state-action pair (s, a):

1. Sample `num_data_per_state` continuous states uniformly within the bin
2. For each sample, set the environment to that state and take action a
3. Observe the next state s' and reward r
4. Accumulate empirical transition counts P[s, a, s'] and rewards R[s, a]
5. Normalize to get transition probabilities and average rewards

The returned MDP is a dictionary containing:

- `P`: Transition probability matrix (S × A × S)
- `R`: Reward matrix (S × A)
- `pos_grid`, `vel_grid`: Bin boundaries for discretization
- `actions`: List of discrete actions (numpy array)
- `state_index()`: Function to map continuous (pos, vel) to discrete state index
- `state_index_bound()`: Function to map discrete state index to bin boundaries

### Finite-Horizon vs Infinite-Horizon

| Aspect | Finite-Horizon | Infinite-Horizon |
|--------|---------------|------------------|
| **Policy** | Time-dependent $\pi_t(s)$ | Stationary $\pi(s)$ |
| **Horizon** | Fixed $T$ steps | Unlimited |
| **Value Function** | $V_t(s)$ depends on time | $V(s)$ time-independent |
| **Discount** | Often $\gamma=1$ (no discount) | Usually $\gamma < 1$ (discount future) |
| **Use Case** | Fixed-length episodes | Continuing tasks |
| **Algorithm** | Backward induction | Iterative methods (VI, PI) |

<!-- ### Why Different Algorithms?

- **Mountain Car**: Episodic task with natural time limit (car must reach goal in 150 steps) → Finite-horizon DP
- **Pendulum**: Continuing task with no natural end → Infinite-horizon value iteration -->

<!-- ## Project Structure

```
code/
├── MDP.py                                  # Core DP algorithms and MDP construction
│   ├── dynamic_programming_finite_horizon() # Finite-horizon DP for Mountain Car
│   ├── value_iteration()                    # Infinite-horizon value iteration
│   ├── policy_iteration()                   # Infinite-horizon policy iteration
│   ├── get_MDP()                            # Build discrete MDP from continuous env
│   ├── get_P_R()                            # Extract transition/reward for policy
│   └── get_v()                              # Solve for value function
├── mountain_car_dynamic_programming.py      # Mountain Car experiment
├── pendulum_dynamic_programming.py          # Pendulum experiment
├── requirements.txt                         # Python dependencies
└── README.md                                
``` -->
