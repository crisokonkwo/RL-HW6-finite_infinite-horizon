import gymnasium as gym
import random
import numpy as np
from functools import partial


# functions for converting between (pos, vel) and state index.
def _state_index_from_grids(pos, vel, pos_grid, vel_grid):
    pos_idx = np.digitize([pos], pos_grid[1:-1])[0]
    vel_idx = np.digitize([vel], vel_grid[1:-1])[0]
    
    return pos_idx * (len(vel_grid) - 1) + vel_idx


def _state_index_bound_from_grids(s, pos_grid, vel_grid):
    pos_bins = len(pos_grid) - 1
    vel_bins = len(vel_grid) - 1
    
    pos_bin_idx = s // vel_bins
    vel_bin_idx = s % vel_bins
        
    (pos_low, pos_high) = (pos_grid[pos_bin_idx], pos_grid[pos_bin_idx + 1])
    (vel_low, vel_high) = (vel_grid[vel_bin_idx], vel_grid[vel_bin_idx + 1])
    
    return (pos_low, pos_high), (vel_low, vel_high)
    

# the action_space_env variable is a list of the actions that the environment
# accepts (e.g., [1] instead of 1)
def dynamic_programming_finite_horizon(MDP, action_space_env, gamma, max_iterations):
    P = MDP["P"]
    R = MDP["R"]
    S, A = R.shape
    T = max_iterations

    V = np.zeros((T + 1, S), dtype=np.float64)
    pi_t = np.zeros((T, S), dtype=np.int32)

    for t in range(T - 1, -1, -1):
        V_next = V[t + 1]
        Q = np.zeros((S, A), dtype=np.float64)
        
        # compute Q-values for all state-action pairs at time step t
        for a in range(A):
            Q[:, a] = R[:, a] + gamma * P[:, a, :].dot(V_next)
        
        # value function at time step t for each state
        V[t, :] = np.max(Q, axis=1) # dynamic programming update - use future value function V_next to compute current V
        # best action at time step t for each state
        pi_t[t, :] = np.argmax(Q, axis=1)

    return V, pi_t


def policy_iteration(MDP, action_space_env, gamma, max_iterations):
    P = MDP["P"]
    R = MDP["R"]
    S, A = R.shape
    
    pi = np.argmax(R, axis=1)
    Q_s = np.zeros(A, dtype=np.float64)

    for _ in range(max_iterations):
        # Policy Evaluation
        P_pi, R_pi = get_P_R(MDP, pi)
        V = get_v(P_pi, R_pi, gamma)
        
        # Policy Improvement
        policy_stable = True
        for s in range(S):
            
            for a in range(A):
                Q_s[a] = R[s, a] + gamma * P[s, a, :].dot(V)
            best_action = int(np.argmax(Q_s))
            if best_action != pi[s]:
                pi[s] = best_action
                policy_stable = False
        if policy_stable:
            break

    return V, pi


def value_iteration(MDP, action_space_env, gamma, eps, max_iterations):
    P = MDP["P"]
    R = MDP["R"]
    S, A = R.shape
    
    V = np.zeros(S, dtype=np.float64)
    Q = np.zeros((S, A), dtype=np.float64)

    for _ in range(max_iterations):

        # compute Q-values for all state-action pairs
        for a in range(A):
            Q[:, a] = R[:, a] + gamma * P[:, a, :].dot(V)
        
        # update value function
        V_new = np.max(Q, axis=1)
        delta = np.max(np.abs(V_new - V))
        
        # check for convergence
        if delta < eps:
            V = V_new
            break
        V = V_new
    # derive policy from value function
    pi = np.argmax(Q, axis=1)

    return V, pi


def get_P_R(MDP, pi):
    P = MDP["P"]
    R = MDP["R"]
    S, A = R.shape
    
    assert len(pi) == S, "Policy length must match number of states."
    
    P_pi = np.zeros((S, S), dtype=np.float64)
    R_pi = np.zeros((S,), dtype=np.float64)
    for s, a in enumerate(pi):
        P_pi[s, :] = P[s, a, :]
        R_pi[s] = R[s, a]

    return P_pi, R_pi


def get_v(P, R, gamma, max_iterations=10000):
    S = P.shape[0]
    I = np.eye(S)
    try:
        V = np.linalg.solve(I - gamma * P, R)
        return V
    except np.linalg.LinAlgError:
        # If the system is singular, use iterative policy evaluation to compute V
        V = np.zeros(S, dtype=np.float64)
        for _ in range(max_iterations):
            V_new = R + gamma * P.dot(V)
            delta = np.max(np.abs(V_new - V))
            if delta < 1e-3:
                return V_new
            V = V_new
        return V


# obs is a function handle -- if it's not None (as in the case of the pendulum),
# you can call the provided function to get the state from the observation
def get_MDP(env, MDP_state_bounds, action_space, num_data_per_state, obs=None):
    pos_min, pos_max = MDP_state_bounds["pos_bounds"]
    vel_min, vel_max = MDP_state_bounds["vel_bounds"]
    pos_bins = int(MDP_state_bounds["pos_bins"])
    vel_bins = int(MDP_state_bounds["vel_bins"])
    pos_label = MDP_state_bounds.get("pos_label", "position")
    vel_label = MDP_state_bounds.get("vel_label", "velocity")

    print("\nBuilding MDP with:")
    print("- {} bins for {}".format(pos_bins, pos_label))
    print("- {} bins for {}".format(vel_bins, vel_label))
    
    print("State bounds:")
    print("- {}: [{}, {}]".format(pos_label, pos_min, pos_max))
    print("- {}: [{}, {}]".format(vel_label, vel_min, vel_max))
    print("- action space: {}".format(action_space))
    
    # position and velocity bins
    pos_grid = np.linspace(pos_min, pos_max, pos_bins + 1)
    vel_grid = np.linspace(vel_min, vel_max, vel_bins + 1)
    
    S = pos_bins * vel_bins
    A = len(action_space)
    print("Total states: {}, Total actions: {}".format(S, A))
    
    # Ensure action is a numpy array with correct shape
    if isinstance(action_space, (list, tuple)):
        action_list = np.array(action_space, dtype=float)
    elif isinstance(action_space, np.ndarray):
        action_list = action_space.astype(float)
    else:
        action_list = np.array([float(action_space)], dtype=float)
    
    # Initialize transition and reward matrices
    P = np.zeros((S, A, S), dtype=np.float64)
    R = np.zeros((S, A), dtype=np.float64)
    samples_per_state_action = np.zeros((S, A), dtype=np.int32)
    
    print("\nCollecting data to build MDP...")
    def get_state_index(observation):
        if obs is not None:
            pos_vel = obs(observation)
            pos, vel = pos_vel[0], pos_vel[1]
        else:
            pos, vel = float(observation[0]), float(observation[1])

        # clip to bounds just to avoid out-of-bounds issues
        pos = np.clip(pos, pos_grid[0], pos_grid[-1] - 1e-12)
        vel = np.clip(vel, vel_grid[0], vel_grid[-1] - 1e-12)
        
        # map position and velocity to a single state index using the grids/bin indices
        return _state_index_from_grids(pos, vel, pos_grid, vel_grid)

    print("- Total data to collect: {} samples".format(S * A * num_data_per_state))

    for s in range(S):
        (pos_low, pos_high), (vel_low, vel_high) = _state_index_bound_from_grids(s, pos_grid, vel_grid)
        
        for act_idx, action in enumerate(action_list):
            for _ in range(num_data_per_state):
                # Sample a random state within the bin
                pos_sample = random.uniform(pos_low, pos_high)
                vel_sample = random.uniform(vel_low, vel_high)
                
                env.state = np.array([pos_sample, vel_sample], dtype=np.float64)

                next_observation, reward, done, _, _ = env.step([action])

                next_s_idx = get_state_index(next_observation)

                P[s, act_idx, next_s_idx] += 1.0
                R[s, act_idx] += float(reward)
                samples_per_state_action[s, act_idx] += 1

    # Normalize the transition probabilities and rewards
    P_norm = np.zeros_like(P, dtype=np.float64)
    R_norm = np.zeros_like(R, dtype=np.float64)
    for s in range(S):
        for a in range(A):
            total_samples = samples_per_state_action[s, a]
            if total_samples > 0:
                P_norm[s, a, :] = P[s, a, :] / total_samples
                R_norm[s, a] = R[s, a] / total_samples
            # If no samples were collected, set to zero
            else:
                P_norm[s, a, :] = np.zeros(S)
                R_norm[s, a] = 0.0
    
    # print("Transition matrix P shape: {}".format(P_norm.shape))
    # print("Reward matrix R shape: {}".format(R_norm.shape))

    state_index = partial(_state_index_from_grids, pos_grid=pos_grid, vel_grid=vel_grid)
    state_index_bound = partial(_state_index_bound_from_grids, pos_grid=pos_grid, vel_grid=vel_grid)
    MDP = {
        "P": P_norm,
        "R": R_norm,
        "pos_grid": pos_grid,
        "vel_grid": vel_grid,
        "actions": action_list,
        "state_index": state_index,
        "state_index_bound": state_index_bound
    }

    return MDP