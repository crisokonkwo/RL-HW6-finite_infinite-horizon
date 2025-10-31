import gymnasium as gym
import random
import numpy as np
import matplotlib.pyplot as plt
from MDP import get_MDP
from MDP import dynamic_programming_finite_horizon
import sys
import pickle
import time



if __name__ == '__main__':

    np.random.seed(1)

    num_data_per_state = 100
    env = gym.make('MountainCarContinuous-v0').env.unwrapped

    pos_min = env.observation_space.low[0]
    pos_max = env.observation_space.high[0]

    vel_min = env.observation_space.low[1]
    vel_max = env.observation_space.high[1]

    act_min = env.action_space.low[0]
    act_max = env.action_space.high[0]
    
    print("Position min: {}, max: {}".format(pos_min, pos_max))
    print("Velocity min: {}, max: {}".format(vel_min, vel_max))
    print("Action min: {}, max: {}".format(act_min, act_max))


    # YOUR CODE GOES HERE

    MDP_state_bounds = {
        "pos_bounds":(pos_min, pos_max),
        "pos_bins": 20, 
        "vel_bounds":(vel_min, vel_max),
        "vel_bins": 24,
        "pos_label": "position",
        "vel_label": "velocity"
    }
    action_space = [-1.0, -0.5, -0.25, -0.22, 0.0, 0.22, 0.25, 0.5, 1.0]

    # if you don't want to rebuild your MDP every time, you
    # can save/load it as a pickle, for example
    # your submitted code should also build the MDP and run for less than 10 minutes total
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        with open(filename, 'rb') as f:
            MC_MDP = pickle.load(f)
        print("Loaded MountainCar MDP from disk.")
        # print(MC_MDP["state_index"](0.0, 0.0))
        # print(MC_MDP["state_index_bound"](MC_MDP["state_index"](0.0, 0.0)))

    else:
        MC_MDP = get_MDP(env=env,
                         MDP_state_bounds = MDP_state_bounds,
                         action_space = action_space,
                         num_data_per_state = 100,
                         obs = None)

        print("Finished building MountainCar MDP.")
        # print(MC_MDP["state_index"](0.0, 0.0))
        # print(MC_MDP["state_index_bound"](MC_MDP["state_index"](0.0, 0.0)))

        filename = 'mc_mdp.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(MC_MDP, f, pickle.HIGHEST_PROTOCOL)

    gamma = 1
    T = 150
    num_episodes = 100

    # dynamic programming to get optimal policy
    V, pi_t = dynamic_programming_finite_horizon(MC_MDP, MC_MDP["actions"], gamma, max_iterations=150)
    print("pi_t shape:", pi_t.shape)
    print("Finished computing optimal policy with finite-horizon dynamic programming.")
    
    # evaluate policy
    all_rewards = []
    
    for _ in range(num_episodes):
        observation, _ = env.reset()
        total_reward = 0.0
        
        for t in range(T):
            s = MC_MDP['state_index'](observation[0], observation[1])
            a_idx = int(pi_t[t, s])
            action = MC_MDP['actions'][a_idx]
            # env.state = np.array([observation[0], observation[1]], dtype=np.float64)
            observation, reward, done, _, _ = env.step([action])
            total_reward += float(reward)
            if done:
                break
        all_rewards.append(total_reward)

    avg_reward = np.mean(all_rewards)
    print(f"MountainCar average reward over {num_episodes} runs: {avg_reward:.2f} (std {np.std(all_rewards):.2f})")
    
    # use this if you would like to render the environment
    # env = gym.make('MountainCarContinuous-v0', render_mode = 'human').env.unwrapped
    # observation, info = env.reset()
    # for _ in range(1000):
    #     action = env.action_space.sample()
    #     observation, reward, done, _, info = env.step(action)
    #     time.sleep(0.02)
    #     if done:
    #         # observation, info = env.reset()
    #         print("Episode finished.")
    #         break

    # env.close()
