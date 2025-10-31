import gymnasium as gym
import random
import numpy as np
import matplotlib.pyplot as plt
from MDP import get_MDP
from MDP import policy_iteration
from MDP import value_iteration
# from six.moves import cPickle as pickle
import pickle
import sys
import time


def get_state_from_observation(obs):

    cos_theta = float(obs[0])
    sin_theta = float(obs[1])
    vel = float(obs[2])

    theta = np.arcsin(sin_theta)

    if sin_theta > 0 and cos_theta < 0:
        theta = np.pi - theta

    elif sin_theta < 0 and cos_theta < 0:
        theta = -np.pi - theta

    return (theta, vel) 


if __name__ == '__main__':

    np.random.seed(1)

    num_data_per_state = 100
    env = gym.make('Pendulum-v1').env.unwrapped

    theta_min = -np.pi
    theta_max = np.pi

    vel_min = env.observation_space.low[2]
    vel_max = env.observation_space.high[2]

    act_min = env.action_space.low[0]
    act_max = env.action_space.high[0]

    # YOUR CODE GOES HERE
    
    MDP_state_bounds = {
        "pos_bounds":(theta_min, theta_max),
        "pos_bins": 31, 
        "vel_bounds":(vel_min, vel_max),
        "vel_bins": 31,
        "pos_label": "angle",
        "vel_label": "angular_velocity"
    }

    action_space = [-2, -1.33, -0.67, 0, 0.67, 1.33, 2]
    # [-2,-1.5,-1,-0.5,0,0.5,1,1.5,2]

    # if you don't want to rebuild your MDP every time, you
    # can save/load it as a pickle, for example
    # your submitted code should also build the MDP and run for less than 10 minutes total
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        with open(filename, 'rb') as f:
            Pendulum_MDP = pickle.load(f)

    else:
        # This is just a placeholder
        Pendulum_MDP = get_MDP(
            env=env, 
            MDP_state_bounds=MDP_state_bounds, 
            action_space=action_space, 
            num_data_per_state=num_data_per_state, 
            obs=get_state_from_observation
        )

        filename = 'pendulum_mdp.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(Pendulum_MDP, f, pickle.HIGHEST_PROTOCOL)

    train_gamma = 0.9
    # sim_gamma = 1
    T = 1000
    num_episodes = 100

    V, pi = value_iteration(Pendulum_MDP, Pendulum_MDP["actions"], train_gamma, eps=1e-3, max_iterations=10000)
    # V, pi = policy_iteration(Pendulum_MDP, Pendulum_MDP["actions"], train_gamma, max_iterations=10000)
    
    # evaluate policy
    all_rewards = []
    
    for _ in range(num_episodes):
        observation, _ = env.reset()
        total_reward = 0.0
        for _ in range(T):
            theta, theta_dot = get_state_from_observation(observation)
            s = Pendulum_MDP["state_index"](theta, theta_dot)
            a_idx = int(pi[s])
            action = Pendulum_MDP['actions'][a_idx]
            observation, reward, done, _, _ = env.step([action])
            total_reward += float(reward)
            
            if done:
                break
            
        all_rewards.append(total_reward)

    print(f"Pendulum average reward over {num_episodes} runs: {np.mean(all_rewards):.2f}")

    # use this if you would like to render the environment
    # env = gym.make('Pendulum-v1', render_mode = 'human').env.unwrapped
    # observation, info = env.reset()
    # for _ in range(100):
    #     action = env.action_space.sample()
    #     observation, reward, done, _, info = env.step(action)
    #     time.sleep(0.02)
    #     if done:
    #         # observation, info = env.reset()
    #         print("Episode finished.")
    #         break

    # env.close()

    