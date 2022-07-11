#!/usr/bin/python3

import gym
import numpy as np
import yaml
import sys

from continuous_mountain_car import Continuous_MountainCarEnv

def main():
    # create gym environment
    env2 = gym.make('MountainCarContinuous-v0').env
    env = Continuous_MountainCarEnv()
    
    trajectories = 100
    episodes = 100
    true_states = []
    pred_states = []

    for traj in range(trajectories):

        # reset the environment to a random new position
        observation = env.reset()
        #observation = env.setState(np.array([-0.5, 0]))

        total_reward = 0

        for e in range(episodes):

            # comment this out if you want to simulate many trajectories
            # env.render()

            action = env.action_space.sample()
            print(action)
            #predict(model, observation.reshape(1,len(observation)))

            # observation stores the car states (position, velocity) during each step
            observation, reward, done, info = env.step(action)

            total_reward += reward

            if observation[0] > 0.45:
                print('Success!')
                print('Number of steps: ' + str(e))
                break

        print('Total reward: ' + str(total_reward))
        true_states.append(env.true_state)
        pred_states.append(env.pred_state)

    np.savetxt("true_state.csv", np.vstack(true_states), delimiter=",")
    np.savetxt("pred_state.csv", np.vstack(pred_states), delimiter=",")

if __name__ == '__main__':
    main()
