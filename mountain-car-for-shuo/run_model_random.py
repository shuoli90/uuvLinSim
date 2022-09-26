#!/usr/bin/python3

import gym
import numpy as np
import yaml
import sys
# from keras.models import Sequential
# from keras import models
import pygame
from tqdm import tqdm

from continuous_mountain_car import Continuous_MountainCarEnv

def predict(model, inputs):
    weights = {}
    offsets = {}

    layerCount = 0
    activations = []

    for layer in range(1, len(model['weights']) + 1):
        
        weights[layer] = np.array(model['weights'][layer])
        offsets[layer] = np.array(model['offsets'][layer])

        layerCount += 1
        activations.append(model['activations'][layer])

    curNeurons = inputs

    for layer in range(layerCount):

        curNeurons = curNeurons.dot(weights[layer + 1].T) + offsets[layer + 1]

        if 'Sigmoid' in activations[layer]:
            curNeurons = sigmoid(curNeurons)
        elif 'Tanh' in activations[layer]:
            curNeurons = np.tanh(curNeurons)

    return curNeurons    

def sigmoid(x):

    sigm = 1. / (1. + np.exp(-x))

    return sigm

def estimate_safety(sampled_trajectories, safety, current_intervals, delta=0.1):
    indices = np.arange(sampled_trajectories.shape[0])
    num_traj = sampled_trajectories.shape[0]
    for idx, interval in enumerate(current_intervals):
        indices= np.where(np.logical_and(sampled_trajectories[:, idx]>=interval[0], 
                                               sampled_trajectories[:, idx]<=interval[1]))
        sampled_trajectories = sampled_trajectories[indices]
        safety = safety[indices]
    M = sampled_trajectories.shape[0] / num_traj
    bound = M * np.sqrt(np.log(2/delta) / 2 / num_traj)
    return np.clip(np.mean(safety) - bound, 0, 1), np.clip(np.mean(safety) + bound, 0, 1)


def main():
    # create gym environment
    env2 = gym.make('MountainCarContinuous-v0').env
    env = Continuous_MountainCarEnv()

    sampled_init_states = np.loadtxt('init_stat_1000.csv', delimiter=',')
    safety = np.loadtxt("done_1000.csv", delimiter=",")

    q_x = 0.022
    q_v = 0.012
    q_c = 0.8574
    q_d = 0.0025
    
    trajectories = 1000
    episodes = 110
    true_states = []
    pred_states = []

    init_states = []
    predicted_inits = []
    dones = []

    input_filename = "sig16x16.yml"

    # load controller
    with open(input_filename, 'rb') as f:
        model = yaml.load(f)

    for traj in tqdm(range(trajectories)):
        print("Trajectory: ", traj)

        # reset the environment to a random new position
        observation = env.reset()
        x_interval = [observation[0] - q_x, observation[0] + q_x]
        v_interval = [observation[1] - q_v, observation[1] + q_v]
        c_interval = [env.truec - q_c, env.truec + q_c]
        d_interval = [env.trued - q_d, env.trued + q_d]
        # intervals = [x_interval, v_interval, c_interval, d_interval]
        intervals = [x_interval, v_interval]
        safety = estimate_safety(sampled_init_states, safety, intervals)
        print(env.trueinitpos, 0.0, env.truec, env.trued)
        print('estimated safety interval', safety)
        breakpoint()

        init_state = [env.trueinitpos, 0.0, env.truec, env.trued]
        c = env.truec
        d = env.trued
        init_states.append(init_state)
        predicted_init = env.obs_to_true(env.trueinitpos, 0.0, c, d)

        total_reward = 0

        for e in range(episodes):

            # comment this out if you want to simulate many trajectories
            # env.render()

            # action = env.action_space.sample()
            action = predict(model, observation.reshape(1,len(observation)))
            # action = np.radians(5) * model.predict(observation.reshape(1,len(observation)))[0]
            # print(action)
            #predict(model, observation.reshape(1,len(observation)))

            # observation stores the car states (position, velocity) during each step
            observation, reward, done, info = env.step(action)
            predicted_init =  env.obs_to_true(observation[0], observation[1], c, d)
            predicted_inits.append(np.array([init_state[0], 
                                             init_state[1], 
                                             predicted_init[0],
                                             predicted_init[1]]))

            total_reward += reward

            if observation[0] > 0.45:
                print('Success!')
                print('Number of steps: ' + str(e))
                break

        dones.append(done)

        print('Total reward: ' + str(total_reward))
        true_states.append(env.true_state)
        pred_states.append(env.pred_state)

    np.savetxt("true_state_1000.csv", np.vstack(true_states), delimiter=",")
    np.savetxt("pred_state_1000.csv", np.vstack(pred_states), delimiter=",")
    np.savetxt("init_stat_1000.csv", np.vstack(init_states), delimiter=",")
    np.savetxt("true_vs_predicted_init_1000.csv", np.vstack(predicted_inits), delimiter=",")
    np.savetxt("done_1000.csv", np.hstack(dones), delimiter=",")

if __name__ == '__main__':
    main()
