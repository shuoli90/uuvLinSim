#from Fish_cont import World
from Fish_oneHz import World
import numpy as np
import random
from keras.models import Sequential
from keras import models
from keras import optimizers
from keras.layers.core import Dense, Dropout, Activation
# from keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
import pickle
import math
import sys
from continuous_mountain_car import Continuous_MountainCarEnv

def relu(x):
    relu = np.maximum(0, x)

    return relu

#this is just for testing purposes
def relu_predict(model, inputs):
    weights = {}
    offsets = {}

    layerCount = 1

    for layer in model.layers:
        if len(layer.get_weights()) > 0:
            weights[layerCount] = layer.get_weights()[0]
            offsets[layerCount] = layer.get_weights()[1]

            layerCount += 1

    curNeurons = inputs

    for layer in range(layerCount-1):
        curNeurons = curNeurons.dot(weights[layer + 1]) + offsets[layer + 1]

        if layer <= layerCount - 3:
            curNeurons = relu(curNeurons)

    return curNeurons

#this is just for testing purposes
def tanh_predict(model, inputs):
    weights = {}
    offsets = {}

    layerCount = 1

    for layer in model.layers:
        
        if len(layer.get_weights()) > 0:
            weights[layerCount] = layer.get_weights()[0]
            offsets[layerCount] = layer.get_weights()[1]

            layerCount += 1

    curNeurons = inputs

    for layer in range(layerCount-1):
        curNeurons = curNeurons.dot(weights[layer + 1]) + offsets[layer + 1]

        curNeurons = np.tanh(curNeurons)

    return curNeurons

def sigmoid(x):
    sigm = 1. / (1. + np.exp(-x))

    return sigm

def swish_predict(model, inputs):
    weights = {}
    offsets = {}

    layerCount = 1

    for layer in model.layers:
        if len(layer.get_weights()) > 0:
            weights[layerCount] = layer.get_weights()[0]
            offsets[layerCount] = layer.get_weights()[1]

            layerCount += 1

    curNeurons = inputs

    for layer in range(layerCount-1):
        curNeurons = curNeurons.dot(weights[layer + 1]) + offsets[layer + 1]

        if layer <= layerCount - 3:
            curNeurons = curNeurons * sigmoid(curNeurons)
            #curNeurons = relu(curNeurons)

    return curNeurons

def normalize(s):
    mean = [2.5]
    spread = [5.0]
    return (s - mean) / spread

def main(argv):

    input_filename = argv[0]
    
    model = models.load_model(input_filename)

    y_pos = [35, 40]
    heading = [math.radians(-5), math.radians(5)]
    episode_length = 50

    # obs_pos_x = [25, 30]
    # obs_pos_y = [30, 35]
    distance = [45, 50]
    obs_vel = 0.514
    obs_radius = 1.0

    parameters = {'y_pos': y_pos, 
                  'heading': heading,
                  'episode_length': episode_length,
                  'distance': distance,
                  'obs vel': obs_vel,
                  'obstacle radius': obs_radius
                  }

    with open('parameters_2.pickle', 'wb') as handle:
        pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('parameters.pickle', 'rb') as handle:
    #     b = pickle.load(handle)
    
    # env = World(0.0, y_pos, heading, episode_length, distance, obs_vel, obs_radius)
    # env2 = gym.make('MountainCarContinuous-v0').env
    env = Continuous_MountainCarEnv()

    # set seeds
    np.random.seed(10)

    rew = 0
    u = np.array([0, 0.48556, 45])    

    allX = []
    allY = []
    allR = []

    allXObs = []
    allYObs = []

    numTrajectories = 1000
    collisions = []
    fails = []
    trajectories = 1000
    episodes = 100
    true_states = []
    pred_states = []

    init_states = []
    predicted_inits = []
    dones = []

    for step in range(numTrajectories):

        print("Trajectory: ", step)

        # reset the environment to a random new position
        observation = env.reset()
        init_state = [env.trueinitpos, 0.0, env.truec, env.trued]
        c = env.truec
        d = env.trued
        init_states.append(init_state)
        predicted_init = env.obs_to_true(env.trueinitpos, 0.0, c, d)
        breakpoint()

        for e in range(episode_length):

            u = np.radians(5) * model.predict(observation.reshape(1,len(observation)))[0]
            observation, reward, done, info = env.step(u)
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

    np.savetxt("true_state.csv", np.vstack(true_states), delimiter=",")
    np.savetxt("pred_state.csv", np.vstack(pred_states), delimiter=",")
    np.savetxt("init_stat.csv", np.vstack(init_states), delimiter=",")
    np.savetxt("true_vs_predicted_init.csv", np.vstack(predicted_inits), delimiter=",")
    np.savetxt("done.csv", np.hstack(dones), delimiter=",")
    
    
if __name__ == '__main__':
    main(sys.argv[1:])
