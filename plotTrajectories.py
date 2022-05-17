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

    y_pos = [25, 35]
    heading = [math.radians(-10), math.radians(10)]
    episode_length = 130

    # obs_pos_x = [25, 30]
    # obs_pos_y = [30, 35]
    distance = [20, 50]
    obs_vel = 0.514
    obs_radius = 0.5

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
    
    env = World(0.0, y_pos, heading, episode_length, distance, obs_vel, obs_radius)

    # set seeds
    np.random.seed(10)

    rew = 0

    # prev_pos_y = env.pos_y

    #observation = env.reset()
    # observation = np.array([0, -y_pos, y_pos])

    u = np.array([0, 0.48556, 45])    

    allX = []
    allY = []
    allR = []

    allXObs = []
    allYObs = []

    numTrajectories = 10
    collisions = []
    fails = []

    for step in range(numTrajectories):

        print('Trajectory: ', step)

        observation = env.reset()
        init = observation
        init_obs = [env.obs_pos_x, env.obs_pos_y]

        rew = 0

        for e in range(episode_length):

            observation, reward, done, collision = env.step(u)
            
            u = np.radians(5) * model.predict(observation.reshape(1,len(observation)))[0]

            if done:
                if e < episode_length - 1:
                    print(init)
                    print(init_obs)
                    fails.append(True)
                break

            rew += reward

        allX.append(env.allX)
        allY.append(env.allY)
        allXObs.append(env.allXObs)
        allYObs.append(env.allYObs)
        allR.append(rew)
        collisions.append(collision)


    #print(np.mean(allR))
    #print('number of crashes: ' + str(num_unsafe))
    
    fig = plt.figure(figsize=(12,10))
    
    #plt.ylim((-1,11))
    #plt.xlim((-1.75,10.25))
    #plt.suptitle('Simulated trajectories of the F1/10 Car', fontsize=30)
    #plt.tick_params(labelsize=20)

    env.plot_pipe()
    for i in range(numTrajectories):
        plt.plot(allX[i], allY[i], 'r-')
        plt.plot(allXObs[i], allYObs[i], 'b*')
        # plt.plot(np.array([0.0, env.PIPE_LENGTH]), np.array([0.0, 0.0]), 'b', linewidth=3)

    # plt.savefig('simulations_1.pdf', format='pdf', bbox_inches = 'tight', pad_inches = 0)
    plt.savefig("trajectories_2.png")
    # plt.show()
    
    with open('collisions_2', 'wb') as fp:
        pickle.dump(collisions, fp)

    with open('fails_1', 'wb') as fp:
        pickle.dump(fails, fp)

    #w.plot_lidar()
    
if __name__ == '__main__':
    main(sys.argv[1:])
