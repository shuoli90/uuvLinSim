#!/usr/bin/python3


""" 
A script to run a DNN controller on a local continuous mountain car model. 

2 command line arguments: 
    1: (mandatory) the name of the YAML file with the DNN controller 
    2: (mandatory) the directory where to save the logs  
    3: (optional) whether the run should be rendered. Providing any string here would stop the rendering. 

""" 

import gym
import numpy as np
import yaml
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# my model
from continuous_mountain_car import Continuous_MountainCarEnv

# for saving traces 
#from gym_recording.wrappers import TraceRecordingWrapper

from datetime import datetime



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


def main(argv):

    # command line processing
    if len(argv) > 3 or len(argv) < 2:
        print('Incorrect number of command line arguments')  
        exit() 

    input_filename = argv[0]
    working_dir = argv[1]
    if working_dir[-1] != '/': 
        working_dir += '/'

    # create directory if it does not exist
    Path(working_dir).mkdir(parents=True, exist_ok=True)

    render = len(argv) <= 2 

    # load controller
    with open(input_filename, 'rb') as f:

       model = yaml.load(f)

    # create gym environment
    #env = gym.make('MountainCarContinuous-v0').env
    env = Continuous_MountainCarEnv() # use our version 
    #env = TraceRecordingWrapper(env)  # for recording 

    episodes = 110#1000

    # reset the environment to a random new position
    observation = env.reset()

    total_reward = 0

    # create empty shaped arrays: 1 row, 2 cols
    obs_log = np.empty([0,2])
    act_log = np.empty([0,1])

    #initprobs = np.array([])
    avgc = np.array([])
    avgd = np.array([])
    stdc = np.array([])
    stdd = np.array([])
    uncertc = np.array([])
    uncertd = np.array([])
    uncertagg1 = np.array([])
    uncertagg2 = np.array([])
    #probc = np.array([])
    #probd = np.array([])
    #probinitpos = np.array([])
    probassnpre = np.array([])
    probassnpost1 = np.array([])
    probassnpost2 = np.array([])

    particle_log_c = []
    particle_log_d = [] 
    weight_log = [] 

    #print("Before the simulation ", str(act_log))

    for e in range(episodes):

        #print("Step ", str(e), ", action log:", str(act_log))

        particle_log_c.append(env.particles_c)
        particle_log_d.append(env.particles_d)
        weight_log.append(env.weights) 

        # comment this out if you want to simulate many trajectories
        if render: 
            env.render()

        #print("PREDICTIING FOR ", str(observation)) 
        #print(observation.reshape(1,len(observation))) 
        action = predict(model, observation.reshape(1,len(observation)))
        #print("PREDICTED ACTION ", str(action))

        # observation stores the car states (position, velocity) during each step
        observation, reward, done, info = env.step(action)

        total_reward += reward
        #print(observation) 
        #initprobs = np.append(initprobs, info) 
        # TODO convert list to tuple, assign in one line 
        avgc = np.append(avgc, info[0]) 
        avgd = np.append(avgd, info[1]) 
        stdc = np.append(stdc, info[2]) 
        stdd = np.append(stdd, info[3]) 
        uncertc = np.append(uncertc, info[4]) 
        uncertd = np.append(uncertd, info[5]) 
        uncertagg1 = np.append(uncertagg1, info[6]) 
        uncertagg2 = np.append(uncertagg2, info[7]) 
        probassnpre = np.append(probassnpre, info[8]) 
        probassnpost1 = np.append(probassnpost1, info[9]) 
        probassnpost2 = np.append(probassnpost2, info[10]) 

        # save a row of new values 
        obs_log = np.append(obs_log, observation.reshape(1,2) , axis = 0) 
        act_log = np.append(act_log, action, axis = 0) 

        #if observation[0] > 0.45:
        if done: 
            #print('Success!')
            #print('Number of steps: ' + str(e))
            #print(env.directory) 
            break

    print('Number of steps: ' + str(e))
    print('Total reward: ' + str(total_reward))

    # create outcome string: whether it reached the flag, how many steps it took, and the total reward accrued 
    outcome_str = ''
    if e < episodes-1: 
        outcome_str += 'True,' 
        print('Success!')
    else: 
        outcome_str += 'False,' 
        print('Failure!') 

    outcome_str += str(e) + ',' + str(total_reward) + '\n' 
    
    # whether verification assumptions hold, actual drawn noise parameters, other car model details, and the seed value
    outcome_str += str(env.VERIFASSN3(env.truec, env.trued, env.trueinitpos) and env.steepness == 0.0025) + ',' +  str(env.VERIFASSN3(env.truec, env.trued, env.trueinitpos)) + ',' + str(env.steepness == 0.0025) + ',' + str(env.truec) + ',' + str(env.trued) + ',' + str(env.trueinitpos) + ',' + str(env.steepness) + ',' + str(env.POSNOISESTD) + ',' + str(env.VELNOISESTD) + ',' + str(env.seed_saved) + '\n'

    # mission hyperparameters
    outcome_str += str(env.TRUECLEFT) + ',' + str(env.TRUECRIGHT) + ',' + str(env.TRUEDLEFT) + ',' + str(env.TRUEDRIGHT) + ',' + str(env.INITPOSLEFT) + ',' + str(env.INITPOSRIGHT) + '\n'

    # PF hyperparameters
    outcome_str += str(env.PARTCT) + ','  + str(env.OVERALLWEIGHT) + ',' + str(env.PFVELWEIGHT) + ',' + str(env.PFINITCLEFT) + ',' + str(env.PFINITCRIGHT) + ',' + str(env.PFINITDLEFT) + ',' + str(env.PFINITDRIGHT) + '\n'

    # Saving data into files 
    # working_dir = '/home/ivan/mountainCarLogs/'
    nowstr = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

    # save outcomes 
    outcome_file = open(working_dir + nowstr + '_out.csv', 'w') 
    outcome_file.write(outcome_str)
    #outcome_file.write
    outcome_file.close() 

    # save logs
    np.savetxt(working_dir + nowstr + '_obs.csv', obs_log, delimiter=',')
    np.savetxt(working_dir + nowstr + '_act.csv', act_log, delimiter=',')
    np.savetxt(working_dir + nowstr + '_mc.csv', probassnpre, delimiter=',')
    #np.savetxt(working_dir + nowstr + '_monassnpre.csv', probassnpre, delimiter=',')
    #np.savetxt(working_dir + nowstr + '_monassnpost.csv', probassnpost, delimiter=',')

    # Plot the outcomes 
    if render: 
        # PLOT 1: the estimated averages
        plt.ylim(-2, 2) 
        plt.plot(avgc) 
        plt.fill_between(range(len(avgc)), avgc-stdc, avgc+stdc, color='aquamarine')
        plt.axhline(y=env.truec, color='steelblue', linestyle='dashed')
        plt.plot(avgd*20) 
        plt.fill_between(range(len(avgd)), 20*(avgd-stdd), 20*(avgd+stdd), color='mistyrose')
        plt.axhline(y=env.trued*20, color='orangered', linestyle='dashed')
        plt.legend(["Mean estimate of c","True value of c", "Mean estimate of d*20", "True value of d*20"])
        plt.ylabel('Parameter value')
        plt.xlabel('Steps')

        # transparent high-weight particles
        for i in range(env.time): # over time points
            for j in range(env.PARTCT): # over particles
                if False:#weight_log[i][j]>0.005: 
                    plt.scatter(i, particle_log_c[i][j], s=5, c="red", alpha=min(1, 10*weight_log[i][j]))
                #plt.scatter([i]*env.PARTCT, particle_log_c[i], s=5, cmap="viridis", alpha=0.3)
        #plt.scatter(20, 0.5, s=5) # last argument is point size 
        plt.show()

        # PLOT 2: the uncertainties 
        plt.clf() 
        plt.ylim(0, 1.4) 
        plt.plot(uncertc)
        plt.plot(uncertd)
        plt.plot(uncertagg1)
        plt.plot(uncertagg2)
        plt.legend(["Uncertainty of c estimate", "Uncertainty in d estimate", "Aggregate uncertainty (arithm mean)", "Aggregate uncertainty (geom mean)"])
        #plt.show()

        # PLOT 3: monitored probabilities in intervals
        plt.clf()
        plt.ylim(0, 1.1) 
        #plt.plot(probc)
        #plt.plot(probd)
        #plt.plot(probinitpos)
        plt.plot(probassnpre)
        plt.plot(probassnpost1)
        plt.plot(probassnpost2)
        plt.legend(["Prob of noise assn being satisfied (no uncert)", "Prob of assn being satisfied (with uncert, arithm mean)", "Prob of assn being satisfied (with uncert, geom mean)"])
        #plt.legend(["Prob of c in " + str(env.MONINTC) ,"Prob of d in " + str(env.MONINTD), "Prob of init pos in " + str(env.MONINTINITPOS), "Prob of assn being satisfied"])
        #plt.legend("Prob of assn being satistied")
        plt.show()

if __name__ == '__main__':
    main(sys.argv[1:])
