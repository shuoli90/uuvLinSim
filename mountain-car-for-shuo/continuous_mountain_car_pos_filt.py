#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@author: Olivier Sigaud

A merge between two sources:

* Adaptation of the MountainCar Environment from the "FAReinforcement" library
of Jose Antonio Martin H. (version 1.0), adapted by  'Tom Schaul, tom@idsia.ch'
and then modified by Arnaud de Broissia

* the OpenAI/gym MountainCar environment
itself from
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math

import numpy as np
from scipy.optimize import minimize 
from scipy.optimize import Bounds 

import gym
from gym import spaces
from gym.utils import seeding

class Continuous_MountainCarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, goal_velocity = 0):
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.45 # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        self.goal_velocity = goal_velocity
        self.power = 0.0015

        self.low_state = np.array([self.min_position, -self.max_speed])
        self.high_state = np.array([self.max_position, self.max_speed])

        self.partct = 100

        self.viewer = None

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action,
                                       shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,
                                            dtype=np.float32)

        # NEW noise functions
        self.noise_pos = lambda pos, vel : 0
        self.noise_vel = lambda pos, vel : -0.01*pos

        # OLD OBSERVATIONS 
        # adding sensing noise for position
        # pos_noise = 0 
        # stochastic 
        #pos_noise = self.np_random.uniform(low=-0.15, high=0.15) # easy 
        #pos_noise = self.np_random.normal(0.0, 0.1) # easy 
        #pos_noise = self.np_random.normal(0.75, 0.1) # medium 
        #pos_noise = self.np_random.normal(0.3, 0.1) # easy 
        #pos_noise = self.np_random.normal(-velocity[0]*5, abs(velocity[0]*3) )
        # deterministic 
        #pos_noise = 5*velocity[0]

        # adding sensing noise for velocity 
        #vel_noise = 0
        # stochastic 
        #vel_noise = self.np_random.uniform(low=-0.05, high=0.05) # easy
        #vel_noise = self.np_random.normal(0, 0.02) # easy 
        #vel_noise = self.np_random.normal(-0.01, 0.02) # medium
        #vel_noise = self.np_random.normal(-0.01, abs(position[0]/40))  # medium 
        # deterministic 
        #vel_noise = -0.01*position[0] 

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # original model: initialization with 0 speed and coordinate between -0.6 and -0.4
        self.state = np.array([self.np_random.uniform(low=-0.51, high=-0.5), 0])
        # simplified: fixed initialization 
        # self.state = np.array([-0.5, 0])

        #self.obs_pos_log = np.array([]) 
        #self.obs_vel_log = np.array([]) 

        self.time = 0

        # Initialize particles  
        self.particles_pos = self.np_random.uniform(low=-0.6, high=-0.4, size=self.partct)
        self.particles_pos_init = self.particles_pos.copy() # save for traceability 
        self.particles_vel = np.array([0.]*self.partct)
        self.weights = np.array([1.]*self.partct)
        
        self.particles_c = self.np_random.uniform(low=-5, high=5, size=self.partct)
        self.particles_d = self.np_random.uniform(low=-0.1, high=0.1, size=self.partct)

        # Structure for storing particle history
#        self.particles_pos_log = np.empty(self.partct, dtype=object)
#        self.particles_vel_log = np.empty(self.partct, dtype=object)
#        for i in range(self.partct): 
#            self.particles_pos_log[i] = []
#            self.particles_vel_log[i] = []

        print("Resetting particles and weights:") 
        #print(self.particles_pos) 
        #print(self.particles_vel) 
        #print(self.weights) 

        return np.array(self.state)

    def step(self, action):
        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], -1.0), 1.0)

        # process noise cor the true model
        proc_noise = [0, 0]   
        #proc_noise = self.np_random.normal([0, 0], [0.001, 0.0001]) 
        #print(proc_noise)
           
        # particle propagation 
        for i in range(self.partct): 
            #part_proc_noise = [0, 0] # none 
            #part_proc_noise = [self.np_random.uniform(low=-0.01, high=0.01), 0] # ok, increasing a bit 
            #part_proc_noise = [self.np_random.uniform(low=-0.1, high=0.1), 0] # poor
            #part_proc_noise = [0, self.np_random.uniform(low=-0.001, high=0.001)] # ok, increasing a bit 
            #part_proc_noise = [0, self.np_random.uniform(low=-0.01, high=0.01)] # high variance, can increase a lot 
            #part_proc_noise = [0, self.np_random.normal(0.0, 0.01)] # bad

            #posarr, velarr = self.model_step(np.array([self.particles_pos[i]]), np.array([self.particles_vel[i]]), force, part_proc_noise)
            #print(posarr, velarr)
            #self.particles_pos[i] = posarr[0]
            #self.particles_vel[i] = velarr[0]
            #print(self.particles_pos[i], self.particles_vel[i])
            #self.particles_pos_log[i] = np.append(self.particles_pos_log[i], posarr[0])
            #self.particles_vel_log[i] = np.append(self.particles_vel_log[i], velarr[0])
            self.particles_c[i] += self.np_random.normal(0.0, 0.1) 
            self.particles_d[i] += self.np_random.normal(0.0, 0.01) 

        # GT state propagation
        position, velocity = self.model_step(position, velocity, force, proc_noise) 
        self.state = np.array([position, velocity])

        # updating the simulation state 
        done = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        reward = 0
        if done:
            reward = 100.0
        reward-= math.pow(action[0],2)*0.1

        #if self.time > 77: 
        #    done = True
        
        # observed values
        obs_pos = position + self.noise_pos(position[0], velocity[0]) 
        obs_vel = velocity + self.noise_vel(position[0], velocity[0]) 
        obs_state = np.array([obs_pos, obs_vel])
        #self.obs_pos_log = np.append(self.obs_pos_log, obs_pos) 
        #self.obs_vel_log = np.append(self.obs_vel_log, obs_vel)


        # fitting noise function for each particle 
        #obs_fns_vel = lambda pos, d: d*pos
#        for i in range(self.partct): 
#            step1 = self.particles_vel_log[i] + obs_fns_vel(self.particles_vel_log[i], -0.01)
#            step2 = self.obs_vel_log - step1 
#            step3 = step2*step2
#            res = minimize(
#                    lambda d: pow(self.particles_vel_log[i] + obs_fns_vel(self.particles_vel_log[i], d) - self.obs_vel_log, 2), 
#                    [0], bounds=Bounds(-0.01, 0.01)) 
#            print(res.x[0])
            

        # Updating partcle weights based on differences from the observations 
        for i in range(self.partct): 
            # predicted observations based on particles 
            popos = (self.particles_c[i]*obs_vel - obs_pos)/(self.particles_c[i]*self.particles_d[i] - 1)
            povel = obs_vel - self.particles_d[i]*popos 
            self.weights[i] = self.weights[i]*math.exp(-abs(obs_pos - popos) - abs(obs_vel - povel))
            #ppos = self.particles_pos[i] 
            #pvel = self.particles_vel[i] 
            #self.weights[i] = self.weights[i]*math.exp(-abs(obs_pos - ppos - self.noise_pos(ppos, pvel)) - abs(obs_vel - pvel - self.noise_vel(ppos, pvel)))

        # normalize weights
        self.weights = self.partct*self.weights/np.sum(self.weights) 
        #initprob =  self.estimate_init_prob([-0.51, -0.5])

        # PRINTOUT 
        print("True values:")
        print(self.state) 

        print("Observed values:") 
        print(obs_state) 

        print("Stepped particles:") 
        print(self.particles_c) 
        print(self.particles_d) 
        
        print("Weights:") 
        print(self.weights) 

        #print("Estimated prob of having started in [-0.51, -0.5]:", initprob)

        self.time += 1 
        #return self.state, reward, done, {}
        return obs_state, reward, done, {}

    # computes a step of the dynamical 
    # no measurement noise here, just the pure dynamics
    # input: ([position], [velocity], [control action], 2-dim process noise) 
    # output: ([new pos], [new vel]) 
    def model_step(self, oldpos, oldvel, act, proc_noise): 

        # updating state values 
        newvel = oldvel + act*self.power -0.0025 * math.cos(3*oldpos) + proc_noise[1]
        if (newvel > self.max_speed): newvel = np.array([self.max_speed])
        if (newvel < -self.max_speed): newvel = np.array([-self.max_speed]) 

        newpos = oldpos + newvel + proc_noise[0]
        if (newpos > self.max_position): newpos = np.array([self.max_position])
        if (newpos < self.min_position): newpos = np.array([self.min_position]) 
        if (newpos == self.min_position and newvel<0): newvel = np.array([0])

        return newpos, newvel

    # estimate the probability based on the current particle weights 
    # inputs: [a, b], where a <= b
    # outputs: float between 0 and 1
    def estimate_init_prob(self, interval):
        #print("Initial particles:", self.particles_pos_init)
        idxRight = np.where(self.particles_pos_init >= interval[0])[0]
        idxLeft = np.where(self.particles_pos_init  <= interval[1])[0]
        idxInside = np.intersect1d(idxRight, idxLeft) 

        #print("IDs of selected particles:", idxInside)
        #idxInside = np.where(self.particles_pos_init >= interval[0] and self.particles_pos_init <= interval[1])[0]
        return np.sum(self.weights[idxInside])/np.sum(self.weights)

    def _height(self, xs):
        return np.sin(3 * xs)*.45+.55

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width/world_width
        carwidth=40
        carheight=20


        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_position)*scale, ys*scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight/2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight/2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position-self.min_position)*scale
            flagy1 = self._height(self.goal_position)*scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
            flag.set_color(.8,.8,0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation((pos-self.min_position)*scale, self._height(pos)*scale)
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
