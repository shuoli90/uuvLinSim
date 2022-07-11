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

        # noise functions
        self.noise_pos = lambda pos, vel : -1*vel
        self.noise_vel = lambda pos, vel : 0.01*pos
        self.noise_pos_gen = lambda pos, vel, c: c*vel
        self.noise_vel_gen = lambda pos, vel, d: d*pos

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # original model: initialization with 0 speed and coordinate between -0.6 and -0.4
        self.state = np.array([self.np_random.uniform(low=-0.52, high=-0.5), 0])
        # simplified: fixed initialization 
        #self.state = np.array([-0.4, 0])
        self.time = 0

        # Remember the initial observation: 
        self.init_pos_obs = self.state[0] + self.noise_pos(self.state[0], self.state[1])
        self.init_vel_obs = self.state[1] + self.noise_vel(self.state[0], self.state[1])

        # Initialize particles  
        #self.particles_pos = self.np_random.uniform(low=-0.6, high=-0.4, size=self.partct)
        #self.particles_pos_init = self.particles_pos.copy() # save for traceability 
        #self.particles_vel = np.array([0.]*self.partct)
        self.weights = np.array([1.]*self.partct)
        self.particles_c = self.np_random.uniform(low=-2, high=2, size=self.partct)
        self.particles_d = self.np_random.uniform(low=-0.05, high=0.05, size=self.partct)

        self.force_log = np.array([])

        print("Resetting particles and weights, initial observations: ", self.init_pos_obs, self.init_vel_obs) 


        return np.array(self.state)

    def step(self, action):
        print("Step for time =", self.time)

        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], -1.0), 1.0)
        # save to the log
        self.force_log = np.append(self.force_log, force)

        #if self.time > 77: 
        #    done = True

        # process noise for the true model
        proc_noise = [0, 0]   
        #proc_noise = self.np_random.normal([0, 0], [0.001, 0.0001]) 
        #print(proc_noise)

        # GT state propagation
        position, velocity = self.model_step(position, velocity, force, proc_noise) 
        self.state = np.array([position, velocity])

        # updating the simulation state 
        done = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        reward = 0
        if done:
            reward = 100.0
        reward-= math.pow(action[0],2)*0.1

        # new observed values
        obs_pos = position + self.noise_pos(position[0], velocity[0]) 
        obs_vel = velocity + self.noise_vel(position[0], velocity[0]) 
        obs_state = np.array([obs_pos, obs_vel])
                   
        # particle process noise 
        for i in range(self.partct): 
            self.particles_c[i] += self.np_random.normal(0.0, 0.001) # jumpy with var 0.05, steady with 0.01
            self.particles_d[i] += self.np_random.normal(0.0, 0.001) # jumpy with var 0.01, steady with 0.001

        # Updating particle weights based on propagation from the beginning
        for i in range(self.partct): 
            # get the supposed GT at the first step 
            pos_gt, vel_gt = self.obs_to_true(self.init_pos_obs, self.init_vel_obs, self.particles_c[i], self.particles_d[i]) 

            # propagate the old GT to new GT 
            for j in range(self.time + 1):
                pos_gt, vel_gt = self.model_step(np.array([pos_gt]), np.array([vel_gt]), self.force_log[j], [0,0])
                pos_gt = pos_gt[0]
                vel_gt = vel_gt[0]

            # get the expected observations from the new GT 
            pred_pos = pos_gt + self.noise_pos_gen(pos_gt, vel_gt, self.particles_c[i]) 
            pred_vel = vel_gt + self.noise_vel_gen(pos_gt, vel_gt, self.particles_d[i]) 
            
            # adjust weights exponentially to observation deltas
            self.weights[i] = math.exp(-abs(obs_pos - pred_pos) - abs(obs_vel - pred_vel))

            #print("For particle", self.particles_c[i], self.particles_d[i])
            #print("GT estimates", pos_gt, vel_gt) 
            #print("Obs predictions", pred_pos, pred_vel) 
            #print("Weight:", self.weights[i])

        # normalize weights
        self.weights = self.weights/np.sum(self.weights)

        # resample particles 
        assert(np.abs(np.sum(self.weights) - 1) < 1e-6) # make sure there are no precision issues
        self.particles_c = np.random.choice(self.particles_c, self.partct, p=self.weights) 
        self.particles_d = np.random.choice(self.particles_d, self.partct, p=self.weights) 

        # compute the means/stds, weighted
        avgc = np.average(self.particles_c, weights=self.weights)
        avgd = np.average(self.particles_d, weights=self.weights)
        stdc = math.sqrt(np.average((self.particles_c-avgc)**2, weights=self.weights))
        stdd = math.sqrt(np.average((self.particles_d-avgd)**2, weights=self.weights))

        # the desired intervals
        interval_c = [-1, 1] 
        #interval_c = [-3, 3] 
        interval_d = [-0.01, 0.01] 
        #interval_d = [-0.02, 0.02] 
        interval_init_pos = [-0.52, -0.48] 

        prob_c = self.estimate_part_in_interval(self.particles_c, interval_c) 
        prob_d = self.estimate_part_in_interval(self.particles_d, interval_d) 

        # get an array of estimated positions, velocities
        poss, vels = self.obs_to_true(self.init_pos_obs, self.init_vel_obs, self.particles_c, self.particles_d) 
        prob_init_pos = self.estimate_part_in_interval(poss, interval_init_pos) # implicitly uses weights, too

        assert(0 <= prob_c <= 1)
        assert(0 <= prob_d <= 1)
        assert(0 <= prob_init_pos <= 1)

        # PRINTOUT 
        print("True values:")
        print(self.state) 

        print("Observed values:") 
        print(obs_state) 

        #print("Stepped particles:") 
        #print(self.particles_c) 
        #print(self.particles_d) 
        
        #print("Weights:") 
        #print(self.weights) 

        print("Mean c:", avgc) 
        print("Mean d:", avgd) 
        print("Stdev c:", stdc) 
        print("Stdev d:", stdd) 

        print("Prob of c in ", interval_c, ": ", prob_c) 
        print("Prob of d in ", interval_d, ": ", prob_d) 
        print("Prob of init pos in ", interval_init_pos, ": ", prob_init_pos) 

        #print("Estimated prob of having started in [-0.51, -0.5]:", initprob)

        self.time += 1 
        #return self.state, reward, done, {}
        return obs_state, reward, done, [avgc, avgd, stdc, stdd, prob_c, prob_d, prob_init_pos]

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

    # computes the supposed true state given the observation and the C/D parameters
    def obs_to_true(self, obspos, obsvel, c, d): 
            gtpos = (c*obsvel - obspos)/(c*d - 1)
            gtvel = (d*obspos - obsvel)/(c*d - 1)
            return gtpos, gtvel


    # estimates the probability of particles in an interval, using the current weights 
    # inputs: 
        # list of 1D particles 
        # interval [a, b], where a <= b
    def estimate_part_in_interval(self, particles, interval): 
        idxRight = np.where(particles  >= interval[0])[0]
        idxLeft = np.where(particles <= interval[1])[0]
        idxInside = np.intersect1d(idxRight, idxLeft) 
        return np.sum(self.weights[idxInside])/np.sum(self.weights)

       

    # estimate the chance the initial position was in an interval, from C/D particles 
    # inputs: [a, b], where a <= b
    def estimate_init_prob_from_cd(self, interval): 
        poss, vels = self.obs_to_true(self.init_pos, self.init_vel, self.particles_c, self.particles_d) 
        self.estimate_part_in_interval(self, poss, interval) 
 


    # estimate the probability about the initial position based on the current particle weights 
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
