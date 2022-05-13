import gym
from gym import spaces
import numpy as np
import scipy.io as sio
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math

# Fish parameters
DEPTH = 45.0
MIN_HEADING_ACTION = math.radians(-5.0)
MAX_HEADING_ACTION = math.radians(5.0)
MIN_SPEED_ACTION = 0.0
MAX_SPEED_ACTION = 1.5433
TIME_STEP = 1.0#s
SONAR_RANGE = 50#m

MIN_SPEED = 0.51444
MAX_SPEED = 2.50
FIXED_SPEED_ACTION = 0.48556

# training parameters
STEP_REWARD_GAIN = 2#0.5
HEADING_REWARD_GAIN = 5
INPUT_REWARD_GAIN = -0.5
RANGE_REWARD_PENALTY = -0.1
CRASH_PENALTY = -100

GOAL_RANGE = 30.0

MAX_DISTANCE = 50
MIN_DISTANCE = 10

PIPE_LENGTH = 400

INIT_X_NOISE = 5
INIT_Y_NOISE = 2.5

OBSTACLE_SPAWN_X_NOISE = 1
OBSTACLE_SPAWN_Y_NOISE = 0.5
OBSTACLE_SPAWN_VEL_X_NOISE = 0.001
OBSTACLE_SPAWN_VEL_Y_NOISE = 0.001

OBSTACLE_RADIUS = 1.5

def truncate(A):

    new_A = A

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):

            #new_A[i][j] = math.trunc(1e15 * A[i][j]) / 1.0e15
            new_A[i][j] = float('%.15f'%(A[i][j]))

    return new_A

class World:

    def __init__(self, pos_x, pos_y, heading, episode_length, obs_pos_x, obs_pos_y, obs_vel_x, obs_vel_y):

        model = sio.loadmat('model_oneHz.mat')

        self.A = model['A']
        self.B = model['B']
        self.C = model['C']
        self.D = model['D']
        self.K = model['K']

        self.x = np.array([0.0, 0.0, 0.0, 0.0])
        self.e = np.array([0.0, 0.0, 0.0])
        self.y = np.array([0, 0, DEPTH])

        self.pos_x = pos_x
        # self.pos_y = pos_y
        # self.heading = heading
        self.pos_y_low = pos_y[0]
        self.pos_y_high = pos_y[1]
        self.heading_low = heading[0]
        self.heading_high = heading[1]
        
        self.init_pos_x = pos_x
        self.init_pos_y = pos_y

        self.obs_pos_x = obs_pos_x
        self.obs_pos_y = obs_pos_y
        self.obs_vel_x = obs_vel_x
        self.obs_vel_y = obs_vel_y

        self.init_obs_pos_x_low = obs_pos_x[0]
        self.init_obs_pos_x_high = obs_pos_x[1]
        self.init_obs_pos_y_low = obs_pos_y[0]
        self.init_obs_pos_y_high = obs_pos_y[1]
        self.init_obs_vel_x = obs_vel_x
        self.init_obs_vel_y = obs_vel_y

        # step parameters
        self.cur_step = 0
        self.episode_length = episode_length
        self.time_step = 1

        # storage
        self.allX = []
        self.allY = []
        self.allHeading = []
        # self.allX.append(self.pos_x)
        # self.allY.append(pos_y)
        # self.allHeading.append(heading)

        self.allXObs = []
        self.allYObs = []
        self.allXObs.append(self.obs_pos_x)
        self.allYObs.append(self.obs_pos_y)


        # parameters needed for consistency with gym environments
        self.obs_low = np.array( [math.radians(-180.0), -1.0 * MAX_DISTANCE, MIN_DISTANCE, 0])
        self.obs_high = np.array( [math.radians(180.0), -1.0 * MIN_DISTANCE, MAX_DISTANCE, SONAR_RANGE])

        self.action_space = spaces.Box(low=MIN_HEADING_ACTION, high=MAX_HEADING_ACTION, shape=(1,))
        self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high)

        self._max_episode_steps = episode_length

    def reset(self):
        self.x = np.array([0.0, 0.0, 0.0, 0.0])
        self.e = np.array([0.0, 0.0, 0.0])
        
        self.cur_step = 0
        self.pos_x = (np.random.random() * INIT_X_NOISE)
        # self.pos_y = self.init_pos_y + (2 * (np.random.random() - 0.5) * INIT_Y_NOISE)
        # self.heading = 0.0 #math.radians(-5.0) + (np.random.random() * math.radians(10.0))
        self.y = np.array([0, 0, DEPTH])

        # randomize y and heading
        self.pos_y = np.random.uniform(self.pos_y_low, self.pos_y_high)
        self.heading = np.random.uniform(self.heading_low, self.heading_high)

        init_obs_pos_x = np.random.uniform(self.init_obs_pos_x_low, self.init_obs_pos_x_high)
        init_obs_pos_y = np.random.uniform(self.init_obs_pos_y_low, self.init_obs_pos_y_high)
        heading = np.random.uniform(0, np.pi/4)
        init_obs_vel_x = self.init_obs_vel_x * np.cos(heading)
        init_obs_vel_y = self.init_obs_vel_x * np.sin(heading)


        self.obs_pos_x = init_obs_pos_x + (2 * (np.random.random() - 0.5) * OBSTACLE_SPAWN_X_NOISE)
        self.obs_pos_y = init_obs_pos_y + (2 * (np.random.random() - 0.5) * OBSTACLE_SPAWN_Y_NOISE)
        self.obs_vel_x = init_obs_vel_x + (2 * (np.random.random() - 0.5) * OBSTACLE_SPAWN_VEL_X_NOISE) ## FIXME: adds noise to the obstacle velocity
        self.obs_vel_y = init_obs_vel_y + (2 * (np.random.random() - 0.5) * OBSTACLE_SPAWN_VEL_Y_NOISE) ## FIXME: adds noise to the obstacle velocity

        self.allX = []
        self.allY = []
        self.allX.append(self.pos_x)
        self.allY.append(self.pos_y)
        self.allHeading.append(self.heading)

        self.allXObs = []
        self.allYObs = []
        self.allXObs.append(self.obs_pos_x)
        self.allYObs.append(self.obs_pos_y)

        pipe_heading = -1.0 * self.y[0]
        stbd_range = self.pos_y / math.cos(self.y[0])
        port_range = -1.0 * stbd_range
        obs_dist = self.get_dist_to_obs()
        
        measurements = np.array([pipe_heading, port_range, stbd_range, obs_dist])
        return measurements

    def get_dist_to_obs(self):
        # compute obstacle intersection
        a = np.tan(self.y[0])
        b = -1
        c = self.pos_y - self.pos_x * a
        obs_dist_to_heading = np.abs(a * self.obs_pos_x + b * self.obs_pos_y + c) / np.sqrt(a*a + b*b)

        obs_dist = SONAR_RANGE
        if obs_dist_to_heading <= OBSTACLE_RADIUS and self.pos_x < self.obs_pos_x: # only see the obstacle if to the left of it
            inter_obs_dist = np.sqrt(math.pow(self.pos_x - self.obs_pos_x, 2) + math.pow(self.pos_y - self.obs_pos_y, 2))
            if inter_obs_dist < SONAR_RANGE:
                obs_dist = inter_obs_dist

        return obs_dist

    def step(self, action):
        self.cur_step += 1

        heading_delta = action[0]
        speed = FIXED_SPEED_ACTION

        # Constrain turning input
        if heading_delta > MAX_HEADING_ACTION:
            heading_delta = MAX_HEADING_ACTION

        if heading_delta < MIN_HEADING_ACTION:
            heading_delta = MIN_HEADING_ACTION

        if speed < MIN_SPEED_ACTION:
            speed = MIN_SPEED_ACTION
        
        if speed > MAX_SPEED_ACTION:
            speed = MAX_SPEED_ACTION

        abs_heading = self.y[0] + heading_delta
        abs_heading = abs_heading if abs_heading < math.pi else abs_heading - (2*math.pi)

        u = np.array([abs_heading, MIN_SPEED + speed, DEPTH])
        
        #y = np.dot(self.C,self.x) + np.dot(self.D,u) + self.e
        
        self.x = np.dot(self.A,self.x) + np.dot(self.B,u) + np.dot(self.K,self.e)
        self.y = np.dot(self.C,self.x) + np.dot(self.D,u) + self.e

        self.heading = self.y[0]
        self.heading = self.heading if self.heading < math.pi else self.heading - (2*math.pi)
        self.pos_x += self.y[1] * math.cos(self.heading)
        self.pos_y += self.y[1] * -1.0 * math.sin(self.heading)
            
        self.obs_pos_x = self.obs_pos_x + self.obs_vel_x
        self.obs_pos_y = self.obs_pos_y + self.obs_vel_y

        self.allX.append(self.pos_x)
        self.allY.append(self.pos_y)
        self.allXObs.append(self.obs_pos_x)
        self.allYObs.append(self.obs_pos_y)

        # Measurements
        pipe_heading = -1.0 * self.y[0]
        stbd_range = self.pos_y / math.cos( self.y[0] )
        port_range = -1.0 * stbd_range

        obs_dist = self.get_dist_to_obs()
        
        measurements = np.array([pipe_heading, port_range, stbd_range, obs_dist])
        
        # Compute reward
        terminal = False
        reward = 0.0
        collision = False
        
        if self.pos_x > PIPE_LENGTH:
            #print("off the end")
            terminal = True
        elif self.pos_x < -10.0:
            #print("off the beginning")
            terminal = True

        if self.pos_y > MAX_DISTANCE or self.pos_y < MIN_DISTANCE:
            #print("too far")
            terminal = True
            reward += CRASH_PENALTY

        if np.sqrt(math.pow(self.pos_x - self.obs_pos_x, 2) + math.pow(self.pos_y - self.obs_pos_y, 2)) < OBSTACLE_RADIUS:
            print("crash into obstacle")
            terminal = True
            reward += CRASH_PENALTY
            collision = True

        if self.cur_step == self.episode_length:
            terminal = True

        
        reward += STEP_REWARD_GAIN
        reward += INPUT_REWARD_GAIN * abs(heading_delta)
        
        if( abs(pipe_heading) < math.radians(5.0) ):
            pass#reward += HEADING_REWARD_GAIN - abs(pipe_heading)

        reward += RANGE_REWARD_PENALTY * abs(stbd_range - GOAL_RANGE)

        # return measurements, reward, terminal, -1
        return measurements, reward, terminal, collision

    def plot_trajectory(self):
        fig = plt.figure()

        plt.plot(np.array([0.0, PIPE_LENGTH]), np.array([0.0, 0.0]), 'b', linewidth=3)

        plt.plot(self.allX, self.allY, 'r--')

        plt.plot(self.allXObs, self.allYObs, 'b--')

        plt.show()

    def plot_pipe(self):
        plt.plot(np.array([0.0, PIPE_LENGTH]), np.array([0.0, 0.0]), 'b', linewidth=3)
