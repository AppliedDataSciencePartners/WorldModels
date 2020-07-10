import sys, math
import numpy as np

from PIL import Image
import gym

from gym import spaces
from gym.spaces.box import Box
from gym.utils import seeding
from gym.envs.classic_control import rendering

import pyglet
from pyglet import gl

import tensorflow as tf
import tensorflow.keras.backend as K

from model import make_model


FPS = 50

SCREEN_X = 64
SCREEN_Y = 64
FACTOR = 8

HIDDEN_UNITS = 256
GAUSSIAN_MIXTURES = 5
Z_DIM = 32

initial_z = np.load('./data/initial_z.npz')
initial_mu = initial_z['initial_mu']
initial_log_var = initial_z['initial_log_var']

initial_mu_log_var = [list(elem) for elem in zip(initial_mu, initial_log_var)]


def get_pi_idx(x, pdf):
  # samples from a categorial distribution
  N = pdf.size
  accumulate = 0
  for i in range(0, N):
    accumulate += pdf[i]
    if (accumulate >= x):
      return i
  random_value = np.random.randint(N)
  #print('error with sampling ensemble, returning random', random_value)
  return random_value


class CarRacingDream(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    def __init__(self, model):
        self.observation_space = Box(low=-50., high=50., shape=(model.rnn.z_dim,) , dtype = np.float32) # , dtype=np.float32
        self.action_space = spaces.Box( np.array([-1,0,0]), np.array([+1,+1,+1]) , dtype = np.float32)  # steer, gas, brake
        
        self.seed()

        self.model = model

        self.viewer = None
        self.t = None

        self.z = None
        self.h = None
        self.c = None
        self.previous_reward = None
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def sample_z(self, mu, log_sigma):
        z =  mu + (np.exp(log_sigma)) * self.np_random.randn(*log_sigma.shape) 
        return z


    def reset(self):
        idx = self.np_random.randint(0, len(initial_mu_log_var))
        init_mu, init_log_var = initial_mu_log_var[idx]

        init_log_sigma = init_log_var / 2

        self.z = self.sample_z(init_mu, init_log_sigma)
        self.h = np.zeros(HIDDEN_UNITS)
        self.c = np.zeros(HIDDEN_UNITS)
        self.previous_reward = 0

        self.t = 0
        return self.z

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


    def get_mixture_coef(self, z_pred):

        log_pi, mu, log_sigma = np.split(z_pred, 3, 1)
        log_pi = log_pi - np.log(np.sum(np.exp(log_pi), axis = 1, keepdims = True))

        return log_pi, mu, log_sigma

    def sample_next_mdn_output(self, action):  


        d = GAUSSIAN_MIXTURES * Z_DIM

        z_dim = self.model.rnn.z_dim

        input_to_rnn = [np.array([[np.concatenate([self.z, action, [self.previous_reward]])]]),np.array([self.h]),np.array([self.c])]

        out = self.model.rnn.forward.predict(input_to_rnn)

        y_pred = out[0][0][0]
        new_h = out[1][0]
        new_c = out[2][0]

        mdn_pred = y_pred[:(3*d)]
        rew_pred = y_pred[-1]
        
        mdn_pred = np.reshape(mdn_pred, [-1, GAUSSIAN_MIXTURES * 3])

        log_pi, mu, log_sigma = self.get_mixture_coef(mdn_pred)

        chosen_log_pi = np.zeros(z_dim)
        chosen_mu = np.zeros(z_dim)
        chosen_log_sigma = np.zeros(z_dim)

        # adjust temperatures
        pi = np.copy(log_pi)
        # pi -= pi.max()
        pi = np.exp(pi)
        pi /= pi.sum(axis=1).reshape(z_dim, 1)

        for j in range(z_dim):
          idx = get_pi_idx(self.np_random.rand(), pi[j])
          chosen_log_pi[j] = idx
          chosen_mu[j] = mu[j, idx]
          chosen_log_sigma[j] = log_sigma[j,idx]

        next_z = self.sample_z(chosen_mu, chosen_log_sigma)

        # print(next_z)
        # print(rew_pred)
        if rew_pred > 0:
            next_reward = 1
        else:
            next_reward = 0



        # if done > 0:
        #     next_done = True
        # else:
        #     next_done = False

        self.h = new_h
        self.c = new_c
        self.previous_reward = next_reward



        return next_z, next_reward #, next_done


    def step(self, action):
        # print(self.t)
        self.t += 1
        next_z, next_reward = self.sample_next_mdn_output(action) #, next_done
        next_done = False
        if self.t > 1000:
          next_done = True
        self.z = next_z
        return next_z, next_reward, next_done, {}


    def render(self, mode='human', close=False):

        if close:
          if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
          return

        img = self.model.vae.decoder.predict(np.array([self.z]))[0]

        img = np.array(Image(img).resize(int(np.round(SCREEN_Y*FACTOR)), int(np.round(SCREEN_X*FACTOR))))

        if self.t > 0:
          pass
        if mode == 'rgb_array':
          return img

        elif mode == 'human':
          
          if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()
          self.viewer.imshow(img)




if __name__=="__main__":
    from pyglet.window import key
    a = np.array( [0.0, 0.0, 0.0] )
    def key_press(k, mod):
        global restart
        if k==0xff0d: restart = True
        if k==key.LEFT:  a[0] = -1.0
        if k==key.RIGHT: a[0] = +1.0
        if k==key.UP:    a[1] = +1.0
        if k==key.DOWN:  a[2] = +0.8   # set 1.0 for wheels to block to zero rotation
    def key_release(k, mod):
        if k==key.LEFT  and a[0]==-1.0: a[0] = 0
        if k==key.RIGHT and a[0]==+1.0: a[0] = 0
        if k==key.UP:    a[1] = 0
        if k==key.DOWN:  a[2] = 0


    dream_model = make_model()

    env = CarRacingDream(dream_model)
    env.render()
    record_video = False
    if record_video:
        env.monitor.start('/tmp/video-test', force=True)
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    while True:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            s, r, done, info = env.step(a)
            total_reward += r
            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
                #import matplotlib.pyplot as plt
                #plt.imshow(s)
                #plt.savefig("test.jpeg")
            steps += 1
            if not record_video: # Faster, but you can as well call env.render() every time to play full window.
                env.render()
            if done or restart: break
    env.close()
