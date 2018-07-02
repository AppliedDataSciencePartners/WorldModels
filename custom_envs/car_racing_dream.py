import sys, math
import numpy as np

from scipy.misc import imresize as resize
from scipy.misc import toimage as toimage

import gym

from gym import spaces
from gym.spaces.box import Box
from gym.utils import seeding
from gym.envs.classic_control import rendering

import pyglet
from pyglet import gl

import tensorflow as tf
import keras.backend as K


FPS = 50

SCREEN_X = 64
SCREEN_Y = 64
FACTOR = 8

GAUSSIAN_MIXTURES = 5
Z_DIM = 32


initial_mu = np.load('./data/initial_mu_' + str(0) + '.npy')
initial_logvar = np.load('./data/initial_logvar_' + str(0) + '.npy')

initial_mu_logvar = [list(elem) for elem in zip(initial_mu, initial_logvar)]



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
        self.observation_space = Box(low=-50., high=50., shape=(model.rnn.z_dim,)) # , dtype=np.float32
        self.action_space = spaces.Box( np.array([-1,0,0]), np.array([+1,+1,+1]))  # steer, gas, brake
        
        self.seed()

        self.model = model

        self.viewer = None
        self.t = None

        self.z = None
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def sample_z(self, mu, sigma):
        z = mu + sigma * self.np_random.randn(*sigma.shape)
        return z



    def reset(self):
        idx = self.np_random.randint(0, len(initial_mu_logvar))
        init_mu, init_logvar = initial_mu_logvar[idx]

        init_sigma = np.exp(init_logvar / 2)

        self.z = self.sample_z(init_mu, init_sigma)
        self.t = 0
        return self.z

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None



    def get_mixture_coef(self, y_pred):

        d = GAUSSIAN_MIXTURES * Z_DIM
        
        rollout_length = y_pred.shape[1]
        
        pi = y_pred[:,:,:d]
        mu = y_pred[:,:,d:(2*d)]
        log_sigma = y_pred[:,:,(2*d):(3*d)]
        reward = y_pred[:,:,(3*d):(3*d+1)]
        done = y_pred[:,:,(3*d+1):]
        
        pi = np.reshape(pi, [-1, rollout_length, GAUSSIAN_MIXTURES, Z_DIM])
        mu = np.reshape(mu, [-1, rollout_length, GAUSSIAN_MIXTURES, Z_DIM])
        log_sigma = np.reshape(log_sigma, [-1, rollout_length, GAUSSIAN_MIXTURES, Z_DIM])

        pi = np.exp(pi) / np.sum(np.exp(pi), axis=2, keepdims=True)
        sigma = np.exp(log_sigma)

        return pi, mu, sigma, reward, done

    def sample_next_mdn_output(self, action):  

        z_dim = self.model.rnn.z_dim

        input_to_rnn = [np.array([[np.concatenate([self.z, action])]])]
        
        y_pred = self.model.rnn.model.predict(input_to_rnn)

        pi, mu, sigma, reward, done = self.get_mixture_coef(y_pred)

        pi = pi[0,0,:,:]
        mu = mu[0,0,:,:]
        sigma = sigma[0,0,:,:]
        reward = reward[0,0,:]
        done = done[0,0,:]

        chosen_pi = np.zeros(z_dim)
        chosen_mu = np.zeros(z_dim)
        chosen_sigma = np.zeros(z_dim)

        for j in range(z_dim):
          idx = get_pi_idx(self.np_random.rand(), pi[:,j])
          chosen_pi[j] = idx
          chosen_mu[j] = mu[idx, j]
          chosen_sigma[j] = sigma[idx, j]

        next_z = self.sample_z(chosen_mu, chosen_sigma)

        reward = np.exp(reward) / (1 + np.exp(reward))
        done = np.exp(done) / (1 + np.exp(done))

        if reward > 0.5:
            next_reward = 3.2
        else:
            next_reward = -0.1

        if done > 0.5:
            next_done = True
        else:
            next_done = False

        return next_z, next_reward, next_done


    def step(self, action):
        # print(self.t)
        self.t += 1
        next_z, next_reward, next_done = self.sample_next_mdn_output(action)
        if self.t > 3000:
          next_done = True
        self.z = next_z
        return next_z, next_reward, next_done, {}

    def decode_obs(self, z):
        # decode the latent vector
        img = self.model.vae.decoder.predict(np.array([z]))[0] * 255.
        img = np.round(img).astype(np.uint8)
        img = img.reshape(*self.model.vae.input_dim)
        return img



    def render(self, mode='human', close=False):

        if close:
          if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
          return

        img = self.model.vae.decoder.predict(np.array([self.z]))[0]

        img = resize(img, (int(np.round(SCREEN_Y*FACTOR)), int(np.round(SCREEN_X*FACTOR))))

        if self.t > 0:
          pass
          #toimage(img, cmin=0, cmax=255).save('output/'+str(self.t)+'.png')

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
    env = CarRacing()
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
