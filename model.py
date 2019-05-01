#python model.py car_racing --filename ./controller/car_racing.cma.4.32.best.json --render_mode --record_video
#xvfb-run -a -s "-screen 0 1400x900x24" python model.py car_racing --filename ./controller/car_racing.cma.4.32.best.json --render_mode --record_video


import numpy as np
import random
import json
import sys
import time
import importlib
import argparse

import config

from gym.wrappers import Monitor

from env import make_env
from vae.arch import VAE
from rnn.arch import RNN
from controller.arch import Controller

final_mode = False
render_mode = False
generate_data_mode = False
dream_mode = False
RENDER_DELAY = False
record_video = False
ADD_NOISE = False

def make_model():

  vae = VAE()
  vae.set_weights('./vae/weights.h5')

  rnn = RNN()
  rnn.set_weights('./rnn/weights.h5')

  controller = Controller()

  model = Model(controller, vae, rnn)
  return model

class Model:
  def __init__(self, controller, vae, rnn):

    self.input_size = vae.input_dim
    self.vae = vae
    self.rnn = rnn

    self.output_noise = controller.output_noise
    self.sigma_bias = controller.noise_bias # bias in stdev of output
    self.sigma_factor = 0.5 # multiplicative in stdev of output

    if controller.time_factor > 0:    
      self.time_factor = float(controller.time_factor)    
      self.time_input = 1
    else:
      self.time_input = 0
    
    self.output_size = controller.output_size
    
    self.sample_output = False
    self.activations = controller.activations

    self.weight = []
    self.bias = []
    self.bias_log_std = []
    self.bias_std = []
    self.param_count = 0

    self.hidden = np.zeros(self.rnn.hidden_units)
    self.cell_values = np.zeros(self.rnn.hidden_units)

    self.shapes = [(self.rnn.hidden_units + self.rnn.z_dim, self.output_size)]

    idx = 0
    for shape in self.shapes:
      self.weight.append(np.zeros(shape=shape))
      self.bias.append(np.zeros(shape=shape[1]))
      self.param_count += (np.product(shape) + shape[1])
      if self.output_noise[idx]:
        self.param_count += shape[1]
      log_std = np.zeros(shape=shape[1])
      self.bias_log_std.append(log_std)
      out_std = np.exp(self.sigma_factor*log_std + self.sigma_bias)
      self.bias_std.append(out_std)
      idx += 1

    self.render_mode = False

  def make_env(self, env_name, seed=-1, render_mode=False, model = None):
    self.render_mode = render_mode
    self.env_name = env_name
    self.env = make_env(env_name, seed=seed, render_mode=render_mode, model = model)


  def get_action(self, x, t=0, add_noise=False):
    # if add_noise = True, ignore sampling.
    h = np.array(x).flatten()
    if self.time_input == 1:
      time_signal = float(t) / self.time_factor
      h = np.concatenate([h, [time_signal]])
    num_layers = len(self.weight)
    for i in range(num_layers):
      w = self.weight[i]
      b = self.bias[i]
      h = np.matmul(h, w) + b
      if (self.output_noise[i] and add_noise):
        out_size = self.shapes[i][1]
        out_std = self.bias_std[i]
        output_noise = np.random.randn(out_size)*out_std
        h += output_noise
      h = self.activations(h)

    if self.sample_output:
      h = sample(h)

    return h



  def set_model_params(self, model_params):
    pointer = 0
    for i in range(len(self.shapes)):
      w_shape = self.shapes[i]
      b_shape = self.shapes[i][1]
      s_w = np.product(w_shape)
      s = s_w + b_shape
      chunk = np.array(model_params[pointer:pointer+s])
      self.weight[i] = chunk[:s_w].reshape(w_shape)
      self.bias[i] = chunk[s_w:].reshape(b_shape)
      pointer += s
      if self.output_noise[i]:
        s = b_shape
        self.bias_log_std[i] = np.array(model_params[pointer:pointer+s])
        self.bias_std[i] = np.exp(self.sigma_factor*self.bias_log_std[i] + self.sigma_bias)
        if self.render_mode:
          print("bias_std, layer", i, self.bias_std[i])
        pointer += s

  def load_model(self, filename):
    with open(filename) as f:    
      data = json.load(f)
    print('loading file %s' % (filename))
    self.data = data
    model_params = np.array(data[0]) # assuming other stuff is in data
    self.set_model_params(model_params)

  def get_random_model_params(self, stdev=0.1):
    return np.random.randn(self.param_count)*stdev

  def reset(self):
    self.hidden = np.zeros(self.rnn.hidden_units)
    self.cell_values = np.zeros(self.rnn.hidden_units) 

  def update(self, obs, t):
    if obs.shape == self.vae.input_dim:
      return self.vae.encoder.predict(np.array([obs]))[0]
    else:
      return obs

def evaluate(model, num_episode, max_len):

  reward, t = simulate(model, num_episode=num_episode, max_len=max_len)

  total_reward = np.mean(reward)

  return reward, total_reward

def compress_input_dct(obs):
  new_obs = np.zeros((8, 8))
  for i in range(obs.shape[2]):
    new_obs = +compress_2d(obs[:, :, i] / 255., shape=(8, 8))
  new_obs /= float(obs.shape[2])
  return new_obs.flatten()


def simulate(model, num_episode=5, seed=-1, max_len=-1, generate_data_mode = False, render_mode = False):

  reward_list = []
  t_list = []

  max_episode_length = 1000

  if max_len > 0:
    if max_len < max_episode_length:
      max_episode_length = max_len

  if (seed >= 0):
    random.seed(seed)
    np.random.seed(seed)
    model.env.seed(seed)

  for episode in range(num_episode):

    model.reset()

    obs = model.env.reset()
    reward = 0
    action = np.array([0,0,0])


    if obs is None:
      obs = np.zeros(model.input_size)

    total_reward = 0.0
    

    model.env.render("rgb_array")

    for t in range(max_episode_length):

      if obs.shape == model.vae.input_dim: ### running in real environment
        obs = config.adjust_obs(obs)
        reward = config.adjust_reward(reward)

      if render_mode:
        model.env.render("human")
        if RENDER_DELAY:
          time.sleep(0.1)
      # else:
      #   model.env.render('rgb_array')

      vae_encoded_obs = model.update(obs, t)

    
      input_to_rnn = [np.array([[np.concatenate([vae_encoded_obs, action, [reward]])]]),np.array([model.hidden]),np.array([model.cell_values])]

      out = model.rnn.forward.predict(input_to_rnn)

      y_pred = out[0][0][0]
      model.hidden = out[1][0]
      model.cell_values = out[2][0]

      controller_obs = np.concatenate([vae_encoded_obs,model.hidden])

      if generate_data_mode:
        action = config.generate_data_action(t=t, env = model.env)
      else:
        action = model.get_action(controller_obs, t=t, add_noise=ADD_NOISE)


      # print(action)
      # action = [-0.1,1,0]

      obs, reward, done, info = model.env.step(action)

      total_reward += reward

      if done:
        break

    if render_mode:
      print("reward", total_reward, "timesteps", t)
    
    reward_list.append(total_reward)
    t_list.append(t)
    model.env.close()

  return reward_list, t_list

def main(args):

  global RENDER_DELAY

  env_name = args.env_name
  filename = args.filename
  the_seed = args.seed
  final_mode = args.final_mode
  generate_data_mode = args.generate_data_mode
  dream_mode = args.dream_mode
  render_mode = args.render_mode
  record_video = args.record_video
  max_length = args.max_length
  

  if env_name.startswith("bullet"):
    RENDER_DELAY = True

  use_model = False

  model = make_model()
  print('model size', model.param_count)

  if dream_mode:
    dream_model = make_model()
    model.make_env(env_name + '_dream', render_mode=render_mode, model = dream_model)
  else:
    model.make_env(env_name, render_mode=render_mode)

  if len(filename) > 0:
    model.load_model(filename)
  else:
    params = model.get_random_model_params(stdev=0.1)
    model.set_model_params(params)
  
  if final_mode:
    total_reward = 0.0
    np.random.seed(the_seed)
    model.env.seed(the_seed)

    for i in range(100):

      reward, steps_taken = simulate(model, num_episode=1, max_len = max_length, generate_data_mode = False)
      total_reward += reward[0]
      print("episode" , i, "reward =", reward[0])
    print("seed", the_seed, "average_reward", total_reward/100)
  else:
    if record_video:
      model.env = Monitor(model.env, directory='./videos',video_callable=lambda episode_id: True, write_upon_reset=True, force=True)
    while(1):
      reward, steps_taken = simulate(model, render_mode=render_mode, num_episode=1, max_len = max_length, generate_data_mode = generate_data_mode)
      print ("terminal reward", reward, "average steps taken", np.mean(steps_taken)+1)
      #break

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=('View a trained agent'))
  parser.add_argument('env_name', type=str, help='car_racing etc - this is only used for labelling files etc, the actual environments are defined in train_envs in config.py')
  parser.add_argument('--filename', type=str, default='', help='Path to the trained model json file')
  parser.add_argument('--seed', type = int, default = 111, help='which seed?')
  parser.add_argument('--final_mode', action='store_true', help='select this to test a given controller over 100 trials')
  parser.add_argument('--generate_data_mode', action='store_true', help='uses the pick_random_action function from config')
  parser.add_argument('--dream_mode', action='store_true', help='run the model in the dreams of the agent')
  parser.add_argument('--render_mode', action='store_true', help='render the run')
  parser.add_argument('--record_video', action='store_true', help='record the run to ./videos')
  parser.add_argument('--max_length', type = int, default = -1, help='max_length of an episode')


  
  args = parser.parse_args()

  main(args)

