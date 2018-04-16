import numpy as np
#import gym
from custom_envs.car_racing import CarRacing

def make_env(env_name, seed=-1, render_mode=False):
  if env_name == 'car_racing':
    env = CarRacing()
    if (seed >= 0):
      env.seed(seed)
  else:
    print("couldn't find this env")

  return env
