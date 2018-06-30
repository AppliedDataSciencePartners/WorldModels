import numpy as np
#import gym
from custom_envs.car_racing import CarRacing
from custom_envs.car_racing_dream import CarRacingDream


def make_env(env_name, seed=-1, render_mode=False, model = None):
	if env_name == 'car_racing':
		env = CarRacing()
		if (seed >= 0):
			env.seed(seed)
	elif env_name == 'car_racing_dream':
		env = CarRacingDream(model)
		if (seed >= 0):
			env.seed(seed)
	else:
		print("couldn't find this env")

	return env
