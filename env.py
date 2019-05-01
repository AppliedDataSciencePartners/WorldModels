import numpy as np
#import gym




def make_env(env_name, seed=-1, render_mode=False, model = None):
	if env_name == 'car_racing':
		from custom_envs.car_racing import CarRacing
		env = CarRacing()
		if (seed >= 0):
			env.seed(seed)
	elif env_name == 'car_racing_dream':
		from custom_envs.car_racing_dream import CarRacingDream
		env = CarRacingDream(model)
		if (seed >= 0):
			env.seed(seed)
	else:
		print("couldn't find this env")

	return env
