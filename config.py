import numpy as np
import random

train_envs = ['car_racing']
test_envs = ['car_racing']

def generate_data_action(t, env):

    a = env.action_space.sample()


    # if t < 20:
    #     a = np.array([-0.1,1,0])

    # else:  
    #     a = env.action_space.sample()

    #     rn = random.randint(0,9)

    #     if rn in [0]:
    #         a = np.array([0,0,0])
    #     elif rn in [1,2,3,4]:
    #         a = np.array([0,random.random(),0])
    #     elif rn in [5,6]:
    #         a = np.array([-random.random(),0,0])
    #     elif rn in [7,8]:
    #         a = np.array([random.random(),0,0])
    #     elif rn in [9]:
    #         a = np.array([0,0,random.random()])
    #     else:
    #         pass

    return a


def adjust_obs(obs):
    # obs[obs==0] = 255
    
    return obs.astype('float32') / 255.
    
