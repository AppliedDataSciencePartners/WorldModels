import numpy as np
import random

train_envs = ['car_racing']
test_envs = ['car_racing']

def generate_data_action(t, env):
    #for the car racing example, it is important to give the car a 'push' early in the exploration so that it can find different examples of curved track.
    if t < 20:
        a = np.array([-0.1,1,0])

    else:  
        a = env.action_space.sample()

        rn = random.randint(0,9)

        if rn in [0]: #do nothing
            a = np.array([0,0,0])
        elif rn in [1,2,3,4]: #accelerate
            a = np.array([0,random.random(),0])
        elif rn in [5,6]: #left
            a = np.array([-random.random(),0,0])
        elif rn in [7,8]: #right
            a = np.array([random.random(),0,0])
        elif rn in [9]: #brake
            a = np.array([0,0,random.random()])
        else:
            pass

    #uncomment this line for truly random actions
    #a = env.action_space.sample()

    return a


def adjust_obs(obs):
    # obs[obs==0] = 255
    
    return obs.astype('float32') / 255.
    

def adjust_reward(reward):
    if reward > 0:
        reward = 1
    else:
        reward = 0
    return reward
    
