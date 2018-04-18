#xvfb-run -s "-screen 0 1400x900x24" python generate_data.py car-racing --total_episodes 200 --start_batch 0 --time_steps 300

import numpy as np
import random
import config
import matplotlib.pyplot as plt

from env import make_env

import argparse

def main(args):

    env_name = args.env_name
    total_episodes = args.total_episodes
    start_batch = args.start_batch
    time_steps = args.time_steps
    render = args.render

    env = make_env(env_name)
    s = 0
    batch = start_batch


    while s < total_episodes:
        obs_data = []
        action_data = []

        for i_episode in range(200):
            print('-----')
            observation = env.reset()
            observation = config.adjust_obs(observation)
            if render:
                plt.imshow(observation)
                plt.show()

            env.render()
            done = False
            action = env.action_space.sample()
            t = 0
            obs_sequence = []
            action_sequence = []

            while t < time_steps and not done:
                t = t + 1
                
                action = config.generate_data_action(t, action)
                
                obs_sequence.append(observation)
                action_sequence.append(action)

                observation, reward, done, info = env.step(action)
                observation = config.adjust_obs(observation)

                if render:
                    env.render()

            obs_data.append(obs_sequence)
            action_data.append(action_sequence)
            
            print("Batch {} Episode {} finished after {} timesteps".format(batch, i_episode, t+1))
            print("Current dataset contains {} observations".format(sum(map(len, obs_data))))

            s = s + 1

        print("Saving dataset for batch {}".format(batch))
        np.save('./data/obs_data_' + str(batch), obs_data)
        np.save('./data/action_data_' + str(batch), action_data)

        batch = batch + 1

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=('Create new training data'))
  parser.add_argument('env_name', type=str, help='name of environment')
  parser.add_argument('--total_episodes', type=int, default = 200, help='total number of episodes to generate')
  parser.add_argument('--start_batch', type=int, default = 0, help='start_batch number')
  parser.add_argument('--time_steps', type=int, default = 300, help='how many timesteps at start of episode?')
  parser.add_argument('--render', action='store_true', help='render the env as data is generated')


  args = parser.parse_args()
  main(args)


  config