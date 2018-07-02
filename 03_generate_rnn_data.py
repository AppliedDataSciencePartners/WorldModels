#python 03_generate_rnn_data.py

from vae.arch import VAE
import argparse
import config
import numpy as np

def main(args):

    start_batch = args.start_batch
    max_batch = args.max_batch

    vae = VAE()

    try:
      vae.set_weights('./vae/weights.h5')
    except:
      print("./vae/weights.h5 does not exist - ensure you have run 02_train_vae.py first")
      raise

    for batch_num in range(start_batch, max_batch + 1):
      first_item = True
      print('Generating batch {}...'.format(batch_num))

      for env_name in config.train_envs:
        try:
          new_obs_data = np.load('./data/obs_data_' + env_name + '_'  + str(batch_num) + '.npy') 
          new_action_data = np.load('./data/action_data_' + env_name + '_'  + str(batch_num) + '.npy')
          new_reward_data = np.load('./data/reward_data_' + env_name + '_'  + str(batch_num) + '.npy') 
          new_done_data = np.load('./data/done_data_' + env_name + '_'  + str(batch_num) + '.npy')
          if first_item:
            obs_data = new_obs_data
            action_data = new_action_data
            reward_data = new_reward_data
            done_data = new_done_data
            first_item = False
          else:
            obs_data = np.concatenate([obs_data, new_obs_data])
            action_data = np.concatenate([action_data, new_action_data])
            reward_data = np.concatenate([reward_data, new_reward_data])
            done_data = np.concatenate([done_data, new_done_data])

          print('Found {}...current data size = {} episodes'.format(env_name, len(obs_data)))
        except:
          pass
      
      if first_item == False:
        rnn_input, rnn_output, initial_mu, initial_logvar = vae.generate_rnn_data(obs_data, action_data, reward_data, done_data)
        np.save('./data/rnn_input_' + str(batch_num), rnn_input)
        np.save('./data/rnn_output_' + str(batch_num), rnn_output)
        np.save('./data/initial_mu_' + str(batch_num), initial_mu)
        np.save('./data/initial_logvar_' + str(batch_num), initial_logvar)
      else:
        print('no data found for batch number {}'.format(batch_num))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=('Generate RNN data'))
  parser.add_argument('--start_batch', type=int, default = 0, help='The start batch number')
  parser.add_argument('--max_batch', type=int, default = 0, help='The max batch number')

  args = parser.parse_args()

  main(args)
