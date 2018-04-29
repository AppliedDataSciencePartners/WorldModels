#python 02_train_vae.py --new_model

from vae.arch import VAE
import argparse
import numpy as np
import config

def main(args):

  start_batch = args.start_batch
  max_batch = args.max_batch
  new_model = args.new_model

  vae = VAE()

  

  if not new_model:
    try:
      vae.set_weights('./vae/weights.h5')
    except:
      print("Either set --new_model or ensure ./vae/weights.h5 exists")
      raise

  for batch_num in range(start_batch, max_batch + 1):
    print('Building batch {}...'.format(batch_num))
    first_item = True

    for env_name in config.train_envs:
      try:
        new_data = np.load('./data/obs_data_' + env_name + '_' + str(batch_num) + '.npy')
        if first_item:
          data = new_data
          first_item = False
        else:
          data = np.concatenate([data, new_data])
        print('Found {}...current data size = {} episodes'.format(env_name, len(data)))
      except:
        pass

  train_data = np.array([item for obs in data for item in obs])
  vae.train(train_data)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=('Train VAE'))
  parser.add_argument('--start_batch', type=int, default = 0, help='The start batch number')
  parser.add_argument('--max_batch', type=int, default = 0, help='The max batch number')
  parser.add_argument('--new_model', action='store_true', help='start a new model from scratch?')
  args = parser.parse_args()

  main(args)
