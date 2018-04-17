#python 02_train_vae.py --new_model

from vae.arch import VAE
import argparse
import numpy as np


def main(args):

  start_batch = args.start_batch
  max_batch = args.max_batch
  new_model = args.new_model
  # epochs = args.epochs

  vae = VAE()

  if not new_model:
    try:
      vae.set_weights('./vae/weights.h5')
    except:
      print("Either set --new_model or ensure ./vae/weights.h5 exists")
      raise

  for batch_num in range(start_batch, max_batch + 1):
    print('Building batch {}...'.format(batch_num))
    data = np.load('./data/obs_data_' + str(batch_num) + '.npy')
    data = np.array([item for obs in data for item in obs])
    vae.train(data)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=('Train VAE'))
  parser.add_argument('--start_batch', type=int, default = 0, help='The start batch number')
  parser.add_argument('--max_batch', type=int, default = 0, help='The max batch number')

  parser.add_argument('--new_model', action='store_true', help='start a new model from scratch?')
  args = parser.parse_args()

  main(args)
