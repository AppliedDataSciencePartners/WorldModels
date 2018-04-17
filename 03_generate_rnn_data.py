#python 03_generate_rnn_data.py --max_batch 0

from vae.arch import VAE
import argparse

def main(args):

    start_batch = args.start_batch
    max_batch = args.max_batch

    vae = VAE()
    try:
      vae.set_weights('./vae/weights.h5')
    except:
      print("./vae/weights.h5 does not exist - ensure you have run 02_train_vae.py first")
      raise

    for i in range(start_batch, max_batch + 1):
        print('Generating batch {}...'.format(i))
        vae.generate_rnn_data(i)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=('Train VAE'))
  parser.add_argument('--start_batch', type=int, default = 0, help='The max batch number')
  parser.add_argument('--max_batch', type=int, default = 0, help='The max batch number')

  args = parser.parse_args()

  main(args)
