#python 04_train_rnn.py --new_model

from rnn.arch import RNN
import argparse
import numpy as np

def main(args):
    
    start_batch = args.start_batch
    max_batch = args.max_batch
    new_model = args.new_model

    rnn = RNN()

    if not new_model:
        try:
          rnn.set_weights('./rnn/weights.h5')
        except:
          print("Either set --new_model or ensure ./rnn/weights.h5 exists")
          raise

    for batch_num in range(start_batch, max_batch + 1):
        print('Building batch {}...'.format(batch_num))
        new_rnn_input = np.load('./data/rnn_input_' + str(batch_num) + '.npy') 
        new_rnn_output = np.load( './data/rnn_output_' + str(batch_num) + '.npy')

        if batch_num>start_batch:
          rnn_input = np.concatenate([rnn_input, new_rnn_input])
          rnn_output = np.concatenate([rnn_output, new_rnn_output])
        else:
          rnn_input = new_rnn_input
          rnn_output = new_rnn_output


    rnn.train(rnn_input, rnn_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train RNN'))
    parser.add_argument('--start_batch', type=int, default = 0, help='The start batch number')
    parser.add_argument('--max_batch', type=int, default = 0, help='The max batch number')
    parser.add_argument('--new_model', action='store_true', help='start a new model from scratch?')

    args = parser.parse_args()

    main(args)
