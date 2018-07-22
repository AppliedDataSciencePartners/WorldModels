#python 04_train_rnn.py --new_model --batch_size 200
# python 04_train_rnn.py --new_model --batch_size 100

from rnn.arch import RNN
import argparse
import numpy as np
import os

ROOT_DIR_NAME = './data/'
ROLLOUT_DIR_NAME = './data/series/'


def get_filelist(N):
    filelist = os.listdir(ROLLOUT_DIR_NAME)
    filelist = [x for x in filelist if x != '.DS_Store']
    filelist.sort()
    length_filelist = len(filelist)


    if length_filelist > N:
      filelist = filelist[:N]

    if length_filelist < N:
      N = length_filelist

    return filelist, N


def random_batch(filelist, batch_size):
	N_data = len(filelist)
	indices = np.random.permutation(N_data)[0:batch_size]

	z_list = []
	action_list = []
	rew_list = []
	done_list = []

	for i in indices:
		new_data = np.load(ROLLOUT_DIR_NAME + filelist[i])

		mu = new_data['mu']
		log_var = new_data['log_var']
		action = new_data['action']
		reward = new_data['reward']
		done = new_data['done']

		reward = np.expand_dims(reward, axis=2)
		done = np.expand_dims(done, axis=2)


		s = log_var.shape

		z = mu + np.exp(log_var/2.0) * np.random.randn(*s)

		z_list.append(z)
		action_list.append(action)
		rew_list.append(reward)
		done_list.append(done)

	z_list = np.array(z_list)
	action_list = np.array(action_list)
	rew_list = np.array(rew_list)
	done_list = np.array(done_list)

	return z_list, action_list, rew_list, done_list

def main(args):
	
	new_model = args.new_model
	N = int(args.N)
	steps = int(args.steps)
	batch_size = int(args.batch_size)

	rnn = RNN() #learning_rate = LEARNING_RATE

	if not new_model:
		try:
			rnn.set_weights('./rnn/weights.h5')
		except:
			print("Either set --new_model or ensure ./rnn/weights.h5 exists")
			raise


	filelist, N = get_filelist(N)


	for step in range(steps):
		print('STEP ' + str(step))

		z, action, rew ,done = random_batch(filelist, batch_size)

		rnn_input = np.concatenate([z[:, :-1, :], action[:, :-1, :], rew[:, :-1, :]], axis = 2)
		rnn_output = np.concatenate([z[:, 1:, :], rew[:, 1:, :]], axis = 2) #, done[:, 1:, :]

		if step == 0:
			np.savez_compressed(ROOT_DIR_NAME + 'rnn_files.npz', rnn_input = rnn_input, rnn_output = rnn_output)

		rnn.train(rnn_input, rnn_output)

		if step % 10 == 0:

			rnn.model.save_weights('./rnn/weights.h5')

	rnn.model.save_weights('./rnn/weights.h5')




if __name__ == "__main__":
		parser = argparse.ArgumentParser(description=('Train RNN'))
		parser.add_argument('--N',default = 10000, help='number of episodes to use to train')
		parser.add_argument('--new_model', action='store_true', help='start a new model from scratch?')
		parser.add_argument('--steps', default = 4000, help='how many rnn batches to train over')
		parser.add_argument('--batch_size', default = 100, help='how many episodes in a batch?')

		args = parser.parse_args()

		main(args)
