#python 04_train_rnn.py --new_model

from rnn.arch import RNN
import argparse
import numpy as np


# LEARNING_RATE = 0.001
# MIN_LEARNING_RATE = 0.001



def random_batch(data_mu, data_logvar, data_action, data_rew, data_done, batch_size):
		N_data = len(data_mu)
		indices = np.random.permutation(N_data)[0:batch_size]

		mu = data_mu[indices]
		logvar = data_logvar[indices]
		action = data_action[indices]
		rew = data_rew[indices]
		done = data_done[indices]

		s = logvar.shape
		z = mu + np.exp(logvar/2.0) * np.random.randn(*s)

		rew = np.expand_dims(rew, axis=2)
		done = np.expand_dims(done, axis=2)

		return z, action, rew, done

def main(args):
		
		start_batch = args.start_batch
		max_batch = args.max_batch
		new_model = args.new_model

		rnn = RNN() #learning_rate = LEARNING_RATE

		if not new_model:
				try:
					rnn.set_weights('./rnn/weights.h5')
				except:
					print("Either set --new_model or ensure ./rnn/weights.h5 exists")
					raise

		for batch_num in range(start_batch, max_batch + 1):
				print('Building batch {}...'.format(batch_num))
				new_mu = np.load('./data/mu_' + str(batch_num) + '.npy') 
				new_log_var = np.load( './data/log_var_' + str(batch_num) + '.npy')
				new_action= np.load('./data/action_'  + str(batch_num) + '.npy')
				new_reward = np.load('./data/reward_' + str(batch_num) + '.npy') 
				new_done = np.load('./data/done_' + str(batch_num) + '.npy')

				if batch_num>start_batch:
					mu_data = np.concatenate([mu_data, new_mu])
					log_var_data = np.concatenate([log_var_data, new_log_var])
					action_data = np.concatenate([action_data, new_action])
					rew_data = np.concatenate([rew_data, new_reward])
					done_data = np.concatenate([done_data, new_done])
				else:
					mu_data = new_mu
					log_var_data = new_log_var
					action_data = new_action
					rew_data = new_reward
					done_data = new_done


		for epoch in range(rnn.epochs):
			print('EPOCH ' + str(epoch))

			z, action, rew ,done = random_batch(mu_data, log_var_data, action_data, rew_data, done_data, rnn.batch_size)
			
			rnn_input = np.concatenate([z[:, :-1, :], action[:, :-1, :]], axis = 2)
			rnn_output = np.concatenate([z[:, 1:, :], rew[:, 1:, :], done[:, 1:, :]], axis = 2)

			
			np.save('./data/rnn_input_' + str(epoch), rnn_input)
			np.save('./data/rnn_output_' + str(epoch), rnn_output)

			# curr_learning_rate = (LEARNING_RATE-MIN_LEARNING_RATE) * (DECAY_RATE) ** epoch + hps.min_learning_rate


			rnn.train(rnn_input, rnn_output)

			rnn.model.save_weights('./rnn/weights_' + str(epoch) +  '.h5')




if __name__ == "__main__":
		parser = argparse.ArgumentParser(description=('Train RNN'))
		parser.add_argument('--start_batch', type=int, default = 0, help='The start batch number')
		parser.add_argument('--max_batch', type=int, default = 0, help='The max batch number')
		parser.add_argument('--new_model', action='store_true', help='start a new model from scratch?')

		args = parser.parse_args()

		main(args)
