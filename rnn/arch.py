import math
import numpy as np

from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping

from keras import losses

import tensorflow as tf

Z_DIM = 32
ACTION_DIM = 3

HIDDEN_UNITS = 256
GAUSSIAN_MIXTURES = 5

BATCH_SIZE =100
EPOCHS = 5

RESTART_FACTOR = 1
REWARD_FACTOR = 1



class RNN():
	def __init__(self):
		self.models = self._build()
		self.model = self.models[0]
		self.forward = self.models[1]
		self.z_dim = Z_DIM
		self.action_dim = ACTION_DIM
		self.hidden_units = HIDDEN_UNITS
		self.gaussian_mixtures = GAUSSIAN_MIXTURES
		self.restart_factor = RESTART_FACTOR
		self.reward_factor = REWARD_FACTOR
		self.batch_size = BATCH_SIZE
		self.epochs = EPOCHS


	def _build(self):

		#### THE MODEL THAT WILL BE TRAINED
		rnn_x = Input(shape=(None, Z_DIM + ACTION_DIM))
		lstm = LSTM(HIDDEN_UNITS, return_sequences=True, return_state = True)

		lstm_output, _ , _ = lstm(rnn_x)
		mdn = Dense(GAUSSIAN_MIXTURES * (3*Z_DIM) + 2)(lstm_output) #

		rnn = Model(rnn_x, mdn)

		#### THE MODEL USED DURING PREDICTION
		state_input_h = Input(shape=(HIDDEN_UNITS,))
		state_input_c = Input(shape=(HIDDEN_UNITS,))

		_ , state_h, state_c = lstm(rnn_x, initial_state = [state_input_h, state_input_c])

		forward = Model([rnn_x] + [state_input_h, state_input_c], [state_h, state_c])

		#### LOSS FUNCTION

		def rnn_z_loss(y_true, y_pred):
			
			z_true, rew_true, done_true = self.get_responses(y_true)

			d = GAUSSIAN_MIXTURES * Z_DIM
			z_pred = y_pred[:,:,:3*d]

			z_pred = K.reshape(z_pred, [-1, GAUSSIAN_MIXTURES * 3])


			log_pi, mu, log_sigma = self.get_mixture_coef(z_pred)

			flat_target_data = K.reshape(z_true,[-1, 1])
			z_loss = log_pi + self.tf_lognormal(flat_target_data, mu, log_sigma)
			z_loss = -K.log(K.sum(K.exp(z_loss), 1, keepdims=True))

			z_loss = K.mean(z_loss) # mean over rollout length and z dim

			return z_loss

		def rnn_rew_loss(y_true, y_pred):
		
			z_true, rew_true, done_true = self.get_responses(y_true)

			d = GAUSSIAN_MIXTURES * Z_DIM
			reward_pred = y_pred[:,:,(3*d):(3*d+1)]

			rew_loss =  K.binary_crossentropy(rew_true, reward_pred)
			
			rew_loss = K.mean(rew_loss)

			return rew_loss

		def rnn_done_loss(y_true, y_pred):
			z_true, rew_true, done_true = self.get_responses(y_true)

			d = GAUSSIAN_MIXTURES * Z_DIM
			done_pred = y_pred[:,:,(3*d+1):]
		
			done_loss = K.binary_crossentropy(done_true, done_pred)
			done_loss = K.mean(done_loss)

			return done_loss


		def rnn_loss(y_true, y_pred):

			z_loss = rnn_z_loss(y_true, y_pred)
			rew_loss = rnn_rew_loss(y_true, y_pred)
			done_loss = rnn_done_loss(y_true, y_pred)

			return z_loss + REWARD_FACTOR * rew_loss + RESTART_FACTOR * done_loss  #+ rnn_kl_loss(y_true, y_pred)


		rnn.compile(loss=rnn_loss, optimizer='rmsprop', metrics = [rnn_z_loss, rnn_rew_loss, rnn_done_loss])

		return (rnn,forward)

	def set_weights(self, filepath):
		self.model.load_weights(filepath)

	def train(self, rnn_input, rnn_output):

		self.model.fit(rnn_input, rnn_output,
			shuffle=False,
			epochs=1,
			batch_size=1)

		self.model.save_weights('./rnn/weights.h5')

	def save_weights(self, filepath):
		self.model.save_weights(filepath)

	def get_responses(self, y_true):

		z_true = y_true[:,:,:Z_DIM]
		rew_true = y_true[:,:,Z_DIM: (Z_DIM + 1) ]
		done_true = y_true[:,:,(Z_DIM + 1):]

		return z_true, rew_true, done_true


	def get_mixture_coef(self, z_pred):

		# d = GAUSSIAN_MIXTURES

		# y_pred = K.reshape(y_pred, [-1, 3*GAUSSIAN_MIXTURES])
		
		# rollout_length = K.shape(y_pred)[1]
		
		# log_pi = y_pred[:,:d]
		# mu = y_pred[:,d:(2*d)]
		# log_sigma = y_pred[:,(2*d):(3*d)]

		log_pi, mu, log_sigma = tf.split(z_pred, 3, 1)
		

		# log_pi = K.reshape(log_pi, [-1, rollout_length, GAUSSIAN_MIXTURES, Z_DIM])
		# mu = K.reshape(mu, [-1, rollout_length, GAUSSIAN_MIXTURES, Z_DIM])
		# log_sigma = K.reshape(log_sigma, [-1, rollout_length, GAUSSIAN_MIXTURES, Z_DIM])

		log_pi = log_pi - K.log(K.sum(K.exp(log_pi), axis = 1, keepdims = True))
		# sigma = K.exp(log_sigma)

		return log_pi, mu, log_sigma


	def tf_lognormal(self, y_true, mu, log_sigma):

		logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))
		return -0.5 * ((y_true - mu) / K.exp(log_sigma)) ** 2 - log_sigma - logSqrtTwoPI


	def get_pi_idx(self, x, pdf):
		# samples from a categorial distribution
		N = pdf.shape[0]
		accumulate = 0
		for i in range(0, N):
			accumulate += pdf[i]
		if K.eval(K.greater(accumulate, x)):
			return i
		print('error with sampling ensemble')
		return -1

