import math
import numpy as np

from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping

from keras import losses
from keras.optimizers import Adam

import tensorflow as tf

Z_DIM = 32
ACTION_DIM = 3

HIDDEN_UNITS = 256
GAUSSIAN_MIXTURES = 5

BATCH_SIZE =100
EPOCHS = 4000

REWARD_FACTOR = 1
#RESTART_FACTOR = 0

LEARNING_RATE = 0.001
# MIN_LEARNING_RATE = 0.001
# DECAY_RATE = 1.0


class RNN():
	def __init__(self): #, learning_rate = 0.001
		
		self.z_dim = Z_DIM
		self.action_dim = ACTION_DIM
		self.hidden_units = HIDDEN_UNITS
		self.gaussian_mixtures = GAUSSIAN_MIXTURES
		#self.restart_factor = RESTART_FACTOR
		self.reward_factor = REWARD_FACTOR
		self.batch_size = BATCH_SIZE
		self.epochs = EPOCHS
		self.learning_rate = LEARNING_RATE

		self.models = self._build()
		self.model = self.models[0]
		self.forward = self.models[1]


	def _build(self):

		#### THE MODEL THAT WILL BE TRAINED
		rnn_x = Input(shape=(None, Z_DIM + ACTION_DIM))
		lstm = LSTM(HIDDEN_UNITS, return_sequences=True, return_state = True)

		lstm_output_model, _ , _ = lstm(rnn_x)
		mdn = Dense(GAUSSIAN_MIXTURES * (3*Z_DIM) + 1) 

		mdn_model = mdn(lstm_output_model)

		model = Model(rnn_x, mdn_model)

		#### THE MODEL USED DURING PREDICTION
		state_input_h = Input(shape=(HIDDEN_UNITS,))
		state_input_c = Input(shape=(HIDDEN_UNITS,))

		lstm_output_forward , state_h, state_c = lstm(rnn_x, initial_state = [state_input_h, state_input_c])

		mdn_forward = mdn(lstm_output_forward)

		forward = Model([rnn_x] + [state_input_h, state_input_c], [mdn_forward, state_h, state_c])

		#### LOSS FUNCTION

		def rnn_z_loss(y_true, y_pred):
			
			z_true, rew_true = self.get_responses(y_true) #, done_true 

			d = GAUSSIAN_MIXTURES * Z_DIM
			z_pred = y_pred[:,:,:(3*d)]
			z_pred = K.reshape(z_pred, [-1, GAUSSIAN_MIXTURES * 3])

			log_pi, mu, log_sigma = self.get_mixture_coef(z_pred)

			flat_target_data = K.reshape(z_true,[-1, 1])

			z_loss = log_pi + self.tf_lognormal(flat_target_data, mu, log_sigma)
			z_loss = -K.log(K.sum(K.exp(z_loss), 1, keepdims=True))

			z_loss = K.mean(z_loss) # mean over rollout length and z dim

			return z_loss

		def rnn_rew_loss(y_true, y_pred):
		
			z_true, rew_true = self.get_responses(y_true) #, done_true

			d = GAUSSIAN_MIXTURES * Z_DIM
			reward_pred = y_pred[:,:,(3*d):(3*d+1)]

			rew_loss =  K.binary_crossentropy(rew_true, reward_pred)
			
			rew_loss = K.mean(rew_loss)

			return REWARD_FACTOR * rew_loss

		# def rnn_done_loss(y_true, y_pred):
		# 	z_true, rew_true = self.get_responses(y_true) #, done_true

		# 	d = GAUSSIAN_MIXTURES * Z_DIM
		# 	done_pred = y_pred[:,:,(3*d+1):]
		
		# 	done_loss = K.binary_crossentropy(done_true, done_pred)
		# 	done_loss = K.mean(done_loss)

		#	return RESTART_FACTOR * done_loss


		def rnn_loss(y_true, y_pred):

			z_loss = rnn_z_loss(y_true, y_pred)
			rew_loss = rnn_rew_loss(y_true, y_pred)
			#done_loss = rnn_done_loss(y_true, y_pred)

			return z_loss + rew_loss # + done_loss  #+ rnn_kl_loss(y_true, y_pred)

		opti = Adam(lr=LEARNING_RATE)
		model.compile(loss=rnn_loss, optimizer=opti, metrics = [rnn_z_loss, rnn_rew_loss]) #, rnn_done_loss
		# model.compile(loss=rnn_loss, optimizer='rmsprop', metrics = [rnn_z_loss, rnn_rew_loss, rnn_done_loss])

		return (model,forward)

	def set_weights(self, filepath):
		self.model.load_weights(filepath)

	def train(self, rnn_input, rnn_output):

		self.model.fit(rnn_input, rnn_output,
			shuffle=False,
			epochs=1,
			batch_size=BATCH_SIZE)


	def save_weights(self, filepath):
		self.model.save_weights(filepath)

	def get_responses(self, y_true):

		z_true = y_true[:,:,:Z_DIM]
		rew_true = y_true[:,:,Z_DIM: (Z_DIM + 1) ]
		# done_true = y_true[:,:,(Z_DIM + 1):]

		return z_true, rew_true #, done_true


	def get_mixture_coef(self, z_pred):

		log_pi, mu, log_sigma = tf.split(z_pred, 3, 1)
		log_pi = log_pi - K.log(K.sum(K.exp(log_pi), axis = 1, keepdims = True))

		return log_pi, mu, log_sigma


	def tf_lognormal(self, z_true, mu, log_sigma):

		logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))
		return -0.5 * ((z_true - mu) / K.exp(log_sigma)) ** 2 - log_sigma - logSqrtTwoPI



