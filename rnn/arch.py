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

BATCH_SIZE =1
EPOCHS = 50




class RNN():
	def __init__(self):
		self.models = self._build()
		self.model = self.models[0]
		self.forward = self.models[1]
		self.z_dim = Z_DIM
		self.action_dim = ACTION_DIM
		self.hidden_units = HIDDEN_UNITS
		self.gaussian_mixtures = GAUSSIAN_MIXTURES

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

			pi, mu, sigma, rew_pred, done_pred = self.get_mixture_coef(y_pred)
			z_true, rew_true, done_true = self.get_responses(y_true)
		
		
			z_loss = self.tf_normal(z_true, mu, sigma, pi)
			z_loss = -K.log(z_loss + 1e-8)
			z_loss = K.mean(z_loss, axis = (1,2)) # mean over rollout length and z dim

			return z_loss

		def rnn_rew_loss(y_true, y_pred):

			pi, mu, sigma, rew_pred, done_pred = self.get_mixture_coef(y_pred)
			z_true, rew_true, done_true = self.get_responses(y_true)


			rew_loss =  K.binary_crossentropy(rew_true, rew_pred)
			rew_loss = K.mean(rew_loss, axis = (1,2))

			return rew_loss

		def rnn_done_loss(y_true, y_pred):

			pi, mu, sigma, rew_pred, done_pred = self.get_mixture_coef(y_pred)
			z_true, rew_true, done_true = self.get_responses(y_true)
		
			done_loss = K.binary_crossentropy(done_true, done_pred)
			done_loss = K.mean(done_loss, axis = (1,2))

			return done_loss


		def rnn_kl_loss(y_true, y_pred):

			pi, mu, sigma, rew_pred, done_pred = self.get_mixture_coef(y_pred)

			mu = K.flatten(mu)
			sigma = K.flatten(sigma)

			kl_loss = - 0.5 * K.mean(1 + K.log(K.square(sigma)) - K.square(mu) - K.square(sigma), axis = -1)
			return kl_loss

		def rnn_loss(y_true, y_pred):
			pi, mu, sigma, rew_pred, done_pred = self.get_mixture_coef(y_pred)
			z_true, rew_true, done_true = self.get_responses(y_true)



			z_loss = rnn_z_loss(y_true, y_pred)
			rew_loss = rnn_rew_loss(y_true, y_pred)
			done_loss = rnn_done_loss(y_true, y_pred)

			return z_loss + rew_loss + done_loss  #+ rnn_kl_loss(y_true, y_pred)


		rnn.compile(loss=rnn_loss, optimizer='rmsprop', metrics = [rnn_z_loss, rnn_rew_loss, rnn_done_loss, rnn_kl_loss])

		return (rnn,forward)

	def set_weights(self, filepath):
		self.model.load_weights(filepath)

	def train(self, rnn_input, rnn_output, validation_split = 0.2):

		earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')
		callbacks_list = [earlystop]

		self.model.fit(rnn_input, rnn_output,
			shuffle=True,
			epochs=EPOCHS,
			batch_size=BATCH_SIZE,
			validation_split=validation_split,
			callbacks=callbacks_list)

		self.model.save_weights('./rnn/weights.h5')

	def save_weights(self, filepath):
		self.model.save_weights(filepath)

	def get_responses(self, y_true):

		z_true = y_true[:,:,:Z_DIM]
		rew_true = y_true[:,:,Z_DIM: (Z_DIM + 1) ]
		done_true = y_true[:,:,(Z_DIM + 1):]

		return z_true, rew_true, done_true


	def get_mixture_coef(self, y_pred):
		
		d = GAUSSIAN_MIXTURES * Z_DIM
		
		rollout_length = K.shape(y_pred)[1]
		
		pi = y_pred[:,:,:d]
		mu = y_pred[:,:,d:(2*d)]
		log_sigma = y_pred[:,:,(2*d):(3*d)]
		reward = y_pred[:,:,(3*d):(3*d+1)]
		done = y_pred[:,:,(3*d+1):]

		pi = K.reshape(pi, [-1, rollout_length, GAUSSIAN_MIXTURES, Z_DIM])
		mu = K.reshape(mu, [-1, rollout_length, GAUSSIAN_MIXTURES, Z_DIM])
		log_sigma = K.reshape(log_sigma, [-1, rollout_length, GAUSSIAN_MIXTURES, Z_DIM])

		pi = K.exp(pi) / K.sum(K.exp(pi), axis=2, keepdims=True)
		sigma = K.exp(log_sigma)

		return pi, mu, sigma, reward, done


	def tf_normal(self, y_true, mu, sigma, pi):

		rollout_length = K.shape(y_true)[1]
		y_true = K.tile(y_true,(1,1,GAUSSIAN_MIXTURES))
		y_true = K.reshape(y_true, [-1, rollout_length, GAUSSIAN_MIXTURES,Z_DIM])

		oneDivSqrtTwoPI = 1 / math.sqrt(2*math.pi)
		result = y_true - mu
	#   result = K.permute_dimensions(result, [2,1,0])
		result = result * (1 / (sigma + 1e-8))
		result = -K.square(result)/2
		result = K.exp(result) * (1/(sigma + 1e-8))*oneDivSqrtTwoPI
		result = result * pi
		result = K.sum(result, axis=2) #### sum over gaussians
		#result = K.prod(result, axis=2) #### multiply over latent dims
		return result


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

