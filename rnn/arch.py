import math
import numpy as np

from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping

Z_DIM = 32
ACTION_DIM = 3

HIDDEN_UNITS = 256
GAUSSIAN_MIXTURES = 5

BATCH_SIZE =32
EPOCHS = 20

def get_mixture_coef(y_pred):
    
    d = GAUSSIAN_MIXTURES * Z_DIM
    
    rollout_length = K.shape(y_pred)[1]
    
    pi = y_pred[:,:,:d]
    mu = y_pred[:,:,d:(2*d)]
    log_sigma = y_pred[:,:,(2*d):(3*d)]
    #discrete = y_pred[:,3*GAUSSIAN_MIXTURES:]
    
    pi = K.reshape(pi, [-1, rollout_length, GAUSSIAN_MIXTURES, Z_DIM])
    mu = K.reshape(mu, [-1, rollout_length, GAUSSIAN_MIXTURES, Z_DIM])
    log_sigma = K.reshape(log_sigma, [-1, rollout_length, GAUSSIAN_MIXTURES, Z_DIM])

    pi = K.exp(pi) / K.sum(K.exp(pi), axis=2, keepdims=True)
    sigma = K.exp(log_sigma)
    
    return pi, mu, sigma#, discrete


def tf_normal(y_true, mu, sigma, pi):

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
        mdn = Dense(GAUSSIAN_MIXTURES * (3*Z_DIM))(lstm_output) #+ discrete_dim

        rnn = Model(rnn_x, mdn)

        #### THE MODEL USED DURING PREDICTION
        state_input_h = Input(shape=(HIDDEN_UNITS,))
        state_input_c = Input(shape=(HIDDEN_UNITS,))
        state_inputs = [state_input_h, state_input_c]
        _ , state_h, state_c = lstm(rnn_x, initial_state = [state_input_h, state_input_c])

        forward = Model([rnn_x] + state_inputs, [state_h, state_c])

        #### LOSS FUNCTION

        def rnn_r_loss(y_true, y_pred):

            pi, mu, sigma = get_mixture_coef(y_pred)
        
            result = tf_normal(y_true, mu, sigma, pi)
            
            result = -K.log(result + 1e-8)
            result = K.mean(result, axis = (1,2)) # mean over rollout length and z dim

            return result
    
        def rnn_kl_loss(y_true, y_pred):
            pi, mu, sigma = get_mixture_coef(y_pred)
            kl_loss = - 0.5 * K.mean(1 + K.log(K.square(sigma)) - K.square(mu) - K.square(sigma), axis = [1,2,3])
            return kl_loss

        def rnn_loss(y_true, y_pred):
            return rnn_r_loss(y_true, y_pred) #+ rnn_kl_loss(y_true, y_pred)


        rnn.compile(loss=rnn_loss, optimizer='rmsprop', metrics = [rnn_r_loss, rnn_kl_loss])

        return (rnn,forward)

    def set_weights(self, filepath):
        self.model.load_weights(filepath)

    def train(self, rnn_input, rnn_output):

        earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')
        callbacks_list = [earlystop]

        self.model.fit(rnn_input, rnn_output,
            shuffle=True,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.2,
            callbacks=callbacks_list)

        self.model.save_weights('./rnn/weights.h5')

    def save_weights(self, filepath):
        self.model.save_weights(filepath)

