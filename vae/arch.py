import numpy as np

from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Lambda, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

INPUT_DIM = (64,64,3)

CONV_FILTERS = [32,64,64, 128]
CONV_KERNEL_SIZES = [4,4,4,4]
CONV_STRIDES = [2,2,2,2]
CONV_ACTIVATIONS = ['relu','relu','relu','relu']

DENSE_SIZE = 1024

CONV_T_FILTERS = [64,64,32,3]
CONV_T_KERNEL_SIZES = [5,5,6,6]
CONV_T_STRIDES = [2,2,2,2]
CONV_T_ACTIVATIONS = ['relu','relu','relu','sigmoid']

Z_DIM = 32

BATCH_SIZE = 100
LEARNING_RATE = 0.0001
KL_TOLERANCE = 0.5

def sampling(args):
    z_mean, z_sigma = args
    epsilon = K.random_normal(shape=K.shape(z_sigma), mean=0.,stddev=1.)
    return z_mean + z_sigma * epsilon

def convert_to_sigma(z_log_var):
    return K.exp(z_log_var / 2)

class VAE():
    def __init__(self):
        self.models = self._build()
        self.full_model = self.models[0]
        self.encoder = self.models[1]
        self.encoder_mu_log_var = self.models[2]
        self.decoder = self.models[3]

        self.input_dim = INPUT_DIM
        self.z_dim = Z_DIM
        self.learning_rate = LEARNING_RATE
        self.kl_tolerance = KL_TOLERANCE

    def _build(self):
        vae_x = Input(shape=INPUT_DIM, name='observation_input')
        vae_c1 = Conv2D(filters = CONV_FILTERS[0], kernel_size = CONV_KERNEL_SIZES[0], strides = CONV_STRIDES[0], activation=CONV_ACTIVATIONS[0], name='conv_layer_1')(vae_x)
        vae_c2 = Conv2D(filters = CONV_FILTERS[1], kernel_size = CONV_KERNEL_SIZES[1], strides = CONV_STRIDES[1], activation=CONV_ACTIVATIONS[0], name='conv_layer_2')(vae_c1)
        vae_c3= Conv2D(filters = CONV_FILTERS[2], kernel_size = CONV_KERNEL_SIZES[2], strides = CONV_STRIDES[2], activation=CONV_ACTIVATIONS[0], name='conv_layer_3')(vae_c2)
        vae_c4= Conv2D(filters = CONV_FILTERS[3], kernel_size = CONV_KERNEL_SIZES[3], strides = CONV_STRIDES[3], activation=CONV_ACTIVATIONS[0], name='conv_layer_4')(vae_c3)

        vae_z_in = Flatten()(vae_c4)

        vae_z_mean = Dense(Z_DIM, name='mu')(vae_z_in)
        vae_z_log_var = Dense(Z_DIM, name='log_var')(vae_z_in)
        vae_z_sigma = Lambda(convert_to_sigma, name='sigma')(vae_z_log_var)

        vae_z = Lambda(sampling, name='z')([vae_z_mean, vae_z_sigma])
        
        vae_z_input = Input(shape=(Z_DIM,), name='z_input')

        #### DECODER: we instantiate these layers separately so as to reuse them later
        vae_dense = Dense(1024, name='dense_layer')
        vae_z_out = Reshape((1,1,DENSE_SIZE), name='unflatten')
        vae_d1 = Conv2DTranspose(filters = CONV_T_FILTERS[0], kernel_size = CONV_T_KERNEL_SIZES[0] , strides = CONV_T_STRIDES[0], activation=CONV_T_ACTIVATIONS[0], name='deconv_layer_1')
        vae_d2 = Conv2DTranspose(filters = CONV_T_FILTERS[1], kernel_size = CONV_T_KERNEL_SIZES[1] , strides = CONV_T_STRIDES[1], activation=CONV_T_ACTIVATIONS[1], name='deconv_layer_2')
        vae_d3 = Conv2DTranspose(filters = CONV_T_FILTERS[2], kernel_size = CONV_T_KERNEL_SIZES[2] , strides = CONV_T_STRIDES[2], activation=CONV_T_ACTIVATIONS[2], name='deconv_layer_3')
        vae_d4 = Conv2DTranspose(filters = CONV_T_FILTERS[3], kernel_size = CONV_T_KERNEL_SIZES[3] , strides = CONV_T_STRIDES[3], activation=CONV_T_ACTIVATIONS[3], name='deconv_layer_4')
        
        #### DECODER IN FULL MODEL
        vae_dense_model = vae_dense(vae_z)
        vae_z_out_model = vae_z_out(vae_dense_model)

        vae_d1_model = vae_d1(vae_z_out_model)
        vae_d2_model = vae_d2(vae_d1_model)
        vae_d3_model = vae_d3(vae_d2_model)
        vae_d4_model = vae_d4(vae_d3_model)

        #### DECODER ONLY
        vae_dense_decoder = vae_dense(vae_z_input)
        vae_z_out_decoder = vae_z_out(vae_dense_decoder)

        vae_d1_decoder = vae_d1(vae_z_out_decoder)
        vae_d2_decoder = vae_d2(vae_d1_decoder)
        vae_d3_decoder = vae_d3(vae_d2_decoder)
        vae_d4_decoder = vae_d4(vae_d3_decoder)

        #### MODELS

        vae_full = Model(vae_x, vae_d4_model)
        vae_encoder = Model(vae_x, vae_z)
        vae_encoder_mu_log_var = Model(vae_x, (vae_z_mean, vae_z_log_var))
        vae_decoder = Model(vae_z_input, vae_d4_decoder)

        def vae_r_loss(y_true, y_pred):
            ######## y_true.shape = (batch size, 3, 64, 64)

            # y_true_flat = K.flatten(y_true)
            # y_pred_flat = K.flatten(y_pred)

            r_loss = K.sum(K.square(y_true - y_pred), axis = [1,2,3])
            return r_loss

        def vae_kl_loss(y_true, y_pred):

            kl_loss = - 0.5 * K.sum(1 + vae_z_log_var - K.square(vae_z_mean) - K.exp(vae_z_log_var), axis = 1)
            kl_loss = K.maximum(kl_loss, KL_TOLERANCE * Z_DIM)
            return kl_loss

        def vae_loss(y_true, y_pred):
            return vae_r_loss(y_true, y_pred) + vae_kl_loss(y_true, y_pred)
        
        opti = Adam(lr=LEARNING_RATE)
        vae_full.compile(optimizer=opti, loss = vae_loss,  metrics = [vae_r_loss, vae_kl_loss])

        return (vae_full,vae_encoder, vae_encoder_mu_log_var, vae_decoder)

    def set_weights(self, filepath):
        self.full_model.load_weights(filepath)

    def train(self, data):

        self.full_model.fit(data, data,
                shuffle=True,
                epochs=1,
                batch_size=BATCH_SIZE)
        
    def save_weights(self, filepath):
        self.full_model.save_weights(filepath)
