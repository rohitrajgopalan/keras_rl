import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense

from keras_rl.utils.utils import get_keras_optimizer, get_hidden_layer_sizes


class DuelingTDNetwork(keras.Model):
    def __init__(self, input_dims, n_actions, network_args, optimizer_type, optimizer_args={}, use_mse=True):
        super(DuelingTDNetwork, self).__init__()

        self.input_dim_len = len(input_dims)
        self.total_fc_input_dims = 1
        for dim in input_dims:
            self.total_fc_input_dims *= dim

        fc_dims = network_args['fc_dims']
        fc1_dims, fc2_dims = get_hidden_layer_sizes(fc_dims)

        self.fc1 = Dense(fc1_dims, activation='relu', input_shape=(self.total_fc_input_dims,))
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.V = Dense(1, activation=None)
        self.A = Dense(n_actions, activation=None)

        self.optimizer = get_keras_optimizer(optimizer_type, optimizer_args)
        self.loss = keras.losses.MeanSquaredError() if use_mse else keras.losses.Huber()
        gpu_device_name = tf.test.gpu_device_name()
        self.device = gpu_device_name if len(gpu_device_name) > 0 else '/device:cpu:0'

    def call(self, state, training=None, mask=None):
        if len(state.shape) == self.input_dim_len:
            states = np.array([state])
        else:
            states = np.zeros((state.shape[0], self.total_fc_input_dims))
            for i, s in enumerate(state):
                states[i] = s.flatten()
        x = self.fc1(states)
        x = self.fc2(x)
        V = self.V(x)
        A = self.A(x)

        Q = (V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True)))

        return Q

    def advantage(self, state):
        if len(state.shape) == self.input_dim_len:
            states = np.array([state])
        else:
            states = np.zeros((state.shape[0], self.input_dim_len))
            for i, s in enumerate(state):
                states[i] = s.flatten()
        x = self.fc1(states)
        x = self.fc2(x)
        A = self.A(x)

        return A
