import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from keras_rl.utils.utils import get_keras_optimizer, get_hidden_layer_sizes


class PolicyNetwork(keras.Model):
    def __init__(self, input_dims, n_actions, network_args, optimizer_type, optimizer_args):
        super(PolicyNetwork, self).__init__()

        self.input_dim_len = len(input_dims)
        self.total_fc_input_dims = 1
        for dim in input_dims:
            self.total_fc_input_dims *= dim

        fc_dims = network_args['fc_dims']
        fc1_dims, fc2_dims = get_hidden_layer_sizes(fc_dims)
        self.fc1 = Dense(fc1_dims, activation='relu', input_shape=(self.total_fc_input_dims,))
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.fc3 = Dense(n_actions, activation='tanh')

        self.optimizer = get_keras_optimizer(optimizer_type, optimizer_args)
        gpu_device_name = tf.test.gpu_device_name()
        self.device = gpu_device_name if len(gpu_device_name) > 0 else '/device:cpu:0'

    def call(self, state, training=None, mask=None):
        if len(state.shape) == self.input_dim_len:
            state = state.flatten()
            x = self.fc1(state)
        else:
            states = np.zeros((state.shape[0], self.input_dim_len))
            for i, s in enumerate(state):
                states[i] = s.flatten()
            x = self.fc1(states)

        x = self.fc2(x)
        return self.fc3(x)


class ValueNetwork(keras.Model):
    def __init__(self, input_dims, action_dim, network_args, optimizer_type, optimizer_args):
        super(ValueNetwork, self).__init__()
        self.input_dim_len = len(input_dims)
        self.total_fc_input_dims = 1
        for dim in input_dims:
            self.total_fc_input_dims *= dim
        self.total_fc_input_dims += action_dim[0]

        fc_dims = network_args['fc_dims']
        fc1_dims, fc2_dims = get_hidden_layer_sizes(fc_dims)

        self.fc1 = Dense(fc1_dims, activation='relu', input_shape=(self.total_fc_input_dims,))
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.fc3 = Dense(1, activation=None)

        self.optimizer = get_keras_optimizer(optimizer_type, optimizer_args)
        gpu_device_name = tf.test.gpu_device_name()
        self.device = gpu_device_name if len(gpu_device_name) > 0 else '/device:cpu:0'

    def forward(self, state, action):
        if len(state.shape) == self.input_dim_len:
            state = state.flatten()
            x = tf.concat([state, action], axis=1)
        else:
            states = np.zeros((state.shape[0], self.input_dim_len))
            for i, s in enumerate(state):
                states[i] = s.flatten()
            x = tf.concat([states, action], axis=1)
        action_value = self.fc1(x)
        action_value = self.fc2(action_value)
        return self.fc3(action_value)
