import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Dense


class Network(keras.Model):
    def __init__(self, input_dims, action_dims, network_args):
        super(Network, self).__init__()
        self.input_dim_len = len(input_dims)
        self.total_fc_input_dims = 1
        for dim in input_dims:
            self.total_fc_input_dims *= dim

        self.h_size = network_args['fc_dim']
        self.a_size = action_dims[0]

        self.fc1 = Dense(self.total_fc_input_dims, activation='relu')
        self.fc2 = Dense(self.h_size, activation='relu')
        self.mu = Dense(self.a_size, activation='tanh')

        gpu_device_name = tf.test.gpu_device_name()
        self.device = gpu_device_name if len(gpu_device_name) > 0 else '/device:cpu:0'

    def set_weights(self, weights):
        s_size = self.total_fc_input_dims
        h_size = self.h_size
        a_size = self.a_size
        # separate the weights for each layer
        fc1_end = (s_size * h_size) + h_size
        fc1_W = weights[:s_size * h_size].reshape(s_size, h_size)
        fc1_b = weights[s_size * h_size:fc1_end]
        fc2_W = weights[fc1_end:fc1_end + (h_size * a_size)].reshape(h_size, a_size)
        fc2_b = weights[fc1_end + (h_size * a_size):]
        # set the weights for each layer
        self.fc1.set_weights([fc1_W, fc1_b])
        self.fc2.set_weights([fc2_W, fc2_b])

    def get_weights_dim(self):
        return (self.s_size + 1) * self.h_size + (self.h_size + 1) * self.a_size

    def call(self, inputs, training=None, mask=None):
        prob = self.fc1(inputs)
        prob = self.fc2(prob)

        return self.mu(prob)
