import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras_rl.utils.utils import get_keras_optimizer, get_hidden_layer_sizes


class TDNetwork:
    def __init__(self, input_dims, n_actions, fc_dims, optimizer_type, optimizer_args={}, use_mse=True):

        fc1_dims, fc2_dims = get_hidden_layer_sizes(fc_dims)

        self.input_dim_len = len(input_dims)
        self.total_fc_dims = 1
        for dim in input_dims:
            self.total_fc_dims *= dim

        self.model = Sequential()

        self.model.add(Dense(fc1_dims, input_shape=(self.total_fc_dims,), activation='relu'))
        self.model.add(Dense(fc2_dims, activation='relu'))
        self.model.add(Dense(n_actions, activation=None))

        self.optimizer = get_keras_optimizer(optimizer_type, optimizer_args)
        self.loss = tf.keras.losses.MeanSquaredError() if use_mse else tf.keras.losses.Huber()
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        gpu_device_name = tf.test.gpu_device_name()
        self.device = gpu_device_name if len(gpu_device_name) > 0 else '/device:cpu:0'

    def forward(self, state):
        if len(state.shape) == self.input_dim_len:
            states = np.array([state])
        else:
            states = np.zeros((state.shape[0], self.total_fc_dims))
            for i, s in enumerate(state):
                states[i] = s.flatten()
        return self.model.predict(states)

    def fit(self, inputs, outputs):
        self.model.fit(inputs, outputs, epochs=1, verbose=0)

    def save_model(self, model_file_name):
        self.model.save(model_file_name, overwrite=True)

    def load_model(self, model_file_name):
        self.model = keras.models.load_model(model_file_name)