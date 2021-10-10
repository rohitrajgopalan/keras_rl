import numpy as np
import tensorflow as tf
from .policy import Policy


class SoftmaxPolicy(Policy):
    def __init__(self, num_actions, tau, move_matrix=None):
        super().__init__(num_actions, move_matrix)
        self.tau = tau

    def get_probs(self, next_states, **args):
        values = args['values']
        if self.move_matrix is None:
            values = values/self.tau
            return tf.keras.activations.softmax(tf.convert_to_tensor(values), dim=1).numpy()
        else:
            probs = np.zeros((next_states.shape[0], self.num_actions))
            for i, next_state in enumerate(next_states):
                available_action_space = self.get_available_actions(next_state)
                filtered_values = np.array([values[a] for a in available_action_space])/self.tau
                softmax_probs = tf.keras.activations.softmax(tf.convert_to_tensor(filtered_values), dim=1).numpy()
                for j, a in enumerate(available_action_space):
                    probs[i, a] = softmax_probs[j]
            return probs

    def get_action(self, train, **args):
        values = args['values']
        if train:
            if self.move_matrix is None:
                values = values / self.tau
                probs = tf.keras.activations.softmax(tf.convert_to_tensor(values), dim=1).numpy()
                return np.random.choice(self.num_actions, p=probs.squeeze())
            else:
                observation = args['observation'] if 'observation' in args else None
                available_action_space = self.get_available_actions(observation)
                filtered_values = np.array([values[a] for a in available_action_space])/self.tau
                softmax_probs = tf.keras.activations.softmax(tf.convert_to_tensor(filtered_values), dim=1).numpy()
                return np.random.choice(available_action_space, p=softmax_probs.squeeze())
        else:
            observation = args['observation'] if 'observation' in args else None
            available_action_space = self.get_available_actions(observation)
            filtered_values = np.array([values[a] for a in available_action_space])
            return np.random.choice(self.actions_with_max_value(filtered_values))
