import random
import numpy as np

from .policy import Policy


class ThompsonSamplingPolicy(Policy):
    def __init__(self, num_actions, min_penalty, move_matrix=None):
        super().__init__(num_actions, move_matrix)
        self.min_penalty = min_penalty
        self.successes = np.zeros((self.num_actions,))
        self.failures = np.zeros((self.num_actions,))

    def get_action(self, train, **args):
        observation = args['observation'] if 'observation' in args else None
        available_action_space = self.get_available_actions(observation)
        if train:
            beta_values = np.array([random.betavariate(self.successes[a] + 1, self.failures[a] + 1)
                                    for a in available_action_space])
            return np.random.choice(self.actions_with_max_value(beta_values))
        else:
            filtered_successes = np.array([self.successes[a] for a in available_action_space])
            return np.random.choice(self.actions_with_max_value(filtered_successes))

    def update(self, **args):
        reward = args['reward']
        action = args['action']
        if reward <= -self.min_penalty:
            self.failures[action] += 1
        else:
            self.successes[action] += 1

    def get_probs(self, next_states, **args):
        if self.move_matrix is None:
            beta_values = np.array([random.betavariate(self.successes[a] + 1, self.failures[a] + 1)
                                    for a in range(self.num_actions)])
            actions_with_max_beta_value = self.actions_with_max_value(beta_values)
            policy_single = np.zeros((self.num_actions,))
            for a in actions_with_max_beta_value:
                policy_single[a] = 1 / len(actions_with_max_beta_value)

            values = args['values']
            policy_2d = np.zeros((values.shape[0], self.num_actions))
            for i in range(values.shape[0]):
                policy_2d[i] = policy_single

            return policy_2d
        else:
            probs = np.zeros((next_states.shape[0], self.num_actions))
            for i, next_state in enumerate(next_states):
                available_action_space = self.get_available_actions(next_state)
                beta_values = np.array([random.betavariate(self.successes[a] + 1, self.failures[a] + 1)
                                        for a in available_action_space])
                actions_with_max_beta_value = self.actions_with_max_value(beta_values)
                for a in actions_with_max_beta_value:
                    probs[i, a] = 1 / len(actions_with_max_beta_value)
            return probs

    def save_snapshot(self, file_name):
        np.save(file='{0}_successes'.format(file_name), arr=self.successes)
        np.save(file='{0}_failures'.format(file_name), arr=self.failures)

    def load_snapshot(self, file_name):
        self.successes = np.load(file='{0}_successes.npy'.format(file_name))
        self.failures = np.load(file='{0}_failures.npy'.format(file_name))
