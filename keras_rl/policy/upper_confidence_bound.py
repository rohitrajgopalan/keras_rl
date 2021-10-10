import numpy as np

from .policy import Policy


class UpperConfidenceBoundPolicy(Policy):
    def __init__(self, num_actions, confidence_factor, move_matrix=None):
        super().__init__(num_actions, move_matrix)
        self.confidence_factor = confidence_factor
        self.t = 0
        self.total_rewards = np.zeros((self.num_actions,))
        self.num_called = np.zeros((self.num_actions,))

    def update(self, **args):
        reward = args['reward']
        action = args['action']
        self.t += 1
        self.total_rewards[action] += reward
        self.num_called[action] += 1

    def get_action(self, train, **args):
        values = self.total_rewards / (self.num_called + 1)
        if train:
            values += (self.confidence_factor * np.sqrt(np.log(self.t + 1) / (self.num_called + 1)))
        observation = args['observation'] if 'observation' in args else None
        available_action_space = self.get_available_actions(observation)
        filtered_values = np.array([values[a] for a in available_action_space])
        return np.random.choice(self.actions_with_max_value(filtered_values))

    def get_probs(self, next_states, **args):
        ucb_values = self.total_rewards / (self.num_called + 1)
        ucb_values += (self.confidence_factor * np.sqrt(np.log(self.t + 1) / (self.num_called + 1)))

        if self.move_matrix is None:
            actions_with_max_ucb_value = self.actions_with_max_value(ucb_values)
            policy_single = np.zeros((self.num_actions,))
            for a in actions_with_max_ucb_value:
                policy_single[a] = 1 / len(actions_with_max_ucb_value)

            values = args['values']
            policy_2d = np.zeros((values.shape[0], self.num_actions))
            for i in range(values.shape[0]):
                policy_2d[i] = policy_single

            return policy_2d
        else:
            probs = np.zeros((next_states.shape[0], self.num_actions))
            for i, next_state in enumerate(next_states):
                available_action_space = self.get_available_actions(next_state)
                filtered_ucb_values = np.array([ucb_values[a] for a in available_action_space])
                actions_with_max_ucb_value = self.actions_with_max_value(filtered_ucb_values)
                for a in actions_with_max_ucb_value:
                    probs[i, a] = 1 / len(actions_with_max_ucb_value)
            return probs

    def save_snapshot(self, file_name):
        np.save(file='{0}_total_rewards'.format(file_name), arr=self.total_rewards)
        np.save(file='{0}_num_called'.format(file_name), arr=self.num_called)

    def load_snapshot(self, file_name):
        self.total_rewards = np.load(file='{0}_total_rewards.npy'.format(file_name))
        self.num_called = np.load(file='{0}_num_called.npy'.format(file_name))
