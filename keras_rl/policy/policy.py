import numpy as np


class Policy:
    def __init__(self, num_actions, move_matrix=None):
        self.num_actions = num_actions
        self.move_matrix = move_matrix

    def get_available_actions(self, observation=None):
        if self.move_matrix is not None and observation is not None:
            if type(observation) == np.ndarray:
                observation = tuple(observation)
            if len(observation) == 1:
                observation = observation[0]
            if observation in self.move_matrix:
                return self.move_matrix[observation]
        else:
            return list(range(self.num_actions))

    def actions_with_max_value(self, values):
        actions_with_max_value = []

        max_value = np.max(values)

        for a in range(values.shape[0]):
            if values[a] == max_value:
                actions_with_max_value.append(a)

        return actions_with_max_value

    def get_action(self, train, **args):
        if train:
            observation = args['observation'] if 'observation' in args else None
            return np.random.choice(self.get_available_actions(observation))
        else:
            values = args['values']
            observation = args['observation'] if 'observation' in args else None
            available_action_space = self.get_available_actions(observation)
            filtered_values = np.array([values[a] for a in available_action_space])
            return np.random.choice(self.actions_with_max_value(filtered_values))

    def update(self, **args):
        pass

    def get_probs(self, next_states, **args):
        if self.move_matrix is None:
            return np.full((next_states.shape[0], self.num_actions), 1/self.num_actions)
        else:
            probs = np.zeros((next_states.shape[0], self.num_actions))
            for i, next_state in enumerate(next_states):
                available_action_space = self.get_available_actions(next_state)
                for a in available_action_space:
                    probs[i, a] = 1/len(available_action_space)
            return probs

    def save_snapshot(self, file_name):
        pass

    def load_snapshot(self, file_name):
        pass
