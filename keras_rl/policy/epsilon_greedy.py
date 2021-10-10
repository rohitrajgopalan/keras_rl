import numpy as np
from .policy import Policy


class EpsilonGreedyPolicy(Policy):
    def __init__(self, num_actions, enable_decay, eps_start, eps_min, eps_dec, move_matrix=None):
        super().__init__(num_actions, move_matrix)
        self.enable_decay = enable_decay
        self.epsilon = eps_start
        self.eps_min = eps_min
        self.eps_dec = eps_dec

    def get_action(self, train, **args):
        observation = args['observation'] if 'observation' in args else None
        available_action_space = self.get_available_actions(observation)
        if not train or np.random.random() > self.epsilon:
            values = args['values']
            filtered_values = np.array([values[a] for a in available_action_space])
            return np.random.choice(self.actions_with_max_value(filtered_values))
        else:
            return np.random.choice(available_action_space)

    def update(self, **args):
        if self.enable_decay:
            self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def get_probs(self, next_states, **args):
        values = args['values']
        policy = np.zeros((values.shape[0], self.num_actions))
        for i, q_values in enumerate(values):
            available_action_space = self.get_available_actions(next_states[i])
            filtered_values = np.array([q_values[a] for a in available_action_space])
            actions_with_max_value = self.actions_with_max_value(filtered_values)
            for a in available_action_space:
                if a in actions_with_max_value:
                    policy[i, a] = (1 - self.epsilon) / len(actions_with_max_value)
                else:
                    policy[i, a] = self.epsilon / (len(available_action_space) - len(actions_with_max_value))
        return policy
