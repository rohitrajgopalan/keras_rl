import joblib
import numpy as np
from gym.spaces import Discrete
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from keras_rl.replay.replay import ReplayBuffer
from keras_rl.utils.types import LearningType


class ActionBlocker:
    def __init__(self, action_space, **args):
        assert type(action_space) == Discrete

        self.action_space = action_space
        self.learning_type = None

        if 'model_name' in args and args['model_name'] is not None:
            self.model = joblib.load(args['model_name'])
        else:
            model_type = args['model_type'] if 'model_type' in args else 'decision_tree'
            if model_type == 'decision_tree':
                self.model = DecisionTreeClassifier()
            else:
                self.model = RandomForestClassifier()

        if 'memory' in args and type(args['memory']) == ReplayBuffer:
            self.memory = args['memory']

        self.penalty = args['penalty'] if 'penalty' in args else 0.01

        self.condensed_input_dims = 0

        self.optimize()

    def assign_learning_type(self, learning_type=LearningType.ONLINE):
        self.learning_type = learning_type

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def optimize(self):
        if self.memory is None or self.memory.mem_cntr == 0:
            pass

        states, actions, rewards, _, _ = self.memory.sample_buffer(randomized=False)
        if len(self.memory.input_shape) > 1:
            self.condensed_input_dims = 1
            for dim in self.memory.input_shape:
                self.condensed_input_dims *= dim
            state_input = np.zeros((min(self.memory.mem_cntr, self.memory.mem_size), self.condensed_input_dims))
            for i, state in enumerate(states):
                state_input[i] = state.flatten()
        else:
            state_input = states
            self.condensed_input_dims = self.memory.input_shape

        actions = actions.reshape(-1, 1)

        inputs = np.concatenate((state_input, actions), axis=1)
        outputs = (rewards <= -self.penalty).astype(int)

        self.model.fit(inputs, outputs)

    def block_action(self, env, state, action):
        if self.learning_type == LearningType.OFFLINE:
            # Let us find out later whether a particular action should be blocked.
            return False
        else:
            if len(state.shape) > 1:
                state = state.flatten()
            state = state.reshape(-1, self.condensed_input_dims)
            action = np.array([action]).reshape(-1, 1)

            input_arr = np.concatenate((state, action), axis=1)
            output = self.model.predict(np.array([input_arr]))[0]

            return output == 1

    def find_safe_action(self, env, state, initial_action, ):
        if self.block_action(env, state, initial_action):
            remaining_actions = [a for a in range(self.action_space.n) if a != initial_action]
            for a in remaining_actions:
                if not self.block_action(env, state, a):
                    return a
            return None
        else:
            return initial_action
