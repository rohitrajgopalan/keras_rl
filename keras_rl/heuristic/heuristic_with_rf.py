import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .heuristic_with_ml import HeuristicWithML

import joblib


class HeuristicWithRF(HeuristicWithML):
    def __init__(self, input_dims, heuristic_func, use_model_only, action_space, enable_action_blocking=False, min_penalty=0,
                 preloaded_memory=None, action_blocker_model_name=None,
                 model_name=None, action_blocker_timesteps=1000000, action_blocker_model_type=None, **args):
        super().__init__(input_dims, heuristic_func, use_model_only, action_space, enable_action_blocking, min_penalty,
                         preloaded_memory, action_blocker_model_name,
                         action_blocker_timesteps, action_blocker_model_type, **args)
        if model_name is None:
            self.model = RandomForestRegressor() if self.is_continuous else RandomForestClassifier()
        else:
            self.load_model(model_name)
        self.states = []
        self.actions = []

    def predict_action(self, observation, train, **args):
        if type(observation) == np.ndarray and len(observation.shape) > 1:
            observation = observation.flatten()
        predicted_action = self.model.predict(np.array([observation]))[0]
        if self.is_continuous and not type(predicted_action) == np.ndarray:
            predicted_action = np.array([predicted_action])
        return predicted_action

    def store_transition(self, state, action, reward, state_, done):
        if type(state) == np.ndarray and len(state.shape) > 1:
            state = state.flatten()
        self.states.append(state)
        self.actions.append(action)
        super().store_transition(state, action, reward, state_, done)

    def optimize(self, env, learning_type):
        self.model.fit(np.array(self.states), np.array(self.actions))
        super().optimize(env, learning_type)

    def __str__(self):
        return 'Heuristic driven RF Agent {0}'.format('only using models' if self.use_model_only else 'alternating '
                                                                                                      'between '
                                                                                                      'models and '
                                                                                                      'heuristic')

    def load_model(self, model_name):
        self.model = joblib.load('{0}.pkl'.format(model_name))

    def save_model(self, model_name):
        joblib.dump(self.model, '{0}.pkl'.format(model_name))
