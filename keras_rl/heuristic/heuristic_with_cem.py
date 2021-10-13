import math

import numpy as np

from .heuristic_with_ml import HeuristicWithML
from ..cem.agent import CEMAgent


class HeuristicWithCEM(HeuristicWithML, CEMAgent):
    def __init__(self, heuristic_func, use_model_only,
                 input_dims, action_space, network_args, gamma=1.0, pop_size=50, elite_frac=0.2, sigma=0.5,
                 goal=None, **args):
        HeuristicWithML.__init__(self, input_dims, heuristic_func, use_model_only, action_space, False, 0, **args)
        CEMAgent.__init__(input_dims, action_space, network_args, gamma, pop_size, elite_frac, sigma, goal)

    def predict_action(self, observation, train, **args):
        return CEMAgent.choose_action(self, observation, train)

    def evaluate_weights_with_learning_type(self, env, learning_type, weights):
        self.network.set_weights(weights)
        episode_return = 0.0
        state = env.reset()
        done = False
        t = 0
        while not done:
            action = self.get_action(env, learning_type, state, train=True)
            state, reward, done, _ = env.step(action)
            episode_return += reward * math.pow(self.gamma, t)
            t += 1

        return episode_return

    def optimize(self, env, learning_type):
        weights_pop = [self.best_weights + (self.sigma * np.random.randn(self.network.get_weights_dim()))
                       for _ in range(self.pop_size)]
        rewards = np.array([self.evaluate_weights_with_learning_type(env, learning_type, weights) for weights in weights_pop])
        elite_idxs = rewards.argsort()[-self.n_elite:]
        elite_weights = [weights_pop[i] for i in elite_idxs]
        self.best_weights = np.array(elite_weights).mean(axis=0)