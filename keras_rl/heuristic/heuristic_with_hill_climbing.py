from .heuristic_with_ml import HeuristicWithML
from keras_rl.hill_climbing.agent import HillClimbingAgent


class HeuristicWithHillClimbing(HeuristicWithML, HillClimbingAgent):
    def __init__(self, heuristic_func, use_model_only, input_dims, action_space, gamma, noise_scale=1e-2,
                 enable_action_blocking=False, min_penalty=0,
                 preloaded_memory=None, action_blocker_model_name=None,
                 action_blocker_timesteps=1000000, action_blocker_model_type=None,
                 **args):
        HeuristicWithML.__init__(self, input_dims, heuristic_func, use_model_only, action_space, enable_action_blocking,
                                 min_penalty, preloaded_memory,
                                 action_blocker_model_name, action_blocker_timesteps, action_blocker_model_type, **args)
        HillClimbingAgent.__init__(self, input_dims, action_space, gamma, noise_scale, False, 0, False, None, None)

    def optimize(self, env, learning_type):
        HillClimbingAgent.learn(self)
        HeuristicWithML.optimize(self, env, learning_type)

    def predict_action(self, observation, train, **args):
        return HillClimbingAgent.choose_policy_action(observation, train)

    def store_transition(self, state, action, reward, state_, done):
        HillClimbingAgent.store_transition(self, state, action, reward, state_, done)
        HeuristicWithML.store_transition(self, state, action, reward, state_, done)
