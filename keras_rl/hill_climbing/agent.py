import numpy as np

from keras_rl.action_blocker.action_blocker import ActionBlocker
from keras_rl.replay.replay import ReplayBuffer


class HillClimbingAgent:
    def __init__(self, input_dims, action_space, gamma, noise_scale=1e-2, enable_action_blocking=False, min_penalty=0,
                 action_blocker_memory=None, action_blocker_model_name=None,
                 action_blocker_timesteps=1000000, action_blocker_model_type=None):

        self.num_actions = action_space.n

        self.flatten_state = len(input_dims) > 1
        self.total_fc_input_dims = 1
        for dim in input_dims:
            self.total_fc_input_dims *= dim
        self.w = 1e-4 * np.random.rand(self.total_fc_input_dims, self.num_actions)

        self.enable_action_blocking = enable_action_blocking
        self.initial_action_blocked = False
        self.initial_action = None
        if self.enable_action_blocking:
            if action_blocker_memory is None:
                action_blocker_memory = ReplayBuffer(input_shape=input_dims, max_size=action_blocker_timesteps)
            else:
                action_blocker_memory.add_more_memory(extra_mem_size=action_blocker_timesteps)
            self.action_blocker = ActionBlocker(action_space, penalty=min_penalty, memory=action_blocker_memory,
                                                model_name=action_blocker_model_name,
                                                model_type=action_blocker_model_type)

        self.gamma = gamma
        self.noise_scale = noise_scale

        self.best_R = -np.inf
        self.best_w = self.w

        self.rewards = []

    def store_transition(self, state, action, reward, state_, done):
        self.rewards.append(reward)
        if type(self.action_blocker) == ActionBlocker:
            self.action_blocker.store_transition(state, action, reward, state_, done)

    def learn(self):
        discounts = [self.gamma ** i for i in range(len(self.rewards) + 1)]
        R = sum([a * b for a, b in zip(discounts, self.rewards)])

        if R >= self.best_R:  # found better weights
            self.best_R = R
            self.best_w = self.w
            self.noise_scale = max(1e-3, self.noise_scale / 2)
            self.w += self.noise_scale * np.random.rand(*self.w.shape)
        else:  # did not find better weights
            self.noise_scale = min(2.0, self.noise_scale * 2)
            self.w = self.best_w + self.noise_scale * np.random.rand(*self.w.shape)

        self.rewards = []
        if type(self.action_blocker) == ActionBlocker:
            self.action_blocker.optimize()

    def choose_action(self, env, learning_type, observation, train=True):
        self.initial_action = self.choose_policy_action(observation, train)
        if self.enable_action_blocking:
            self.action_blocker.assign_learning_type(learning_type)
            actual_action = self.action_blocker.find_safe_action(env, observation, self.initial_action)
            self.initial_action_blocked = (actual_action is None or actual_action != self.initial_action)
            if actual_action is None:
                print('WARNING: No valid policy action found, running original action')
            return self.initial_action if actual_action is None else actual_action
        else:
            return self.initial_action

    def choose_policy_action(self, observation, train=True):
        if self.flatten_state:
            observation = observation.flatten()

        x = np.dot(observation, self.w)
        probs = np.exp(x)/sum(np.exp(x))

        if train:
            return np.random.choice(self.num_actions, p=probs)
        else:
            return np.argmax(probs)
