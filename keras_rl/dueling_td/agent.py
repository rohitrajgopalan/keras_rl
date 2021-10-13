import numpy as np
import tensorflow as tf

from keras_rl.replay.priority_replay import PriorityReplayBuffer
from keras_rl.replay.replay import ReplayBuffer
from keras_rl.utils.types import PolicyType, TDAlgorithmType
from keras_rl.utils.utils import choose_policy
from .network import DuelingTDNetwork
from ..action_blocker.action_blocker import ActionBlocker


class DuelingTDAgent:
    def __init__(self, algorithm_type, is_double, gamma, action_space, input_dims,
                 mem_size, batch_size, network_args, optimizer_type, policy_type, policy_args={},
                 replace=1000, optimizer_args={}, enable_action_blocking=False, min_penalty=0,
                 pre_loaded_memory=None, action_blocker_model_name=None,
                 goal=None, assign_priority=False, model_name=None, action_blocker_timesteps=1000000,
                 action_blocker_model_type=None, use_mse=True):
        self.algorithm_type = algorithm_type
        self.is_double = is_double
        self.gamma = gamma
        self.n_actions = action_space.n
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.policy_type = policy_type
        self.policy = choose_policy(self.n_actions, self.policy_type, policy_args)

        self.replace_target_cnt = replace
        self.learn_step_counter = 0

        self.goal = goal
        if self.goal is not None:
            if not type(self.goal) == np.ndarray:
                self.goal = np.array([self.goal]).astype(np.float32)
            self.q_eval = DuelingTDNetwork(tuple(np.add(self.input_dims, self.goal.shape)), self.n_actions,
                                           network_args, optimizer_type,
                                           optimizer_args, use_mse)
            self.q_next = DuelingTDNetwork(tuple(np.add(self.input_dims, self.goal.shape)), self.n_actions,
                                           network_args, optimizer_type,
                                           optimizer_args, use_mse)
        else:
            self.q_eval = DuelingTDNetwork(self.input_dims, self.n_actions, network_args, optimizer_type,
                                           optimizer_args, use_mse)
            self.q_next = DuelingTDNetwork(self.input_dims, self.n_actions, network_args, optimizer_type,
                                           optimizer_args, use_mse)

        self.q_eval.compile(optimizer=self.q_eval.optimizer, loss=self.q_eval.loss)
        self.q_next.compile(optimizer=self.q_next.optimizer, loss=self.q_next.loss)

        if assign_priority:
            self.memory = PriorityReplayBuffer(mem_size, input_dims, goal=self.goal)
        else:
            self.memory = ReplayBuffer(mem_size, input_dims, goal=self.goal)

        self.enable_action_blocking = enable_action_blocking
        self.action_blocker = None

        self.learn(pre_loaded_memory)

        if self.enable_action_blocking:
            if pre_loaded_memory is None:
                pre_loaded_memory = ReplayBuffer(input_shape=input_dims, max_size=action_blocker_timesteps)
            else:
                pre_loaded_memory.add_more_memory(extra_mem_size=action_blocker_timesteps)
            self.action_blocker = ActionBlocker(action_space, penalty=min_penalty, memory=pre_loaded_memory,
                                                model_name=action_blocker_model_name,
                                                model_type=action_blocker_model_type)

        self.initial_action_blocked = False
        self.initial_action = None

        if model_name is not None:
            self.load_model(model_name)

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

    def choose_policy_action(self, observation, goal=None, train=True):
        with tf.device(self.q_eval.device):
            if goal is None:
                goal = self.goal

            if not type(observation) == np.ndarray:
                observation = np.array([observation]).astype(np.float32)

            if goal is not None:
                inputs = np.concatenate((observation, goal), axis=None)
            else:
                inputs = observation
            _, advantage = self.q_eval.forward(inputs)
            return self.policy.get_action(train, values=advantage)

    def get_weighted_sum(self, q_values_arr, next_states):
        policy = self.policy.get_probs(values=q_values_arr, next_states=next_states)
        return np.sum(policy * q_values_arr, axis=1, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done,
                                     error_val=self.determine_error(state, action, reward, state_, done))
        if self.enable_action_blocking:
            self.action_blocker.store_transition(state, action, reward, state_, done)

    def get_q_values(self, network, observation, goal=None, advantage_only=True):
        with tf.device(self.q_eval.device):
            if goal is None:
                goal = self.goal

            if not type(observation) == np.ndarray:
                observation = np.array([observation]).astype(np.float32)

            if goal is not None:
                inputs = np.concatenate((observation, goal), axis=None)
            else:
                inputs = observation
            return network.advantage(inputs).numpy() if advantage_only else network(inputs).numpy()

    def get_target_value(self, action, reward, state_, done, advantage_only=True):
        with tf.device(self.q_eval.device):
            if done:
                return reward
            else:
                q_next = self.get_q_values(self.q_next, state_, self.goal, advantage_only)
                if self.is_double:
                    q_eval = self.get_q_values(self.q_eval, state_, self.goal, advantage_only)
                    if self.algorithm_type == TDAlgorithmType.SARSA:
                        next_q_value = q_next[self.policy.get_action(True, values=q_eval)]
                    elif self.algorithm_type == TDAlgorithmType.Q:
                        next_q_value = np.max(q_next)
                    elif self.algorithm_type == TDAlgorithmType.EXPECTED_SARSA:
                        Q_eval = q_eval.reshape(-1, self.n_actions)
                        policy = self.policy.get_probs(values=Q_eval, next_states=np.array([state_]))
                        next_q_value = np.sum(policy * Q_eval, axis=1)[0]
                    else:
                        next_q_value = q_next[action]
                else:
                    if self.algorithm_type == TDAlgorithmType.SARSA:
                        next_q_value = q_next[self.policy.get_action(True, values=q_next)]
                    elif self.algorithm_type == TDAlgorithmType.Q:
                        next_q_value = np.max(q_next)
                    elif self.algorithm_type == TDAlgorithmType.EXPECTED_SARSA:
                        Q_eval = q_next.reshape(-1, self.n_actions)
                        policy = self.policy.get_probs(values=Q_eval, next_states=np.array([state_]))
                        next_q_value = np.sum(policy * Q_eval, axis=1)[0]
                    else:
                        next_q_value = q_next[action]
                return reward + (self.gamma * next_q_value)

    def determine_error(self, state, action, reward, state_, done):
        return self.get_target_value(action, reward, state_, done, False) - self.get_q_values(self.q_eval, state, self.goal, False)

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.set_weights(self.q_eval.get_weights())

    def learn(self, pre_loaded_memory=None):
        if self.memory.mem_cntr < self.batch_size and pre_loaded_memory is None:
            return

        self.replace_target_network()

        with tf.device(self.q_eval.device):
            if pre_loaded_memory is not None:
                pre_loaded_memory.goal = self.goal
                states, actions, rewards, new_states, dones, goals = pre_loaded_memory.sample_buffer(randomized=False)
            else:
                states, actions, rewards, new_states, dones, goals = self.memory.sample_buffer(self.batch_size)

            if goals is None:
                goals = [None] * self.batch_size

            for state, action, reward, new_state, done, goal in zip(states, actions, rewards, new_states, dones, goals):
                if not type(state) == np.ndarray:
                    state = np.array([state]).astype(np.float32)

                if goal is not None:
                    inputs = np.concatenate((state, goal), axis=None)
                else:
                    inputs = state

                target = self.get_target_value(action, reward, new_state, done, advantage_only=False)

                # Construct the target vector as follows:
                # 1. Use the current model to output the Q-value predictions
                target_f = self.get_q_values(self.q_eval, state, goal, advantage_only=False)

                # 2. Rewrite the chosen action value with the computed target
                target_f[0][action] = target

                self.q_next.fit(inputs, target_f)

        self.learn_step_counter += 1

        if self.policy_type == PolicyType.EPSILON_GREEDY:
            self.policy.update()
        elif self.policy_type == PolicyType.UCB:
            for reward, action in zip(rewards, actions):
                self.policy.update(reward=reward, action=action)
        elif self.policy_type == PolicyType.THOMPSON_SAMPLING:
            for reward in rewards:
                self.policy.update(reward=reward)

        if type(self.action_blocker) == ActionBlocker:
            self.action_blocker.optimize()

    def load_model(self, model_name):
        self.q_eval = tf.keras.models.load_model('{0}_q_eval'.format(model_name))
        self.q_next = tf.keras.models.load_model('{0}_q_next'.format(model_name))
        self.policy.load_snapshot(model_name)

    def save_model(self, model_name):
        self.q_eval.save('{0}_q_eval'.format(model_name))
        self.q_next.save('{0}_q_next'.format(model_name))
        self.policy.save_snapshot(model_name)

    def __str__(self):
        return '{0}Dueling Deep {1} Agent using {2} policy'.format('Double ' if self.is_double else '',
                                                                   self.algorithm_type.name,
                                                                   self.policy_type.name)
