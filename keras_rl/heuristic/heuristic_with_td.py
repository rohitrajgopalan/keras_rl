import math

import numpy as np
import tensorflow as tf

from keras_rl.td.agent import TDAgent
from .heuristic_with_ml import HeuristicWithML
from ..utils.types import TDAlgorithmType, PolicyType, LearningType


class HeuristicWithTD(HeuristicWithML, TDAgent):
    def __init__(self, heuristic_func, use_model_only, algorithm_type, is_double, gamma, action_space, input_dims,
                 mem_size, batch_size, network_args, optimizer_type, policy_type, policy_args={},
                 replace=1000, optimizer_args={}, goal=None, enable_action_blocking=False, min_penalty=0,
                 pre_loaded_memory=None, action_blocker_model_name=None,
                 action_blocker_timesteps=1000000, action_blocker_model_type=None,
                 model_name=None, use_mse=True, **args):
        HeuristicWithML.__init__(self, input_dims, heuristic_func, use_model_only, action_space, enable_action_blocking,
                                 min_penalty, pre_loaded_memory,
                                 action_blocker_model_name, action_blocker_timesteps, action_blocker_model_type, **args)
        TDAgent.__init__(self, algorithm_type, is_double, gamma, action_space, input_dims, mem_size, batch_size,
                         network_args,
                         optimizer_type, policy_type, policy_args, replace, optimizer_args, False, 0, None, None,
                         goal, False, model_name, use_mse)

    def obtain_target_value(self, action, reward, new_state, done, env, learning_type):
        with tf.device(self.q_eval.device):
            if done:
                return reward
            else:
                Q_ = self.get_q_values(self.q_next, new_state, self.goal).squeeze()

                if self.is_double:
                    Q_eval = self.get_q_values(self.q_eval, new_state, self.goal).squeeze()
                    if self.algorithm_type == TDAlgorithmType.SARSA:
                        next_q_value = Q_[self.determine_next_action(env, learning_type, new_state, Q_eval)]
                    elif self.algorithm_type == TDAlgorithmType.Q:
                        next_q_value = np.max(Q_eval)
                    elif self.algorithm_type == TDAlgorithmType.EXPECTED_SARSA:
                        Q_eval = Q_eval.reshape(-1, self.n_actions)
                        policy = self.policy.get_probs(values=Q_eval, next_states=np.array([new_state]))
                        next_q_value = np.sum(policy * Q_eval, axis=1)[0]
                    else:
                        next_q_value = Q_[action]
                else:
                    if self.algorithm_type == TDAlgorithmType.SARSA:
                        next_q_value = Q_[self.determine_next_action(env, learning_type, new_state, Q_)]
                    elif self.algorithm_type == TDAlgorithmType.Q:
                        next_q_value = np.max(Q_)
                    elif self.algorithm_type == TDAlgorithmType.EXPECTED_SARSA:
                        Q_eval = Q_.reshape(-1, self.n_actions)
                        policy = self.policy.get_probs(values=Q_eval, next_states=np.array([new_state]))
                        next_q_value = np.sum(policy * Q_eval, axis=1)[0]
                    else:
                        next_q_value = Q_[action]
                return reward + (self.gamma * next_q_value)

    def optimize(self, env, learning_type):
        num_updates = int(math.ceil(self.memory.mem_cntr / self.batch_size))
        for i in range(num_updates):
            self.q_next.optimizer.step()

            start = (self.batch_size * (i - 1)) if i >= 1 else 0
            if i == num_updates - 1:
                end = self.memory.mem_cntr
            else:
                end = i * self.batch_size

            with tf.device(self.q_eval.device):
                states, actions, rewards, new_states, dones, goals = self.memory.sample_buffer(randomized=False,
                                                                                               start=start,
                                                                                               end=end)

                if goals is None:
                    goals = [None] * self.batch_size

                for state, action, reward, new_state, done, goal in zip(states, actions, rewards, new_states, dones,
                                                                        goals):
                    if not type(state) == np.ndarray:
                        state = np.array([state]).astype(np.float32)

                    if goal is not None:
                        inputs = np.concatenate((state, goal), axis=None)
                    else:
                        inputs = state

                    target = self.obtain_target_value(action, reward, new_state, done, env, learning_type)

                    # Construct the target vector as follows:
                    # 1. Use the current model to output the Q-value predictions
                    target_f = self.get_q_values(self.q_eval, state, goal)

                    # 2. Rewrite the chosen action value with the computed target
                    target_f[0][action] = target

                    self.q_next.fit(np.array([inputs]), target_f)

            self.learn_step_counter += 1

            self.q_next.model.set_weights(self.q_eval.model.get_weights())

            if self.policy_type == PolicyType.EPSILON_GREEDY:
                self.policy.update()
            elif self.policy_type == PolicyType.UCB:
                for action, reward in zip(actions, rewards):
                    self.policy.update(action=action, reward=reward)

        self.memory.mem_cntr = 0
        super().optimize(env, learning_type)

    def determine_next_action(self, env, learning_type, next_state, q_values=None):
        if learning_type == LearningType.OFFLINE:
            return self.heuristic_func(next_state)
        else:
            if q_values is not None:
                return self.policy.get_action(True, values=q_values)
            else:
                return self.get_action(env, LearningType.ONLINE, next_state, True)

    def predict_action(self, observation, train, **args):
        return TDAgent.choose_policy_action(observation, train)

    def store_transition(self, state, action, reward, state_, done):
        TDAgent.store_transition(self, state, action, reward, state_, done)

    def __str__(self):
        return 'Heuristic driven {0}Deep {1} Agent using {2} policy {3}'.format(
            'Double ' if self.is_double else '',
            self.algorithm_type.name,
            self.policy_type.name,
            'only using models' if self.use_model_only else 'alternating between models and heuristic '
        )
