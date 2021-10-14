import math

import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras_rl.td3.agent import TD3Agent
from .heuristic_with_ml import HeuristicWithML


class HeuristicWithTD3(HeuristicWithML, TD3Agent):
    def __init__(self, heuristic_func, use_model_only, input_dims, action_space, tau, network_args,
                 actor_optimizer_type,
                 critic_optimizer_type, actor_optimizer_args={}, critic_optimizer_args={}, gamma=0.99,
                 max_size=1000000, batch_size=64, policy_update_interval=2, noise_std=0.2,
                 noise_clip=0.5, goal=None, model_name=None, pre_loaded_memory=None,
                 use_mse=True, **args):
        HeuristicWithML.__init__(self, input_dims, heuristic_func, use_model_only, action_space, False, 0, **args)
        TD3Agent.__init__(input_dims, action_space, tau, network_args, actor_optimizer_type, critic_optimizer_type,
                          actor_optimizer_args, critic_optimizer_args, gamma,
                          max_size, batch_size, policy_update_interval, noise_std,
                          noise_clip, goal, False, model_name, pre_loaded_memory, use_mse)

    def optimize(self, env, learning_type):
        num_updates = int(math.ceil(self.memory.mem_cntr / self.batch_size))
        for i in range(num_updates):
            start = (self.batch_size * (i - 1)) if i >= 1 else 0
            if i == num_updates - 1:
                end = self.memory.mem_cntr
            else:
                end = i * self.batch_size

            state, action, reward, new_state, done, goals = self.memory.sample_buffer(randomized=False, start=start,
                                                                                      end=end)

            with tf.device(self.critic1.device):
                states = tf.convert_to_tensor(state, dtype=tf.float32)
                states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
                rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
                actions = tf.convert_to_tensor(action, dtype=tf.float32)

                if goals is not None:
                    goals = tf.convert_to_tensor(goals, dtype=tf.float32)
                    inputs = tf.concat([states, goals], axis=1)
                    inputs_ = tf.concat([states_, goals], axis=1)
                else:
                    inputs = states
                    inputs_ = states_

                with tf.GradientTape() as tape:
                    target_actions = self.target_actor.forward(inputs_)
                    target_actions = target_actions + tf.clip_by_value(
                        tf.convert_to_tensor(np.random.normal(scale=self.noise_clip)), -self.noise_clip,
                        self.noise_clip)
                    target_actions = tf.clip_by_value(target_actions, self.action_space.low[0],
                                                      self.action_space.high[0])
                    critic_value1_ = self.target_critic1.forward(inputs_, target_actions)
                    critic_value2_ = self.target_critic2.forward(inputs_, target_actions)
                    critic_value1 = self.critic1.forward(inputs, actions)
                    critic_value2 = self.critic2.forward(inputs, actions)
                    critic_value_ = tf.math.minimum(critic_value1_, critic_value2_)

                    target = rewards + self.gamma * critic_value_ * (1 - done)
                    if self.use_mse:
                        critic_loss1 = keras.losses.MSE(target, critic_value1)
                        critic_loss2 = keras.losses.MSE(target, critic_value2)
                    else:
                        critic_loss1 = keras.losses.Huber(target, critic_value1)
                        critic_loss2 = keras.losses.Huber(target, critic_value2)

                critic_loss = critic_loss1 + critic_loss2
                critic_network_gradient1 = tape.gradient(critic_loss, self.critic1.trainable_variables)
                critic_network_gradient2 = tape.gradient(critic_loss, self.critic2.trainable_variables)
                self.critic1.optimizer.apply_gradients(zip(critic_network_gradient1, self.critic1.trainable_variables))
                self.critic2.optimizer.apply_gradients(zip(critic_network_gradient2, self.critic2.trainable_variables))

                if self.learn_step_cntr % self.policy_update_interval != 0:
                    return

                with tf.GradientTape() as tape:
                    new_policy_actions = self.actor.forward(inputs)
                    actor_loss = -self.critic1.forward(inputs, new_policy_actions)
                    actor_loss = tf.math.reduce_mean(actor_loss)

                actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
                self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))

            self.update_network_parameters()

            self.learn_step_cntr += 1

    def predict_action(self, observation, train, **args):
        return TD3Agent.choose_action(observation, args['t'], train)

    def store_transition(self, state, action, reward, state_, done):
        TD3Agent.store_transition(self, state, action, reward, state_, done)

    def __str__(self):
        return 'Heuristic driven TD3 Agent {0}'.format('only using models' if self.use_model_only else 'alternating '
                                                                                                       'between '
                                                                                                       'models and '
                                                                                                       'heuristic')
