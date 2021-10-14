import math

import tensorflow as tf
from tensorflow import keras

from keras_rl.ddpg.agent import DDPGAgent
from .heuristic_with_ml import HeuristicWithML


class HeuristicWithDDPG(HeuristicWithML, DDPGAgent):
    def __init__(self, heuristic_func, use_model_only, input_dims, action_space, tau, network_args, actor_optimizer_type,
                 critic_optimizer_type, actor_optimizer_args={}, critic_optimizer_args={}, gamma=0.99,
                 max_size=1000000, batch_size=64, goal=None, model_name=None, pre_loaded_memory=None,
                 use_mse=True, **args):
        HeuristicWithML.__init__(self, input_dims, heuristic_func, use_model_only, action_space, False, 0, **args)
        DDPGAgent.__init__(input_dims, action_space, tau, network_args, actor_optimizer_type, critic_optimizer_type,
                           actor_optimizer_args, critic_optimizer_args, gamma,
                           max_size, batch_size, goal, False, model_name, pre_loaded_memory, use_mse)

    def optimize(self, env, learning_type):
        num_updates = int(math.ceil(self.memory.mem_cntr / self.batch_size))
        for i in range(num_updates):
            start = (self.batch_size * (i - 1)) if i >= 1 else 0
            if i == num_updates - 1:
                end = self.memory.mem_cntr
            else:
                end = i * self.batch_size

            with tf.device(self.critic.device):
                state, action, reward, new_state, done, goals = self.memory.sample_buffer(randomized=False, start=start,
                                                                                          end=end)

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
                    critic_value_ = self.target_critic.forward(inputs_, target_actions)
                    critic_value = self.critic.forward(inputs, actions)
                    target = rewards + self.gamma * critic_value_ * (1 - done)
                    if self.use_mse:
                        critic_loss = keras.losses.MSE(target, critic_value)
                    else:
                        critic_loss = keras.losses.Huber(target, critic_value)

                critic_network_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
                self.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.trainable_variables))

                with tf.GradientTape() as tape:
                    new_policy_actions = self.actor.forward(inputs)
                    actor_loss = -self.critic.forward(inputs, new_policy_actions)
                    actor_loss = tf.math.reduce_mean(actor_loss)

                actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
                self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))

            self.update_network_parameters()

    def predict_action(self, observation, train, **args):
        return DDPGAgent.choose_action(observation, args['t'], train)

    def store_transition(self, state, action, reward, state_, done):
        DDPGAgent.store_transition(self, state, action, reward, state_, done)

    def __str__(self):
        return 'Heuristic driven DDPG Agent {0}'.format('only using models' if self.use_model_only else 'alternating '
                                                                                                        'between '
                                                                                                        'models and '
                                                                                                        'heuristic')
    
    