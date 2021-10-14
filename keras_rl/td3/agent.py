import tensorflow as tf
from tensorflow import keras
import numpy as np

from keras_rl.ddpg.network import PolicyNetwork, ValueNetwork
from keras_rl.noise.gaussian import GaussianExploration
from keras_rl.replay.priority_replay import PriorityReplayBuffer
from keras_rl.replay.replay import ReplayBuffer


class TD3Agent:
    def __init__(self, input_dims, action_space, tau, network_args, actor_optimizer_type, critic_optimizer_type,
                 actor_optimizer_args={}, critic_optimizer_args={}, gamma=0.99,
                 max_size=1000000, batch_size=64, policy_update_interval=2, noise_std=0.2,
                 noise_clip=0.5, goal=None, assign_priority=False, model_name=None, pre_loaded_memory=None,
                 use_mse=True):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.policy_update_interval = policy_update_interval
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.use_mse = use_mse

        self.noise = GaussianExploration(action_space)

        self.goal = goal
        if self.goal is not None:
            if not type(self.goal) == np.ndarray:
                self.goal = np.array([self.goal]).astype(np.float32)
            self.actor = PolicyNetwork(tuple(np.add(input_dims, self.goal.shape)),
                                       action_space.shape[0], network_args, actor_optimizer_type,
                                       actor_optimizer_args)

            self.critic1 = ValueNetwork(tuple(np.add(input_dims, self.goal.shape)),
                                        action_space.shape, network_args, critic_optimizer_type,
                                        critic_optimizer_args)

            self.critic2 = ValueNetwork(tuple(np.add(input_dims, self.goal.shape)),
                                        action_space.shape, network_args, critic_optimizer_type,
                                        critic_optimizer_args)

            self.target_actor = PolicyNetwork(tuple(np.add(input_dims, self.goal.shape)),
                                              action_space.shape[0], network_args, actor_optimizer_type,
                                              actor_optimizer_args)

            self.target_critic1 = ValueNetwork(tuple(np.add(input_dims, self.goal.shape)),
                                               action_space.shape, network_args, critic_optimizer_type,
                                               critic_optimizer_args)
            self.target_critic2 = ValueNetwork(tuple(np.add(input_dims, self.goal.shape)),
                                               action_space.shape, network_args, critic_optimizer_type,
                                               critic_optimizer_args)
        else:
            self.actor = PolicyNetwork(input_dims, action_space.shape[0], network_args, actor_optimizer_type,
                                       actor_optimizer_args)

            self.critic1 = ValueNetwork(input_dims, action_space.shape, network_args, critic_optimizer_type,
                                        critic_optimizer_args)

            self.critic2 = ValueNetwork(input_dims, action_space.shape, network_args, critic_optimizer_type,
                                        critic_optimizer_args)

            self.target_actor = PolicyNetwork(input_dims, action_space.shape[0], network_args, actor_optimizer_type,
                                              actor_optimizer_args)

            self.target_critic1 = ValueNetwork(input_dims, action_space.shape, network_args, critic_optimizer_type,
                                               critic_optimizer_args)
            self.target_critic2 = ValueNetwork(input_dims, action_space.shape, network_args, critic_optimizer_type,
                                               critic_optimizer_args)

        self.update_network_parameters(soft_tau=1)

        if assign_priority:
            self.memory = PriorityReplayBuffer(max_size, input_dims, action_space.shape[0], self.goal)
        else:
            self.memory = ReplayBuffer(max_size, input_dims, action_space.shape[0], self.goal)

        self.learn(pre_loaded_memory)

        self.learn_step_cntr = 0
        self.action_space = action_space

        if model_name is not None:
            self.load_model(model_name)

        self.actor.compile(self.actor.optimizer)
        self.critic1.compile(self.critic1.optimizer)
        self.critic2.compile(self.critic2.optimizer)
        self.target_actor.compile(self.target_actor.optimizer)
        self.target_critic1.compile(self.target_critic1.optimizer)
        self.target_critic2.compile(self.target_critic2.optimizer)

        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic1.weights
        for i, weight in enumerate(self.critic1.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_critic1.set_weights(weights)

        weights = []
        targets = self.target_critic2.weights
        for i, weight in enumerate(self.critic2.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_critic2.set_weights(weights)

    def choose_action(self, observation, t=0, train=True):
        with tf.device(self.actor.device):
            state = tf.convert_to_tensor(observation, dtype=tf.float32)
            if self.goal is not None:
                goal = tf.convert_to_tensor(self.goal, dtype=tf.float32)
                inputs = tf.concat([observation, goal], axis=0)
            else:
                inputs = state

        actions = self.actor.forward(inputs).numpy()
        action = actions[0]
        if train:
            return self.noise.get_action(action, t)
        else:
            return np.clip(action, self.action_space.low, self.action_space.high)

    def get_critic_value(self, observation, action, use_target=False):
        with tf.device(self.target_actor.device if use_target else self.actor.device):
            state = tf.convert_to_tensor(observation, dtype=tf.float32)
            if self.goal is not None:
                goal = tf.convert_to_tensor(self.goal, dtype=tf.float32)
                inputs = tf.concat([observation, goal], axis=0)
            else:
                inputs = state
            action = tf.convert_to_tensor(action, dtype=tf.float32)

            if use_target:
                values1 = self.target_critic1.forward(inputs, action)
                values2 = self.target_critic2.forward(inputs, action)
            else:
                values1 = self.critic1.forward(inputs, action)
                values2 = self.critic2.forward(inputs, action)

            value1 = values1.numpy()[0]
            value2 = values2.numpy()[0]

            return value1 if value1 < value2 else value2

    def store_transition(self, state, action, reward, state_, done, t=0):
        self.memory.store_transition(state, action, reward, state_, done,
                                     error_val=self.determine_error(state, action, reward, state_, done, t))

    def get_target_value(self, reward, state_, done, t=0):
        if done:
            return reward
        else:
            target_action = self.choose_action(state_, t, True)
            next_critic_value = self.get_critic_value(state_, target_action, True)
            return reward + (self.gamma * next_critic_value)

    def determine_error(self, state, action, reward, state_, done, t=0):
        return self.get_target_value(reward, state_, done, t) - self.get_critic_value(state, action, False)

    def learn(self, pre_loaded_memory=None):
        if self.memory.mem_cntr < self.batch_size and pre_loaded_memory is None:
            return

        with tf.device(self.critic1.device):
            if pre_loaded_memory is not None:
                pre_loaded_memory.goal = self.goal
                states, actions, rewards, new_states, dones, goals = pre_loaded_memory.sample_buffer(randomized=False)
            else:
                states, actions, rewards, new_states, dones, goals = self.memory.sample_buffer(self.batch_size)

            states = tf.convert_to_tensor(states, dtype=tf.float32)
            states_ = tf.convert_to_tensor(new_states, dtype=tf.float32)
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.float32)

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
                target_actions = tf.clip_by_value(target_actions, self.action_space.low[0], self.action_space.high[0])
                critic_value1_ = self.target_critic1.forward(inputs_, target_actions)
                critic_value2_ = self.target_critic2.forward(inputs_, target_actions)
                critic_value1 = self.critic1.forward(inputs, actions)
                critic_value2 = self.critic2.forward(inputs, actions)
                critic_value_ = tf.math.minimum(critic_value1_, critic_value2_)

                target = rewards + self.gamma * critic_value_ * (1 - dones)
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

    def load_model(self, model_name):
        self.actor = keras.model.load_model('{0}_actor'.format(model_name))
        self.target_actor = keras.model.load_model('{0}_target_actor'.format(model_name))
        self.critic1 = keras.model.load_model('{0}_critic'.format(model_name))
        self.critic2 = keras.model.load_model('{0}_critic'.format(model_name))
        self.target_critic1 = keras.model.load_model('{0}_target_critic'.format(model_name))
        self.target_critic2 = keras.model.load_model('{0}_target_critic'.format(model_name))

    def save_model(self, model_name):
        self.actor.save('{0}_actor'.format(model_name))
        self.target_actor.save('{0}_target_actor'.format(model_name))
        self.critic1.save('{0}_critic'.format(model_name))
        self.critic2.save('{0}_critic'.format(model_name))
        self.target_critic1.save('{0}_target_critic'.format(model_name))
        self.target_critic2.save('{0}_target_critic'.format(model_name))

    def apply_transfer_learning(self, ddpg_actor_model, ddpg_critic_model,
                                ddpg_target_actor_model=None, ddpg_target_critic_model=None):
        self.actor = keras.models.load_model(ddpg_actor_model)
        self.critic1 = keras.models.load_model(ddpg_critic_model)
        self.critic2 = keras.models.load_model(ddpg_critic_model)
        if ddpg_target_actor_model is not None:
            self.target_actor = keras.models.load_model(ddpg_target_actor_model)
        if ddpg_target_critic_model is not None:
            self.target_critic1 = keras.models.load_model(ddpg_target_critic_model)
            self.target_critic2 = keras.models.load_model(ddpg_target_critic_model)

    def __str__(self):
        return "TD3 Agent"
