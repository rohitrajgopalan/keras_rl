import numpy as np
import tensorflow as tf
from tensorflow import keras
from .network import PolicyNetwork, ValueNetwork
from keras_rl.noise.ou import OUNoise
from keras_rl.replay.replay import ReplayBuffer
from keras_rl.replay.priority_replay import PriorityReplayBuffer


class DDPGAgent:
    def __init__(self, input_dims, action_space, tau, network_args, actor_optimizer_type, critic_optimizer_type,
                 actor_optimizer_args={}, critic_optimizer_args={}, gamma=0.99,
                 max_size=1000000, batch_size=64, goal=None, assign_priority=False, model_name=None,
                 pre_loaded_memory=None, use_mse=True):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.use_mse = use_mse

        self.noise = OUNoise(action_space, mu=np.zeros(action_space.shape))

        self.goal = goal
        if self.goal is not None:
            if not type(self.goal) == np.ndarray:
                self.goal = np.array([self.goal]).astype(np.float32)
            self.actor = PolicyNetwork(tuple(np.add(input_dims, self.goal.shape)),
                                       action_space.shape[0], network_args, actor_optimizer_type,
                                       actor_optimizer_args)

            self.critic = ValueNetwork(tuple(np.add(input_dims, self.goal.shape)),
                                       action_space.shape, network_args, critic_optimizer_type,
                                       critic_optimizer_args)

            self.target_actor = PolicyNetwork(tuple(np.add(input_dims, self.goal.shape)),
                                              action_space.shape[0], network_args, actor_optimizer_type,
                                              actor_optimizer_args)

            self.target_critic = ValueNetwork(tuple(np.add(input_dims, self.goal.shape)),
                                              action_space.shape, network_args, critic_optimizer_type,
                                              critic_optimizer_args)
        else:
            self.actor = PolicyNetwork(input_dims, action_space.shape[0], network_args, actor_optimizer_type,
                                       actor_optimizer_args)

            self.critic = ValueNetwork(input_dims, action_space.shape, network_args, critic_optimizer_type,
                                       critic_optimizer_args)

            self.target_actor = PolicyNetwork(input_dims, action_space.shape[0], network_args, actor_optimizer_type,
                                              actor_optimizer_args)

            self.target_critic = ValueNetwork(input_dims, action_space.shape, network_args, critic_optimizer_type,
                                              critic_optimizer_args)


        if assign_priority:
            self.memory = PriorityReplayBuffer(max_size, input_dims, action_space.shape[0], self.goal)
        else:
            self.memory = ReplayBuffer(max_size, input_dims, action_space.shape[0], self.goal)

        self.learn(pre_loaded_memory)

        self.action_space = action_space

        if model_name is not None:
            self.load_model(model_name)

        self.actor.compile(self.actor.optimizer)
        self.critic.compile(self.critic.optimizer)
        self.target_actor.compile(self.target_actor.optimizer)
        self.target_critic.compile(self.target_critic.optimizer)

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
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_critic.set_weights(weights)

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
                values = self.target_critic.forward(inputs, action).numpy()[0]
            else:
                values = self.critic.forward(inputs, action).numpy()[0]
            return values

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

        with tf.device(self.critic.device):
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
                critic_value_ = self.target_critic.forward(inputs_, target_actions)
                critic_value = self.critic.forward(inputs, actions)
                target = rewards + self.gamma * critic_value_ * (1 - dones)
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

    def load_model(self, model_name):
        self.actor = keras.models.load_model('{0}_actor'.format(model_name))
        self.target_actor = keras.models.load_model('{0}_target_actor'.format(model_name))
        self.critic = keras.models.load_model('{0}_critic'.format(model_name))
        self.target_critic = keras.models.load_model('{0}_target_critic'.format(model_name))

    def save_model(self, model_name):
        self.actor.save('{0}_actor'.format(model_name))
        self.target_actor.save('{0}_target_actor'.format(model_name))
        self.critic.save('{0}_critic'.format(model_name))
        self.target_critic.save('{0}_target_critic'.format(model_name))

    def apply_transfer_learning(self, td3_actor_model, td3_critic_model,
                                td3_target_actor_model=None, td3_target_critic_model=None):
        self.actor = keras.models.load_model(td3_actor_model)
        self.critic = keras.models.load_model(td3_critic_model)
        if td3_target_actor_model is not None:
            self.target_actor = keras.models.load_model(td3_target_actor_model)
        if td3_target_critic_model is not None:
            self.target_critic = keras.models.load_model(td3_target_critic_model)

    def __str__(self):
        return "DDPG Agent"
