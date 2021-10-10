from queue import PriorityQueue
import numpy as np


class PriorityReplayBuffer:
    def __init__(self, max_size, input_shape, n_action_dims=1, goal=None):
        self.input_shape = input_shape
        self.n_action_dims = n_action_dims

        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = PriorityQueue(maxsize=self.mem_size)
        self.new_state_memory = PriorityQueue(maxsize=self.mem_size)

        self.action_memory = PriorityQueue(maxsize=self.mem_size)

        self.reward_memory = PriorityQueue(maxsize=self.mem_size)
        self.terminal_memory = PriorityQueue(maxsize=self.mem_size)

        self.goal = goal

    def store_transition(self, state, action, reward, state_, done, **args):
        error_val = args['error_val']
        if error_val != 0.0:
            error_val = -error_val

        if self.mem_cntr == self.mem_size:
            self.mem_cntr -= 1
            self.state_memory.get()
            self.new_state_memory.get()
            self.action_memory.get()
            self.reward_memory.get()
            self.terminal_memory.get()

        self.state_memory.put((error_val, state))
        self.new_state_memory.put((error_val, state_))
        self.action_memory.put((error_val, action))
        self.reward_memory.put((error_val, reward))
        self.terminal_memory.put((error_val, done))

        self.mem_cntr += 1

    def sample_buffer(self, batch_size, **args):

        states = np.zeros((batch_size, *self.input_shape), dtype=np.float32)
        states_ = np.zeros((batch_size, *self.input_shape), dtype=np.float32)

        if self.n_action_dims == 1:
            actions = np.zeros(batch_size, dtype=np.int64)
        else:
            actions = np.zeros((batch_size, self.n_action_dims), dtype=np.float32)

        rewards = np.zeros(batch_size, dtype=np.float32)
        dones = np.zeros(batch_size, dtype=np.bool)

        if self.goal is not None:
            goals = np.full((batch_size, self.goal.shape[0]), self.goal, dtype=np.float32)
        else:
            goals = None

        for i in range(batch_size):
            states[i] = self.state_memory.get()
            states_[i] = self.new_state_memory.get()
            actions[i] = self.action_memory.get()
            rewards[i] = self.reward_memory.get()
            dones[i] = self.terminal_memory.get()

        self.mem_cntr -= batch_size

        return states, actions, rewards, states_, dones, goals
