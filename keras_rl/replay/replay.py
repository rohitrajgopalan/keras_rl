import numpy as np


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_action_dims=1, goal=None):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.input_shape = input_shape
        self.n_action_dims = n_action_dims
        self.goal = goal

        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)

        if n_action_dims == 1:
            self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        else:
            self.action_memory = np.zeros((self.mem_size, n_action_dims), dtype=np.float32)

        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

        if goal is not None:
            self.goal_memory = np.full((self.mem_size, goal.shape[0]), goal, dtype=np.float32)
        else:
            self.goal_memory = None

    def store_transition(self, state, action, reward, state_, done, **args):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size=0, **args):
        randomized = args['randomized'] if 'randomized' in args else False
        start = args['start'] if 'start' in args else -1
        end = args['end'] if 'end' in args else -1
        max_mem = min(self.mem_cntr, self.mem_size)

        if batch_size > 0:
            if randomized:
                batch = np.random.choice(max_mem, batch_size)
            else:
                batch = np.arange(batch_size)
        else:
            if end >= 0:
                if 0 <= start < end:
                    batch = np.arange(start, end)
                else:
                    batch = np.arange(end)
            else:
                batch = np.arange(max_mem)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        if self.goal_memory is None:
            goals = None
        else:
            goals = self.goal_memory[batch]

        return states, actions, rewards, states_, dones, goals

    def add_more_memory(self, extra_mem_size):
        old_mem_size = self.mem_size
        self.mem_size += extra_mem_size

        self.state_memory.resize((self.mem_size, *self.input_shape))
        self.new_state_memory.resize((self.mem_size, *self.input_shape))

        if self.n_action_dims == 1:
            self.action_memory.resize(self.mem_size)
        else:
            self.action_memory.resize((self.mem_size, self.n_action_dims))

        self.reward_memory.resize(self.mem_size)
        self.terminal_memory.resize(self.mem_size)

        if self.goal is not None:
            self.goal_memory.resize((self.mem_size, self.goal.shape[0]))
            for i in range(old_mem_size, self.mem_size):
                self.goal_memory[i] = self.goal
