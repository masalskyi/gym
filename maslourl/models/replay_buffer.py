import numpy as np


class ReplayBuffer(object):
    def __init__(self, mem_size, input_shape, n_actions, discrete=True, create_memory_arrays=True):
        self.terminal_memory = None
        self.reward_memory = None
        self.action_memory = None
        self.new_state_memory = None
        self.state_memory = None
        self.mem_size = mem_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.input_shape = input_shape
        self.n_actions = n_actions
        if create_memory_arrays:
            self.clear()

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state.reshape(self.input_shape)
        self.new_state_memory[index] = state_.reshape(self.input_shape)
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]
        return states, actions, rewards, new_states, terminal

    def clear(self):
        self.state_memory = np.zeros((self.mem_size, *self.input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *self.input_shape))
        dtype = np.int16 if self.discrete else float
        if self.discrete:
            self.action_memory = np.zeros(self.mem_size, dtype=dtype)
        else:
            self.action_memory = np.zeros((self.mem_size, self.n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=int)

    def save(self, path_to_save):
        np.savez(path_to_save,
                 mem_size=self.mem_size, 
                 mem_cntr=self.mem_cntr, 
                 discrete=self.discrete, 
                 input_shape=self.input_shape,
                 n_actions=self.n_actions,
                 state_memory=self.state_memory,
                 new_state_memory=self.new_state_memory,
                 action_memory=self.action_memory,
                 reward_memory=self.reward_memory,
                 terminal_memory=self.terminal_memory)
    
    @staticmethod
    def load(path_to_file):
        data = np.load(path_to_file, allow_pickle=True)
        buffer = ReplayBuffer(data["mem_size"], data["input_shape"], data["n_actions"], data["discrete"], False)
        buffer.state_memory = data["state_memory"]
        buffer.new_state_memory = data["new_state_memory"]
        buffer.action_memory = data["action_memory"]
        buffer.reward_memory = data["reward_memory"]
        buffer.terminal_memory = data["terminal_memory"]
        buffer.mem_cntr = data["mem_cntr"]
        return buffer
