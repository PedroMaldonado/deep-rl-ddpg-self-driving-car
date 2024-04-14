import random
import numpy as np
# Is a double-ended queue that allows append and pop operations from both ends. 
# It's used here because it's efficient for the operations required by the replay buffer
from collections import deque

# Store finite number of experiences
# past experiences are reused to train an agent

# Experience replay
class ReplayBuffer(object):
    # Memory Size
    # Input shape of the environment
    # Number of possible actions (number of componentes in the action space)
    def __init__(self, max_size, input_shape, n_actions):
        self.max_size = max_size #  The maximum number of experiences the buffer can hold.
        self.cur_size = 0 #  The current number of experiences in the buffer.
        self.buffer = deque() #  The buffer that holds the experiences
        # self.new_buffer = deque() 
        # self.action_memory = deque() # np.zeros((self.men_size, n_actions))
        # self.reward_memory = deque() # np.zeros(self.mem_size)
        # self.terminal_memory = deque() # np.zeros(self.mem_size, dtype=np.bool)

    def add(self, experience):
        # Add experience to the buffer
        # If the buffer is not full, add the experience
        if self.cur_size < self.max_size:
            self.buffer.append(experience)
            # Increment the current size
            self.cur_size += 1
        else:
            # If the buffer is full, replace the oldest experience with the new one (remove from the left end of the deque)
            self.buffer.popleft()
            # This will insert to the right end of the deque
            self.buffer.append(experience)

    def get_batch(self, size):
        # Get a batch of experiences from the buffer

        # If the batch size is smaller or equal to the current size of the buffer
        # Sample the batch size
        if size <= self.cur_size:
            return random.sample(self.buffer, size)
        else:
            # If not, then sample the current size
            return random.sample(self.buffer, self.cur_size)

        
        # sample_size = size if size <= self.cur_size else self.cur_size
        # return random.sample(self.buffer, sample_size)
    
    def __len__(self):
        return self.cur_size

    def clear(self):
        self.buffer.clear()
        self.cur_size = 0
