import random
import numpy as np

from transaction import Transition
import config


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.TD_error_memory = []
        self.index = 0

    def push(self, state, action, state_next, reward, td_error):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.TD_error_memory.append(None)

        self.memory[self.index] = Transition(state, action, state_next, reward)
        self.TD_error_memory[self.index] = td_error
        self.index = (self.index + 1) % self.capacity

    # new_errors is a Tensor
    def update_td_errors(self, new_errors):
        self.TD_error_memory = new_errors.tolist()
        if len(self.memory) != len(self.TD_error_memory):
            print('The memory item number is different from the error item number.')

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def get_prioritized_sample(self, batch_size):
        sum_absolute_td_error = np.sum(np.absolute(self.TD_error_memory))
        sum_absolute_td_error += config.TD_ERROR_EPSILON * len(self.TD_error_memory)

        rand_list = np.random.uniform(0, sum_absolute_td_error, batch_size)
        rand_list = np.sort(rand_list)

        indexes = []
        idx = 0
        tmp_sum_absolute_td_error = 0
        for rand_num in rand_list:
            while tmp_sum_absolute_td_error < rand_num:
                tmp_sum_absolute_td_error += (abs(self.TD_error_memory[idx]) + config.TD_ERROR_EPSILON)
                idx += 1

            if idx >= len(self.TD_error_memory):
                idx = len(self.TD_error_memory) - 1
            indexes.append(idx)

        prioritised_transactions = [self.memory[n] for n in indexes]
        return prioritised_transactions

    def __len__(self):
        return len(self.memory)
