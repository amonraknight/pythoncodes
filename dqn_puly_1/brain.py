import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from transaction import Transition
import numpy as np

import config
from memoryreplayer import ReplayMemory


class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions
        self.memory = ReplayMemory(config.CAPACITY)
        self.model = nn.Sequential()
        self.model.add_module('fc1', nn.Linear(num_states, 32))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('fc2', nn.Linear(32, 32))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('fc3', nn.Linear(32, num_actions))

        print(self.model)

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)

    def replay(self):
        if len(self.memory) < config.BATCH_SIZE:
            return

        # Get states, actions and results from the memory
        transactions = self.memory.sample(config.BATCH_SIZE)
        batch = Transition(*zip(*transactions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # evaluation mode
        self.model.eval()

        state_action_values = self.model(state_batch).gather(1, action_batch)
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
        next_state_values = torch.zeros(config.BATCH_SIZE)

        # Get the largest Q
        # "tensor.detach()" means to have a new tensor pointer pointing to the original data in memory
        # and have requires_grad=False.
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = reward_batch + config.GAMMA * next_state_values

        # training mode
        self.model.train()
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decide_action(self, state, episode):

        # Get a probablity of a random action.
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.model.eval()
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1, 1)

        else:
            # Give a random value.
            action = torch.LongTensor([[random.randrange(self.num_actions)]])

        return action
