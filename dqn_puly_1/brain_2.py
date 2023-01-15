import random
import torch
from torch import optim
import torch.nn.functional as F
from transaction import Transition
import numpy as np

import config
from memoryreplayer2 import ReplayMemory
from dueling_net import Net


class Brain:
    def __init__(self, num_states, num_actions):
        # Prepare the device:
        # self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        # print(f"Training on device {self.device}.")

        self.state_action_values = None
        self.expected_state_action_values = None
        self.non_final_next_states = None
        self.reward_batch = None
        self.action_batch = None
        self.state_batch = None
        self.batch = None
        self.num_actions = num_actions
        self.memory = ReplayMemory(config.CAPACITY)
        n_in, n_mid, n_out = num_states, 32, num_actions
        self.main_q_network = Net(n_in, n_mid, n_out)
        self.target_q_network = Net(n_in, n_mid, n_out)
        print(self.main_q_network)
        self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=config.LEARNING_RATE)

    def push(self, state, action, state_next, reward, td_error):
        '''
        state = state.to(device=self.device)
        action = action.to(device=self.device)
        if state_next is not None:
            state_next = state_next.to(device=self.device)
        reward = reward.to(device=self.device)
        '''

        self.memory.push(state, action, state_next, reward, td_error)

    def replay(self, episode):
        if len(self.memory) < config.BATCH_SIZE:
            return

        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = \
            self.make_minibatch(episode)
        self.expected_state_action_values = self.get_expected_state_action_values()
        self.update_main_q_network()

    def decide_action(self, state, episode):
        epsilon = 0.5 * (1 / (episode + 1))
        if epsilon <= np.random.uniform(0, 1):
            self.main_q_network.eval()
            with torch.no_grad():
                action = self.main_q_network(state).max(1)[1].view(1, 1)

        else:
            # Give a random value.
            action = torch.LongTensor([[random.randrange(self.num_actions)]])

        return action

    def make_minibatch(self, episode):

        if episode < 30:
            transactions = self.memory.sample(config.BATCH_SIZE)
        else:
            transactions = self.memory.get_prioritized_sample(config.BATCH_SIZE)

        batch = Transition(*zip(*transactions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        return batch, state_batch, action_batch, reward_batch, non_final_next_states

    def get_expected_state_action_values(self):
        # Evaluation mode:
        self.main_q_network.eval()
        self.target_q_network.eval()

        self.state_action_values = self.main_q_network(self.state_batch).gather(1, self.action_batch)
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, self.batch.next_state)))
        next_state_values = torch.zeros(config.BATCH_SIZE)
        a_m = torch.zeros(config.BATCH_SIZE).type(torch.LongTensor)

        non_final_mask = non_final_mask.bool()
        a_m[non_final_mask] = self.main_q_network(self.non_final_next_states).detach().max(1)[1]
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)
        next_state_values[non_final_mask] = \
            self.target_q_network(self.non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()

        expected_state_action_values = self.reward_batch + config.GAMMA * next_state_values
        return expected_state_action_values

    def update_main_q_network(self):
        # Why self.state_action_values and self.expected_state_action_values are None?

        # Train mode:
        self.main_q_network.train()

        loss = F.smooth_l1_loss(self.state_action_values, self.expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())

    def update_td_error_memory(self):
        # Evaluation mode:
        self.main_q_network.eval()
        self.target_q_network.eval()

        # Take all the memory items:
        transactions = self.memory.memory
        batch = Transition(*zip(*transactions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # Get Q(s_t, a_t)
        state_action_values = self.main_q_network(state_batch).gather(1, action_batch)

        # Mask index
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
        non_final_mask = non_final_mask.bool()

        next_state_values = torch.zeros(self.memory.__len__())
        a_m = torch.zeros(self.memory.__len__()).type(torch.LongTensor)
        a_m[non_final_mask] = self.main_q_network(non_final_next_states).detach().max(1)[1]
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)
        next_state_values[non_final_mask] = \
            self.target_q_network(non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()

        td_errors = (reward_batch + config.GAMMA * next_state_values) - state_action_values.squeeze()
        self.memory.update_td_errors(td_errors)





