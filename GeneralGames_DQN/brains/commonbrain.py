import torch
import torch.nn as nn

import config
from common.transaction import Transition
from memory.memory_replayer import Buffer


class CommonBrain:
    def __init__(self):
        self.non_final_next_states = None
        self.state_action_values = None
        self.expected_state_action_values = None
        self.next_states = None
        self.reward_batch = None
        self.action_batch = None
        self.state_batch = None
        self.batch = None

        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        print(f"Training on device {self.device}.")

        self.buffer = Buffer(config.CAPACITY)

        self.main_q_network = None
        self.target_q_network = None
        self.optimizer = None
        self.loss_func = nn.MSELoss()

        # Define this part at the child classes.
        '''
        self.main_q_network = ConvNet1(config.DIM_IN, config.DIM_OUT)
        self.target_q_network = ConvNet1(config.DIM_IN, config.DIM_OUT)
        print(self.main_q_network)
        self.main_q_network.to(device=self.device)
        self.target_q_network.to(device=self.device)
        self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=config.LEARNING_RATE)
        '''

    def push(self, state, action, state_next, reward):
        state = state.to(device=self.device)
        action = action.to(device=self.device)
        if state_next is not None:
            state_next = state_next.to(device=self.device)
        reward = reward.to(device=self.device)
        self.buffer.push(state, action, state_next, reward)

    def replay(self):
        if self.buffer.__len__() < config.CAPACITY:
            return

        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states \
            = self.make_minibatch()

        self.expected_state_action_values = self.get_expected_state_action_values()
        self.update_main_q_network()

    def make_minibatch(self):
        transactions = self.buffer.sample(config.BATCH_SIZE)
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
        next_state_values = torch.zeros(config.BATCH_SIZE).to(device=self.device)
        a_m = torch.zeros(config.BATCH_SIZE).type(torch.LongTensor).to(device=self.device)

        non_final_mask = non_final_mask.bool()
        next_action_scores = self.main_q_network(self.non_final_next_states)
        a_m[non_final_mask] = next_action_scores.detach().max(1)[1]
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)
        next_state_values[non_final_mask] = \
            self.target_q_network(self.non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()

        expected_state_action_values = self.reward_batch + config.GAMMA * next_state_values

        return expected_state_action_values

    def update_main_q_network(self):
        # Train mode:
        self.main_q_network.train()

        loss = self.loss_func(self.state_action_values, self.expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())

    # Don't support greedy. Do it in the child brain.
    def predict_action(self, state):
        state = state.to(device=self.device)
        self.main_q_network.eval()
        with torch.no_grad():
            action = self.main_q_network(state)

        return action
