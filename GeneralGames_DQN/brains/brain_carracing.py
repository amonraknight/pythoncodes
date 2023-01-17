import torch
from torch import optim
import numpy as np
import random

from brains.commonbrain import CommonBrain
from networks.networks import ConvNet4CarRacing
import config


class CarRacingBrain(CommonBrain):
    def __init__(self):
        super(CarRacingBrain, self).__init__()
        self.main_q_network = ConvNet4CarRacing(config.DIM_IN, config.DIM_OUT)
        self.target_q_network = ConvNet4CarRacing(config.DIM_IN, config.DIM_OUT)
        print(self.main_q_network)
        self.main_q_network.to(device=self.device)
        self.target_q_network.to(device=self.device)
        self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=config.LEARNING_RATE)

    # Return an array of four
    def decide_action(self, state, episode):
        epsilon = config.RANDOM_CHANCE * (1 - episode / config.NUM_EPISODES)
        if epsilon <= np.random.uniform(0, 1):
            action = super(CarRacingBrain, self).predict_action(state)
            action = action.max(1)[1].view(1, 1)
        else:
            if config.DONOTHING_CHANCE <= np.random.uniform(0, 1):
                action = torch.IntTensor([[0]])
            else:
                action = torch.IntTensor([[random.randint(0, 10)]])
        return action

