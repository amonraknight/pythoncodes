from brains.brain_carracing import CarRacingBrain
from agent.module_saveload import ModuleSaveLoad
import config


class Agent:
    def __init__(self):
        if config.GAME_NAME == 'CarRacing-v2':
            self.brain = CarRacingBrain()
        self.saveLoader = ModuleSaveLoad()

    def update_q_function(self):
        self.brain.replay()

    def get_action_explore(self, state, episode):
        action = self.brain.decide_action_explore(state, episode)
        return action

    def get_action(self, state):
        action = self.brain.decide_action(state)
        return action

    # This is a list of the 3 input parameters.
    def get_action_merge_possibility(self, state):
        action = self.brain.acquire_merged_action(state)
        return action

    def memorize(self, state, action, state_next, reward):
        self.brain.push(state, action, state_next, reward)

    def update_target_q_function(self):
        self.brain.update_target_q_network()

    def save_network(self, episode=0):
        self.saveLoader.save_module(self.brain.main_q_network, episode)

    def read_latest_module(self):
        module_backup, previous_episode = self.saveLoader.load_module()
        if module_backup and previous_episode:
            self.brain.main_q_network.load_state_dict(module_backup)
            self.brain.update_target_q_network()

        return previous_episode
