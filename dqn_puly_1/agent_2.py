from brain_2 import Brain


class Agent:
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)

    def update_q_function(self, episode):
        self.brain.replay(episode)

    def get_action(self, state, episode):
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward, td_error):
        self.brain.push(state, action, state_next, reward, td_error)

    def update_target_q_function(self):
        self.brain.update_target_q_network()

    def update_td_error_memory(self):
        self.brain.update_td_error_memory()
