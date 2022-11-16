import numpy as np
import math
import copy
import torch

import game_logic as logic
import config as c
from agent.agent import Agent


class Game2048NoDisplay:
    def __init__(self):
        self.matrix = logic.new_game(c.GRID_LEN)
        self.step_count = 0

        # DQN items
        num_states = c.GRID_LEN * c.GRID_LEN
        num_actions = len(c.ACTION_NUMBERS)
        self.agent = Agent(num_states, num_actions)

        self.commands = {
            0: logic.up,
            1: logic.down,
            2: logic.left,
            3: logic.right
        }

    def observe(self):
        # number of each cell
        cells = np.array(self.matrix).flatten()
        cells = np.array(list(map(lambda x: math.log2(max(x, 1)) / 16, cells)))
        # game is over or not. 'win', 'lose', 'not over'
        status = logic.game_state(self.matrix)
        # possible actions
        matrix_copy = copy.deepcopy(self.matrix)
        action_l = logic.get_possible_actions(matrix_copy)
        return cells, status, action_l

    def reset(self):
        self.step_count = 0
        self.matrix = logic.new_game(c.GRID_LEN)
        return self.observe()

    def dqn_solve(self):
        print('DQN solution training start...')
        complete_episodes = 0
        starting_episode = 0

        # Find if there are half-way modules.
        previous_episode = self.agent.read_latest_module()
        if previous_episode:
            starting_episode = previous_episode + 1

        # Each episode:
        for episode in range(starting_episode, c.NUM_EPISODES):
            # Reset the game:
            cells, status, action_l = self.reset()

            # Prepare the state in tensor[[]]
            state = torch.from_numpy(cells).type(torch.FloatTensor)
            state = torch.unsqueeze(state, 0)

            step = 0
            while status == 'not over':
                action = self.agent.get_action(state, action_l, episode)

                if action_l[action] == 1:
                    # Commit the step: tensor.item converts the tensor with a single item to a single value.
                    state_next, status, action_l, reward = self.step(action.item())
                    step += 1
                else:
                    state_next = cells
                    reward = torch.FloatTensor([[-0.01]])

                state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
                state_next = torch.unsqueeze(state_next, 0)

                self.agent.memorize(state, action, state_next, reward)
                # Update the neronet after a configured interval.
                if episode % c.TRAIN_INTERVAL == 0:
                    self.agent.update_q_function()

                state = state_next

                if status == 'win' or status == 'lose':
                    if status == 'win':
                        complete_episodes += 1

                    elif status == 'lose':
                        complete_episodes = 0

                    print('%d Episode: Finished after %d steps in status %s.' % (episode, step, status))
                    if episode % c.TARGET_NET_UPDATE_INTERVAL == 0:
                        self.agent.update_target_q_function()
                    break

            if (episode > 0 and episode % c.BACKUP_INTERVAL == 0) or complete_episodes >= c.ACCEPT_THRESHOLD:
                self.agent.save_network(episode)
                if complete_episodes >= c.ACCEPT_THRESHOLD:
                    print('{} successful episodes.'.format(c.ACCEPT_THRESHOLD))
                    break

    # actions: 0 up, 1 down, 2 left, 3 right
    def step(self, action):
        reward = 0.0
        if 'mono-sequential' in c.REWARD_STRATEGY:
            reward = reward + logic.get_general_score(self.matrix)

        self.matrix, done, step_score = self.commands[action](self.matrix)

        if 'mono-sequential' in c.REWARD_STRATEGY:
            reward = reward - logic.get_general_score(self.matrix)

        if 'merged cells' in c.REWARD_STRATEGY:
            reward = math.log2(max(reward, 1)) + math.log2(max(step_score, 1))
        reward = torch.FloatTensor([[max(reward, 0)]])

        # Add a random cell.
        self.matrix = logic.add_two_or_four(self.matrix)

        observation, status, action_l = self.observe()
        return observation, status, action_l, reward


if __name__ == "__main__":
    game = Game2048NoDisplay()
    game.dqn_solve()
