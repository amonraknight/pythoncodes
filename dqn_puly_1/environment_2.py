import gym
import numpy as np
import torch
import time

import config
from agent_2 import Agent


class Environment:

    def __init__(self):
        self.env = gym.make(config.ENV)
        num_states = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n

        self.agent = Agent(num_states, num_actions)

    def run(self):

        # Keep the last 10 steps:
        episode_10_list = np.zeros(10)
        complete_episodes = 0

        # Each episode:
        for episode in range(config.NUM_EPISODES):
            # Reset the game:
            observation = self.env.reset()
            state = observation
            # Convert the state into a tensor of 1*4.
            state = torch.from_numpy(state).type(torch.FloatTensor)
            state = torch.unsqueeze(state, 0)

            # Take steps:
            for step in range(config.MAX_STEPS):

                action = self.agent.get_action(state, episode)

                observation_next, _, done, _ = self.env.step(action.item())

                self.env.render()
                time.sleep(1 / 30)  # FPS

                # If the stick is down within the step limit, done is True.
                if done:
                    state_next = None
                    # Add the last to the 10 most recent list.
                    episode_10_list = np.hstack((episode_10_list[1:], step + 1))

                    if step < config.MAX_STEPS - 5:
                        # Reward is -1 when the stick is down before 200 steps.
                        reward = torch.FloatTensor([-1.0])

                    else:
                        # Reward is +1 when
                        reward = torch.FloatTensor([1.0])

                else:
                    # Reward is 0 if the stick is not down in step limit.
                    reward = torch.FloatTensor([0.0])
                    state_next = observation_next
                    # Convert the state into a tensor of 1*4.
                    state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
                    state_next = torch.unsqueeze(state_next, 0)

                # Memorize a 0 as TD_error
                self.agent.memorize(state, action, state_next, reward, 0)
                self.agent.update_q_function(episode)
                state = state_next

                if done:
                    print('%d Episode: Finished after %d steps: average steps = %.1lf' % (
                        episode, step + 1, episode_10_list.mean()))
                    complete_episodes = 0
                    self.agent.update_td_error_memory()

                    if episode % 2 == 0:
                        self.agent.update_target_q_function()
                    break

                if step == config.MAX_STEPS - 1:
                    print('%d Episode: pole stands after %d steps.' % (episode, step))
                    complete_episodes = complete_episodes + 1

            if complete_episodes >= config.ACCEPT_THRESHOLD:
                print('{} successful episodes.'.format(config.ACCEPT_THRESHOLD))
                break
