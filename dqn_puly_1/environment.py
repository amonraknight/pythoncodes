import gym
import numpy as np
import torch

import config
from agent import Agent
import displayer


class Environment:

    def __init__(self):
        self.env = gym.make(config.ENV)
        self.num_states = self.env.observation_space.shape[0]

        self.num_actions = self.env.action_space.n

        self.agent = Agent(self.num_states, self.num_actions)

    def run(self):
        # Keep the last 10 steps:
        episode_10_list = np.zeros(10)
        complete_episodes = 0
        episode_final = False
        frames = []

        # Each episode:
        for episode in range(config.NUM_EPISODES):
            # Reset the game:
            observation = self.env.reset()

            state = observation[0]
            # Convert the state into a tensor of 1*4.
            state = torch.from_numpy(state).type(torch.FloatTensor)
            state = torch.unsqueeze(state, 0)

            # Take steps:
            for step in range(config.MAX_STEPS):
                if episode_final is True:
                    # frames.append(self.env.render(mode='rgb_array'))
                    frames.append(self.env.render())

                action = self.agent.get_action(state, episode)

                observation_next, _, done, _, _ = self.env.step(action.item())

                # If the stick is down within the step limit, done is True.
                if done:
                    state_next = None
                    # Add the last to the 10 most recent list.
                    episode_10_list = np.hstack((episode_10_list[1:], step + 1))

                    if step < 195:
                        # Reward is -1 when the stick is down before 200 steps.
                        reward = torch.FloatTensor([-1.0])
                        complete_episodes = 0
                    else:
                        # Reward is +1 when
                        reward = torch.FloatTensor([1.0])
                        complete_episodes = complete_episodes + 1

                else:
                    # Reward is 0 if the stick is not down in step limit.
                    reward = torch.FloatTensor([0.0])
                    state_next = observation_next
                    # Convert the state into a tensor of 1*4.
                    state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
                    state_next = torch.unsqueeze(state_next, 0)

                self.agent.memorize(state, action, state_next, reward)
                self.agent.update_q_function()
                state = state_next

                if done:
                    print('%d Episode: Finished after %d steps: average steps = %.1lf' % (
                        episode, step + 1, episode_10_list.mean()))
                    break

            if episode_final is True:
                # Save the animation
                displayer.display_frames_as_gif(frames)
                break

            if complete_episodes >= 10:
                print('10 successful episodes.')
                episode_final = True
