import gym
import numpy as np
import torch

from agent.agent import Agent
import config


class CarRacing:
    def __init__(self):
        self.env = gym.make('CarRacing-v2', render_mode=config.RENDER_MODE)
        self.agent = Agent()

    def train(self):
        # Find if there are half-way modules.
        starting_episode = 0
        previous_episode = self.agent.read_latest_module()
        if previous_episode:
            starting_episode = previous_episode + 1

        for episode in range(starting_episode, config.NUM_EPISODES + 1):

            obs = self.env.reset()
            # skip the zooming, do nothing
            for i in range(config.SKIP_FRAMES):
                # 96*96*3
                obs, _, _, _ = self.env.step(config.CARRACING_ACTIONS[0])

            # Collect the initial frames
            # 96*96*3 -> 3*96*96
            obs = obs.transpose(2, 0, 1)
            observation_frames = np.repeat(obs, config.FRAMES_EACH_OBSERVATION, axis=0)
            state = torch.from_numpy(observation_frames).type(torch.FloatTensor)
            state = state.unsqueeze(0)

            done = False
            step = 0
            total_score = 0
            while not done:
                action = self.agent.get_action_explore(state, episode)
                action_no = action.item()
                observation_next, reward, done, info = self.env.step(config.CARRACING_ACTIONS[action_no])

                step += 1
                total_score += reward

                '''
                if action_no == 0 or action_no == 3 or action_no == 4:
                    reward = reward - 0.1
                '''

                observation_next = observation_next.transpose(2, 0, 1)
                observation_frames = np.concatenate((observation_next, observation_frames[:6, :, :]), axis=0)

                if done:
                    state_next = None

                else:
                    state_next = torch.from_numpy(observation_frames).type(torch.FloatTensor)
                    state_next = state_next.unsqueeze(0)

                reward = torch.FloatTensor([reward])
                self.agent.memorize(state, action, state_next, reward)
                if episode % config.TRAIN_INTERVAL == 0:
                    self.agent.update_q_function()
                state = state_next

                if done:
                    print('%d Episode: Finished after %d steps: score %d' % (episode, step+1, total_score))
                    if episode % config.TARGET_RENEW_RATE == 0:
                        self.agent.update_target_q_function()
                    break

            if episode > 0 and episode % config.BACKUP_INTERVAL == 0:
                self.agent.save_network(episode)

    def play(self):
        self.agent.read_latest_module()
        for episode in range(0, 30):
            obs = self.env.reset()
            # skip the zooming, do nothing
            for i in range(config.SKIP_FRAMES):
                # 96*96*3
                obs, _, _, _ = self.env.step(config.CARRACING_ACTIONS[0])

            obs = obs.transpose(2, 0, 1)
            observation_frames = np.repeat(obs, config.FRAMES_EACH_OBSERVATION, axis=0)
            state = torch.from_numpy(observation_frames).type(torch.FloatTensor)
            state = state.unsqueeze(0)

            done = False
            step = 0
            total_score = 0
            while not done:
                action = self.agent.get_action(state)
                action_no = action.item()
                observation_next, reward, done, info = self.env.step(config.CARRACING_ACTIONS[action_no])
                step += 1
                total_score += reward

                observation_next = observation_next.transpose(2, 0, 1)
                observation_frames = np.concatenate((observation_next, observation_frames[:6, :, :]), axis=0)

                if done:
                    state_next = None

                else:
                    state_next = torch.from_numpy(observation_frames).type(torch.FloatTensor)
                    state_next = state_next.unsqueeze(0)

                state = state_next

                if done:
                    print('%d Episode: Finished after %d steps: score %d' % (episode, step+1, total_score))


if __name__ == "__main__":
    game = CarRacing()
    # game.train()
    game.play()
