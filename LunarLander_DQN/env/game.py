import gym
import torch

from agent.agent import Agent
import config


class LunarLander:
    def __init__(self):
        self.env = gym.make(
            "LunarLander-v2",
            continuous=False,
            gravity=-10.0,
            enable_wind=False,
            wind_power=15.0,
            turbulence_power=1.5,
            render_mode=config.RENDER_MODE
        )
        self.agent = Agent()

    def train(self):
        starting_episode = 0
        # Get the existing module
        previous_episode = self.agent.read_latest_module()
        if previous_episode:
            starting_episode = previous_episode + 1

        for episode in range(starting_episode, config.NUM_EPISODES + 1):

            final_score = 0

            obs = self.env.reset()
            state = torch.from_numpy(obs).type(torch.FloatTensor)
            state = state.unsqueeze(0)

            done = False
            step_count = 0

            action_count = [0, 0, 0, 0]

            while not done:
                action = self.agent.get_action(state, episode)
                observation_next, reward, done, info = self.env.step(action.item())
                action_count[action.item()] = action_count[action.item()] + 1

                final_score += reward
                step_count += 1

                if done:
                    state_next = None
                else:
                    state_next = torch.from_numpy(observation_next).type(torch.FloatTensor)
                    state_next = state_next.unsqueeze(0)

                reward = torch.FloatTensor([reward])

                self.agent.memorize(state, action, state_next, reward)

                if episode % config.TRAIN_INTERVAL == 0:
                    self.agent.update_q_function()
                state = state_next

                if done:
                    print(
                        '%d Episode: Finished after %d steps: score %d' % (
                            episode, step_count, final_score))
                    print(action_count)

                    # update target net
                    if episode % config.TARGET_RENEW_RATE == 0:
                        self.agent.update_target_q_function()

                    break

            if episode > 0 and episode % config.BACKUP_INTERVAL == 0:
                self.agent.save_network(episode)

    def play(self):
        '''
        Play the game
        :return: None
        '''
        starting_episode = 0
        # Get the existing module
        previous_episode = self.agent.read_latest_module()
        if previous_episode:
            starting_episode = previous_episode + 1

        for episode in range(starting_episode, config.NUM_EPISODES + 1):

            final_score = 0
            obs = self.env.reset()
            state = torch.from_numpy(obs).type(torch.FloatTensor)
            state = state.unsqueeze(0)

            done = False
            step_count = 0

            action_count = [0, 0, 0, 0]

            while not done:
                action = self.agent.get_action(state, episode, True)
                observation_next, reward, done, info = self.env.step(action.item())
                action_count[action.item()] = action_count[action.item()] + 1

                final_score += reward
                step_count += 1

                if done:
                    state_next = None
                else:
                    state_next = torch.from_numpy(observation_next).type(torch.FloatTensor)
                    state_next = state_next.unsqueeze(0)

                state = state_next

                if done:
                    print(
                        '%d Episode: Finished after %d steps: score %d' % (
                            episode, step_count, final_score))
                    print(action_count)

                    # update target net
                    break


if __name__ == "__main__":
    game = LunarLander()
    # game.train()
    game.play()
