import time
import gym

env = gym.make('CarRacing-v2', render_mode='human')

obs = env.reset()
while True:
    # Next action:
    # (feed the observation to your agent here)
    action = env.action_space.sample()

    action[1] = 1
    # action[2] = 0
    # Processing:
    obs, reward, done, info = env.step(action)
    print(action, reward)
    # Rendering the game:
    # (remove this two lines during training)
    '''
    env.render()
    time.sleep(1 / 30)  # FPS
    '''

    # Checking if the player is still alive
    if done:
        break

env.close()
