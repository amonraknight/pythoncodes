import time
import gym

env = gym.make(
    "LunarLander-v2",
    continuous=False,
    gravity=-10.0,
    enable_wind=False,
    wind_power=15.0,
    turbulence_power=1.5,
    render_mode='human'
)

obs = env.reset()
while True:
    # Next action:
    # (feed the observation to your agent here)
    action = env.action_space.sample()

    # Processing:
    obs, reward, done, info = env.step(action)
    print(obs, reward)
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
