# Please keep all settings here:
# Paths
OUTPUT_PATH = r'E:\\development\\GitRepository\\pythoncodes\\\dqn_puly_1\\output\\'
ANIMATION_SAVE_PATH = OUTPUT_PATH+'movie_cartpole_DQN.mp4'

# Constant values:

# The environment CartPole-v0 is out of date. You should consider upgrading to version `v1`.
ENV = 'CartPole-v1'
GAMMA = 0.99
MAX_STEPS = 200
NUM_EPISODES = 500

BATCH_SIZE = 32
CAPACITY = 10000
LEARNING_RATE = 0.0001

ACCEPT_THRESHOLD = 10

# Prioritized Sampling:
TD_ERROR_EPSILON = 0.0001

RENDER = False
