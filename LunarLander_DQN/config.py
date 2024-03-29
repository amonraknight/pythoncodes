
# DQN parameters
LEARNING_RATE = 1e-4
GAMMA = 0.99
CAPACITY = 30000
BATCH_SIZE = 512
NUM_EPISODES = 3000
RANDOM_CHANCE = 0.2
FLAP_CHANCE = 0.1
TARGET_RENEW_RATE = 2
TRAIN_INTERVAL = 1
DONOTHING_CHANCE = 0.3

# Game
GAME_NAME = 'LunarLander-v2'


# Model settings
DIM_IN = 8
DIM_OUT = 4

FC_LAYER_1_SIZE = 64
FC_LAYER_2_SIZE = 64

# RENDER_MODE = None
RENDER_MODE = 'human'

# Backup
OUTPUT_PATH = '..\\outputs\\'
MODULE_PATH = OUTPUT_PATH + 'module_{}.pt'
MODULE_LIST_RECORD = OUTPUT_PATH + 'MODULE_LIST'
BACKUP_INTERVAL = 200
BACKUP_AMOUNT = 5
