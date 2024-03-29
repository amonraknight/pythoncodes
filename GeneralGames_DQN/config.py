# DQN parameters
LEARNING_RATE = 0.0001
GAMMA = 0.99
CAPACITY = 5000
BATCH_SIZE = 128
NUM_EPISODES = 30000
RANDOM_CHANCE = 0.2
FLAP_CHANCE = 0.1
TARGET_RENEW_RATE = 2
TRAIN_INTERVAL = 1
DONOTHING_CHANCE = 0.1

# Game
GAME_NAME = 'CarRacing-v2'

SKIP_FRAMES = 40

CARRACING_ACTIONS = [
    [0, 0, 0],      # 0, do nothing
    [0, 1, 0],      # 1, full gas
    [0, 0.5, 0],    # 2, half gas
    [0, 0, 1],    # 3, forceful break
    [0, 0, 0.5],    # 4, gentle break
    [-1, 0, 0],     # 5, full left
    [-0.5, 0, 0],   # 6, half left
    [-0.5, 0.5, 0],     # 7, half left/half gas
    [1, 0, 0],      # 8, full right
    [0.5, 0, 0],    # 9, half right
    [0.5, 0.5, 0]   # 10, half right/half gas
]

# Getting the action parameter list by taking the softmax results as possibilities to mock continuous action domain.
# But this is to have a lower score.
USE_MERGE_POSSIBILITY = False

# Model settings
# 4 frames * 3 channels
FRAMES_EACH_OBSERVATION = 3
DIM_IN = FRAMES_EACH_OBSERVATION*3

# [left steering, right steering, gas, break] All are positive values
DIM_OUT = len(CARRACING_ACTIONS)

FC_LAYER_1_SIZE = 64
FC_LAYER_2_SIZE = 64

RENDER_MODE = None
# RENDER_MODE = 'human'


# Backup
OUTPUT_PATH = '..\\outputs\\'
MODULE_PATH = OUTPUT_PATH + 'module_{}.pt'
MODULE_LIST_RECORD = OUTPUT_PATH + 'MODULE_LIST'
BACKUP_INTERVAL = 30
BACKUP_AMOUNT = 5
