# Paths
PATH_SIKU_VECTORS_ORI = r'E:\\development\\GitRepository\\pythoncodes\\PoemGenerator_GAN_RNN\\corpus\\sgns' \
                        r'.sikuquanshu.word'

PATH_SIKU_CHARS = r'E:\\development\\GitRepository\\pythoncodes\\PoemGenerator_GAN_RNN\\corpus\\siku_chars.npy'
PATH_SIKU_VECTORS = r'E:\\development\\GitRepository\\pythoncodes\\PoemGenerator_GAN_RNN\\corpus\\siku_vectors.npy'

PATH_SINGLE_POEMS = r'E:\\development\\GitRepository\\pythoncodes\\PoemGenerator_GAN_RNN\\corpus\\corpus_poem'
PATH_POEM_TABLE = r'E:\\development\\GitRepository\\pythoncodes\\PoemGenerator_GAN_RNN\\corpus\\corpus_poem.csv'
PATH_POEM_NDARRAY = r'E:\\development\\GitRepository\\pythoncodes\\PoemGenerator_GAN_RNN\\corpus\\corpus_poem.npy'

PATH_SHIJING_ORI = r'E:\\development\\GitRepository\\pythoncodes\\PoemGenerator_GAN_RNN\\corpus\\shijing2.txt'
PATH_SHIJING_NDARRAY = r'E:\\development\\GitRepository\\pythoncodes\\PoemGenerator_GAN_RNN\\corpus\\corpus_shijing.npy'

PATH_X = r'E:\\development\\GitRepository\\pythoncodes\\PoemGenerator_GAN_RNN\\traindata\\X.npy'
PATH_Y = r'E:\\development\\GitRepository\\pythoncodes\\PoemGenerator_GAN_RNN\\traindata\\Y.npy'

PATH_MODULE_BACKUP = r'E:\\development\\GitRepository\\pythoncodes\\PoemGenerator_GAN_RNN\\modelpackup\\module_{}.pt'
PATH_MODULE_LIST = r'E:\\development\\GitRepository\\pythoncodes\\PoemGenerator_GAN_RNN\\modelpackup\\MODULE_LIST'


# Regexp
REGEX_POEM_DYNASTY = 'dynasty:(.+)?'
REGEX_POEM_AUTHOR = 'author:(.+)?'
REGEX_POEM_TAGS = 'tags:(.+)?'
REGEX_POEM_STAR = 'star:(.+)?'
REGEX_POEM_AUTHOR_STARS = 'author_stars:(.+)?'
REGEX_POEM_TITLE = 'title:(.+)?'
REGEX_POEM_CONTENT = 'content:(.+)?'

# char_index
# 'P', 'U', 'S', 'E'
IDX_P = 0
IDX_U = 1
IDX_S = 2
IDX_E = 3


# Model
BATCH_SIZE = 8
EMBEDDING_SIZE = 300
HIDDEN_SIZE = 100
OUTPUT_SIZE = 1
DROP_OUT = 0.1
LEAK = 0.2
# The number of heads in the transformer.
HEAD_NUM = 10

HIDDEN_SIZE_2 = 1024
LAYER_SIZE_2 = 3
TEACHER_FORCING_RATE = 0.6

LAYER_SIZE_D = 2
# The encoder size and the decoder size must be the same.
LAYER_SIZE_G = 3

# Model backup
BACKUP_AMOUNT = 5

# Training
IS_DEBUG = False
WINDOW_SIZE = 40
EPOCH = 100
LEARNING_RATE = 0.001
CLIP = 0.5
GAN_RATE_IDX = 0.98
LEARNING_RATE_D = 0.00001
LEARNING_RATE_G = 0.0001
D_TRAIN_THRESHOLD = 0.015
ROLLOUT_UPDATE_RATE = 0.8
ROLLOUT_NUMBER = 2

# Poem Generation:
LINE_MAX_LENGTH = 9

MANUAL_TEST_LINES = [
    '大风起兮云飞扬',
    '春雨惊春清谷天',
    '路漫漫其修远兮',
    '春初早被相思染',
    '不见春风不肯开',
    '不知细叶谁裁出',
    '春江花月夜',
    '爱在西元前',
    '打个响指吧',
    '白日依山尽'
]

