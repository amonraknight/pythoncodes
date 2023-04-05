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

# Model
BATCH_SIZE = 20
EMBEDDING_SIZE = 300
HIDDEN_SIZE = 100
OUTPUT_SIZE = 2
DROP_OUT = 0.1

# Model backup
BACKUP_AMOUNT = 5

# Training
WINDOW_SIZE = 40
EPOCH = 5
LEARNING_RATE = 0.0001


