import numpy as np
import torch
import torch.utils.data as tdata
import os
import re
from tqdm import tqdm

import utilities.corpus_util as corpus_util
import common.config as config


def prepare_train_test_data():
    # X, Y = None, None
    if not os.path.exists(config.PATH_X) or not os.path.exists(config.PATH_Y):
        libai_corpus = corpus_util.get_poem_corpus(dynasty='唐代', author='李白')
        shijing_corpus = corpus_util.get_shijing_corpus()

        # [0] is LiBai, [1] is ShiJing
        libai_y = np.repeat(np.array([0]), len(libai_corpus))
        shijing_y = np.repeat(np.array([1]), len(shijing_corpus))

        X = np.append(libai_corpus, shijing_corpus)
        Y = np.append(libai_y, shijing_y, axis=0)

        char_array, vectors = corpus_util.get_char_vectors()
        # Add <Pad> and <Unknown> as 0 and 1:
        vector_pad = np.zeros(vectors.shape[1])
        vector_unknown = vector_pad + 1

        char_array = np.insert(char_array, 0, '<Unknown>')
        char_array = np.insert(char_array, 0, '<Pad>')
        vectors = np.insert(vectors, 0, vector_unknown, axis=0)
        vectors = np.insert(vectors, 0, vector_pad, axis=0)

        X = vectorize_texts(X, char_array, vectors)

        np.save(config.PATH_X, X)
        np.save(config.PATH_Y, Y)

    else:
        X = np.load(config.PATH_X, allow_pickle=True)
        Y = np.load(config.PATH_Y, allow_pickle=True)

    train_idx, val_idx = tdata.random_split(X, [1000, X.shape[0] - 1000])
    # train_ds = tdata.TensorDataset()
    print('len(train_idx): {}, len(val_idx): {}'.format(len(train_idx), len(val_idx)))

    X_train = X[train_idx.indices]
    Y_train = Y[train_idx.indices]
    X_validate = X[val_idx.indices]
    Y_validate = Y[val_idx.indices]

    return X_train, Y_train, X_validate, Y_validate


def vectorize_texts(text_array, char_array, vectors):
    vector_sequence_array = None
    char_list = list(char_array)
    for each_poem in text_array:
        chars = list(each_poem)
        index_list = np.zeros(config.WINDOW_SIZE, dtype=np.int16)
        if len(each_poem) > config.WINDOW_SIZE:
            chars = chars[0:config.WINDOW_SIZE]

        for i, each_char in enumerate(chars):
            if each_char in char_list:
                index_list[i] = char_list.index(each_char)
            else:
                index_list[i] = 1

        # print(index_list)
        vector_sequence = np.array([[vectors[idx] for idx in index_list]])

        if vector_sequence_array is None:
            vector_sequence_array = vector_sequence
        else:
            vector_sequence_array = np.append(vector_sequence_array, vector_sequence, axis=0)

    return vector_sequence_array


# For line-to-line generator
# Prepare the training data in [upper context, target line].
def prepare_train_test_data_2():

    # Prepare the vectors and indexes.
    char_array, vectors = corpus_util.get_char_vectors()
    vectors = np.clip(vectors, -0.9999, 0.9999)
    char_array, vectors = expend_char_vectors(char_array, vectors)

    if not os.path.exists(config.PATH_X):
        poems = corpus_util.get_poem_corpus_v2(dynasties=None, authors=None, length=100)
        context_target_index_list = []

        for each_poem in tqdm(poems):
            if each_poem.endswith('，') or each_poem.endswith('。') or each_poem.endswith('？'):
                each_poem = each_poem[0:len(each_poem)-1]

            lines = re.split('[，。？]', each_poem)
            # print(lines)
            if len(lines) % 2 == 0:
                for i in range(0, len(lines) - 1, 2):
                    if (len(lines[i]) == 5 or len(lines[i]) == 7) and len(lines[i]) == len(lines[i+1]):
                        upper_context, target_line = convert_line_to_indexes(lines[i], char_array), \
                                                     convert_line_to_indexes(lines[i+1], char_array)
                        each_context_target = [upper_context, target_line]
                        context_target_index_list.append(each_context_target)
                    else:
                        break

        np.save(config.PATH_X, context_target_index_list)
    else:
        context_target_index_list = np.load(config.PATH_X, allow_pickle=True)

    return context_target_index_list, char_array, vectors


# Each line of the poem is a text of 5 or 7 chars.
# char_array should have <start>, <end>, <pad> and <unknown>.
def convert_line_to_indexes(line, char_array):
    char_list = list(char_array)
    index_list = np.zeros(9, dtype=np.int16)
    index_list[0] = 2  #<Start>
    chars = list(line)
    for i, each_char in enumerate(chars):
        if each_char in char_list:
            index_list[i+1] = char_list.index(each_char)
        else:
            index_list[i+1] = 1

    index_list[len(line)+1] = 3  # <End>
    return index_list


def expend_char_vectors(char_array, vectors):
    #'<Pad>' 0, '<Unknown>' 1, '<Start>' 2, '<End>' 3
    char_array = np.insert(char_array, 0, ['P', 'U', 'S', 'E'])

    vector_pad = np.zeros(vectors.shape[1])
    vector_unknown = vector_pad + 0.999
    vector_start = vector_pad - 0.999
    vector_end = vector_pad + 0.5

    vectors = np.insert(vectors, 0, [vector_pad, vector_unknown, vector_start, vector_end], axis=0)
    return char_array, vectors



if __name__ == "__main__":
    prepare_train_test_data_2()
