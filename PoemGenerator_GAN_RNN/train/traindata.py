import numpy as np
import torch.utils.data as tdata
import os

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


if __name__ == "__main__":
    prepare_train_test_data()
