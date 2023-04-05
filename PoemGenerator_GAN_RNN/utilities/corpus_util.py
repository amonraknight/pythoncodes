# UTF-8
# This utility is to make the conversions
#       (1) from a Chinese character to an index
#       (2) from an index to a Chinese character
#       (3) from an index to a vector
# Poems corpus


import numpy as np
import pathlib
import re
import os

import common.config as config


def get_char_vectors():

    if not os.path.exists(config.PATH_SIKU_CHARS) or not os.path.exists(config.PATH_SIKU_VECTORS):
        ori_vec_array = np.loadtxt(fname=config.PATH_SIKU_VECTORS_ORI, delimiter=' ', dtype='str', encoding='UTF-8')
        char_array = ori_vec_array[:, 0:1].flatten()
        vectors = ori_vec_array[:, 1:-1]
        vectors = vectors.astype('float32')
        np.save(config.PATH_SIKU_CHARS, char_array)
        np.save(config.PATH_SIKU_VECTORS, vectors)
        return char_array, vectors
    else:
        char_array = np.load(config.PATH_SIKU_CHARS, allow_pickle=True)
        vectors = np.load(config.PATH_SIKU_VECTORS, allow_pickle=True)

    return char_array, vectors


def concatenate_poems():
    poem_array = read_poem_corpus()
    np.savetxt(config.PATH_POEM_TABLE, poem_array, fmt='%s', delimiter=',', encoding='UTF-8')
    np.save(config.PATH_POEM_NDARRAY, poem_array)


def read_poem_corpus():
    single_poem_dir = pathlib.Path(config.PATH_SINGLE_POEMS)
    poem_table = None

    i = 0
    for single_poem in single_poem_dir.iterdir():
        each_row = extract_each_poem(single_poem)
        if poem_table is None:
            poem_table = each_row
        else:
            poem_table = np.concatenate((poem_table, each_row), axis=0)

        i += 1
        if i % 500 == 0:
            print('{} files read.'.format(i))
    return poem_table


def extract_each_poem(single_path):
    single_file = open(single_path, 'r', encoding='utf-8')
    lines = single_file.readlines()

    dynasty = re.match(config.REGEX_POEM_DYNASTY, lines[0]).group(1)
    author = re.match(config.REGEX_POEM_AUTHOR, lines[1]).group(1)
    tags = re.match(config.REGEX_POEM_TAGS, lines[2]).group(1)
    star = re.match(config.REGEX_POEM_STAR, lines[3]).group(1)
    author_stars = re.match(config.REGEX_POEM_AUTHOR_STARS, lines[4]).group(1)
    title = re.match(config.REGEX_POEM_TITLE, lines[5]).group(1)
    content = re.match(config.REGEX_POEM_CONTENT, lines[6]).group(1)

    if tags is not None:
        tags = tags.replace(',', '，')

    if title is not None:
        title = title.replace(',', '，')

    if content is not None:
        content = content.replace(',', '，')

    array = np.array([[dynasty, author, tags, star, author_stars, title, content]])
    return array


def read_concatenated_poem_corpus(corpus_path):
    array = None

    with open(corpus_path, 'r', encoding='UTF-8') as corpus:
        lines = corpus.readlines()
        for each_line in lines:
            each_row_data = np.array([each_line.split(',')])
            if array is None:
                array = each_row_data
            else:
                array = np.append(array, each_row_data, axis=0)

    return array


def get_poem_corpus(dynasty=None, author=None):
    if not os.path.exists(config.PATH_POEM_NDARRAY):
        concatenate_poems()

    # poem_array = read_concatenated_poem_corpus(config.PATH_POEM_TABLE)
    poem_array = np.load(config.PATH_POEM_NDARRAY, allow_pickle=True)
    filters = np.array([])
    if dynasty is not None:
        filters = np.where(poem_array[:, 0] == dynasty, True, False)

    if author is not None:
        filters = np.logical_and(filters, np.where(poem_array[:, 1] == author, True, False))

    if len(filters) == poem_array.shape[0]:
        poem_array = poem_array[filters]

    return poem_array[:, 6]


def get_shijing_corpus():
    array = None
    if not os.path.exists(config.PATH_SHIJING_NDARRAY):
        with open(config.PATH_SHIJING_ORI, 'r', encoding='UTF-8') as corpus:
            lines = corpus.readlines()
            for each_line in lines:
                each_row_data = np.array([each_line.split(',')])
                if array is None:
                    array = each_row_data
                else:
                    array = np.append(array, each_row_data, axis=0)

    return array[:, 1]

if __name__ == "__main__":
    # get_poem_corpus(dynasty='唐代', author='李白')
    get_char_vectors()

