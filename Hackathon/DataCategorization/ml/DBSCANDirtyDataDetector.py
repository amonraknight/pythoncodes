import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.cluster import DBSCAN

import common.Config as cfg
from common.EditDistanceCalculator import edit_dist
from preprocessing.SampleFileReader import get_samples_having_dirty_data


def find_ill_formed_distance(distance_matrix, distance_threshold):
    print(type(distance_matrix))
    dbscan = DBSCAN(eps=distance_threshold, metric='precomputed')
    dbscan.fit_predict(distance_matrix)
    idx = dbscan.labels_
    return np.where(idx == -1)


def prepare_distance_matrix(in_str_array):
    print(type(in_str_array))
    dist_l = pdist(in_str_array, metric=lambda a, b: edit_dist(a[0], b[0]))
    dist_m = squareform(dist_l)
    return dist_m


def find_ill_formed_str(str_df):

    str_df[1] = str_df[0].map(len)
    len_array = str_df[1].values
    avg_len = round(np.average(len_array))
    dirty_data_array = str_df[0].values.reshape(-1, 1)

    dist_matrix = prepare_distance_matrix(dirty_data_array)

    '''
    scattered_idx_list = []
    for diff in range(int(avg_len), int(avg_len/2), -1):

        scattered_idx_list = find_ill_formed_distance(dist_matrix, diff)
        if scattered_idx_list is not None and len(scattered_idx_list) > 0:
            break
    '''

    scattered_idx_list = find_ill_formed_distance(dist_matrix, int(avg_len/2))
    print(scattered_idx_list)


dirty_data_df = get_samples_having_dirty_data(cfg.DIRTY_DATA_IN_ID)
find_ill_formed_str(dirty_data_df)
