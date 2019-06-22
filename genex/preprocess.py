import math
from itertools import chain, combinations
from pyspark import SparkContext
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from .cluster import cluster
from .sequence import sequence

def filter_sublists(input_list, length):
    """

    :param input_list: list of raw data
    :param length:
    :return: generator object with each entry being [start, end, [data]]
    """
    if length > len(input_list):
        length = len(input_list)
        # raise Exception('genex: preprocess: filtered_sublist: given length is greater than the size of input list')
    return ([i, i + length - 1, input_list[i: i + length]] for i in range(len(input_list) - length + 1))


def filter_sublists_with_id(input_list, length):
    """

    :param input_list: [id, start, end, list of raw data]
    :param length:
    """
    if length > len(input_list[1]):
        length = len(input_list[1])
    return ([input_list[0], i, i + length - 1, input_list[1][i: i + length]] for i in
            range(len(input_list[1]) - length + 1))


def filter_sublists_with_id_length(input_list, length):
    """

    :param input_list: [id, start, end, list of raw data]
    :param length:
    """
    if length > len(input_list[1]):
        length = len(input_list[1])
    return ((length, sequence(input_list[0], i, i + length - 1, input_list[1][i: i + length])) for i in
            range(len(input_list[1]) - length + 1))


def all_sublists(input_list):
    tmp = []
    for i in range(1, len(input_list) + 1):
        tmp.append(list(filter_sublists(input_list, i)))
    return tmp


def all_sublists_with_id(input_list):
    tmp = []
    for i in range(1, len(input_list[1]) + 1):
        tmp.append(list(filter_sublists_with_id(input_list, i)))
    return [y for x in tmp for y in x]  # flatten the list


def all_sublists_with_id_length(input_list, loi):
    tmp = []

    if loi[0] < 0:
        raise Exception('Genex: preprocess: all_sublists_with_id_length: Start can not be 0 in the Length of Interest')
    if loi[1] > len(input_list[1]) + 1:
        loi[1] = len(input_list[1]) + 1
    else:
        loi[1] += 1  # length offset

    if loi[0] >= loi[1]:
        raise Exception('Genex: preprocess: all_sublists_with_id_length: Start must be greater than end in the '
                        'Length of Interest')

    for i in range(loi[0], loi[1]):
        tmp.append(list(filter_sublists_with_id_length(input_list, i)))
    return [y for x in tmp for y in x]  # flatten the list


def preprocess(input_list: list, loi: tuple, sc: SparkContext, similarity_threshhold: float = 0.1, dist_type: str='eu', normalize: bool=True, del_data: bool = False):
    """

    :param input_list:
    :param sc:
    :param loi: length of interets, ceiled at maximum length
    :param normalize:
    """
    input_list = min_max_normalize(input_list)
    input_rdd = sc.parallelize(input_list, numSlices=16)

    input_rdd = input_rdd.flatMap(lambda x: all_sublists_with_id_length(x, loi))  # get subsequences of all possible length
    input_rdd = input_rdd.groupByKey().mapValues(list)  # group the subsequences by length

    # cluster the input
    input_rdd = input_rdd.map(lambda x: cluster(x, similarity_threshhold, dist_type, del_data))
    return input_rdd.collect()


def min_max_normalize(input_list):
    scaler = MinMaxScaler(feature_range=(0, 1))
    input_array = np.array(list(map(lambda x: x[1], input_list)))
    input_array_scaled = scaler.fit_transform(input_array)

    normalize_list = []

    for i in range(len(input_list)):
        normalize_list.append([input_list[i][0], input_array_scaled[i]])

    return normalize_list
