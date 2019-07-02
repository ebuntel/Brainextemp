from pyspark import SparkContext
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from .cluster import _cluster
from genex.classes.Gcluster import Gcluster
from genex.classes.Sequence import Sequence

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
    return ((length, Sequence(input_list[0], i, i + length - 1, input_list[1][i: i + length])) for i in
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


def _all_sublists_with_id_length(input_list:list, loi:list):
    tmp = []

    if loi[1] > len(input_list[1]) + 1:
        print('Warning: given loi exceeds maximum sequence length, setting end point to sequence length')
        loi[1] = len(input_list[1]) + 1
    else:
        loi[1] += 1  # length offset

    for i in range(loi[0], loi[1]):
        tmp.append(list(filter_sublists_with_id_length(input_list, i)))
    return [y for x in tmp for y in x]  # flatten the list


def do_gcluster(input_list: list, loi: list, sc: SparkContext,
                similarity_threshold: float = 0.1, dist_type: str= 'eu', normalize: bool=True, del_data: bool = False, data_slices:int=16, isCollect: bool=False):
    """
    :param input_list:
    :param loi: length of interets, ceiled at maximum length
    :param sc:

    :param similarity_threshold:
    :param dist_type:
    :param normalize:
    :param del_data:
    :param data_slices:
    :param isCollect:

    :return:
    """
    # inputs validation
    # validate input exists
    if len(input_list) == 0:
        raise Exception('do_gcluster: nothing in input_list to cluster.')
    # validate key value pairs
    try:
        dict(input_list)  # validate by converting input_list into a dict
    except (TypeError, ValueError):
        raise Exception('do_gcluster: input_list is not key-value pair.')
    # validate the length of interest
    if loi[0] <= 0:
        raise Exception('do_gcluster: first element of loi must be equal to or greater than 1')
    if loi[0] >= loi[1]:
        raise Exception('do_gcluster: Start must be greater than end in the '
                        'Length of Interest')

    if similarity_threshold <= 0 or similarity_threshold >= 1:
        raise Exception('do_gcluster: similarity_threshold must be greater 0 and less than 1')

    input_list = _min_max_normalize(input_list)
    input_rdd = sc.parallelize(input_list, numSlices=data_slices)

    input_rdd = input_rdd.flatMap(lambda x: _all_sublists_with_id_length(x, loi))  # get subsequences of all possible length
    input_rdd = input_rdd.groupByKey().mapValues(list)  # group the subsequences by length

    # cluster the input
    input_rdd = input_rdd.map(lambda x: _cluster(x, similarity_threshold, dist_type, del_data))

    if isCollect:
        return Gcluster(dict(input_rdd.collect()), collected=True)
    else:
        return Gcluster(input_rdd, collected=False)


def _min_max_normalize(input_list):
    scaler = MinMaxScaler(feature_range=(0, 1))
    input_array = np.array(list(map(lambda x: x[1], input_list)))
    input_array_scaled = scaler.fit_transform(input_array)

    normalize_list = []

    for i in range(len(input_list)):
        normalize_list.append([input_list[i][0], input_array_scaled[i]])

    return normalize_list


