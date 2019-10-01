import math
from pyspark import SparkContext
import numpy as np

from genex.cluster import _cluster, filter_cluster
from genex.classes.Sequence import Sequence


def flatten(l):
    flat_list = []
    for sublist in l:
        for item in sublist:
            flat_list.append(item)
    return flat_list


def filter_sublists(input_list, length):
    """

    :param input_list: list of raw clusters
    :param length:
    :return: generator object with each entry being [start, end, [clusters]]
    """
    if length > len(input_list):
        length = len(input_list)
        # raise Exception('genex: preprocess: filtered_sublist: given length is greater than the size of input list')
    return ([i, i + length - 1, input_list[i: i + length]] for i in range(len(input_list) - length + 1))


def filter_sublists_with_id(input_list, length):
    """

    :param input_list: [id, start, end, list of raw clusters]
    :param length:
    """
    if length > len(input_list[1]):
        length = len(input_list[1])
    return ([input_list[0], i, i + length - 1, input_list[1][i: i + length]] for i in
            range(len(input_list[1]) - length + 1))


def filter_sublists_with_id_length(input_list, length):
    """

    :param input_list: [id, start, end, list of raw clusters]
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


def all_sublists_with_id_length(input_list: list, loi: slice):
    # TODO refactor the loi  parameter, using slice notation
    tmp = []

    for i in range(loi.start, loi.end):
        tmp.append(list(filter_sublists_with_id_length(input_list, i)))
    return [y for x in tmp for y in x]  # flatten the list


def group_inputs(input_lists: list, loi: list):
    result = []

    for input_list in input_lists:

        tmp = []
        if len(loi) == 1:
            loi.append(math.inf)

        if loi[1] > len(input_list[1]) + 1:
            print('Warning: given loi exceeds maximum sequence length, setting end point to sequence length')
            loi[1] = len(input_list[1])  # + 1

        for i in range(loi[0], loi[1] + 1):
            tmp.append(list(filter_sublists_with_id_length(input_list, i)))
        result.append([y for x in tmp for y in x])  # flatten the list

    return result


def do_gcluster(input_list: list, loi: list, sc: SparkContext, num_cores: int,
                similarity_threshold: float = 0.1, dist_type: str = 'eu', is_collect: bool = True, log_level: int = 1):
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

    # normalize the input list, keep global_max and global min to return later
    normalized_input_list, global_max, global_min = min_max_normalize(input_list)
    input_rdd = sc.parallelize(normalized_input_list, numSlices=num_cores)
    group_rdd = input_rdd.flatMap(lambda x: all_sublists_with_id_length(x, loi))
    cluster_rdd = group_rdd.mapPartitions(
        lambda x: filter_cluster(groups=x, st=similarity_threshold, log_level=log_level, dist_type=dist_type),
        preservesPartitioning=False).cache()

    # if is_collect:
    #     return Gcluster(feature_list=feature_list,
    #                     data=input_list, data_normalized=normalized_input_list, st=similarity_threshold,
    #                     cluster_dict=dict(input_rdd.collect()), collected=True,
    #                     # this two attribute are different based on is_collect set to true or false
    #                     global_max=global_max, global_min=global_min)
    # else:
    #     return Gcluster(feature_list=feature_list,
    #                     data=input_list, data_normalized=normalized_input_list, st=similarity_threshold,
    #                     cluster_dict=input_rdd, collected=False,
    #                     global_max=global_max, global_min=global_min)


# def do_gcluster_legacy(input_list: list, loi: list, sc: SparkContext,
#                 similarity_threshold: float = 0.1, dist_type: str = 'eu', normalize: bool = True,
#                 del_data: bool = False, data_slices: int = 16, is_collect: bool = True, log_level: int = 1):
#     """
#     :param input_list:
#     :param loi: length of interets, ceiled at maximum length
#     :param sc:
#
#     :param similarity_threshold:
#     :param dist_type:
#     :param normalize:
#     :param del_data:
#     :param data_slices:
#     :param is_collect:
#
#     :return:
#     """
#     # inputs validation
#     # validate input exists
#     if len(input_list) == 0:
#         raise Exception('do_gcluster: nothing in input_list to cluster.')
#     # validate key value pairs
#     try:
#         dict(input_list)  # validate by converting input_list into a dict
#     except (TypeError, ValueError):
#         raise Exception('do_gcluster: input_list is not key-value pair.')
#     # validate the length of interest
#     if loi[0] <= 0:
#         raise Exception('do_gcluster: first element of loi must be equal to or greater than 1')
#     if loi[0] >= loi[1]:
#         raise Exception('do_gcluster: Start must be greater than end in the '
#                         'Length of Interest')
#
#     # validate the data length
#     all_ts = list(map(lambda x: x[1], input_list))
#     try:
#         assert not all(loi[1] > len(ts) for ts in all_ts)
#     except AssertionError as ae:
#         raise Exception('Given loi exceeds all input time series length')
#
#     if similarity_threshold <= 0 or similarity_threshold >= 1:
#         raise Exception('do_gcluster: similarity_threshold must be greater 0 and less than 1')
#
#     normalized_input_list, global_max, global_min = min_max_normalize(input_list)
#
#     # is normalization if enable, replace the input list with normalized input list
#     if normalize:
#         input_list = normalized_input_list
#
#     # create feature list
#     feature_list = flatten(map(lambda x: x[0], input_list))
#
#     # input_list = _min_max_normalize(input_list)
#     input_rdd = sc.parallelize(input_list, numSlices=data_slices)
#
#     input_rdd = input_rdd.flatMap(
#         lambda x: all_sublists_with_id_length(x, loi))  # get subsequences of all possible length
#     input_rdd = input_rdd.groupByKey().mapValues(list)  # group the subsequences by length
#
#     # cluster the input
#     input_rdd = input_rdd.map(
#         lambda x: _cluster(x, st=similarity_threshold, log_level=log_level, dist_type=dist_type, del_data=del_data))
#
#     if is_collect:
#         return genex_database(feature_list=feature_list,
#                               data=input_list, data_normalized=normalized_input_list, st=similarity_threshold,
#                               cluster_dict=dict(input_rdd.collect()), collected=True,
#                               # this two attribute are different based on is_collect set to true or false
#                               global_max=global_max, global_min=global_min)
#     else:
#         return genex_database(feature_list=feature_list,
#                               data=input_list, data_normalized=normalized_input_list, st=similarity_threshold,
#                               cluster_dict=input_rdd, collected=False,
#                               global_max=global_max, global_min=global_min)


def normalize_num(num, global_max, global_min):
    return (num - global_min) / (global_max - global_min)


def min_max_normalize(input_list, z_normalization=False):
    # scaler = MinMaxScaler(feature_range=(0, 1))
    #
    #
    #
    # input_array = np.array(list(map(lambda x: x[1], input_list)), dtype=np.float64)
    #
    # input_array_scaled = scaler.fit_transform(input_array)

    if z_normalization:
        input_list = z_normalize(input_list)

    flattened_list = np.array(flatten(list(map(lambda x: x[1], input_list))))
    global_max = flattened_list.max()
    global_min = flattened_list.min()

    normalized_list = map(lambda id_sequence:
                          [id_sequence[0],
                           list(map(lambda num: normalize_num(num, global_max, global_min), id_sequence[1]))]
                          , input_list)

    return list(normalized_list), global_max, global_min


def z_normalize(input_list):
    normalized_list = [(x[0], (x[1] - np.mean(x[1])) / np.std(x[1])) for x in input_list]
    return normalized_list
