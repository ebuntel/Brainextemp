import math
import multiprocessing
from functools import reduce

from genex.op.cluster_op import _randomize, _cluster_groups, _cluster_to_meta, _cluster_reduce_func
from genex.op.query_op import get_dist_query
from genex.utils.utils import flatten
from process_utils import _grouper, _group_time_series, reduce_by_key, get_second


def _partitioner(data, slice_num):
    _slice_size = math.floor(len(data) / slice_num)
    return _grouper(_slice_size, data)


def __partition_and_group(data, slice_num, start, end, p: multiprocessing.pool, shuffle=True):
    data_partition = _partitioner(data, p._processes)
    group_arg_partition = [(x, start, end) for x in data_partition]
    group_partition = p.starmap(_group_time_series, group_arg_partition)
    return group_partition


def _cluster_multi_process(p: multiprocessing.pool, data_normalized, start, end, st, dist_func, verbose):
    group_partition = __partition_and_group(data_normalized, p._processes, start, end, p)
    cluster_arg_partition = [(x, st, dist_func, verbose) for x in group_partition]
    """
    Linear Cluster for debug purposes
    # cluster_partition = []
    # for arg in cluster_arg_partition:
    #     cluster_partition.append(_cluster_groups(*arg))

    """
    cluster_partition = p.starmap(_cluster_groups, cluster_arg_partition)
    cluster_meta_dict = _cluster_to_meta_mp(cluster_partition, p)

    return cluster_partition, cluster_meta_dict


def _cluster_to_meta_mp(cluster_partition: list, p: multiprocessing.pool):
    clusters = flatten(cluster_partition)
    temp = p.map(_cluster_to_meta, clusters)
    return list(reduce_by_key(_cluster_reduce_func, temp))


def _query_bf_mp(query, p: multiprocessing.pool, data_normalized: list, start, end, dt_index):
    group_partition = flatten( __partition_and_group(data_normalized, p._processes, start, end, p))
    subsequences = flatten(p.map(get_second, group_partition))
    dist_subsequences_arg = [(query, x, dt_index) for x in subsequences]
    dist_subsequences = p.starmap(get_dist_query, dist_subsequences_arg)
    return dist_subsequences


