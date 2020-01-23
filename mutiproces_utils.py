import math
import multiprocessing
from functools import reduce

from genex.cluster_operations import _randomize, _cluster_groups, _cluster_to_meta, _cluster_reduce_func
from genex.utils import flatten
from process_utils import _grouper, _group_time_series, reduce_by_key


def _partitioner(data, slice_num, shuffle=True):
    if shuffle:
        data = _randomize(data)
    _slice_size = math.floor(len(data) / slice_num)
    return _grouper(_slice_size, data)


def _cluster_multi_process(p: multiprocessing.pool, data_normalized, start, end, st, dist_func, verbose):
    data_partition = _partitioner(data_normalized, p._processes)
    group_arg_partition = [(x, start, end) for x in data_partition]
    group_partition = p.starmap(_group_time_series, group_arg_partition)
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
