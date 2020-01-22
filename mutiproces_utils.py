import math
import multiprocessing

from genex.cluster import _randomize, _cluster_groups
from process_utils import _grouper, _group_time_series


def _partitioner(data, slice_num, shuffle=True):
    if shuffle:
        data = _randomize(data)
    _slice_size = math.floor(len(data) / slice_num)
    return _grouper(_slice_size, data)


def _cluster_multi_process(data_normalized, slice_num, start, end, st, dist_func, verbose):
    p = multiprocessing.Pool(slice_num, maxtasksperchild=1)
    data_partition = _partitioner(data_normalized, slice_num)
    group_arg_partition = [(x, start, end) for x in data_partition]
    group_partition = p.starmap(_group_time_series, group_arg_partition)
    cluster_arg_partition = [(x, st, dist_func, verbose) for x in group_partition]
    cluster_partition = p.starmap(_cluster_groups, cluster_arg_partition)
    pass