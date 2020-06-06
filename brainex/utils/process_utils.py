from operator import itemgetter

import numpy as np
from functools import reduce
from itertools import groupby
from brainex.classes.Sequence import Sequence
from brainex.utils.utils import flatten


def dss_multiple(begin_index_iter, ts_list, start, end, parallelism):
    begin_index_iter = list(begin_index_iter)
    merged_g = []
    for ts_index, ts in enumerate(ts_list):
        g = dss((begin_index_iter[0] + ts_index) % parallelism, ts, start, end, parallelism)
        merged_g.append(g)

    merged_g = merge_dicts(merged_g, merge_func=lambda x, y: x + y)
    return list(merged_g.items())


def dss(begin_index, time_series, start, end, parallelism):
    """
    used when a single time series is given
    :param parallelism:
    :param begin_index:
    :param time_series:
    :param start:
    :param end:
    """

    ts_id = time_series[0]
    ts_data = time_series[1]

    rtn = dict()

    for ts_index in range(begin_index, len(ts_data), parallelism):  # step by distribution
        max_len = min(end, len(ts_data) - ts_index)
        for length in range(start - 1, min(end, len(ts_data) - ts_index)):  # get appropriate
            seq_len = length + 1
            if seq_len not in rtn.keys():
                rtn[seq_len] = []
            rtn[seq_len].append(Sequence(start=ts_index, end=ts_index + length, seq_id=ts_id))
    return rtn


def _group_time_series(time_series, start, end):
    """
    This function groups the raw time series data_original into sub sequences of all possible length within the given grouping
    range

    :param time_series: set of raw time series sequences
    :param start: starting index for grouping range
    :param end: end index for grouping range

    :return: a list of lists containing groups of subsequences of different length indexed by the group length
    """

    # start must be greater than 1, this is asserted in genex_databse._process_loi
    # time_series = kwargs['time_series']
    # start = kwargs['start']
    # end = kwargs['end']

    rtn = dict()

    for ts in time_series:
        ts_id = ts[0]
        ts_data = ts[1]
        # we take min because min can be math.inf
        for i in range(start - 1, min(end, len(ts_data))):
            target_length = i + 1
            if target_length not in rtn.keys():
                rtn[target_length] = []
            rtn[target_length] += _get_sublist_as_sequences(single_time_series=ts_data, data_id=ts_id, length=i)
    return list(rtn.items())


def _get_sublist_as_sequences(single_time_series, data_id, length):
    # if given length is greater than the size of the data_list itself, the
    # function returns an empty list
    rtn = []
    for i in range(0, len(single_time_series) - length):
        # if the second number in range() is less than 1, the iteration will not run
        # data_list[i:i+length]  # for debug purposes
        rtn.append(Sequence(start=i, end=i + length, seq_id=data_id))
    return rtn


def _slice_time_series(time_series, start, end):
    """
    This function slices raw time series data_original into sub sequences of all possible length.
    :param time_series: set of raw time series data_original
    :param start: start index of length range
    :param end: end index of length range

    :return: list containing  subsequences of all possible lengths
    """
    # start must be greater than 1, this is asserted in genex_databse._process_loi
    rtn = list()

    for ts in time_series:
        ts_id = ts[0]
        ts_data = ts[1]
        # we take min because min can be math.inf
        for i in range(start, min(end, len(ts_data))):
            rtn += _get_sublist_as_sequences(single_time_series=ts_data, data_id=ts_id, length=i)
    return rtn


def _grouper(n, iterable):
    return [iterable[x:x + n] for x in range(0, len(iterable), n)]


def get_first(p):
    return p[0]


def get_second(p):
    return p[1]


def reduce_by_key(func, iterable):
    """Reduce by key.
    ApocalyVec adopted from https://gist.github.com/Juanlu001/562d1ec55be970403442
    Equivalent to the Spark counterpart
    Inspired by http://stackoverflow.com/q/33648581/554319
    1. Sort by key
    2. Group by key yielding (key, grouper)
    3. For each pair yield (key, reduce(func, last element of each grouper))
    """
    # iterable.groupBy(_._1).map(l => (l._1, l._2.map(_._2).reduce(func)))
    return map(
        lambda l: (l[0], reduce(func, map(get_second, l[1]))),
        groupby(sorted(iterable, key=get_first), get_first)
    )


def merge_dicts(dict_list: list, merge_func):
    ''' Merge dictionaries and merge values by merge_funct'''
    rtn = {}
    for d in dict_list:
        for k, v in d.items():
            if k in rtn.keys():
                rtn[k] = merge_func(rtn[k], v)
            else:
                rtn[k] = v
    return rtn


def equal_ignore_order(a, b):
    """ Use only when elements are neither hashable nor sortable! """
    unmatched = list(b)
    for element in a:
        try:
            unmatched.remove(element)
        except ValueError:
            return False
    return not unmatched