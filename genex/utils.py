import heapq
import math

# from genex.Gcluster_utils import _isOverlap
from sklearn.preprocessing import MinMaxScaler

from genex.Gcluster_utils import _isOverlap
from genex.classes.Sequence import Sequence
from genex.cluster import sim_between_seq, lb_kim_sequence, lb_keogh_sequence
from genex.preprocess import normalize_num
import numpy as np


def normalize_sequence(seq: Sequence, max, min, z_normalize=True):
    if seq.data is None:
        raise Exception('Given sequence does not have data set, use fetch_data to set its data first')
    data = seq.data
    if z_normalize:
        data = [(x - np.mean(data) / np.std(data)) for x in data]

    normalized_data = list(map(lambda num: normalize_num(num, max, min), data))

    seq.set_data(normalized_data)


def scale(ts_df, feature_num):
    time_series = ts_df.iloc[:, feature_num:].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    num_time_series = len(time_series)
    time_series = time_series.reshape(-1, 1)
    time_series = scaler.fit_transform(time_series)
    time_series = time_series.reshape(num_time_series, -1)

    df_normalized = ts_df.copy()
    df_normalized.iloc[:, feature_num:] = time_series

    return df_normalized, scaler


# def reduction_to_mult(reduction_factor):
# if the reduction factor is a string, reduction is performed based on the cluster size
# if the reduction factor is a number n, reduction is performed by taking the top n
# the reduction factors are only for experiment purposes
#     if type(reduction_factor) == int:
#         return reduction_factor
#     elif type(reduction_factor) == str:
#         if reduction_factor == 'half':
#             rtn = 0.5
#         elif reduction_factor == '1quater':
#             rtn = 0.25
#         elif reduction_factor == '3quater':
#             rtn = 0.75
#         else:
#             raise Exception('reduction factor must be one of the specified string, given: ' + reduction_factor)
#     else:
#         raise Exception('reduction facotr must be an int or one of the specified string, given: ' + reduction_factor)
#     return rtn

def prune_by_lbh(seq_list: list, seq_length: int, q: Sequence, kim_reduction: float = 0.75,
                 keogh_reduction: float = 0.25):
    # prune using lb_kim
    seq_list = [(x, lb_kim_sequence(x.data, q.data)) for x in seq_list]  # (seq, lb_kim_dist)
    seq_list.sort(key=lambda x: x[1])
    seq_list = seq_list[:int(len(seq_list) * kim_reduction)]  # take the top half

    # prune using lb_keogh
    if seq_length != len(q):
        seq_list = (
            (x[0], np.interp(np.linspace(0, len(x[0].data), len(q)), np.arrange(len(x[0].data)), x[0].data))
            for x in seq_list)  # now entries are (seq, interp_data)
    seq_list = [(x[0], lb_keogh_sequence(x[1], q.data)) for x in seq_list]  # (seq, lb_keogh_dist)
    seq_list.sort(key=lambda x: x[1])
    seq_list = [x[0] for x in seq_list[:int(len(seq_list) * keogh_reduction)]]  # take the top half

    return seq_list


def _query_partition(cluster, q, k: int, data_normalized, dist_type,
                     _lb_opt_cluster: str, repr_kim_rf: float, repr_keogh_rf: float,
                     _lb_opt_repr: str, cluster_kim_rf: float, cluster_keogh_rf: float,
                     loi=None, exclude_same_id: bool = False, overlap: float = 1.0):
    """

    :param cluster:
    :param q:
    :param k:
    :param data_normalized:
    :param dist_type:
    :param _lb_opt_cluster: 'bsf', 'lbh', 'lbh_bst, 'none'
    :param _lb_opt_repr: 'lbh', 'none'
    :param loi:
    :param exclude_same_id:
    :param overlap:
    :return:
    """
    q = q.value
    data_normalized = data_normalized.value

    q_length = len(q.data)

    # data_normalized = data_normalized.value

    if loi is not None:
        cluster_dict = dict(x for x in cluster if x[0] in range(loi[0], loi[1]))
    else:
        cluster_dict = dict(cluster)
    # get the seq length of the query, start query in the cluster that has the same length as the query

    target_length = len(q)

    # get the seq length Range of the partition
    try:
        len_range = (min(cluster_dict.keys()), max(cluster_dict.keys()))
    except ValueError as ve:
        raise Exception('cluster does not have the given query loi!')

    # if given query is longer than the longest cluster sequence,
    # set starting clusterSeq length to be of the same length as the longest sequence in the partition
    target_length = max(min(target_length, len_range[1]), len_range[0])

    query_result = []
    prune_count = 0
    # temperoray variable that decides whether to look up or down when a cluster of a specific length is exhausted

    while len(cluster_dict) > 0:
        target_cluster = cluster_dict[target_length]
        # query_threshold = math.sqrt(target_length) * st / 2
        target_reprs = target_cluster.keys()
        # fetch data for representatives
        [x.fetch_and_set_data(data_normalized) for x in target_reprs]

        if _lb_opt_repr == 'lbh':
            target_reprs = prune_by_lbh(seq_list=target_reprs, seq_length=target_length, q=q,
                                        kim_reduction=repr_kim_rf, keogh_reduction=repr_keogh_rf)

        target_reprs = [(sim_between_seq(x, q, dist_type=dist_type), x) for x in target_reprs]
        heapq.heapify(target_reprs)

        while len(target_reprs) > 0:
            querying_repr = heapq.heappop(target_reprs)
            querying_cluster = target_cluster[querying_repr[1]]

            # filter by id
            if exclude_same_id:
                querying_cluster = (x for x in querying_cluster if x.id != q.id)
                if len(querying_cluster) == 0:  # if after filtering, there's no sequence left, then simply continue
                    # to the next iteration
                    continue

            # fetch data for the target cluster
            [x.fetch_and_set_data(data_normalized) for x in querying_cluster]

            if (_lb_opt_cluster == 'lbh' or _lb_opt_cluster == 'lbh_bsf') and \
                    len(querying_cluster) > (k / cluster_keogh_rf) / cluster_kim_rf:
                querying_cluster = prune_by_lbh(seq_list=querying_cluster, seq_length=target_length, q=q,
                                                kim_reduction=cluster_kim_rf, keogh_reduction=cluster_keogh_rf)

            if _lb_opt_cluster == 'bsf' or _lb_opt_cluster == 'lbh_bsf':
                # use ranked heap
                query_result = list()
                # print('Num seq in the querying cluster: ' + str(len(querying_cluster)))
                for candidate in querying_cluster:
                    # print('Using bsf')
                    if len(query_result) < k:
                        # take the negative distance so to have a maxheap
                        heapq.heappush(query_result, (-sim_between_seq(q, candidate, dist_type), candidate))
                    else:  # len(dist_heap) == k or >= k
                        # if the new seq is better than the heap head
                        # a = -lb_kim_sequence(candidate.data, q.data)
                        if -lb_kim_sequence(candidate.data, q.data) < query_result[0][0]:
                            prune_count += 1
                            continue
                        # interpolate for keogh calculaton
                        if target_length != q_length:
                            candidate_interp_data = np.interp(np.linspace(0, 1, q_length),
                                                              np.linspace(0, 1, len(candidate.data)), candidate.data)
                        # b = -lb_keogh_sequence(candidate_interp_data, q.data)
                        if -lb_keogh_sequence(candidate_interp_data, q.data) < query_result[0][0]:
                            prune_count += 1
                            continue
                        # c = -lb_keogh_sequence(q.data, candidate_interp_data)
                        if -lb_keogh_sequence(q.data, candidate_interp_data) < query_result[0][0]:
                            prune_count += 1
                            continue
                        dist = -sim_between_seq(q, candidate, dist_type)
                        if dist > query_result[0][0]:  # first index denotes the top of the heap, second gets the dist
                            heapq.heappop(query_result)
                            heapq.heappush(query_result, (dist, candidate))
                if (len(query_result)) >= k:
                    print('Found k matches, returning query result')
                    print(str(prune_count) + ' of ' + str(len(querying_cluster)) + ' pruned')
                    return [(-x[0], x[1]) for x in query_result]
            else:
                # print('Not using bsf')
                # calculate the distances between the query and all the sequences in the cluster
                dist_seq_list = [(sim_between_seq(x, q, dist_type), x) for x in querying_cluster]
                heapq.heapify(dist_seq_list)

                while len(dist_seq_list) > 0:
                    dist_seq = heapq.heappop(dist_seq_list)
                    if overlap == 1.0 or exclude_same_id:
                        print('Adding to querying result')
                        query_result.append(dist_seq)
                    else:
                        if not any(_isOverlap(dist_seq[1], prev_match[1], overlap) for prev_match in
                                   query_result):  # check for overlap against all the matches so far
                            print('Adding to querying result')
                            query_result.append(dist_seq)

                    if (len(query_result)) >= k:
                        print('Found k matches, returning query result')
                        return query_result
        cluster_dict.pop(target_length)  # remove this len-cluster just queried

        # find the next closest sequence length
        if len(cluster_dict) != 0:
            target_length = min(list(cluster_dict.keys()), key=lambda x: abs(x - target_length))
        else:
            break
    # print(str(prune_count) + ' sequences pruned')
    return query_result


def _validate_gxdb_build_arguments(args: dict):
    """
    sanity check function for the arguments of build as a class method @ genex_databse object
    :param gxdb:
    :param args:
    :return:
    """
    # TODO finish the exception messages

    if 'loi' in args and args['loi'] is not None:
        try:
            assert args['loi'].step == None
        except AssertionError as ae:
            raise Exception('Build check argument failed: build loi(length of interest) does not support stepping, '
                            'loi.step=' + str(args['loi'].step))
    try:
        assert 0. < args['similarity_threshold'] < 1.
    except AssertionError as ae:
        raise Exception('Build check argument failed: build similarity_threshold must be between 0. and 1. and not '
                        'equal to 0. and 1.')
    print(args)

    return


def _validate_gxdb_query_arguments(args: dict):
    _lb_opt_repr_options = ['lbh', 'none']
    _lb_opt_cluster_options = ['lbh', 'bsf', 'lbh_bst', 'none']
    try:
        assert args['_lb_opt_repr'] in _lb_opt_repr_options
    except AssertionError:
        print(
            'Query check argument failed: query _lb_opt_repr must be one of the following ' + str(_lb_opt_repr_options))
        raise AssertionError

    try:
        assert args['_lb_opt_cluster'] in _lb_opt_cluster_options
    except AssertionError:
        print(
            'Query check argument failed: query _lb_opt_cluster must be one of the following ' + str(
                _lb_opt_cluster_options))
        raise AssertionError


def _df_to_list(df, feature_num):
    df_list = [_row_to_feature_and_data(x, feature_num, df.head()) for x in df.values.tolist()]
    return df_list


def _row_to_feature_and_data(row, feature_num, feature_head):
    # list slicing syntax: ending at the key_num-th element but not include it
    # seq_id = tuple([(name, value) for name, value in zip(feature_head[:feature_num], row[:feature_num])])
    seq_id = tuple([str(x) for x in row[:feature_num]])

    data = [x for x in row[feature_num:] if not np.isnan(x)]
    return seq_id, data


def _process_loi(loi: slice):
    start = 1
    end = math.inf
    if loi is not None:
        if loi.start is not None:
            start = loi.start
        if loi.stop is not None:
            end = loi.stop

    assert start > 0
    return start, end

# if _lb_optimization == 'heuristic':
#     # Sorting sequence using cascading bounds
#     # need to make sure that the query and the candidates are of the same length when calculating LB_keogh
#     if target_length != q_length:
#         print('interpolating')
#         querying_cluster = (
#             (x[0], np.interp(np.linspace(0, 1, q_length), np.linspace(0, 1, len(x[1])), x[1])) for x in
#             querying_cluster)  # now entries are (seq, interp_data)
#
#     # sorting the sequence using LB_KIM bound
#     querying_cluster = [(x[0], x[1], lb_kim_sequence(x[1], q.data)) for x in querying_cluster]
#     querying_cluster.sort(key=lambda x: x[2])
#     # checking how much we reduce the cluster
#     if type(reduction_factor_lbkim) == str:
#         querying_cluster = querying_cluster[:int(len(querying_cluster) * lbkim_mult_factor)]
#     elif type(reduction_factor_lbkim) == int:
#         querying_cluster = querying_cluster[:reduction_factor_lbkim * k]
#     else:
#         raise Exception('Type of reduction factor must be str or int')
#
#     # Sorting the sequence using LB Keogh bound
#     querying_cluster = [(x[0], x[1], lb_keogh_sequence(x[1], q.data)) for x in
#                         querying_cluster]  # now entries are (seq, data, lb_heuristic)
#     querying_cluster.sort(key=lambda x: x[2])  # sort by the lb_heuristic
#     # checking how much we reduce the cluster
#     if type(reduction_factor_lbkeogh) == str:
#         querying_cluster = querying_cluster[:int(len(querying_cluster) * lbkeogh_mult_factor)]
#     elif type(reduction_factor_lbkim) == int:
#         querying_cluster = querying_cluster[:reduction_factor_lbkeogh * k]
#     else:
#         raise Exception('Type of reduction factor must be str or int')
#
#     querying_cluster_reduced = [(sim_between_seq(x[1], q.data, dist_type=dist_type), x[0]) for x in
#                                 querying_cluster]  # now entries are (dist, seq)
