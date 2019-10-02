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
        data = [(x-np.mean(data)/np.std(data)) for x in data]

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


def query_partition(cluster, q, st: float, k: int, normalized_input, dist_type: str = 'eu', loi=None,
                            exclude_same_id: bool = False, overlap: float = 1.0,
                            lb_optimization='heuristic', reduction_factor_lbkim='half', reduction_factor_lbkeogh=2):
    # if the reduction factor is a string, reduction is performed based on the cluster size
    # if the reduction factor is a number n, reduction is performed by taking the top n
    # the reduction factors are only for experiment purposes
    if reduction_factor_lbkim == 'half':
        lbkim_mult_factor = 0.5
    elif reduction_factor_lbkim == '1quater':
        lbkim_mult_factor = 0.25
    elif reduction_factor_lbkim == '3quater':
        lbkim_mult_factor = 0.75
    else:
        raise Exception('reduction factor must be one of the specified string')

    if reduction_factor_lbkeogh == 'half':
        lbkeogh_mult_factor = 0.5
    elif reduction_factor_lbkeogh == '1quater':
        lbkeogh_mult_factor = 0.25
    elif reduction_factor_lbkeogh == '3quater':
        lbkeogh_mult_factor = 0.75
    else:
        raise Exception('reduction factor must be one of the specified string')

    q = q.value
    q_length = len(q.data)

    normalized_input = normalized_input.value

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
    # temperoray variable that decides whether to look up or down when a cluster of a specific length is exhausted

    while len(cluster_dict) > 0:
        target_cluster = cluster_dict[target_length]
        query_threshold = math.sqrt(target_length) * st / 2
        target_cluster_reprs = target_cluster.keys()
        target_cluster_reprs = list(
            map(lambda rpr: [sim_between_seq(rpr.fetch_data(normalized_input), q.data, dist_type=dist_type), rpr],
                target_cluster_reprs))  # calculates the warped distance between the query and the representatives
        target_cluster_reprs = [x for x in target_cluster_reprs if x[0] < query_threshold]

        # add a counter to avoid comparing a Sequence object with another Sequence object
        heapq.heapify(target_cluster_reprs)

        while len(target_cluster_reprs) > 0:
            querying_repr = heapq.heappop(target_cluster_reprs)
            querying_cluster = target_cluster[querying_repr[1]]

            # filter by id
            if exclude_same_id:
                querying_cluster = (x for x in querying_cluster if x.id != q.id)

            # fetch data for the target cluster
            querying_cluster = ((x, x.fetch_data(normalized_input)) for x in
                                querying_cluster)  # now entries are (seq, data)
            if lb_optimization == 'heuristic':
                # Sorting sequence using cascading bounds
                # need to make sure that the query and the candidates are of the same length when calculating LB_keogh
                if target_length != q_length:
                    print('interpolating')
                    querying_cluster = (
                    (x[0], np.interp(np.linspace(0, 1, q_length), np.linspace(0, 1, len(x[1])), x[1])) for x in
                    querying_cluster)  # now entries are (seq, interp_data)

                # sorting the sequence using LB_KIM bound
                querying_cluster = [(x[0], x[1], lb_kim_sequence(x[1], q.data)) for x in querying_cluster]
                querying_cluster.sort(key=lambda x: x[2])
                # checking how much we reduce the cluster
                if type(reduction_factor_lbkim) == str:
                    querying_cluster = querying_cluster[:int(len(querying_cluster) * lbkim_mult_factor)]
                elif type(reduction_factor_lbkim) == int:
                    querying_cluster = querying_cluster[:reduction_factor_lbkim * k]
                else:
                    raise Exception('Type of reduction factor must be str or int')

                # Sorting the sequence using LB Keogh bound
                querying_cluster = [(x[0], x[1], lb_keogh_sequence(x[1], q.data)) for x in
                                    querying_cluster]  # now entries are (seq, data, lb_heuristic)
                querying_cluster.sort(key=lambda x: x[2])  # sort by the lb_heuristic
                # checking how much we reduce the cluster
                if type(reduction_factor_lbkeogh) == str:
                    querying_cluster = querying_cluster[:int(len(querying_cluster) * lbkeogh_mult_factor)]
                elif type(reduction_factor_lbkim) == int:
                    querying_cluster = querying_cluster[:reduction_factor_lbkeogh * k]
                else:
                    raise Exception('Type of reduction factor must be str or int')

                querying_cluster_reduced = [(sim_between_seq(x[1], q.data, dist_type=dist_type), x[0]) for x in
                                    querying_cluster]  # now entries are (dist, seq)


            elif lb_optimization == 'bestSoFar':
                dist_buffer = list()
                querying_cluster_reduced = list()
                if len(dist_buffer) < k:
                    querying_cluster_reduced.append()

            if len(querying_cluster_reduced) == 0:
                continue
            heapq.heapify(querying_cluster_reduced)
            for cur_match in querying_cluster_reduced:
                if overlap != 1.0:
                    if not any(_isOverlap(cur_match[1], prev_match[1], overlap) for prev_match in
                               query_result):  # check for overlap against all the matches so far
                        print('Adding to querying result')
                        query_result.append(cur_match[:2])
                    else:
                        print('Overlapped, Not adding to query result')
                else:
                    print('Not applying overlapping')
                    query_result.append(cur_match[:2])

                if (len(query_result)) >= k:
                    return query_result

        cluster_dict.pop(target_length)  # remove this len-cluster just queried

        # find the next closest sequence length
        if len(cluster_dict) != 0:
            target_length = min(list(cluster_dict.keys()), key=lambda x: abs(x - target_length))
        else:
            break
    return query_result


def _validate_gxdb_build_arguments(gxdb, args: dict):
    """
    sanity check function for the arguments of build as a class method @ genex_databse object
    :param gxdb:
    :param args:
    :return:
    """
    # TODO finish the exception messages
    if 'loi' in args and args['loi']:
        try:
            assert args['loi'].step == None
        except AssertionError as ae:
            raise Exception('Build check argument failed: build loi(length of interest) does not support stepping, '
                            'loi.step=' + str(args['loi'].step))
    print(args)

    return


def _df_to_list(df, feature_num):
    df_list = [_row_to_feature_and_data(x, feature_num) for x in df.values.tolist()]
    return df_list

def _list_to_df(data_list):
    max_seq_len = max([len(x[1]) for x in data_list])
    rtn = np.empty((len(data_list), ))
    rtn[:] = np.nan

def _row_to_feature_and_data(row, feature_num):
    # list slicing syntax: ending at the key_num-th element but not include it
    key = tuple([str(x) for x in row[:feature_num]])
    value = [x for x in row[feature_num:] if not np.isnan(x)]
    return (key, value)
