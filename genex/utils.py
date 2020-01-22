import heapq
import math

# from genex.Gcluster_utils import _isOverlap
import time

from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cityblock
from scipy.spatial.distance import minkowski
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import chebyshev
from genex.Gcluster_utils import _isOverlap
from genex.classes.Sequence import Sequence
from genex.cluster import sim_between_seq, lb_kim_sequence, lb_keogh_sequence
import numpy as np

# dist_func_index = {'eu': np.vectorize(lambda x, y: euclidean(x, y) / np.sqrt(len(x))),
#                    'ma': np.vectorize(lambda x, y: cityblock(x, y) / len(x)),
#                    'min': np.vectorize(lambda x, y: minkowski(x, y) / np.sqrt(len(x))),
#                    'ch': np.vectorize(chebyshev)}


# return function that calculates the corresponding normalized distances
# note that the sequence x, y must be normalized in the first place
dist_func_index = {'eu': lambda x, y: euclidean(x, y) / np.sqrt(len(x)),
                   'ma': lambda x, y: cityblock(x, y) / len(x),
                   'min': lambda x, y: minkowski(x, y) / np.sqrt(len(x)),
                   'ch': chebyshev}


def normalize_sequence(seq: Sequence, max, min, z_normalize=True):
    """
    Use min max and z normalization to normalize time series data_original

    :param seq: Time series sequence
    :param max: maximum value in sequence
    :param min: minimum value in sequence
    :param z_normalize: whether data_original is z normalized or not

    """

    if seq.data is None:
        raise Exception('Given sequence does not have data_original set, use fetch_data to set its data_original first')
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


def prune_by_lbh(seq_list: list, seq_length: int, q: Sequence, kim_reduction: float = 0.75,
                 keogh_reduction: float = 0.25):
    """
    Prune the sequences based on LB_Kim and LB_Keogh lower bound. First the sequences are pruned using LB_Kim
    reduction factor and then using LB_Keogh reduction factor

    :param seq_list: list of candidate time series sequences
    :param seq_length: length of the sequences in the seq_list
    :param q: Query Sequence
    :param kim_reduction: Value of reduction factor for LB_Kim lower bound
    :param keogh_reduction: Value of reduction factor for LB_Keogh lower bound

    :return: a list containing remaining candidate sequences that can not be pruned using LBH
    """

    # prune using lb_kim
    seq_list = [(x, lb_kim_sequence(x.data, q.data)) for x in seq_list]  # (seq, lb_kim_dist)
    seq_list.sort(key=lambda x: x[1])
    seq_list = seq_list[:int(len(seq_list) * kim_reduction)]  # take the top half

    # prune using lb_keogh
    a = len(q)
    if seq_length != len(q):
        seq_list = [
            (x[0], np.interp(np.linspace(0, seq_length, len(q)), np.arange(seq_length), x[0].data))
            for x in seq_list]  # now entries are (seq, interp_data)
        seq_list = [(x[0], lb_keogh_sequence(x[1], q.data)) for x in seq_list]  # (seq, lb_keogh_dist)
    else:
        seq_list = [(x[0], lb_keogh_sequence(x[0].data, q.data)) for x in seq_list]  # (seq, lb_keogh_dist)

    seq_list.sort(key=lambda x: x[1])
    seq_list = [x[0] for x in seq_list[:int(len(seq_list) * keogh_reduction)]]  # take the top half

    return seq_list


def get_trgt_len(l_list, q_len):
    """
    from the l_list, give the closest value to the q_len
    :param l_list:
    :param q_len:
    :return:
    """
    return min(list(l_list), key=lambda x: abs(x - q_len))


def get_trgt_len_within_r(l_list, q_len, radius):
    """
    UNIT TEST NEEDED
    :param l_list:
    :param q_len:
    :param radius:
    :return:
    """
    return [x for x in l_list if q_len - radius <= x <= q_len + radius]
    # trgt_l_list = []
    # l_list_copy = l_list.copy()
    # for r in range(radius + 1):
    #     if l_list_copy:
    #         trgt_l = get_trgt_len(l_list=l_list_copy, q_len=q_len)
    #         trgt_l_list.append(trgt_l)
    #         l_list_copy.remove(trgt_l)
    #     else:
    #         break
    # return trgt_l_list


def naive_search_rspace(q, k, r_list, cluster):
    c_list = []
    target_reprs = [(sim_between_seq(x, q), x) for x in r_list]  # calculate DTW
    heapq.heapify(target_reprs)  # heap sort R-space
    # get enough sequence from the clusters represented to query
    while len(target_reprs) > 0 and len(c_list) < k:
        this_repr = heapq.heappop(target_reprs)[
            1]  # take the second element for the first one is the DTW dist
        c_list += (cluster[this_repr])
    return c_list


def _query_partition(cluster, q, k: int, ke: int, data_normalized, loi: slice,
                     _lb_opt_cluster: str, repr_kim_rf: float, repr_keogh_rf: float,
                     _lb_opt_repr: str, cluster_kim_rf: float, cluster_keogh_rf: float, overlap: float,
                     exclude_same_id: bool, radius: int, st: float):
    """
    This function finds k best matches for given query sequence on the worker node

    :param cluster: cluster being queried
    :param q: Query sequence
    :param k: number of best matches to retrieve
    :param data_normalized:
    :param _lb_opt_cluster:Type of optimization used for clusters ('bsf', 'lbh', 'lbh_bst, 'none')
    :param _lb_opt_repr:Type of optimization used for representatives ('lbh', 'none')
    :param loi: Length of interest, default value is none
    :param exclude_same_id: whether to exclude the query sequence when finding best matches
    :param overlap: Overlapping parameter( must be between 0 and 1 inclusive)

    :return: a list containing retrieved matches for given query sequence on that worker node
    """
    q = q.value
    data_normalized = data_normalized.value
    cluster_filtered = [x for x in cluster if x[0] in range(loi.start, loi.stop)] if loi is not None else cluster
    cluster_dict = dict(list(reduce_by_key(lambda x, y: merge_dict([x, y]), cluster_filtered)))
    try:
        assert cluster_filtered
    except AssertionError as ve:
        print('Given loi is ' + str(loi))
        print('sequence length in the database: ' + str(list(cluster_dict.keys())))
        raise Exception('cluster does not have the given query loi!')
    q_length = len(q.data)

    query_result = []

    # note that we are using ke here
    candidates = []
    while len(cluster_dict) > 0 and len(candidates) < ke:
        available_lens = list(cluster_dict.keys())
        # the specific sized sequences from which the candidates will be extracted, depending on the query length and
        # the radius
        target_l_list = get_trgt_len_within_r(l_list=available_lens, q_len=q_length, radius=radius)
        # print('Num Candidate = ' + str(len(candidates)) + ' cluster key: ' + str(list(cluster_dict.keys())) + '
        # targetlenlist: ' + str(target_l_list))
        for target_l in target_l_list:
            # print('searching')
            target_cluster = cluster_dict[target_l]
            target_reprs = target_cluster.keys()
            [x.fetch_and_set_data(data_normalized) for x in target_reprs]  # fetch data_original for the representatives

            # start = time.time()
            # this_candidates = get_sequences_represented(
            #     bsf_search_rspace(q, k, r_list=target_reprs, cluster=target_cluster, st=st),
            #     cluster=target_cluster)
            # duration_opt = time.time() - start
            # start = time.time()
            # this_candidates = naive_search_rspace(q, k, r_list=target_reprs, cluster=target_cluster)
            # duration_nonopt = time.time() - start

            if _lb_opt_repr == 'bsf':
                this_candidates = bsf_search_rspace(q, k, r_list=target_reprs, cluster=target_cluster, st=st)
            else:
                this_candidates = naive_search_rspace(q, k, r_list=target_reprs, cluster=target_cluster)

            candidates += this_candidates
            cluster_dict.pop(target_l)
        radius += 1 # ready to search the next length

    # process exclude same id
    candidates = (x for x in candidates if x.seq_id != q.seq_id) if exclude_same_id else candidates
    [x.fetch_and_set_data(data_normalized) for x in candidates]  # fetch data_original for the candidates]
    # print('# Sequences in the candidate list:: ' + str(len(candidates)))

    # start = time.time()
    # rtn = bsf_search(q, k, candidates)
    # duration_opt = time.time() - start
    # start = time.time()
    # rtn = naive_search(q, k, candidates, overlap, exclude_same_id)
    # duration_nonopt = time.time() - start

    if _lb_opt_cluster == 'bsf':
        # print('Using bsf')
        return bsf_search(q, k, candidates)
    elif _lb_opt_cluster == 'none':
        # print('Using naive')
        return naive_search(q, k, candidates, overlap, exclude_same_id)
    else:
        raise Exception('_query_partition: unsupported _lb_opt_cluster: ' + str(_lb_opt_cluster))


def naive_search(q: Sequence, k: int, candidates: list, overlap: float, exclude_same_id: bool):
    query_result = []
    c_dist_list = [(sim_between_seq(x, q), x) for x in candidates]
    heapq.heapify(c_dist_list)

    # note that we are using k here
    while len(c_dist_list) > 0 and len(query_result) < k:
        c_dist = heapq.heappop(c_dist_list)
        if overlap == 1.0 or not exclude_same_id:
            query_result.append(c_dist)
        else:
            if not any(_isOverlap(c_dist[1], prev_match[1], overlap) for prev_match in
                       query_result):  # check for overlap against all the matches so far
                query_result.append(c_dist)
    return query_result


def bsf_search(q, k, candidates):
    # use ranked heap
    # prune_count = 0
    query_result = list()
    # print('Num seq in the querying cluster: ' + str(len(querying_cluster)))
    for c in candidates:
        # print('Using bsf')
        if len(query_result) < k:
            # take the negative distance so to have a maxheap
            heapq.heappush(query_result, (-sim_between_seq(q, c), c))
        else:  # len(dist_heap) == k or >= k
            # if the new seq is better than the heap head
            if -lb_kim_sequence(c.data, q.data) < query_result[0][0]:
                # prune_count += 1
                continue
            # interpolate for keogh calculation
            # if len(c) != len(q):
            #     c_interp_data = np.interp(np.linspace(0, 1, len(q)),
            #                                       np.linspace(0, 1, len(c.data)), c.data)
            # else:
            #     c_interp_data = c.data
            # if -lb_keogh_sequence(c_interp_data, q.data) < query_result[0][0]:
            #     # prune_count += 1
            #     continue
            # if -lb_keogh_sequence(q.data, c_interp_data) < query_result[0][0]:
            #     # prune_count += 1
            #     continue
            dist = -sim_between_seq(q, c)
            if dist > query_result[0][0]:  # first index denotes the top of the heap, second gets the dist
                heapq.heappop(query_result)
                heapq.heappush(query_result, (dist, c))
    if (len(query_result)) >= k:
        # print(str(prune_count) + ' of ' + str(len(candidates)) + ' candidate(s) pruned')
        return [(-x[0], x[1]) for x in query_result]


def bsf_search_rspace(q, ke, r_list, cluster, st):
    """

    :param q:
    :param ke:
    :param r_list:
    :param cluster: dict, repr -> list of sequences representated
    :return:
    """
    # use ranked heap
    # prune_count = 0
    result_list = list()
    # print(r_list)
    for r in r_list:
        # print('Using bsf')
        if len(get_sequences_represented([r[1] for r in result_list],
                                         cluster)) < ke:  # keep track of how many sequences are we querying right now
            # take the negative distance so to have a maxheap
            heapq.heappush(result_list, (sim_between_seq(q, r), r))
        else:  # len(dist_heap) == k or >= k
            # a = lb_kim_sequence(r.data, q.data)
            if lb_kim_sequence(r.data, q.data) > st:
                # prune_count += 1
                continue
            # interpolate for keogh calculation
            # if len(r) != len(q):
            #     r_interp_data = np.interp(np.linspace(0, 1, len(q)),
            #                                       np.linspace(0, 1, len(r.data)), r.data)
            # else:
            #     r_interp_data = r.data
            # # b = lb_keogh_sequence(candidate_interp_data, q.data)
            # if lb_keogh_sequence(r_interp_data, q.data) > st:
            #     # prune_count += 1
            #     continue
            # # c = lb_keogh_sequence(q.data, candidate_interp_data)
            # if lb_keogh_sequence(q.data, r_interp_data) > st:
            #     # prune_count += 1
            #     continue
            dist = sim_between_seq(q, r)
            if dist < result_list[0][0]:  # first index denotes the top of the heap, second gets the dist
                heapq.heappop(result_list)
                heapq.heappush(result_list, (dist, r))
    # print(str(prune_count) + ' of ' + str(len(r_list)) + ' representative(s) pruned')
    return get_sequences_represented([r[1] for r in result_list], cluster)  # only return the representatives


def get_sequences_represented(reprs, cluster):
    """

    :param reprs: list of Sequences that are representatives
    :param cluster: repr -> list of sequences represented, mapping from representativs to their clusters
    """
    return flatten([cluster[r] for r in reprs]) if reprs is not None else list()


def flatten(l):
    return [item for sublist in l for item in sublist]


def _validate_gxdb_build_arguments(args: dict):
    """
    sanity check function for the arguments of build as a class method @ genex_databse object
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
    """
    sanity check function for the arguments of query as a class method @ genex_databse object
    :param args:
    :return:
    """

    _lb_opt_repr_options = ['lbh', 'bsf', 'none']
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
    df_list = [_row_to_feature_and_data(x, feature_num) for x in df.values.tolist()]
    return df_list


def _create_f_uuid_map(df, feature_num: int):
    f_uuid_dict = dict()
    for x in df.values.tolist():
        f_uuid_dict[tuple(x[1:feature_num])] = x[0]

    return f_uuid_dict


def _row_to_feature_and_data(row, feature_num):
    # list slicing syntax: ending at the key_num-th element but not include it
    # seq_id = tuple([(name, value) for name, value in zip(feature_head[:feature_num], row[:feature_num])])
    # seq_id = tuple([str(x) for x in row[:feature_num]])
    seq_id = tuple([str(x) for x in row[:feature_num]])
    try:
        data = [x for x in row[feature_num:] if not np.isnan(x)]
    except TypeError as te:
        raise Exception(
            'Genex: this may due to an incorrect feature_num, please check you data_original file for the number '
            'of features\n '
            + 'Exception: ' + str(te))
    return seq_id, data


def _process_loi(loi: slice):
    """
    Process the length of interest parameter to get the start and end index
    :param loi: length of interest parameter
    :return: start and end index
    """
    start = 1
    end = math.inf
    if loi is not None:
        if loi.start is not None:
            start = loi.start
        if loi.stop is not None:
            end = loi.stop

    assert start > 0
    return start, end




from functools import reduce
from itertools import groupby


def reduce_by_key(func, iterable):
    """Reduce by key.
    Equivalent to the Spark counterpart
    Inspired by http://stackoverflow.com/q/33648581/554319
    1. Sort by key
    2. Group by key yielding (key, grouper)
    3. For each pair yield (key, reduce(func, last element of each grouper))
    """
    get_first = lambda p: p[0]
    get_second = lambda p: p[1]
    # iterable.groupBy(._1).map(l => (l._1, l._2.map(._2).reduce(func)))
    return map(
        lambda l: (l[0], reduce(func, map(get_second, l[1]))),
        groupby(sorted(iterable, key=get_first), get_first)
    )


def merge_dict(dicts: list):
    merged_dict = dict()
    merged_len = 0
    for d in dicts:
        merged_len += len(d)
        merged_dict = {**merged_dict, **d}  # make sure there is no replacement of elements
    try:
        assert merged_len == len(merged_dict)
    except AssertionError as ae:
        print(str(ae))
        raise Exception('duplicate dict keys: dict item replaced!')
    return merged_dict


def normalize_num(num, global_max, global_min):
    return (num - global_min) / (global_max - global_min)


def genex_normalize(input_list, z_normalization=False):
    # perform z normalization
    if z_normalization:
        z_normalized_input_list = _z_normalize(input_list)
    else:
        print('Not using z-normalization')
    # get a flatten z normalized list so to obtain the global min and max
    flattened_list = flatten([x[1] for x in z_normalized_input_list])
    global_max = np.max(flattened_list)
    global_min = np.min(flattened_list)

    # perform Min-max normalization
    zmm_normalized_list = _min_max_normalize(z_normalized_input_list, global_max=global_max, global_min=global_min)

    normalized_array = np.asarray([x[1] for x in zmm_normalized_list])
    return zmm_normalized_list, global_max, global_min


def _z_normalize(input_list):
    z_normalized_list = [(x[0], (x[1] - np.mean(x[1])) / np.std(x[1])) for x in input_list]
    return z_normalized_list


def _min_max_normalize(input_list, global_max, global_min):
    mm_normalized_list = [(x[0], (x[1] - global_min) / (global_max - global_min)) for x in input_list]
    return mm_normalized_list


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
    rtn = dict()

    for ts in time_series:
        ts_id = ts[0]
        ts_data = ts[1]
        # we take min because min can be math.inf
        for i in range(start - 1, min(end, len(ts_data))):
            target_length = i + 1
            if target_length not in rtn.keys():
                rtn[target_length] = []
            rtn[target_length] += _get_sublist_as_sequences(data_list=ts_data, data_id=ts_id, length=i)
    return list(rtn.items())


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
            rtn += _get_sublist_as_sequences(data_list=ts_data, data_id=ts_id, length=i)
    return rtn


def _get_sublist_as_sequences(data_list, data_id, length):
    # if given length is greater than the size of the data_list itself, the
    # function returns an empty list
    rtn = []
    for i in range(0, len(data_list) - length):
        # if the second number in range() is less than 1, the iteration will not run
        # data_list[i:i+length]  # for debug purposes
        rtn.append(Sequence(start=i, end=i + length, seq_id=data_id, data=np.array(data_list[i:i + length + 1])))
    return rtn
