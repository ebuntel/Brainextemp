import heapq
import math

import numpy as np
from dtw import dtw
from fastdtw import fastdtw
from pyspark.broadcast import Broadcast

from genex.classes.Sequence import Sequence
from genex.misc import merge_dict
from genex.utils.ts_utils import lb_kim_sequence, lb_keogh_sequence
from genex.utils.utils import get_trgt_len_within_r, get_sequences_represented, _isOverlap, reduce_by_key


def sim_between_seq(seq1: Sequence, seq2: Sequence, pnorm: int):
    """
    calculate the similarity between sequence 1 and sequence 2 using DTW

    :param pnorm: the distance type that can be: 0, 1, or 2
    :param seq1: Time series sequence
    :param seq2: Time series sequence
    :return float: return the Normalized DTW distance between sequence 1 (seq1) and sequence 2 (seq2)
    """
    # dt_pnorm_dict = {'eu': 0,
    #                    'ma': 1,
    #                    'ch': 2,
    #                    'min': 2}
    dist = fastdtw(seq1.get_data(), seq2.get_data(), dist=pnorm)[0]
    if pnorm == 2:
        return np.sqrt(dist / (len(seq1) + len(seq2)))
    elif pnorm == 1:
        return dist / (len(seq1) + len(seq2))
    elif pnorm == math.inf:
        return dist
    else:
        raise Exception('Unsupported dist type in sim_between_seq, this should never happen!')


def sim_between_array(a1: np.ndarray, a2: np.ndarray, pnorm: int):
    """
    calculate the similarity between sequence 1 and sequence 2 using DTW

    :param a1:
    :param a2:
    :param pnorm: the distance type that can be: 0, 1, or 2
    :return float: return the Normalized DTW distance between sequence 1 (seq1) and sequence 2 (seq2)
    """
    # dt_pnorm_dict = {'eu': 0,
    #                    'ma': 1,
    #                    'ch': 2,
    #                    'min': 2}

    dist = fastdtw(a1, a2, dist=pnorm)[0]
    if pnorm == 2:
        return np.sqrt(dist / (len(a1) + len(a2)))
    elif pnorm == 1:
        return dist / (len(a1) + len(a2))
    elif pnorm == math.inf:
        return dist
    else:
        raise Exception('Unsupported dist type in sim_between_seq, this should never happen!')

def get_dist_query(query, target, dt_index):
    return sim_between_seq(query, target, pnorm=dt_index), target


def _query_partition(cluster, q, k: int, ke: int, data_normalized, loi: slice, dt_index: int,
                     _lb_opt_cluster: str, _lb_opt_repr: str,
                     overlap: float, exclude_same_id: bool, radius: int, st: float):
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
    if isinstance(q, Broadcast):
        q = q.value
    if isinstance(data_normalized, Broadcast):
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
                this_candidates = \
                    bsf_search_rspace(q, k, r_list=target_reprs, cluster=target_cluster, st=st, dt_index=dt_index)
            else:
                this_candidates = \
                    naive_search_rspace(q, k, r_list=target_reprs, cluster=target_cluster, dt_index=dt_index)

            candidates += this_candidates
            cluster_dict.pop(target_l)
        radius += 1  # ready to search the next length

    # process exclude same id
    candidates = [x for x in candidates if x.seq_id != q.seq_id] if exclude_same_id else candidates
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
        return bsf_search(q, k, candidates, dt_index=dt_index)
    elif _lb_opt_cluster == 'none':
        # print('Using naive')
        return naive_search(q, k, candidates, overlap, exclude_same_id, dt_index=dt_index)
    else:
        raise Exception('_query_partition: unsupported _lb_opt_cluster: ' + str(_lb_opt_cluster))


def naive_search_rspace(q, k, r_list, cluster, dt_index):
    c_list = []
    target_reprs = [(sim_between_seq(x, q, dt_index), x) for x in r_list]  # calculate DTW
    heapq.heapify(target_reprs)  # heap sort R-space
    # get enough sequence from the clusters represented to query
    while len(target_reprs) > 0 and len(c_list) < k:
        this_repr = heapq.heappop(target_reprs)[
            1]  # take the second element for the first one is the DTW dist
        c_list += (cluster[this_repr])
    return c_list


def naive_search(q: Sequence, k: int, candidates: list, overlap: float, exclude_same_id: bool, dt_index: int):
    query_result = []
    c_dist_list = [(sim_between_seq(x, q, dt_index), x) for x in candidates]
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


def bsf_search(q, k, candidates, dt_index: int):
    # use ranked heap
    # prune_count = 0
    query_result = list()
    # print('Num seq in the querying cluster: ' + str(len(querying_cluster)))
    for c in candidates:
        # print('Using bsf')
        if len(query_result) < k:
            # take the negative distance so to have a maxheap
            heapq.heappush(query_result, (-sim_between_seq(q, c, dt_index), c))
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
            dist = -sim_between_seq(q, c, dt_index)
            if dist > query_result[0][0]:  # first index denotes the top of the heap, second gets the dist
                heapq.heappop(query_result)
                heapq.heappush(query_result, (dist, c))
    if (len(query_result)) >= k:
        # print(str(prune_count) + ' of ' + str(len(candidates)) + ' candidate(s) pruned')
        return [(-x[0], x[1]) for x in query_result]


def bsf_search_rspace(q, ke, r_list, cluster, st, dt_index: int):
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
            heapq.heappush(result_list, (sim_between_seq(q, r, dt_index), r))
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
            dist = sim_between_seq(q, r, dt_index)
            if dist < result_list[0][0]:  # first index denotes the top of the heap, second gets the dist
                heapq.heappop(result_list)
                heapq.heappush(result_list, (dist, r))
    # print(str(prune_count) + ' of ' + str(len(r_list)) + ' representative(s) pruned')
    return get_sequences_represented([r[1] for r in result_list], cluster)  # only return the representatives


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