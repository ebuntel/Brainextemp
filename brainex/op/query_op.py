import heapq
import math

import numpy as np
from pyspark.broadcast import Broadcast

from brainex.classes.Sequence import Sequence
from brainex.misc import merge_dict, fd_workaround
from brainex.utils.ts_utils import lb_kim_sequence, lb_keogh_sequence, paa_compress, sax_compress
from brainex.utils.utils import get_trgt_len_within_r, get_sequences_represented, _isOverlap, reduce_by_key

try:
    from fastdtw import fastdtw
    from fastdtw import dtw
except ImportError:
    fd_workaround()


def sim_between_array(a1: np.ndarray, a2: np.ndarray, pnorm: int, use_fast=True):
    """
    calculate the similarity between sequence 1 and sequence 2 using DTW

    :param use_fast:
    :param paa: the number of segments for time series reduction
    :param a1:
    :param a2:
    :param pnorm: the distance type that can be: 0, 1, or 2
    :return float: return the Normalized DTW distance between sequence 1 (seq1) and sequence 2 (seq2)
    """
    # dt_pnorm_dict = {'eu': 0,
    #                    'ma': 1,
    #                    'ch': 2,
    #                    'min': 2}

    dist = fastdtw(a1, a2, dist=pnorm)[0] if use_fast else dtw(a1, a2, dist=pnorm)[0]
    if pnorm == 2:
        return np.sqrt(dist / (len(a1) + len(a2)))
    elif pnorm == 1:
        return dist / (len(a1) + len(a2))
    elif pnorm == math.inf:
        return dist
    else:
        raise Exception('Unsupported dist type in array, this should never happen!')


def sim_between_array_piecewise(a1: np.ndarray, a2: np.ndarray, pnorm: int, piecewise: str, n_segment, use_fast=True):
    """
    calculate the similarity between sequence 1 and sequence 2 using DTW

    :param n_segment:
    :param piecewise:
    :param use_fast:
    :param paa: the number of segments for time series reduction
    :param a1:
    :param a2:
    :param pnorm: the distance type that can be: 0, 1, or 2
    :return float: return the Normalized DTW distance between sequence 1 (seq1) and sequence 2 (seq2)
    """
    if piecewise == 'paa':
        a1 = paa_compress(a1, n_segment)
        a2 = paa_compress(a2, n_segment)
    elif piecewise == 'sax':
        a1 = sax_compress(a1, n_segment)
        a2 = sax_compress(a2, n_segment)

    dist = fastdtw(a1, a2, dist=pnorm)[0] if use_fast else dtw(a1, a2, dist=pnorm)[0]
    if pnorm == 2:
        return np.sqrt(dist / (len(a1) + len(a2)))
    elif pnorm == 1:
        return dist / (len(a1) + len(a2))
    elif pnorm == math.inf:
        return dist
    else:
        raise Exception('Unsupported dist type in array, this should never happen!')


def _get_dist_sequence(seq1: Sequence, seq2: Sequence, dt_index, data_list):
    """
    the use of paa
    :param seq1:
    :param seq2:
    :param dt_index:
    :param paa:
    :param data_list:
    :return:
    """
    return sim_between_array(seq1.get_data(), seq2.fetch_data(data_list), pnorm=dt_index), seq2


def _get_dist_sequence_piecewise(query_com, candidate: Sequence, dt_index, data_list, piecewise, n_segment, fitter):
    """
    the use of paa
    :param seq1:
    :param candidate:
    :param dt_index:
    :param paa:
    :param data_list:
    :return:
    """
    if piecewise == 'paa':
        candidate_com, _ = paa_compress(a=candidate.fetch_data(data_list), paa_seg=n_segment)
    else:
        candidate_com, _ = sax_compress(a=candidate.fetch_data(data_list), sax_seg=n_segment, sax=fitter)
    return sim_between_array(query_com, candidate_com, dt_index), candidate
    # return sim_between_array_piecewise(seq1.get_data(), candidate.fetch_data(data_list), dt_index, piecewise, n_segment), candidate


def _get_dist_array(a1: np.ndarray, a2: np.ndarray, dt_index):
    # try:
    #     assert len(a2) >= 1 and len(a1) >= 1
    # except (AssertionError, TypeError):
    #     raise Exception('Invalid array length: ' + str(a2))
    return sim_between_array(a1, a2, pnorm=dt_index)


def _query_partition(cluster, q, k: int, ke: int, data_normalized, pnorm: int,
                     lb_opt: bool, exclude_same_id: bool, radius: int, st: float,
                     overlap: float, id_filter, filter_mode, loi,
                     prev_matches: list = []):
    """
    This function finds k best matches for given query sequence on the worker node

    :param cluster: cluster being queried
    :param q: Query sequence
    :param k: number of best matches to retrieve
    :param data_normalized:
    :param exclude_same_id: whether to exclude the query sequence when finding best matches
    :param overlap: Overlapping parameter( must be between 0 and 1 inclusive)

    :return: a list containing retrieved matches for given query sequence on that worker node
    """
    """We automatically use Traditional DTW if optimization is set to True"""

    if isinstance(q, Broadcast):
        q = q.value
    if isinstance(data_normalized, Broadcast):
        data_normalized = data_normalized.value

    cluster_dict = dict(list(reduce_by_key(lambda x, y: merge_dict([x, y]), cluster)))
    q_length = len(q.data)
    candidates = []
    if loi:  # filter by LOI
        cluster_dict = dict([(c_len, c) for c_len, c in cluster_dict.items() if loi[0] <= c_len <= loi[1]])

    # cluster_filtered = [x for x in cluster if x[0] in range(loi.start, loi.stop)] if loi is not None else cluster
    # cluster_dict = dict(list(reduce_by_key(lambda x, y: merge_dict([x, y]), cluster_filtered)))
    # try:
    #     assert cluster_filtered
    # except AssertionError as ve:
    #     print('Given loi is ' + str(loi))
    #     print('sequence length in the database: ' + str(list(cluster_dict.keys())))
    #     raise Exception('cluster does not have the given query loi!')

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
            r_data = [x.fetch_data(data_normalized) for x in
                      target_reprs]  # fetch data_original for the representatives

            # start = time.time()
            # this_candidates = get_sequences_represented(
            #     bsf_search_rspace(q, k, r_list=target_reprs, cluster=target_cluster, st=st),
            #     cluster=target_cluster)
            # duration_opt = time.time() - start
            # start = time.time()
            # this_candidates = naive_search_rspace(q, k, r_list=target_reprs, cluster=target_cluster)
            # duration_nonopt = time.time() - start

            if lb_opt:
                this_candidates = \
                    bsf_search_rspace(q, k, r_data, r_list=target_reprs, cluster=target_cluster, st=st, dt_index=pnorm)
            else:
                this_candidates = \
                    naive_search_rspace(q, k, r_data, r_list=target_reprs, cluster=target_cluster, dt_index=pnorm, overlap=overlap, prev_matches=prev_matches)
            candidates += this_candidates
            cluster_dict.pop(target_l)
        radius += 1  # ready to search the next length

    # process exclude same id
    candidates = [x for x in candidates if x.seq_id != q.seq_id] if exclude_same_id else candidates
    if id_filter:  # filter by seq id
        if filter_mode == 'any':
            candidates = [x for x in candidates if (check_id_any(x.seq_id, id_filter))]
        if filter_mode == 'all':
            candidates = [x for x in candidates if (check_id_all(x.seq_id, id_filter))]
    if len(candidates) == 0:
        return []

    # fetch data_original for the candidates
    c_data = [x.fetch_data(data_normalized) for x in candidates]
    # print('# Sequences in the candidate list:: ' + str(len(candidates)))

    if lb_opt == 'bsf':
        return bsf_search(q, k, c_data, candidates, dt_index=pnorm)
    else:
        return naive_search(q, k, c_data, candidates, dt_index=pnorm)


def check_id_any(ids1: tuple, ids2: tuple):
    """
    check if there are common elements in two id tuple
    :param ids1:
    :param ids2:
    :return:
    """
    return len(set(ids1).intersection(ids2)) > 0


def check_id_all(candidate_ids: tuple, filter_ids: tuple):
    """
    return true only when all the elements in filter_ids are present in candidate_ids
    :param candidate_ids:
    :param filter_ids:
    """
    return set(filter_ids).issubset(set(candidate_ids))

def naive_search_rspace(q, k, r_data, r_list, cluster, dt_index, overlap, prev_matches):
    c_list = []
    target_reprs = [(sim_between_array(rd, q.get_data(), dt_index), r) for rd, r in
                    zip(r_data, r_list)]  # calculate DTW
    heapq.heapify(target_reprs)  # heap sort R-space
    # get enough sequence from the clusters represented to query
    while len(target_reprs) > 0 and len(c_list) < k:
        this_repr = heapq.heappop(target_reprs)[
            1]  # take the second element for the first one is the DTW dist
        # filter by overlap
        target_cluster = cluster[this_repr] if overlap == 1.0 else [c for c in cluster[this_repr] if not any(_isOverlap(c, pv_c[1], overlap) for pv_c in prev_matches)]
        c_list += (target_cluster)
    return c_list


def naive_search(q: Sequence, k: int, c_data, candidates: list, dt_index: int):
    query_result = []
    c_dist_list = [(sim_between_array(cd, q.get_data(), dt_index), c) for cd, c in zip(c_data, candidates)]
    heapq.heapify(c_dist_list)

    # k = k if overlap == 1.0 or exclude_same_id else int((1000 * k) / (1 - overlap))
    # print(k)
    # note that we are using k here
    while len(c_dist_list) > 0 and len(query_result) < k:
        query_result.append(heapq.heappop(c_dist_list))

    return query_result


def bsf_search(q, k, c_data, candidates, dt_index: int):
    # use ranked heap
    # prune_count = 0
    query_result = list()
    # print('Num seq in the querying cluster: ' + str(len(querying_cluster)))
    for cd, c in zip(c_data, candidates):
        # print('Using bsf')
        if len(query_result) < k:
            # take the negative distance so to have a maxheap
            heapq.heappush(query_result, (-sim_between_array(q.get_data(), cd, dt_index, use_fast=True), c))
        else:  # len(dist_heap) == k or >= k
            # if the new seq is better than the heap head
            if -lb_kim_sequence(cd, q.data) < query_result[0][0]:
                # prune_count += 1
                continue
            # interpolate for keogh calculation
            if len(c) != len(q):
                c_interp_data = np.interp(np.linspace(0, 1, len(q)),
                                          np.linspace(0, 1, len(cd)), cd)
            else:
                c_interp_data = cd
            if -lb_keogh_sequence(c_interp_data, q.data) < query_result[0][0]:
                # prune_count += 1
                continue
            if -lb_keogh_sequence(q.data, c_interp_data) < query_result[0][0]:
                # prune_count += 1
                continue
            dist = -sim_between_array(q.get_data(), cd, dt_index, use_fast=True)
            if dist > query_result[0][0]:  # first index denotes the top of the heap, second gets the dist
                heapq.heappop(query_result)
                heapq.heappush(query_result, (dist, c))
    if (len(query_result)) >= k:
        # print(str(prune_count) + ' of ' + str(len(candidates)) + ' candidate(s) pruned')
        return [(-x[0], x[1]) for x in query_result]


def bsf_search_rspace(q, ke, r_data, r_list, cluster, st, dt_index: int):
    """
    Not using fastDTW for the sake of optization
    :param q:
    :param ke:
    :param r_list:
    :param cluster: dict, repr -> list of sequences represented
    :return:
    """
    # use ranked heap
    # prune_count = 0
    result_list = list()
    # print(r_list)
    for rd, r in zip(r_data, r_list):
        # print('Using bsf')
        if len(get_sequences_represented([r[1] for r in result_list],
                                         cluster)) < ke:  # keep track of how many sequences are we querying right now
            # take the negative distance so to have a maxheap
            heapq.heappush(result_list, (sim_between_array(q.get_data(), rd, dt_index, use_fast=False), r))
        else:  # len(dist_heap) == k or >= k
            # a = lb_kim_sequence(r.data, q.data)
            if lb_kim_sequence(rd, q.data) > st:
                # prune_count += 1
                continue
            # interpolate for keogh calculation
            if len(r) != len(q):
                r_interp_data = np.interp(np.linspace(0, 1, len(q)),
                                          np.linspace(0, 1, len(rd)), rd)
            else:
                r_interp_data = rd
            # b = lb_keogh_sequence(candidate_interp_data, q.data)
            if lb_keogh_sequence(r_interp_data, q.data) > st:
                # prune_count += 1
                continue
            # c = lb_keogh_sequence(q.data, candidate_interp_data)
            if lb_keogh_sequence(q.data, r_interp_data) > st:
                # prune_count += 1
                continue
            dist = sim_between_array(q.get_data(), rd, dt_index, use_fast=False)
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
