# import _ucrdtw
import csv
import random
# distance libraries
import time

from dtw import dtw, accelerated_dtw
from fastdtw import fastdtw
from scipy.spatial.distance import cityblock
from scipy.spatial.distance import minkowski
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import chebyshev

import math
import numpy as np
from tslearn import metrics

from genex.classes.Sequence import Sequence


def sim_between_seq(seq1: Sequence, seq2: Sequence):
    """
    calculate the similarity between sequence 1 and sequence 2 using DTW

    :param dist_type: the distance type that can be: 'eu', 'ma', 'mi', 'ch'
    :param seq1: Time series sequence
    :param seq2: Time series sequence
    :return float: return the Normalized DTW distance between sequence 1 (seq1) and sequence 2 (seq2)
    """
    dist = fastdtw(seq1.get_data(), seq2.get_data(), dist=lambda x, y: np.abs(x-y))[0] / (len(seq1) + len(seq2))

    return dist


def lb_keogh_sequence(seq_matching, seq_enveloped):
    """
    calculate lb keogh lower bound between query and sequence with envelope around query
    :param seq_matching:
    :param seq_enveloped:
    :return: lb keogh lower bound distance between query and sequence
    """
    try:
        assert len(seq_matching) == len(seq_enveloped)
    except AssertionError as ae:
        raise Exception('cluster.lb_keogh_sequence: two sequences must be of equal length to calculate lb_keogh')
    envelope_down, envelope_up = metrics.lb_envelope(seq_enveloped, radius=1)
    lb_k_sim = metrics.lb_keogh(seq_matching,
                                envelope_candidate=(envelope_down, envelope_up))
    return lb_k_sim / len(seq_matching)  # normalize


def lb_kim_sequence(candidate_seq, query_sequence):
    """
    Calculate lb kim lower bound between candidate and query sequence
    :param candidate_seq:
    :param query_sequence:
    :return: lb kim lower bound distance between query and sequence
    """

    lb_kim_sim = math.sqrt((candidate_seq[0] - query_sequence[0])**2 + (candidate_seq[-1] - query_sequence[-1])**2)

    return lb_kim_sim / 2.0  # normalize


def randomize(arr, seed=42):
    """
    Apply the randomize in place algorithm to the given array
    Adopted from https://www.geeksforgeeks.org/shuffle-a-given-array-using-fisher-yates-shuffle-algorithm/
    :param array arr: the arr to randomly permute
    :return: the random;uy permuted array
    """
    random.seed(seed)
    if len(arr) == 0:
        return arr
    for i in range(len(arr) - 1, 0, -1):
        # Pick a random index from 0 to i
        j = random.randint(0, i)

        # Swap arr[i] with the element at random index
        arr[i], arr[j] = arr[j], arr[i]
    return arr


def _cluster_groups(groups: list, st: float, dist_func, log_level: int = 1,
                    del_data: bool = True) -> list:
    result = []
    for seq_len, grp in groups:
        result.append(cluster_with_filter(grp, st, seq_len, dist_func=dist_func))

    return result


def cluster_with_filter(group: list, st: float, sequence_len: int, dist_func, log_level: int = 1,
                        del_data: bool = True) -> dict:
    """
    all subsequence in 'group' must be of the same length
    For example:
    [[1,4,2],[6,1,4],[1,2,3],[3,2,1]] is a valid 'sub-sequences'

    :param del_data:
    :param log_level:
    :param sequence_len:
    :param group: list of sebsequences of a specific length, entry = sequence object
    :param int length: the length of the group to be clustered
    :param float st: similarity threshold to determine whether a sub-sequence
    :param float global_min: used for minmax normalization
    :param float global_min: used for minmax normalization
    :param dist_func: distance types including eu = euclidean, ma = mahalanobis, mi = minkowski
    belongs to a group

    :return a dictionary of clusters
    """
    cluster = {}

    # randomize the sequence in the group to remove clusters-related bias
    group = randomize(group)

    for s in group:
        if not cluster.keys():  # if there's no representatives, the first subsequence becomes a representative
            cluster[s] = [s]
        else:
            # find the closest representative
            min_dist = math.inf
            min_representative = None

            for r in list(cluster.keys()):
                dist = dist_func(r.data, s.data)
                if dist < min_dist:
                    min_dist = dist
                    min_representative = r
            # representatives = list(cluster.keys())  # keep a ordered list of representatives
            # dists = [dist_func(r.get_data(), s.get_data()) for r in representatives]  # use the vectorized dist func
            # min_dist = np.min(dists)
            # min_representative = representatives[np.argmin(dists)]

            if min_dist <= st / 2.0:  # if the calculated min similarity is smaller than the
                # similarity threshold, put subsequence in the similarity cluster keyed by the min representative
                cluster[min_representative].append(s)

            else:
                # if the minSim is greater than the similarity threshold, we create a new similarity group
                # with this sequence being its representative
                if s not in cluster.keys():
                    cluster[s] = [s]
    # print('Cluster length: ' + str(sequence_len) + '   Done!----------------------------------------------')

    if del_data:
        for value in cluster.values():
            for s in value:
                s.del_data()

    return sequence_len, cluster
