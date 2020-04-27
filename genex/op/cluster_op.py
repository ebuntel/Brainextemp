# import _ucrdtw
import random
import numpy as np
# distance libraries

import math

from genex.classes.Sequence import Sequence
from genex.utils.ts_utils import lb_kim_sequence


def _randomize(arr, seed=42):
    """
    Apply the randomize in place algorithm to the given array
    Adopted from https://www.geeksforgeeks.org/shuffle-a-given-array-using-fisher-yates-shuffle-algorithm/
    :param seed:
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


def _build_clusters_dynamic(groups: list, st: float, dist_func, data_list, log_level: int, pnorm) -> list:
    """
    the dynamic programming implementation of the clustering algorithm
    :param groups:
    :param st:
    :param dist_func:
    :param data_list:
    :param log_level:
    """
    group_dict = dict(groups)
    subseq_lengths = list(group_dict.keys())
    subseq_lengths.reverse()  # use reversed to start from longer
    clusters = {}  # seq_len -> center_seq -> list of represented seq
    for length in subseq_lengths:
        group_target = group_dict[length]
        if length + 1 not in clusters.keys():  # check if there is a cluster of current length + 1
            cl, c = cluster_group_dist(group_target, st, length, dist_func=dist_func, data_list=data_list)
            clusters[cl] = c
        else:  # if the cluster of length 1+n exists, we use it as heuristics
            reprs_up = np.array(list(clusters[length + 1].keys()), dtype=Sequence)
            reprs_head_off = np.array([x.S(slice(1, None, None)) for x in reprs_up], dtype=Sequence)
            reprs_tail_off = np.array([x.S(slice(None, -1, None)) for x in reprs_up], dtype=Sequence)
            # TODO use the representative dist mt from length +1 to save calculation
            stay_mask_head = coalease_repr(reprs_head_off, diameter=st, dist_func=dist_func, data_list=data_list)
            stay_mask_tail = coalease_repr(reprs_tail_off, diameter=st, dist_func=dist_func, data_list=data_list)
            # we keep the larger list of representatives
            reprs_preformed, stay_mask = (reprs_head_off, stay_mask_head) \
                if np.sum(stay_mask_head) > np.sum(stay_mask_tail) else (reprs_tail_off, stay_mask_tail)
            reprs_preformed = reprs_preformed[stay_mask]
            # add to the new cluster
            c = {}
            # now onto validating the represented sequences
            # r is the centers from the masked (preformed) centers' list
            for r_up, r in zip(reprs_up[stay_mask], reprs_preformed):  # repr_up is the representative from length+1 clusters
                c[r] = []
                for dist, seq_up in clusters[length + 1][r_up]:
                    seq_head_off = seq_up.S(slice(1, None, None))
                    seq_tail_off = seq_up.S(slice(None, -1, None))

                    if check_still_in(r, seq_head_off, st, pnorm):
                        group_target.remove(seq_head_off)
                    if check_still_in(r, seq_tail_off, st, pnorm):  # same deal for tailOff
                        group_target.remove(seq_tail_off)

    pass


def check_still_in():
    return True

def coalease_repr(seqs: list, diameter, dist_func, data_list):
    """
    calculate the distances between the seqs and return a mask of that says which sequences should be ruled out
    are all greater than the half the diameter
    the current implementation rule out sequences in a row-wise fashion, in the future, this should be improved to
    rule out the sequences with the most negative values, thus to achieve more remaining sequences
    :param diameter:
    :param seqs:
    """
    dist_mt = np.full((len(seqs), len(seqs)), math.inf)
    for i in range(len(seqs)):  # TODO use DP to derive this matrix from length + 1 iteration
        for j in range(i + 1):
            dist_mt[i, j] = dist_func(seqs[j].fetch_data(data_list), seqs[i].fetch_data(data_list)) if i != j else math.inf
    stay_mask = dist_mt > diameter / 2.0
    stay_mask = np.all(stay_mask, axis=1)
    return stay_mask


def _build_clusters(groups: list, st: float, dist_func, data_list, log_level: int = 1) -> list:
    result = []
    for seq_len, grp in groups:
        result.append(cluster_group(grp, st, seq_len, dist_func=dist_func, data_list=data_list))
    return result


def cluster_group(group: list, st: float, sequence_len: int, dist_func, data_list, log_level: int = 1):
    """
    all subsequence in 'group' must be of the same length
    For example:
    [[1,4,2],[6,1,4],[1,2,3],[3,2,1]] is a valid 'sub-sequences'

    :param data_list:
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
    group = _randomize(group)

    for s in group:
        if not cluster.keys():  # if there's no representatives, the first subsequence becomes a representative
            cluster[s] = [(s)]
        else:
            # find the closest representative
            min_dist = math.inf
            min_representative = None

            for r in list(cluster.keys()):
                r_data = r.fetch_data(data_list)
                s_data = s.fetch_data(data_list)
                if lb_kim_sequence(r_data, s_data) > min_dist:  # compute the lb_kim
                    continue
                dist = dist_func(r_data, s_data)
                if dist < min_dist:
                    min_dist = dist
                    min_representative = r

            if min_dist <= st / 2.0:  # if the calculated min similarity is smaller than the
                # similarity threshold, put subsequence in the similarity cluster keyed by the min representative
                cluster[min_representative].append(s)

            else:
                # if the minSim is greater than the similarity threshold, we create a new similarity group
                # with this sequence being its representative
                if s not in cluster.keys():
                    cluster[s] = [s]
    # print('Cluster length: ' + str(sequence_len) + '   Done!----------------------------------------------')
    return sequence_len, cluster

def cluster_group_dist(group: list, st: float, sequence_len: int, dist_func, data_list, log_level: int = 1):
    """
    all subsequence in 'group' must be of the same length
    For example:
    [[1,4,2],[6,1,4],[1,2,3],[3,2,1]] is a valid 'sub-sequences'

    :param data_list:
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
    group = _randomize(group)

    for s in group:
        if not cluster.keys():  # if there's no representatives, the first subsequence becomes a representative
            cluster[s] = [(0.0, s)]  # put the distances and the sequence
        else:
            # find the closest representative
            min_dist = math.inf
            min_representative = None

            for r in list(cluster.keys()):
                r_data = r.fetch_data(data_list)
                s_data = s.fetch_data(data_list)
                if lb_kim_sequence(r_data, s_data) > min_dist:  # compute the lb_kim
                    continue
                dist = dist_func(r_data, s_data)
                if dist < min_dist:
                    min_dist = dist
                    min_representative = r

            if min_dist <= st / 2.0:  # if the calculated min similarity is smaller than the
                # similarity threshold, put subsequence in the similarity cluster keyed by the min representative
                cluster[min_representative].append((min_dist, s))

            else:
                # if the minSim is greater than the similarity threshold, we create a new similarity group
                # with this sequence being its representative
                if s not in cluster.keys():
                    cluster[s] = [(0.0, s)]
    # print('Cluster length: ' + str(sequence_len) + '   Done!----------------------------------------------')
    return sequence_len, cluster


def _cluster_to_meta(cluster):
    return cluster[0], {rprs: len(slist) for (rprs, slist) in cluster[1].items()}


def _cluster_reduce_func(v1, v2):
    return {**v1, **v2}
