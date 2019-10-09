# import _ucrdtw
import csv
import random
# distance libraries
import time

from fastdtw import fastdtw
from scipy.spatial.distance import cityblock
from scipy.spatial.distance import minkowski
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import chebyshev

import math
import numpy as np
from tslearn import metrics

from genex.classes import Sequence
from .data_process import get_data


def sim_between_seq(seq1: Sequence, seq2: Sequence, dist_type: str):
    """
    calculate the similarity between sequence 1 and sequence 2 using DTW

    :param seq1:
    :param seq2:
    :return float: return the similarity between sequence 1 and sequence 2
    """
    try:
        assert seq1.data is not None and seq2.data is not None
    except AssertionError as ae:
        raise Exception('sim_between_seq: one or both of the given sequence do(es) not have their(its) data set!')
    if dist_type == 'eu':  #TODO trillion optimizaiton here
        dist = fastdtw(seq1.data, seq2.data, dist=euclidean)[0]
        # print("distance using fastdtw is " + dist)
        return dist  # fastdtw returns a tuple with the first item being the distance

    if dist_type == 'ma':
        return fastdtw(seq1.data, seq2.data, dist=cityblock)[0]  # 1st item is the shortest path
    if dist_type == 'mi':
        return fastdtw(seq1.data, seq2.data, dist=minkowski)[0]
    if dist_type == 'ch':
        return fastdtw(seq1.data, seq2.data, dist=chebyshev)[0]
    else:
        raise Exception("sim_between_seq: cluster: invalid distance type: " + dist_type)

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
    return lb_k_sim

def lb_kim_sequence(candidate_seq, query_sequence):
    lb_kim_sim = math.sqrt((candidate_seq[0] - query_sequence[0])**2 + (candidate_seq[-1] - query_sequence[-1])**2)

    return lb_kim_sim


def randomize(arr):
    """
    Apply the randomize in place algorithm to the given array
    Adopted from https://www.geeksforgeeks.org/shuffle-a-given-array-using-fisher-yates-shuffle-algorithm/
    :param array arr: the arr to randomly permute
    :return: the random;uy permuted array
    """
    if len(arr) == 0:
        return arr
    for i in range(len(arr) - 1, 0, -1):
        # Pick a random index from 0 to i
        j = random.randint(0, i)

        # Swap arr[i] with the element at random index
        arr[i], arr[j] = arr[j], arr[i]
    return arr


# def clusterer_legacy(groups, st):
#     """
#     construct similarity clusters
#     Look at clusters of all length, not using Distributed system
#
#     This is a Legacy function, not used anymore
#     :param dict groups: [key = length, value = array of sebsequences of the length]
#         For example:
#         [[1,4,2],[6,1,4],[1,2,3],[3,2,1]] is a valid 'subsequences'
#     :param float st: similarity threshold
#     :return: dict clusters: [key = representatives, value = similarity cluster: array of sebsequences]
#     """
#     clusters = []
#     for group_len in groups.keys():
#
#         processing_groups = groups[group_len]
#         processing_groups = randomize(
#             processing_groups)  # randomize the sequence in the group to remove clusters-related bias
#
#         for sequence in processing_groups:  # the subsequence that is going to form or be put in a similarity clustyer
#             if not clusters.keys():  # if there is no item in the similarity clusters
#                 clusters[sequence] = [sequence]  # put the first sequence as the representative of the first cluster
#             else:
#                 minSim = math.inf
#                 minRprst = None
#
#                 for rprst in clusters.keys():  # iterate though all the similarity groups, rprst = representative
#                     dist = sim_between_seq(sequence, rprst)
#                     if dist < minSim:
#                         minSim = dist
#                         minRprst = rprst
#
#                 if minSim <= math.sqrt(group_len) * st / 2:  # if the calculated min similarity is smaller than the
#                     # similarity threshold, put subsequence in the similarity cluster keyed by the min representative
#                     clusters[minRprst].append(sequence)
#                 else:  # if the minSim is greater than the similarity threshold, we create a new similarity group
#                     # with this sequence being its representative
#                     if sequence in clusters.keys():
#                         raise Exception('cluster_operations: clusterer_legacy: Trying to create new similarity cluster '
#                                         'due to exceeding similarity threshold, target sequence is already a '
#                                         'representative(key) in clusters. The sequence isz: ' + str(sequence))
#                     clusters[sequence] = [sequence]


# def cluster_two_pass(group, length, st, normalized_ts_dict, dist_type='eu'):
#     """
#     this is an alternative for the regular clustering algorithm.
#     It does two passes in generating clusters. More refined cluster will result from it
#
#     when makeing new representatives (clusters), it checks if the new representative's similarity is greater than
#     twice the similarity threshold (normal cap for creating new representative is just the 1*similarity threshold)
#     :param group:
#     :param length:
#     :param st:
#     :param normalized_ts_dict:
#     :param dist_type:
#     """
#     cluster = dict()
#
#     # get all seubsequences from ts_dict
#     # at one time
#     # ???or maybe during group operation
#     # During group operation is better, because the clusters will be too large if
#     # we retrieve all of it
#
#     ssequences = []
#
#     # waiting list for the sequences that are not close enough to be put into existing clusters, but also not far enough to be their own clusters
#     waiting_ssequences = []
#
#     for g in group:
#         tid = g[0]
#         start_point = g[1]
#         end_point = g[2]
#         ssequences.append(TimeSeriesObj(tid, start_point, end_point))
#
#     print("Clustering length of: " + str(length) + ", number of subsequences is " + str(len(ssequences)))
#
#     # group length validation
#     for time_series in ssequences:
#         # end_point and start_point
#         if time_series.end_point - time_series.start_point != length:
#             raise Exception("cluster_operations: clusterer: group length dismatch, len = " + str(length))
#
#     # randomize the sequence in the group to remove clusters-related bias
#     ssequences = randomize(ssequences)
#
#     delimiter = '_'
#     cluster_count = 0
#     sim = math.sqrt(length) * st / 2
#
#     # first pass
#     for ss in ssequences:
#         if not cluster.keys():
#             # if there is no item in the similarity cluster
#             # future delimiter
#             group_id = str(length) + delimiter + str(cluster_count)
#             ss.set_group_represented(group_id)
#             cluster[ss] = [ss]
#             ss.set_representative()
#             cluster_count += 1
#             # put the first sequence as the representative of the first cluster
#         else:
#             minSim = math.inf
#             minRprst = None
#             # rprst is a time_series obj
#             for rprst in list(cluster.keys()):  # iterate though all the similarity clusters, rprst = representative
#                 # ss is also a time_series obj
#                 ss_raw_data = get_data(ss.id, ss.start_point, ss.end_point, normalized_ts_dict)
#                 rprst_raw_data = get_data(rprst.id, rprst.start_point, rprst.end_point, normalized_ts_dict)
#
#                 # check the distance type
#                 if dist_type == 'eu':
#                     dist = euclidean(np.asarray(ss_raw_data), np.asarray(rprst_raw_data))
#                 elif dist_type == 'ma':
#                     dist = cityblock(np.asarray(ss_raw_data), np.asarray(rprst_raw_data))
#                 elif dist_type == 'mi':
#                     dist = minkowski(np.asarray(ss_raw_data), np.asarray(rprst_raw_data))
#                 else:
#                     raise Exception("cluster_operations: cluster: invalid distance type: " + dist_type)
#
#                 # update the minimal similarity
#                 if dist < minSim:
#                     minSim = dist
#                     minRprst = rprst
#
#             if minSim <= sim:  # if the calculated min similarity is smaller than the
#                 # similarity threshold, put subsequence in the similarity cluster keyed by the min representative
#                 cluster[minRprst].append(ss)
#                 ss.set_group_represented(minRprst.get_group_represented())
#
#             # This is the key different between two-pass clustering and the previous clustering:
#             # We see if the distance is not far enough to be it's own cluster
#             elif minSim <= sim * 2:
#
#                 waiting_ssequences.append(ss)
#
#             else:
#                 if ss not in cluster.keys():
#                     cluster[ss] = [ss]
#                     group_id = str(length) + delimiter + str(cluster_count)
#                     ss.set_group_represented(group_id)
#                     ss.set_representative()
#                     cluster_count += 1
#
#     # second pass
#     for wss in waiting_ssequences:
#         if not cluster.keys():
#             raise Exception("cluster_operations.py: cluster_two_pass: no existing clusters, invalid second pass")
#         else:  # this is exact the same as the first pass, but we are not creating a wait list anymore
#             minSim = math.inf
#             minRprst = None
#             for rprst in list(cluster.keys()):
#                 wss_raw_data = get_data(wss.id, wss.start_point, wss.end_point, normalized_ts_dict)
#                 rprst_raw_data = get_data(rprst.id, rprst.start_point, rprst.end_point, normalized_ts_dict)
#
#                 # check the distance type
#                 if dist_type == 'eu':
#                     dist = euclidean(np.asarray(wss_raw_data), np.asarray(rprst_raw_data))
#                 elif dist_type == 'ma':
#                     dist = cityblock(np.asarray(wss_raw_data), np.asarray(rprst_raw_data))
#                 elif dist_type == 'mi':
#                     dist = minkowski(np.asarray(wss_raw_data), np.asarray(rprst_raw_data))
#                 elif dist_type == 'ch':
#                     dist = chebyshev(np.asarray(wss_raw_data), np.asarray(rprst_raw_data))
#
#                 else:
#                     raise Exception("cluster_operations: cluster: invalid distance type: " + dist_type)
#
#                 # update the minimal similarity
#                 if dist < minSim:
#                     minSim = dist
#                     minRprst = rprst
#
#             if minSim <= sim:  # if the calculated min similarity is smaller than the
#                 cluster[minRprst].append(wss)
#                 wss.set_group_represented(minRprst.get_group_represented())
#             else:
#                 if wss not in cluster.keys():
#                     cluster[wss] = [wss]
#                     group_id = str(length) + delimiter + str(cluster_count)
#                     wss.set_group_represented(group_id)
#                     wss.set_representative()
#                     cluster_count += 1
#     return cluster

from itertools import groupby


def _cluster_groups(groups: list, st: float, dist_type, log_level: int = 1,
                    del_data: bool = True) -> list:
    result = []
    for seq_len, grp in groups:
        result.append(cluster_with_filter(grp, st, seq_len, dist_type=dist_type))

    return result


def cluster_with_filter(group: list, st: float, sequence_len: int, log_level: int = 1, dist_type: str = 'eu',
                        del_data: bool = True) -> dict:
    """
    all subsequence in 'group' must be of the same length
    For example:
    [[1,4,2],[6,1,4],[1,2,3],[3,2,1]] is a valid 'sub-sequences'

    :param group: list of sebsequences of a specific length, entry = sequence object
    :param int length: the length of the group to be clustered
    :param float st: similarity threshold to determine whether a sub-sequence
    :param float global_min: used for minmax normalization
    :param float global_min: used for minmax normalization
    :param dist_type: distance types including eu = euclidean, ma = mahalanobis, mi = minkowski
    belongs to a group

    :return a dictionary of clusters
    """
    cluster = dict()
    length = sequence_len
    subsequences = group
    threshold = math.sqrt(length) * st / 2
    # print("Clustering length of: " + str(length) + ", number of subsequences is " + str(len(group[1])))

    # randomize the sequence in the group to remove clusters-related bias
    subsequences = randomize(subsequences)

    count = 0

    for ss in subsequences:
        if log_level == 1:
            print('Cluster length: ' + str(length) + ':   ' + str(count + 1) + '/' + str(len(group)) + ' Num clusters: ' + str(len(cluster)))
            count += 1

        if not cluster.keys():  # if there's no representatives, the first subsequence becomes a representative
            cluster[ss] = [ss]
        else:
            min_dist = math.inf
            min_rprst = None
            # rprst is a time_series obj
            for rprst in list(cluster.keys()):  # iterate though all the similarity clusters, rprst = representative
                # ss is also a time_series obj
                ss_raw_data = ss.get_data()
                rprst_raw_data = rprst.get_data()

                # check the distance type
                try:
                    if dist_type == 'eu':
                        dist = euclidean(np.asarray(ss_raw_data), np.asarray(rprst_raw_data))
                    elif dist_type == 'ma':
                        dist = cityblock(np.asarray(ss_raw_data), np.asarray(rprst_raw_data))
                    elif dist_type == 'mi':
                        dist = minkowski(np.asarray(ss_raw_data), np.asarray(rprst_raw_data))
                    elif dist_type == 'ch':
                        dist = chebyshev(np.asarray(ss_raw_data), np.asarray(rprst_raw_data))
                    else:
                        raise Exception("cluster_operations: cluster: invalid distance type: " + dist_type)
                except TypeError as te:
                    print('first: ' + str(ss_raw_data))
                    print('type of the first: ' + str(type(ss_raw_data[0])))
                    print('second:' + str(rprst_raw_data))
                    print(te)
                    raise TypeError
                # update the minimal similarity
                if dist < min_dist:
                    min_dist = dist
                    min_rprst = rprst

            if min_dist <= threshold:  # if the calculated min similarity is smaller than the
                # similarity threshold, put subsequence in the similarity cluster keyed by the min representative
                cluster[min_rprst].append(ss)

            else:
                # if the minSim is greater than the similarity threshold, we create a new similarity group
                # with this sequence being its representative
                if ss not in cluster.keys():
                    cluster[ss] = [ss]
    print()
    print('Cluster length: ' + str(length) + '   Done!----------------------------------------------')

    if del_data:
        for value in cluster.values():
            for ss in value:
                ss.del_data()

    return length, cluster


def _cluster(group: list, st: float, log_level: int = 1, dist_type: str = 'eu', del_data: bool = True) -> dict:
    """
    all subsequence in 'group' must be of the same length
    For example:
    [[1,4,2],[6,1,4],[1,2,3],[3,2,1]] is a valid 'sub-sequences'

    :param group: list of sebsequences of a specific length, entry = sequence object
    :param int length: the length of the group to be clustered
    :param float st: similarity threshold to determine whether a sub-sequence
    :param float global_min: used for minmax normalization
    :param float global_min: used for minmax normalization
    :param dist_type: distance types including eu = euclidean, ma = mahalanobis, mi = minkowski
    belongs to a group

    :return a dictionary of clusters
    """
    cluster = dict()
    length = group[0]
    subsequences = group[1]

    # print("Clustering length of: " + str(length) + ", number of subsequences is " + str(len(group[1])))

    # randomize the sequence in the group to remove clusters-related bias
    subsequences = randomize(subsequences)

    delimiter = '_'

    count = 0

    for ss in subsequences:
        if log_level == 1:
            print('Cluster length: ' + str(length) + ':   ' + str(count + 1) + '/' + str(len(group[1])))
            count += 1

        if not cluster.keys():
            cluster[ss] = [ss]
        else:
            minSim = math.inf
            minRprst = None
            # rprst is a time_series obj
            for rprst in list(cluster.keys()):  # iterate though all the similarity clusters, rprst = representative
                # ss is also a time_series obj
                ss_raw_data = ss.get_data()
                rprst_raw_data = rprst.get_data()

                # check the distance type
                if dist_type == 'eu':
                    dist = euclidean(np.asarray(ss_raw_data), np.asarray(rprst_raw_data))
                elif dist_type == 'ma':
                    dist = cityblock(np.asarray(ss_raw_data), np.asarray(rprst_raw_data))
                elif dist_type == 'mi':
                    dist = minkowski(np.asarray(ss_raw_data), np.asarray(rprst_raw_data))
                elif dist_type == 'ch':
                    dist = chebyshev(np.asarray(ss_raw_data), np.asarray(rprst_raw_data))
                else:
                    raise Exception("cluster_operations: cluster: invalid distance type: " + dist_type)

                # update the minimal similarity
                if dist < minSim:
                    minSim = dist
                    minRprst = rprst
            sim = math.sqrt(length) * st / 2
            if minSim <= sim:  # if the calculated min similarity is smaller than the
                # similarity threshold, put subsequence in the similarity cluster keyed by the min representative
                cluster[minRprst].append(ss)

            else:
                # if the minSim is greater than the similarity threshold, we create a new similarity group
                # with this sequence being its representative
                # if ss in cluster.keys():
                #     # should skip
                #     continue
                #     # raise Exception('cluster_operations: clusterer: Trying to create new similarity cluster '
                #     #                 'due to exceeding similarity threshold, target sequence is already a '
                #     #                 'representative(key) in cluster. The sequence isz: ' + ss.toString())

                if ss not in cluster.keys():
                    cluster[ss] = [ss]

    print()
    print('Cluster length: ' + str(length) + '   Done!')

    if del_data:
        for value in cluster.values():
            for ss in value:
                ss.del_data()

    return length, cluster
