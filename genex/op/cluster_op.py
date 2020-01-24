# import _ucrdtw
import random
# distance libraries

import math


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
    group = _randomize(group)

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


def _cluster_to_meta(cluster):
    return cluster[0], {rprs: len(slist) for (rprs, slist) in cluster[1].items()}


def _cluster_reduce_func(v1, v2):
    return {**v1, **v2}
