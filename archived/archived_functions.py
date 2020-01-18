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
            # print('Cluster length: ' + str(length) + ':   ' + str(count + 1) + '/' + str(len(group[1])))
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



# while len(target_reprs) > 0:
#         this_repr = heapq.heappop(target_reprs)
#         querying_cluster_seq = target_cluster[this_repr[1]]
#
#         # filter by id
#         if exclude_same_id:
#             querying_cluster_seq = (x for x in querying_cluster_seq if x.id != q.id)
#             if len(querying_cluster_seq) == 0:  # if after filtering, there's no sequence left, then simply continue
#                 # to the next iteration
#                 continue
#
#         # fetch data_original for the target cluster
#         [x.fetch_and_set_data(data_normalized) for x in querying_cluster_seq]
#
#         if (_lb_opt_cluster == 'lbh' or _lb_opt_cluster == 'lbh_bsf') and \
#                 len(querying_cluster_seq) > (k / cluster_keogh_rf) / cluster_kim_rf:
#             querying_cluster_seq = prune_by_lbh(seq_list=querying_cluster_seq, seq_length=target_length, q=q,
#                                             kim_reduction=cluster_kim_rf, keogh_reduction=cluster_keogh_rf)
#
#         if _lb_opt_cluster == 'bsf' or _lb_opt_cluster == 'lbh_bsf':
#             # use ranked heap
#             query_result = list()
#             # print('Num seq in the querying cluster: ' + str(len(querying_cluster)))
#             for candidate in querying_cluster_seq:
#                 # print('Using bsf')
#                 if len(query_result) < k:
#                     # take the negative distance so to have a maxheap
#                     heapq.heappush(query_result, (-sim_between_seq(q, candidate, dist_type), candidate))
#                 else:  # len(dist_heap) == k or >= k
#                     # if the new seq is better than the heap head
#                     # a = -lb_kim_sequence(candidate.data_original, q.data_original)
#                     if -lb_kim_sequence(candidate.data_original, q.data_original) < query_result[0][0]:
#                         prune_count += 1
#                         continue
#                     # interpolate for keogh calculaton
#                     if target_length != q_length:
#                         candidate_interp_data = np.interp(np.linspace(0, 1, q_length),
#                                                           np.linspace(0, 1, len(candidate.data_original)), candidate.data_original)
#                     # b = -lb_keogh_sequence(candidate_interp_data, q.data_original)
#                     if -lb_keogh_sequence(candidate_interp_data, q.data_original) < query_result[0][0]:
#                         prune_count += 1
#                         continue
#                     # c = -lb_keogh_sequence(q.data_original, candidate_interp_data)
#                     if -lb_keogh_sequence(q.data_original, candidate_interp_data) < query_result[0][0]:
#                         prune_count += 1
#                         continue
#                     dist = -sim_between_seq(q, candidate, dist_type)
#                     if dist > query_result[0][0]:  # first index denotes the top of the heap, second gets the dist
#                         heapq.heappop(query_result)
#                         heapq.heappush(query_result, (dist, candidate))
#             if (len(query_result)) >= k:
#                 print('Found k matches, returning query result')
#                 print(str(prune_count) + ' of ' + str(len(querying_cluster_seq)) + ' pruned')
#                 return [(-x[0], x[1]) for x in query_result]
#         else:
#             # print('Not using bsf')
#             # calculate the distances between the query and all the sequences in the cluster
#             dist_seq_list = [(sim_between_seq(x, q, dist_type), x) for x in querying_cluster_seq]
#             heapq.heapify(dist_seq_list)
#
#             while len(dist_seq_list) > 0:
#                 c_dist = heapq.heappop(dist_seq_list)
#                 if overlap == 1.0 or exclude_same_id:
#                     print('Adding to querying result')
#                     query_result.append(c_dist)
#                 else:
#                     if not any(_isOverlap(c_dist[1], prev_match[1], overlap) for prev_match in
#                                query_result):  # check for overlap against all the matches so far
#                         print('Adding to querying result')
#                         query_result.append(c_dist)
#
#                 if (len(query_result)) >= k:
#                     print('Found k matches, returning query result')
#                     return query_result
#     cluster_dict.pop(target_length)  # remove this len-cluster just queried
#
#     # find the next closest sequence length
#     if len(cluster_dict) != 0:
#         target_length = min(list(cluster_dict.keys()), key=lambda x: abs(x - target_length))
#     else:
#         break
# # print(str(prune_count) + ' sequences pruned')
# return query_result


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
#     querying_cluster = [(x[0], x[1], lb_kim_sequence(x[1], q.data_original)) for x in querying_cluster]
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
#     querying_cluster = [(x[0], x[1], lb_keogh_sequence(x[1], q.data_original)) for x in
#                         querying_cluster]  # now entries are (seq, data_original, lb_heuristic)
#     querying_cluster.sort(key=lambda x: x[2])  # sort by the lb_heuristic
#     # checking how much we reduce the cluster
#     if type(reduction_factor_lbkeogh) == str:
#         querying_cluster = querying_cluster[:int(len(querying_cluster) * lbkeogh_mult_factor)]
#     elif type(reduction_factor_lbkim) == int:
#         querying_cluster = querying_cluster[:reduction_factor_lbkeogh * k]
#     else:
#         raise Exception('Type of reduction factor must be str or int')
#
#     querying_cluster_reduced = [(sim_between_seq(x[1], q.data_original, dist_type=dist_type), x[0]) for x in
#                                 querying_cluster]  # now entries are (dist, seq)
