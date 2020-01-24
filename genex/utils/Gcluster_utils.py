from genex.classes.Sequence import Sequence


# def merge_gclusters(gclusters):
#     # gclusters validation
#     try:
#         iterator = iter(gclusters)
#     except TypeError as te:
#         raise Exception('Given Gclusters is not iterable.')
#
#     try:
#         for gc in gclusters:
#             assert type(gc) is genex_database
#             assert gc.collected is True
#     except AssertionError as ae:
#         raise Exception('Object in the given list must all be Gclusters and have been collected.')
#
#     # validate if the clusters are from the same original data_original
#     try:
#         data_list = list(map(lambda x: x.data_original, gclusters))
#         norm_data_list = list(map(lambda x: x.norm_data, gclusters))
#         all_feature_lists = list(map(lambda x: x.feature_list, gclusters))
#         global_max_list = list(map(lambda x: x.global_max, gclusters))
#         global_min_list = list(map(lambda x: x.global_min, gclusters))
#         st_list = list(map(lambda x: x.st, gclusters))
#
#         assert all(x == data_list[0] for x in data_list) and \
#                all(x == norm_data_list[0] for x in norm_data_list) and \
#                all(x == all_feature_lists[0] for x in all_feature_lists) and \
#                all(x == global_max_list[0] for x in global_max_list) and \
#                all(x == global_min_list[0] for x in global_min_list) and \
#                all(x == st_list[0] for x in st_list)
#     except AssertionError as ae:
#         raise Exception('Gclusters to merge must be from the same original data_original and have the same Similarity Threshold')
#
#     # merge the clusters
#     merged_clusters = {}
#
#     for gc in gclusters:
#         merged_clusters.update(gc.clusters)
#
#     return genex_database(feature_list=all_feature_lists[0],
#                           data_original=data_list[0], data_normalized=norm_data_list[0], st=st_list[0],
#                           cluster_dict=merged_clusters, collected=True,
#                           # this two attribute are different based on is_collect set to true or false
#                           global_max=global_max_list[0], global_min=global_min_list[0])


# def cluster_count(gc: genex_database, sc:SparkContext=None, data_slices=32):
#     # try:
#     #     for gc in gc:
#     #         assert type(gc) is Gcluster
#     #         assert gc.collected is True
#     # except AssertionError as ae:
#     #     raise Exception('Object in the given list must all be Gclusters and have been collected.')
#
#     try:
#         assert type(gc) is genex_database
#     except AssertionError as ae:
#         raise Exception('Object is not Gcluster')
#
#     if sc:
#         clusters_rdd = sc.parallelize(gc.clusters.items(), numSlices=data_slices)
#         clusters_count = clusters_rdd.map(lambda x: (x[0], len(x[1]))).collect()
#         return clusters_count
#     else:
#         return {k: len(v) for k, v in gc.clusters.items()}
#
#
# def gquery(gcluster: genex_database, query_sequence: Sequence, sc: SparkContext,
#            loi=None, foi=None, k: int = 1, dist_type: str = 'eu', data_slices: int = 32,
#            ex_sameID: bool = False, overlap: float = 0.0):
#     # TODO update gquery so that it can utilize past query result to do new queries
#     # input validation
#     try:
#         query_sequence.fetch_data(input_list=gcluster.norm_data)
#     except KeyError as ke:
#         raise Exception('Given query sequence is not present in this Gcluster')
#
#     if overlap != 0.0:
#         try:
#             assert 0.0 <= overlap <= 1.0
#         except AssertionError as e:
#             raise Exception('gquery: overlap factor must be a float between 0.0 and 1.0')
#
#     if loi[0] <= 0:
#         raise Exception('gquery: first element of loi must be equal to or greater than 1')
#     if loi[0] >= loi[1]:
#         raise Exception('gquery: Start must be greater than end in the '
#                         'Length of Interest')
#
#     r_heap = gcluster._gfilter(size=loi, filter_features=foi)  # retrieve cluster sequences of interests
#     r_heap = list(r_heap.items())
#
#     bc_norm_data = sc.broadcast(
#         gcluster.norm_data)  # broadcast the normalized data_original so that the Sequence objects can find data_original faster
#     rheap_rdd = sc.parallelize(r_heap, numSlices=data_slices)
#     rheap_rdd = rheap_rdd.flatMap(lambda x: x[1])  # retrieve all the sequences and flatten
#
#     # calculate the distances, create a key-value pair: key = dist from query to the sequence, value = the sequence
#     # ready to be heapified!
#     rheap_rdd = rheap_rdd.map(lambda x: (
#         sim_between_seq(query_sequence.fetch_data(bc_norm_data.value), x.fetch_data(bc_norm_data.value), dist_type=dist_type), x))
#     r_heap = rheap_rdd.collect()
#     heapq.heapify(r_heap)
#
#     query_result = []
#
#     while len(query_result) < k:
#         # create a cluster to query
#         querying_cluster = []
#         while len(querying_cluster) <= k:
#             try:
#                 top_rep = heapq.heappop(r_heap)
#             except IndexError as ie:
#                 print('Warning: R space exhausted, best k not reached, returning all the matches so far')
#                 return query_result
#
#             querying_cluster = querying_cluster + gcluster.get_cluster(top_rep[1])  # top_rep: (dist to query, rep sequence)
#
#         query_cluster_rdd = sc.parallelize(querying_cluster, numSlices=data_slices)
#
#         if ex_sameID:  # filter by not same id
#             query_cluster_rdd = query_cluster_rdd.filter(lambda x: x.id != query_sequence.id)
#
#         # TODO do not fetch data_original everytime for the query sequence
#         query_cluster_rdd = query_cluster_rdd.map(lambda x: (
#             sim_between_seq(query_sequence.fetch_data(bc_norm_data.value), x.fetch_data(bc_norm_data.value), dist_type=dist_type), x))
#         qheap = query_cluster_rdd.collect()
#
#         try:
#             heapq.heapify(qheap)
#         except TypeError as te:
#             return qheap
#
#         while len(query_result) < k and len(qheap) != 0:
#             current_match = heapq.heappop(qheap)
#
#             if not any(_isOverlap(current_match[1], prev_match[1], overlap) for prev_match in query_result):  # check for overlap against all the matches so far
#                 query_result.append(current_match)
#
#     return query_result

def _isOverlap(seq1: Sequence, seq2: Sequence, overlap: float) -> bool:
    if seq1.seq_id != seq2.seq_id:  # overlap does NOT matter if two seq have different id
        return True
    else:
        of =  _calculate_overlap(seq1, seq2)
        return _calculate_overlap(seq1, seq2) >= overlap


def _calculate_overlap(seq1, seq2) -> float:
    if seq2.end > seq1.end and seq2.start >= seq1.start:
        return (seq1.end - seq2.start + 1) / (seq2.end - seq1.start + 1)
    elif seq1.end > seq2.end and seq1.start >= seq2.start:
        return (seq2.end - seq1.start + 1) / (seq1.end - seq2.start + 1)
    if seq2.end >= seq1.end and seq2.start > seq1.start:
        return (seq1.end - seq2.start + 1) / (seq2.end - seq1.start + 1)
    elif seq1.end >= seq2.end and seq1.start > seq2.start:
        return (seq2.end - seq1.start + 1) / (seq1.end - seq2.start + 1)

    elif seq1.end > seq2.end and seq2.start >= seq1.start:
        return len(seq2) / len(seq1)
    elif seq2.end > seq1.end and seq1.start >= seq2.start:
        return len(seq1) / len(seq2)
    elif seq1.end >= seq2.end and seq2.start > seq1.start:
        return len(seq2) / len(seq1)
    elif seq2.end >= seq1.end and seq1.start > seq2.start:
        return len(seq1) / len(seq2)

    elif seq2.start > seq1.end or seq1.start > seq2.end:  # does not overlap at all
        return 0.0
    else:
        print(seq1)
        print(seq2)
        raise Exception('FATAL: sequence 100% overlap, please report the bug')
