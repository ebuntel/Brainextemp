import math
import csv
from genex.parse import generate_source
from genex.preprocess import min_max_normalize
from genex.utils import normalize_sequence
import heapq
import time
from genex.cluster import sim_between_seq
import matplotlib.pyplot as plt

fn = 'SART2018_HbO_40.csv'

input_list = generate_source(fn, feature_num=5)
# input_list = input_list[:51]


normalized_input_list, global_max, global_min = min_max_normalize(input_list)

from pyspark import SparkContext, SparkConf

num_cores = 8

conf = SparkConf(). \
    setMaster("local[" + str(num_cores) + "]"). \
    setAppName("Genex").set('spark.driver.memory', '15G'). \
    set('spark.driver.maxResultSize', '15G')
sc = SparkContext(conf=conf)

input_rdd = sc.parallelize(normalized_input_list, numSlices=num_cores)
partition_input = input_rdd.glom().collect()

from genex.preprocess import all_sublists_with_id_length

gstart_time = time.time()
group_rdd = input_rdd.flatMap(
    lambda x: all_sublists_with_id_length(x, [150, 190]))
partition_group = group_rdd.glom().collect()
gend_time = time.time()
gtime = gend_time - gstart_time
gtime = time.strftime("%H:%M:%S", time.gmtime(gtime))
print("Group time:------------------------------ " + str(gtime))
file = open('time_log.txt', 'a')
file.write('Group time is :' + gtime)
file.close()

from genex.cluster import filter_cluster

start_time = time.time()
cluster_rdd = group_rdd.mapPartitions(lambda x: filter_cluster(groups=x, st=0.05, log_level=1),
                                      preservesPartitioning=False).cache()
cluster_partition = cluster_rdd.glom().collect()
end_time = time.time()
ctime = end_time - start_time
ctime = time.strftime("%H:%M:%S", time.gmtime(ctime))
print("Cluster time:----------------------------- " + str(ctime))
file = open('time_log.txt', 'a')
file.write('Cluster time is :' + ctime)
file.close()


def query_cluster_partition(cluster, q, st: float, k: int, normalized_input, dist_type: str = 'eu_ucr', loi=None):
    q = q.value
    normalized_input = normalized_input.value

    if loi is not None:
        cluster_dict = dict(x for x in cluster if x[0] in range(loi[0], loi[1]))
    else:
        cluster_dict = dict(cluster)
    # get the seq length of the query, start query in the cluster that has the same length as the query
    querying_clusterSeq_len = len(q)

    # get the seq length Range of the partition
    try:
        len_range = (min(cluster_dict.keys()), max(cluster_dict.keys()))
    except ValueError as ve:
        raise Exception('cluster does not have the given query loi!')

    # if given query is longer than the longest cluster sequence,
    # set starting clusterSeq length to be of the same length as the longest sequence in the partition
    querying_clusterSeq_len = max(min(querying_clusterSeq_len, len_range[1]), len_range[0])

    query_result = []
    # temperoray variable that decides whether to look up or down when a cluster of a specific length is exhausted

    while len(cluster_dict) > 0:
        target_cluster = cluster_dict[querying_clusterSeq_len]
        target_cluster_reprs = target_cluster.keys()
        target_cluster_reprs = list(
            map(lambda rpr: [sim_between_seq(rpr.fetch_data(normalized_input), q.data, dist_type=dist_type),
                             rpr], target_cluster_reprs))
        heapq.heapify(target_cluster_reprs)

        while len(target_cluster_reprs) > 0:
            querying_repr = heapq.heappop(target_cluster_reprs)
            querying_cluster = target_cluster[querying_repr[1]]
            querying_cluster = list(
                map(lambda cluster_seq: [
                    sim_between_seq(cluster_seq.fetch_data(normalized_input), q.data, dist_type=dist_type),
                    cluster_seq],
                    querying_cluster))
            heapq.heapify(querying_cluster)

            for seq in querying_cluster:
                if seq[0] < st:
                    query_result.append((seq[0], seq[1]))
                    if (len(query_result)) >= k:
                        return query_result

        cluster_dict.pop(querying_clusterSeq_len)  # remove this len-cluster just queried

        # find the next closest sequence length
        querying_clusterSeq_len = min(list(cluster_dict.keys()), key=lambda x: abs(x - querying_clusterSeq_len))

    return query_result


from genex.parse import generate_query

# generate the query sets
query_set = generate_query(file_name='queries_test.csv', feature_num=5)
print(query_set)

for query in query_set:
    lst = []
    print(query)
    # randomly pick a sequence as the query from the query sequence, make sure the picked sequence is in the input list
    # query = next((item for item in query_set if item.id in dict(input_list).keys()), None)
    # print(query)
    # fetch the data for the query
    query.set_data(query.fetch_data(input_list))

    normalize_sequence(query, global_max, global_min)
    query_bc = sc.broadcast(query)

    query_st = math.inf
    best_k = 5

    normalized_input_list_bc = sc.broadcast(normalized_input_list)
    query_bc = sc.broadcast(query)

    # a = query_cluster_partition(cluster=cluster_partition[0], q=query_bc, st=query_st, k=best_k,
    #                             normalized_input=normalized_input_list_bc)

    qstart_time = time.time()
    query_rdd = cluster_rdd.mapPartitions(
        lambda x:
        query_cluster_partition(cluster=x, q=query_bc, st=query_st, k=best_k,
                                normalized_input=normalized_input_list_bc)).cache()
    query_partition = query_rdd.glom().collect()
    qend_time = time.time()
    qtime = qend_time - qstart_time
    qtime = time.strftime("%H:%M:%S", time.gmtime(qtime))
    aggre_query_result = query_rdd.collect()
    heapq.heapify(aggre_query_result)
    best_matches = []

    for i in range(best_k):
        best_matches.append(heapq.heappop(aggre_query_result))

    fig = plt.figure(figsize=(15, 15))
    plt.plot(query.fetch_data(input_list), color='cyan', linewidth=3.0, label='Query Sequence' + str(query.id))
    lst.append(query.id)
    for seq in best_matches:
        l = seq[1].id
        lst.append(l)
        lst.append(seq[0])
        plt.plot(seq[1].fetch_data(input_list), label=str(l) + str(seq[0]))
    lst.append(qtime)
    with open('results_Genex_HbO_aaj.csv', 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(lst)
    csvfile.close()

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
