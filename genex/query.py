# TODO finish implementing query

def query(mode:str='genex'):
    """

    :param mode: select query mode, it can be 'genex' or 'bf'
    """
    if mode == 'genex':
        gquery()

    elif mode == 'bf':
        bfquery()

    else:
        raise Exception('Unsupported query mode: ' + mode)


def gquery():
    print()


def bfquery():
    print()


def query(query_sequence, query_range, cluster, k, time_series_dict, exclude_overlap, percentage=1):
    """

    :param query_sequence: list of data: the sequence to be queried
    :param cluster: dict[key = representative, value = list of timeSeriesObj] -> representative is timeSeriesObj
                    the sequences in the cluster are all of the SAME length
    :param k: int
    :return list of time series objects: best k matches. Again note they are all of the SAME length
    """

    # iterate through all the representatives to find which cluster to look at
    min_rprs = None  # the representative that is closest to the query distance
    min_dist = math.inf
    target_cluster = []
    for cur_rprs in cluster.keys():
        # print("actually querying")
        # print('end point is' + str(cur_rprs.end_point))
        # print('start point is' + str(cur_rprs.start_point))
        # TODO do we want to get raw data here, or set the raw in timeSeriesObj before calling query (no parsing)
        if (cur_rprs.end_point - cur_rprs.start_point) in range(query_range[0], query_range[1] + 1):
            # print("it's in")
            cur_dist = sim_between_seq(query_sequence, get_data_for_timeSeriesObj(cur_rprs, time_series_dict))

            if cur_dist < min_dist:
                min_rprs = cur_rprs
                min_dist = cur_dist
        else:
            pass

    if min_rprs:
        print('min representative is ' + min_rprs.id)

        print("Querying Cluster of length: " + str(len(get_data_for_timeSeriesObj(min_rprs, time_series_dict))))
        target_cluster = cluster[min_rprs]
        print('len of cluster is ' + str(len(target_cluster)))
        print("sorting")

        # this sorting is taking a long time!
        target_cluster.sort(key=lambda cluster_sequence: sim_between_seq(query_sequence,
                                                                         get_data_for_timeSeriesObj(cluster_sequence,
                                                                                                    time_series_dict)))
    #     use a heap?
    #     use quickselect
    #     similar question to k closet point to origin

    # where can we get none?
    if len(target_cluster) != 0:
        # print(target_cluster)
        if exclude_overlap:
            target_cluster = exclude_overlapping(target_cluster, percentage, k)
            print("k is" + str(k))
        return target_cluster[0:k]  # return the k most similar sequences
    # else:
    #     raise Exception("No matching found")


#     raise none exception?
#     Note that this function return None for those times series range not in the query range
def custom_query(query_sequences, query_range, cluster, k, time_series_dict):
    """

    :param query_sequences: list of list: the list of sequences to be queried
    :param cluster: dict[key = representative, value = list of timeSeriesObj] -> representative is timeSeriesObj
                    the sequences in the cluster are all of the SAME length
    :param k: int
    :return list of time series objects: best k matches. Again note they are all of the SAME length
    """

    # iterate through all the representatives to find which cluster to look at
    # try :
    #
    query_result = dict()
    if not isinstance(query_sequences, list) or len(query_sequences) == 0:
        raise ValueError("query sequence must be a list and not empty")
    cur_query_number = 0
    if isinstance(query_sequences[0], list):
        print("length of query is [" + str(len(query_sequences)) + "]" + "[" + str(len(query_sequences[0])) + "]")
        print("query is a list of list")
        for cur_query in query_sequences:
            if isinstance(cur_query, list):
                query_result[cur_query_number] = get_most_k_sim(cur_query, query_range, cluster, k, time_series_dict)
                return query_result
    else:
        return get_most_k_sim(query_sequences, query_range, cluster, k, time_series_dict)


def get_most_k_sim(query_sequence, query_range, cluster, k, time_series_dict):
    min_rprs = None  # the representative that is closest to the query distance
    min_dist = math.inf
    target_cluster = []
    for cur_rprs in cluster.keys():

        # TODO do we want to get raw data here, or set the raw in timeSeriesObj before calling query (no parsing)
        if (cur_rprs.end_point - cur_rprs.start_point) in range(query_range[0], query_range[1] + 1):

            cur_dist = sim_between_seq(query_sequence, get_data_for_timeSeriesObj(cur_rprs, time_series_dict))

            if cur_dist < min_dist:
                min_rprs = cur_rprs
                min_dist = cur_dist
        else:
            continue

    if min_rprs:
        print('min representative is ' + min_rprs.id)

        print("Querying Cluster of length: " + str(len(get_data_for_timeSeriesObj(min_rprs, time_series_dict))))
        target_cluster = cluster[min_rprs]
        print('len of cluster is ' + str(len(target_cluster)))
        print("sorting")

        target_cluster.sort(key=lambda cluster_sequence: sim_between_seq(query_sequence,
                                                                         get_data_for_timeSeriesObj(cluster_sequence,
                                                                                                    time_series_dict)))
        k = int(k)
        return target_cluster[0:k]  # return the k most similar sequences


def get_query_sequence_from_file(file):
    resList = []
    with open(file, 'r') as f:
        for i, line in enumerate(f):
            if not i:
                features = list(map(lambda x: strip_function(x),
                                    line.strip()[:-1].split(',')))
            if line != "" and line != "\n":
                data = remove_trailing_zeros(line.split(",")[:-1])
                series_data = data[len(features):]
                resList.append(series_data)
    if len(resList[0]) == 0:
        return resList[1:]
    else:

        return resList


def query_operation(sc, normalized_ts_dict, time_series_dict, res_list, cluster_rdd, exclude_same_id, SAVED_DATASET_DIR,
                    include_in_range, gp_project):
    print("querying ")
    global_dict = sc.broadcast(normalized_ts_dict)
    # change naming here from time_series_dict to global_time_series_dict
    # because it might cause problem when saving
    global_time_series_dict = sc.broadcast(time_series_dict)
    global_dict_rdd = sc.parallelize(res_list[1:],
                                     numSlices=128)  # change the number of slices to mitigate larger datasets

    grouping_range = (1, max([len(v) for v in global_dict.value.values()]))
    # print("grouping_range" + str(grouping_range))
    query_id = '(2013e_001)_(100-0-Back)_(A-DC4)_(232665953.1250)_(232695953.1250)'
    query_sequence = get_data(query_id, 24, 117, global_time_series_dict.value)  # get an example query
    print(len(query_sequence))
    # cluster_rdd.collect()
    # repartition(16).
    # raise exception if the query_range exceeds the grouping range
    # TODO after getting range and filtering, repartition!!
    querying_range = (90, 91)
    k = 5  # looking for k best matches
    print("start query")
    if querying_range[0] < grouping_range[0] or querying_range[1] > grouping_range[1]:
        raise Exception("query_operations: query: Query range does not match group range")
    filter_rdd = cluster_rdd.filter(lambda x: include_in_range(x, querying_range)).filter(
        lambda x: exclude_same_id(x, query_id))

    # clusters = cluster_rdd.collect()
    # query_result = cluster_rdd.filter(lambda x: x).map(lambda clusters: query(query_sequence, querying_range, clusters, k, time_series_dict.value)).collect()
    exclude_overlapping = True
    path_to_save = SAVED_DATASET_DIR + os.sep + gp_project.get_project_name()
    if os.path.isdir(path_to_save + '/filter/') and len(os.listdir(path_to_save + '/filter/')) != 0:
        filter_rdd_back = sc.pickleFile(path_to_save + '/filter/')
        # filter_res_back = filter_rdd_back.collect()
        filter_res = filter_rdd.collect()
        print("load back")
    else:

        filter_rdd.saveAsPickleFile(path_to_save + '/filter/')
        # filter_res = filter_rdd.collect()
        filter_rdd_back = sc.pickleFile(path_to_save + '/filter/')
        print("first time of saving query")
    query_result = filter_rdd_back.repartition(16).map(
        lambda clusters: query(query_sequence, querying_range, clusters, k,
                               global_time_series_dict.value,
                               exclude_overlapping,
                               0.5)).collect()
    # changed here
    # plot_query_result(query_sequence, query_result, global_time_series_dict.value)
    return query_result


def custom_query_operation(sc, normalized_ts_dict, time_series_dict, res_list, cluster_rdd, exclude_same_id,
                           SAVED_DATASET_DIR,
                           include_in_range, gp_project, querying_range, k, file):
    print("custom querying ")
    global_dict = sc.broadcast(normalized_ts_dict)
    # change naming here from time_series_dict to global_time_series_dict
    # because it might cause problem when saving
    global_time_series_dict = sc.broadcast(time_series_dict)
    global_dict_rdd = sc.parallelize(res_list[1:],
                                     numSlices=128)  # change the number of slices to mitigate larger datasets

    grouping_range = (1, max([len(v) for v in global_dict.value.values()]))
    # print("grouping_range" + str(grouping_range))

    # cluster_rdd.collect()
    # repartition(16).
    # raise exception if the query_range exceeds the grouping range
    # TODO after getting range and filtering, repartition!!

    # looking for k best matches
    print("start query")
    querying_range = list(map(int, querying_range))
    if querying_range[0] < grouping_range[0] or querying_range[1] > grouping_range[1]:
        raise Exception("query_operations: query: Query range does not match group range")
    filter_rdd = cluster_rdd.filter(lambda x: include_in_range(x, querying_range))

    path_to_save = SAVED_DATASET_DIR + os.sep + gp_project.get_project_name()
    # todo
    query_sequence = get_query_sequence_from_file(file)[0:2]
    if os.path.isdir(path_to_save + '/custom_query/') and len(os.listdir(path_to_save + '/custom_query/')) != 0:
        filter_rdd_back = sc.pickleFile(path_to_save + '/custom_query/')
        # filter_res_back = filter_rdd_back.collect()
        # filter_res = filter_rdd.collect()
        print("custom query load back")
    else:

        filter_rdd.saveAsPickleFile(path_to_save + '/custom_query/')
        # filter_res = filter_rdd.collect()
        filter_rdd_back = sc.pickleFile(path_to_save + '/custom_query/')
        print("first time of saving query")
    query_result = filter_rdd_back.repartition(16).map(
        lambda clusters: custom_query(query_sequence, querying_range, clusters, k,
                                      global_time_series_dict.value, )).collect()
    # changed here
    # plot_query_result(query_sequence, query_result, global_time_series_dict.value)
    return query_result