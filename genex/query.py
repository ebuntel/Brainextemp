# TODO finish implementing query
import heapq
import os

import math
from pyspark import SparkContext

from genex.cluster import sim_between_seq
from genex.data_process import get_data_for_timeSeriesObj
from genex.parse import strip_function, remove_trailing_zeros
from .classes import Sequence
from .classes import Gcluster


def query(q: Sequence, gc: Gcluster, loi: list, sc: SparkContext,
          k:int=1, ex_sameID: bool=False, overlap: float= 1.0, mode:str='genex'):
    """

    :param q: query sequence
    :param gc: Gcluster in which to query
    :param loi: list of two integer values, specifying the query range, if set to None, is going to query all length
    :param sc: spark context on which to run the query operation

    :param k: integer, specifying to return top k matches
    :param ex_sameID: boolean, whether to include sequences from the time series with the same id as the query sequence
    :param overlap: float, how much overlapping between queries lookups
    :param mode: query mode, supported modes are 'genex' and 'bf' (bf = brute force)
    """
    if mode == 'genex':
        gquery()

    elif mode == 'bf':
        bfquery()

    else:
        raise Exception('Unsupported query mode: ' + mode)

def get_query_from_dict():

    pass
def get_query_sequence_from_file(file: str):
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



def gquery(q: Sequence, gc: Gcluster, loi: list, sc: SparkContext,
          k:int=1, ex_sameID: bool=False, overlap: float= 1.0):
    # get query from id, start, end point
    # get query from csv file
    #

    query_result = custom_query_operation(sc, normalized_ts_dict, time_series_dict, res_list, cluster_rdd,
                                          exclude_same_id, SAVED_DATASET_DIR,
                                          loi, gp_project, querying_range, k, sequence)
    print()


def bfquery():
    print()


def custom_query_operation(q: Sequence, gc: Gcluster, loi: list, sc: SparkContext,
          k:int=1, ex_sameID: bool=False, overlap: float= 1.0):

    query_result = filter_rdd_back.repartition(16).map(
        lambda clusters: custom_query(q, loi, gc, k,
                                      global_time_series_dict.value, ))
    # changed here
    # plot_query_result(query_sequence, query_result, global_time_series_dict.value)
    return query_result

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


def get_most_k_sim(query_sequence: list, query_range, Gcluster, k, time_series_dict):
    min_rprs = None  # the representative that is closest to the query distance
    min_dist = math.inf
    target_cluster = []
    
    for cur_rprs in Gcluster.values().keys():
        heap = []
        # TODO do we want to get raw data here, or set the raw in timeSeriesObj before calling query (no parsing)
        if (cur_rprs.end - cur_rprs.start) in range(query_range[0], query_range[1] + 1):
            # modify here, not use get data from objects, use values
            cur_dist = sim_between_seq(query_sequence, cur_rprs.data)

            if cur_dist < min_dist:
                min_rprs = cur_rprs
                min_dist = cur_dist
        else:
            continue

    if min_rprs:
        print('min representative is ' + min_rprs.id)

        print("Querying Cluster of length: " + str(len(get_data_for_timeSeriesObj(min_rprs, time_series_dict))))
        target_cluster = Gcluster[min_rprs]
        print('len of cluster is ' + str(len(target_cluster)))
        print("sorting")

        target_cluster.sort(key=lambda cluster_sequence: sim_between_seq(query_sequence,
                                                                         get_data_for_timeSeriesObj(cluster_sequence,
                                                                                                    time_series_dict)))
        k = int(k)
        return target_cluster[0:k]  # return the k most similar sequences






