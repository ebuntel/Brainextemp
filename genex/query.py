# TODO finish implementing query

import math
from pyspark import SparkContext

# from genex.cluster import sim_between_seq
from genex.op.query_op import sim_between_seq
from genex.parse import strip_function, remove_trailing_zeros
from .classes import Sequence
from genex.database import genexengine


def query(q: Sequence, gc: genexengine, loi: list, sc: SparkContext,
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



def gquery(query_list: list, gc_data: dict, loi: list, input_list: list,
          k:int=1, ex_sameID: bool=False, overlap: float= 1.0, ):
    """
    Because Gcluster object doesn't have map property, we have to use dict as input
    :param file:
    :param gc_data:
    :param loi:
    :param input_list:
    :param k:
    :param ex_sameID:
    :param overlap:
    :return:
    """
    # get query from id, start, end point
    # get query from csv file
    #

    # query_list = []
    # query_set = get_query_from_csv_with_id(file)
    # print(query_set)
    # for cur_query in query_set:
    #     query_list.append(get_query_from_sequence(cur_query[0], int(cur_query[1]), int(cur_query[2]), input_list))
    # print(query_list)

    return custom_query(query_list, loi, gc_data, k, input_list)



def bfquery():
    print()

#
# def custom_query_operation(q: Sequence, gc: Gcluster, loi: list, sc: SparkContext,
#           k:int=1, ex_sameID: bool=False, overlap: float= 1.0):
#
#     query_result = filter_rdd_back.repartition(16).map(
#         lambda clusters: custom_query(q, loi, gc, k,
#                                       global_time_series_dict.value, ))
#     # changed here
#     # plot_query_result(query_sequence, query_result, global_time_series_dict.value)
#     return query_result



def get_query_from_sequence(id: tuple, start: int, end: int, input_list: list):
    """

    :param id:
    :param start:
    :param end:
    :param input_list:
    :return: a list
    """
    try:
        input_dict = dict(input_list)  # validate by converting input_list into a dict
    except (TypeError, ValueError):
        raise Exception('sequence: fetch_data: input_list is not key-value pair.')

    return input_dict[id][start: end]




def custom_query(query_sequences: list, loi: list, Gcluster_data:dict, k : int, input_list:list):
    # """
    #
    # :param query_sequences: list of list: the list of sequences to be queried
    # :param cluster: dict[key = representative, value = list of timeSeriesObj] -> representative is timeSeriesObj
    #                 the sequences in the cluster are all of the SAME length
    # :param k: int
    # :return list of time series objects: best k matches. Again note they are all of the SAME length
    # """
    """

    :param query_sequences:
    :param query_range:
    :param Gcluster_data:
    :param k:
    :param input_list:
    :return:
    """

    # get query from csv file which contains lists of list of query actual clusters
    # get query from csv file which contains lists of tuple of id, start, endpoint
    query_result = dict()
    if not isinstance(query_sequences, list) or len(query_sequences) == 0:
        raise ValueError("query sequence must be a list and not empty")
    cur_query_number = 0
    if isinstance(query_sequences[0], list):
        print("length of query is [" + str(len(query_sequences)) + "]" + "[" + str(len(query_sequences[0])) + "]")
        print("query is a list of list")
        for cur_query in query_sequences:
            if isinstance(cur_query, list):
                query_result[cur_query_number] = get_most_k_sim(cur_query, loi, Gcluster_data, k, input_list)
                cur_query_number += 1
        return query_result
    else:
        return get_most_k_sim(query_sequences, loi, Gcluster_data, k, input_list)


def get_most_k_sim(query_sequence: list, loi: list, Gcluster_data : dict, k, input_list:list):
    """

    :param query_sequence:
    :param query_range:
    :param Gcluster_data:
    :param k:
    :param input_list:
    :return:
    """

    min_rprs = None  # the representative that is closest to the query distance
    min_dist = math.inf
    target_cluster = []
    print("length of gcluster clusters is " + str(len(Gcluster_data[1])))
    for cur_rprs_seq in Gcluster_data[1].keys():

        # TODO do we want to get raw clusters here, or set the raw in timeSeriesObj before calling query (no parsing)
        if (cur_rprs_seq.end - cur_rprs_seq.start + 1) in range(loi[0], loi[1] + 1):
            # modify here, not use get clusters from objects, use values
            cur_dist = sim_between_seq(query_sequence, cur_rprs_seq.fetch_data(input_list))
            if cur_dist < min_dist:
                min_rprs = cur_rprs_seq
                min_dist = cur_dist
        else:
            break

    if min_rprs:
        print('min representative is ' + min_rprs.__str__())
        print('min dist' + str(min_dist))
        # print("Querying Cluster of length: " + str(len(get_data_for_timeSeriesObj(min_rprs, time_series_dict))))
        target_cluster = Gcluster_data[1].get(min_rprs)
        print('len of cluster is ' + str(len(target_cluster)))
        # print("sorting")
        #
        target_cluster.sort(key=lambda cluster_sequence: sim_between_seq(query_sequence,
                                                                         cluster_sequence.data))
        k = int(k)
        return target_cluster[0:k]  # return the k most similar sequences
    else:
        return None





