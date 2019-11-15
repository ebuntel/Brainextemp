import csv
import time

import genex.database.genex_database as gxdb
from pyspark import SparkContext, SparkConf

from genex.classes.Sequence import Sequence
from genex.cluster import sim_between_seq
from genex.parse import generate_query

import numpy as np
import pandas as pd


# create the spark context
def experiment_genex(data, output, feature_num, num_sample, num_query, add_uuid):
    num_cores = 8
    conf = SparkConf(). \
        setMaster("local[" + str(num_cores) + "]"). \
        setAppName("Genex").set('spark.driver.memory', '15G'). \
        set('spark.driver.maxResultSize', '15G')
    sc = SparkContext(conf=conf)

    # create gxdb from a csv file

    # set up where to save the results
    result_headers = np.array(
        [['cluster_time', 'query', 'gx_time', 'bf_time', 'diff', 'gx_dist', 'gx_match', 'bf_dist', 'bf_match']])
    result_df = pd.DataFrame(columns=result_headers[0, :])

    print('Performing clustering ...')
    mydb = gxdb.from_csv(data, sc=sc, feature_num=feature_num, add_uuid=add_uuid, _rows_to_consider=num_sample)

    print('Generating query of max seq len ...')
    # generate the query sets
    query_set = list()
    # get the number of subsequences
    # randomly pick a sequence as the query from the query sequence, make sure the picked sequence is in the input list
    # this query'id must exist in the database
    for i in range(num_query):
        query_set.append(mydb.get_random_seq_of_len(int(mydb.get_max_seq_len()/2)))

    cluster_start_time = time.time()
    mydb.build(similarity_threshold=0.1)
    cluster_time = time.time() - cluster_start_time
    result_df = result_df.append({'cluster_time': cluster_time}, ignore_index=True)

    # randomly pick a sequence as the query from the query sequence, make sure the picked sequence is in the input list
    # this query'id must exist in the database
    overall_diff_list = []

    print('Evaluating ...')
    for i, q in enumerate(query_set):
        print('Querying #' + str(i) + ' of ' + str(len(query_set)) + '; query = ' + str(q))
        start = time.time()
        print('     Running Genex Query ...')
        query_result_gx = mydb.query(query=q, best_k=15, _ke=3*15)
        gx_time = time.time() - start

        start = time.time()
        print('     Running Brute Force Query ...')
        query_result_bf = mydb.query_brute_force(query=query_set[0], best_k=15)
        bf_time = time.time() - start

        # save the results
        print('     Saving results ...')
        result_df = result_df.append({'query': str(q), 'gx_time': gx_time, 'bf_time': bf_time}, ignore_index=True)
        diff_list = []
        for gx_r, bf_r in zip(query_result_gx, query_result_bf):
            diff = abs(gx_r[0] - bf_r[0])
            diff_list.append(diff)
            overall_diff_list.append(diff)
            result_df = result_df.append({'diff': diff,
                                          'gx_dist': gx_r[0], 'gx_match': gx_r[1],
                                          'bf_dist': bf_r[0], 'bf_match': bf_r[1]}, ignore_index=True)
        result_df = result_df.append({'diff': np.mean(diff_list)}, ignore_index=True)
        result_df.to_csv(output)

        print('     Done')

    # save the overall difference
    result_df = result_df.append({'diff': np.mean(overall_diff_list)}, ignore_index=True)
    result_df.to_csv(output)
    # terminate the spark session
    sc.stop()

    return mydb


# data_file = 'data/test/ItalyPowerDemand_TEST.csv'
# query_file = 'data/test/ItalyPowerDemand_query.csv'
# result_file = 'results/test/ItalyPowerDemand_result_regular.csv'
# experiment_genex(data_file, query_file, result_file)
# TODO run ECG
# Querying #9 of 15; query = (ECG-1)_(Label-2): (61:118)
#      Running Genex Query ...


experiment_set = {'ecgFiveDays': {'data': 'data/ECGFiveDays.csv',
                                  'output': 'results/ECGFiveDays_result.csv',
                                  'feature_num': 2,
                                  'add_uuid': False},
                  'italyPowerDemand': {'data': 'data/ItalyPower.csv',
                                       'output': 'results/ItalyPowerDemand_result.csv',
                                       'feature_num': 2,
                                       'add_uuid': False}, }

mydb = experiment_genex(**experiment_set['italyPowerDemand'], num_sample=40, num_query=40)


# q = Sequence(seq_id=('Italy_power25', '2'), start=7, end=18)
# seq1 = Sequence(seq_id=('Italy_power25', '2'), start=6, end=18)
# seq2 = Sequence(seq_id=('Italy_power25', '2'), start=7, end=17)
# q.fetch_and_set_data(mydb.data_normalized)
# seq1.fetch_and_set_data(mydb.data_normalized)
# seq2.fetch_and_set_data(mydb.data_normalized)
# from dtw import dtw
# import matplotlib.pyplot as plt
# plt.plot(q.data, label='query')
# plt.plot(seq1.data, label='gx')
# plt.plot(seq2.data, label='bf')
# plt.legend()
# plt.show()
# euclidean_norm = lambda x, y: np.abs(x - y)
# x_dist1, cost_matrix1, acc_cost_matrix1, path1 = dtw(q.data, seq1.data, dist=euclidean_norm)
# x_dist2, cost_matrix2, acc_cost_matrix2, path2 = dtw(q.data, seq2.data, dist=euclidean_norm)
# plt.imshow(acc_cost_matrix1.T, origin='lower', cmap='gray', interpolation='nearest')
# plt.plot(path1[0], path1[1], 'w')
# plt.title('query and gx')
# plt.show()
#
# plt.imshow(acc_cost_matrix2.T, origin='lower', cmap='gray', interpolation='nearest')
# plt.plot(path2[0], path2[1], 'w')
# plt.title('query and bf')
# plt.show()
# print('distance between query and gx ' + str(x_dist1))
# print('distance between query and bf ' + str(x_dist2))
# dist1 = sim_between_seq(q, seq1, dist_type='eu')
# dist2 = sim_between_seq(q, seq2, dist_type='eu')

