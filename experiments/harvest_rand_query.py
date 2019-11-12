import csv
import random
import time

import genex.database.genex_database as gxdb
from pyspark import SparkContext, SparkConf
from genex.parse import generate_query

import numpy as np
import pandas as pd


# create the spark context
def experiment_genex(data_file, num_sample, num_query, result_file):
    num_cores = 12
    conf = SparkConf(). \
        setMaster("local[" + str(num_cores) + "]"). \
        setAppName("Genex").set('spark.driver.memory', '64G'). \
        set('spark.driver.maxResultSize', '64G')
    sc = SparkContext(conf=conf)

    # create gxdb from a csv file

    # set up where to save the results
    result_headers = np.array(
        [['cluster_time', 'query', 'gx_time', 'bf_time', 'diff', 'gx_dist', 'gx_match', 'bf_dist', 'bf_match']])
    result_df = pd.DataFrame(columns=result_headers[0, :])

    print('Performing clustering ...')
    mydb = gxdb.from_csv(data_file, sc=sc, feature_num=2, _rows_to_consider=num_sample)

    max_seq_len = mydb.get_max_seq_len()
    print('Generating query of max seq len ...')
    # generate the query sets
    query_set = list()

    for i in range(num_query):
        query_set.append(mydb.get_random_seq_of_len(max_seq_len))

    # perform clustering
    cluster_start_time = time.time()
    mydb.build(similarity_threshold=0.1)
    cluster_time = time.time() - cluster_start_time
    result_df = result_df.append({'cluster_time': cluster_time}, ignore_index=True)

    # get the number of subsequences
    num_seq = mydb.get_num_subsequences()

    # randomly pick a sequence as the query from the query sequence, make sure the picked sequence is in the input list
    # this query'id must exist in the database
    k_to_test = [1, 9, 15]
    for best_k in k_to_test:

        overall_diff_list = []
        print('Evaluating ...')
        for i, q in enumerate(query_set):

            print('Querying #' + str(i) + ' of ' + str(len(query_set)) + '; query = ' + str(q))
            start = time.time()
            print('     Running Genex Query ...')
            query_result_gx = mydb.query(query=q, best_k=best_k)
            gx_time = time.time() - start

            start = time.time()
            print('     Running Brute Force Query ...')
            query_result_bf = mydb.query_brute_force(query=query_set[0], best_k=best_k)
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
            result_df.to_csv(result_file)

        # save the overall difference
        result_df = result_df.append({'diff': np.mean(overall_diff_list)}, ignore_index=True)
        result_df.to_csv(result_file + '_' + str(best_k) + '.csv')
    # terminate the spark session
    sc.stop()


# data_file = 'data/test/ItalyPowerDemand_TEST.csv'
# query_file = 'data/test/ItalyPowerDemand_query.csv'
# result_file = 'results/test/ItalyPowerDemand_result.csv'
# experiment_genex(data_file, query_file, result_file)
# TODO run ECG
# Querying #9 of 15; query = (ECG-1)_(Label-2): (61:118)
#      Running Genex Query ...
data_file = 'data/test/ItalyPowerDemand_TEST.csv'
result_file = 'results/test/ipd/ItalyPowerDemand_result'
experiment_genex(data_file, num_sample=40, num_query=40, result_file=result_file)
