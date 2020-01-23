import csv
import time

import genex.database.genexengine as gxdb
from pyspark import SparkContext, SparkConf
from genex.parse import generate_query

import numpy as np
import pandas as pd


# create the spark context
def experiment_genex(data_file, query_file, result_file):
    num_cores = 32
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

    print('Fetching query ...')
    # generate the query sets
    query_set = generate_query(file_name=query_file, feature_num=2)
    print('Performing clustering ...')
    mydb = gxdb.from_csv(data_file, sc=sc, feature_num=2)
    if not all(mydb.is_seq_exist(x) for x in query_set):
        print('some query not found in the original dataset, aborted.')
        return

    cluster_start_time = time.time()
    mydb.build(st=0.1)
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
        query_result_gx = mydb.query(query=q, best_k=15)
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
        result_df.to_csv(result_file)

        print('     Done')

    # save the overall difference
    result_df = result_df.append({'diff': np.mean(overall_diff_list)}, ignore_index=True)
    result_df.to_csv(result_file)
    # terminate the spark session
    sc.stop()


# data_file = 'data_original/test_result/ItalyPowerDemand_TEST.csv'
# query_file = 'data_original/test_result/ItalyPowerDemand_query.csv'
# result_file = 'results/test_result/ItalyPowerDemand_result_regular.csv'
# experiment_genex(data_file, query_file, result_file)
# TODO run ECG
# Querying #9 of 15; query = (ECG-1)_(Label-2): (61:118)
#      Running Genex Query ...
data_file = 'data_original/test_result/ECGFiveDays_TEST .csv'
query_file = 'data_original/test_result/ECGFiveDays_query_1.csv'
result_file = 'results/test_result/ECGFiveDays_result_1.csv'
experiment_genex(data_file, query_file, result_file)
