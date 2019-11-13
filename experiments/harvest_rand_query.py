import csv
import math
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
        [['cluster_time', 'query', 'gx_time', 'bf_time', 'ke', 'gx_dist', 'gx_match', 'bf_dist', 'bf_match']])
    result_df = pd.DataFrame(columns=result_headers[0, :])

    print('Performing clustering ...')
    mydb = gxdb.from_csv(data_file, sc=sc, feature_num=2, _rows_to_consider=num_sample)

    print('Generating query of max seq len ...')
    # generate the query sets
    query_set = list()

    for i in range(num_query):
        query_set.append(mydb.get_random_seq_of_len(mydb.get_max_seq_len()))

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

    result_dict = dict()

    for best_k in k_to_test:
        overall_diff_list = []
        print('Evaluating ...')

        best_l1_so_far = math.inf
        current_ke = 1
        diff_list = []

        while best_l1_so_far > 0.0001:

            # calculate diff for all queries
            for i, q in enumerate(query_set):

                start = time.time()
                print('     Running Brute Force Query ...')
                query_result_bf = mydb.query_brute_force(query=query_set[0], best_k=best_k)
                bf_time = time.time() - start

                print('Querying #' + str(i) + ' of ' + str(len(query_set)) + '; query = ' + str(q))
                start = time.time()
                print('     Running Genex Query ...')
                query_result_gx = mydb.query(query=q, best_k=best_k, _ke=current_ke)
                gx_time = time.time() - start

                # calculating l1 distance
                for gx_r, bf_r in zip(query_result_gx, query_result_bf):
                    diff_list.append(abs(gx_r[0] - bf_r[0]))

            cur_l1 = np.mean(diff_list)

            if best_k not in result_dict.keys():
                result_dict[best_k] = [[], []]
            print('Current l1 and ke are: ' + str(cur_l1) + '       ' + str(current_ke))
            result_dict[best_k][0].append(cur_l1)
            result_dict[best_k][1].append(current_ke)

            current_ke += mydb.get_max_seq_len() * 0.05  # increment ke
            best_l1_so_far = cur_l1 if cur_l1 < best_l1_so_far else best_l1_so_far  # update bsf

    sc.stop()
    return result_dict


# data_file = 'data/test/ItalyPowerDemand_TEST.csv'
# query_file = 'data/test/ItalyPowerDemand_query.csv'
# result_file = 'results/test/ItalyPowerDemand_result.csv'
# experiment_genex(data_file, query_file, result_file)
# TODO run ECG
# Querying #9 of 15; query = (ECG-1)_(Label-2): (61:118)
#      Running Genex Query ...
data_file = 'data/test/ItalyPowerDemand_TEST.csv'
result_file = 'results/test/ipd/ItalyPowerDemand_result'
result = experiment_genex(data_file, num_sample=40, num_query=40, result_file=result_file)
