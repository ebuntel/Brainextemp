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
def experiment_genex_ke(data_file, num_sample, num_query, best_k, feature_num, add_uuid):
    num_cores = 12
    conf = SparkConf(). \
        setMaster("local[" + str(num_cores) + "]"). \
        setAppName("Genex").set('spark.driver.memory', '32G'). \
        set('spark.driver.maxResultSize', '32G')
    sc = SparkContext(conf=conf)

    # create gxdb from a csv file

    # set up where to save the results

    print('Performing clustering ...')
    mydb = gxdb.from_csv(data_file, sc=sc, feature_num=feature_num, add_uuid=add_uuid, _rows_to_consider=num_sample)

    print('Generating query of max seq len ...')
    # generate the query sets
    query_set = list()
    # get the number of subsequences
    # randomly pick a sequence as the query from the query sequence, make sure the picked sequence is in the input list
    # this query'id must exist in the database
    for i in range(num_query):
        query_set.append(mydb.get_random_seq_of_len(mydb.get_max_seq_len(), seed=i))

    best_l1_so_far = math.inf
    l1_ke_list = [[], []]
    timing_dict = dict()
    current_ke = 1

    # perform clustering
    cluster_start_time = time.time()
    mydb.build(similarity_threshold=0.1)
    timing_dict['cluster time'] = time.time() - cluster_start_time

    bf_result_dict = dict()
    bf_time_list = list()
    for i, q in enumerate(query_set):
        print('Brute Force Querying #' + str(i) + ' of ' + str(len(query_set)) + '; query = ' + str(q))
        start = time.time()
        query_result_bf = mydb.query_brute_force(query=q, best_k=best_k)
        bf_time_list.append(time.time() - start)
        bf_result_dict[q] = query_result_bf
    timing_dict['bf query time'] = np.mean(bf_time_list)

    print('Running Genex Query ...')
    gx_timing_list = list()
    while best_l1_so_far > 0.0001:
        diff_list = []
        # calculate diff for all queries
        for i, q in enumerate(query_set):
            print('Querying #' + str(i) + ' of ' + str(len(query_set)) + '; query = ' + str(q))
            start = time.time()
            query_result_gx = mydb.query(query=q, best_k=best_k, _ke=current_ke)
            gx_timing_list.append(time.time() - start)

            # calculating l1 distance
            for gx_r, bf_r in zip(query_result_gx, bf_result_dict[q]):  # retrive bf result from the result dict
                diff_list.append(abs(gx_r[0] - bf_r[0]))

        print('Diff list is ' + str(diff_list))
        cur_l1 = np.mean(diff_list)

        print('Current l1 and ke are: ' + str(cur_l1) + '       ' + str(current_ke))
        l1_ke_list[0].append(cur_l1)
        l1_ke_list[1].append(current_ke)

        current_ke = int(current_ke + mydb.get_num_subsequences() * 0.05)  # increment ke
        # break the loop if current_ke exceeds the number of subsequences
        if current_ke > mydb.get_num_subsequences():
            break
        best_l1_so_far = cur_l1 if cur_l1 < best_l1_so_far else best_l1_so_far  # update bsf

    timing_dict['gx query time'] = np.mean(gx_timing_list)
    sc.stop()
    return l1_ke_list, timing_dict


# data_file = 'data/test/ItalyPowerDemand_TEST.csv'
# query_file = 'data/test/ItalyPowerDemand_query.csv'
# result_file = 'results/test/ItalyPowerDemand_result_regular.csv'
# experiment_genex(data_file, query_file, result_file)


data_file = 'data/ItalyPower.csv'
result_file = 'results/ipd/ItalyPowerDemand_result'
feature_num = 2
add_uuid = False

k_to_test = [15, 9, 1]
result_dict = dict()
for k in k_to_test:
    result_dict[k] = experiment_genex_ke(data_file, num_sample=40, num_query=40, best_k=k, add_uuid=add_uuid, feature_num=feature_num)
