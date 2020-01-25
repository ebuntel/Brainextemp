import csv
import math
import random
import time

import genex.database.genexengine as gxdb
from pyspark import SparkContext, SparkConf
from genex.parse import generate_query

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import findspark
import os


# spark_location = '/Users/Leo/spark-2.4.3-bin-hadoop2.7' # Set your own
# java8_location = '/Library/Java/JavaVirtualMachines/jdk1.8.0_151.jdk/Contents/Home/jre'
# os.environ['JAVA_HOME'] = java8_location
# findspark.init(spark_home=spark_location)

# create the spark context
def experiment_genex_ke(data_file, num_sample, num_query, best_k, feature_num, add_uuid, dist_type):
    num_cores = 32
    conf = SparkConf(). \
        setMaster("local[" + str(num_cores) + "]"). \
        setAppName("Genex").set('spark.driver.memory', '64G'). \
        set('spark.driver.maxResultSize', '64G')
    sc = SparkContext(conf=conf)

    # create gxdb from a csv file

    # set up where to save the results

    print('Performing clustering ...')
    mydb = gxdb.from_csv(data_file, sc=sc, feature_num=feature_num, add_uuid=add_uuid,
                         _rows_to_consider=num_sample)

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

    # perform clustering
    cluster_start_time = time.time()
    mydb.build(st=0.1, dist_type=dist_type)
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
    current_ke = best_k

    while best_l1_so_far > 0.0001 and current_ke < mydb.get_num_subsequences():
        diff_list = []
        # calculate diff for all queries
        for i, q in enumerate(query_set):
            print(
                'dist_type: ' + dist_type + '. Best k = ' + str(best_k) + '- Querying #' + str(i) + ' of ' + str(len(query_set)) + '; query = ' + str(
                    q))
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

        current_ke = int(current_ke + mydb.get_num_subsequences() * 0.01)  # increment ke
        # break the loop if current_ke exceeds the number of subsequences
        if current_ke > mydb.get_num_subsequences():
            break
        best_l1_so_far = cur_l1 if cur_l1 < best_l1_so_far else best_l1_so_far  # update bsf

    timing_dict['gx query time'] = np.mean(gx_timing_list)
    sc.stop()
    return l1_ke_list, gx_timing_list, bf_time_list, timing_dict,


def harvest_ke_multiple_k(k_to_test, experiment_set):
    result_dict = dict()

    for dataset_name, config in experiment_set.items():
        ke_result_dict = dict()
        l1_ke_k_array = list()
        fig, ax = plt.subplots()
        for k in k_to_test:
            ke_result_dict[k] = experiment_genex_ke(config['data_original'], num_sample=40, num_query=40, best_k=k,
                                                    add_uuid=config['add_uuid'], feature_num=config['feature_num'],
                                                    dist_type=config['dist_type'])

            a = np.transpose(ke_result_dict[k][0])
            b = np.expand_dims([k for i in range(len(a))], axis=1)
            c = np.concatenate([a, b], axis=1)
            l1_ke_k_array.append(c)
            ax.plot(ke_result_dict[k][0][1], ke_result_dict[k][0][0], label='k=' + str(k), marker='o')
        ax.set_ylabel('RMSE')
        ax.set_xlabel('Ke')
        ax.set_title(dataset_name + ' l1-ke')
        ax.legend()
        plt.show()
        result_dict[dataset_name] = [ke_result_dict, l1_ke_k_array]

    return result_dict


k_to_test = [50, 15, 9, 1]
experiment_set_ma = {
    'italyPowerDemand': {'data_original': 'data_original/ItalyPower.csv',
                         'feature_num': 2,
                         'add_uuid': False,
                         'dist_type': 'ma'},

    'ecgFiveDays': {'data_original': 'data_original/ECGFiveDays.csv',
                    'feature_num': 2,
                    'add_uuid': False,
                    'dist_type': 'ma'},

    'Gun_Point_TRAIN': {'data_original': 'data_original/Gun_Point_TRAIN.csv',
                        'feature_num': 1,
                        'add_uuid': True,
                        'dist_type': 'ma'},
    'synthetic_control_TRAIN': {'data_original': 'data_original/synthetic_control_TRAIN.csv',
                                'feature_num': 1,
                                'add_uuid': True,
                                'dist_type': 'ma'},
}

result_dict = harvest_ke_multiple_k(k_to_test, experiment_set_ma)
