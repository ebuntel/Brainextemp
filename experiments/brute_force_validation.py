import random
import time

import genex.database.genex_database as gxdb
from pyspark import SparkContext, SparkConf
import pandas as pd
import csv
import itertools


def validate_brute_force(data_file_path, len_to_test, feature_num, rows_to_consider=None, sample_per_length=1, seed=0):
    random.seed(seed)

    num_cores = 12
    conf = SparkConf(). \
        setMaster("local[" + str(num_cores) + "]"). \
        setAppName("Genex").set('spark.driver.memory', '31G'). \
        set('spark.driver.maxResultSize', '31G')
    sc = SparkContext(conf=conf)

    # create gxdb from a csv file
    mydb = gxdb.from_csv(data_file_path, sc=sc, feature_num=feature_num, _rows_to_consider=rows_to_consider)
    mydb.build(similarity_threshold=0.1, _is_cluster=False)  # just set the build parameters without actually clustering

    query_bf_results = []

    for testing_len in len_to_test:
        for i in range(sample_per_length):
            q = mydb.get_random_seq_of_len(testing_len)

            start = time.time()
            result = mydb.query_brute_force(query=q, best_k=5)
            result.insert(0, (q, time.time() - start))
            query_bf_results.append(list(result))

    sc.stop()
    return query_bf_results


'''
Validation using SART
# dataset_name = 'SART2018_HbO'
# len_to_test = [1, 8, 16, 24]
'''

'''
Validation using Italy Power
# dataset_name = 'ItalyPower'
# len_to_test = [1, 8, 16, 24]

'''

'''
Validation using ECGFiveDays

'''
dataset_name = 'ECGFiveDays'
len_to_test = [4, 16, 64, 136]

sample_per_length = 16

if __name__ == '__main__':
    data_file = 'data/' + dataset_name + '.csv'
    validation_result = validate_brute_force(data_file_path=data_file, feature_num=2, len_to_test=len_to_test,
                                             sample_per_length=sample_per_length)

    flattened = list(itertools.chain(*validation_result))
    flattened = [list(x) for x in flattened]

    pd.DataFrame(flattened).to_csv('results/bf_validation_' + dataset_name + '.csv')
