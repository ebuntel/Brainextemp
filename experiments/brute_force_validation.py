import time

import genex.database.genex_database as gxdb
from pyspark import SparkContext, SparkConf


def validate_brute_force(data_file_path, len_to_test, feature_num, rows_to_consider=None):
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
        q = mydb.get_random_seq_of_len(testing_len)

        start = time.time()
        result = mydb.query_brute_force(query=q, best_k=5)
        result.insert(0, time.time() - start)
        query_bf_results.append(result)
    sc.stop()
    return query_bf_results

validation_result = []

'''
Validation using SART
'''
# data_file = 'data/SART2018_HbO.csv'
# validation_result.append(validate_brute_force(data_file_path=data_file, rows_to_consider=[0, 12], feature_num=5, len_to_test=[4, 16, 64, 256]))

'''
Validation using Italy Power
'''
data_file = 'data/ItalyPower.csv'
validation_result.append(validate_brute_force(data_file_path=data_file, feature_num=2, len_to_test=[1, 8, 16, 24]))

