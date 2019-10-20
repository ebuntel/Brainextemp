import time

import genex.database.genex_database as gxdb
from pyspark import SparkContext, SparkConf
from genex.parse import generate_query

import numpy as np


# create the spark context
def validate_brute_force(data_file_path, query_file_path):
    num_cores = 32
    conf = SparkConf(). \
        setMaster("local[" + str(num_cores) + "]"). \
        setAppName("Genex").set('spark.driver.memory', '64G'). \
        set('spark.driver.maxResultSize', '64G')
    sc = SparkContext(conf=conf)

    # create gxdb from a csv file
    data_file = '/home/apocalyvec/PycharmProjects/Genex/ECGFiveDays.csv'
    query_set = generate_query(file_name='/home/apocalyvec/PycharmProjects/Genex/ECG_Queries_set.csv', feature_num=2)

    mydb = gxdb.from_csv(data_file, sc=sc, feature_num=2)
    mydb.build(similarity_threshold=0.1, _is_cluster=False)

    query_bf_results = []

    for i, q in enumerate(query_set):
        start = time.time()
        result = mydb.query_brute_force(query=q, best_k=1)
        query_bf_results.append(result.insert(0, time.time() - start))
    sc.stop()
    return query_bf_results

data_file = '/home/apocalyvec/PycharmProjects/Genex/ECGFiveDays.csv'
query_set = generate_query(file_name='/home/apocalyvec/PycharmProjects/Genex/ECG_Queries_set_reduced.csv', feature_num=2)

validate_brute_force(data_file_path=data_file, query_file_path=query_set)