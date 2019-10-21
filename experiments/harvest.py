import csv
import time

import genex.database.genex_database as gxdb
from pyspark import SparkContext, SparkConf
from genex.parse import generate_query

import numpy as np


# create the spark context
num_cores = 32
conf = SparkConf(). \
    setMaster("local[" + str(num_cores) + "]"). \
    setAppName("Genex").set('spark.driver.memory', '64G'). \
    set('spark.driver.maxResultSize', '64G')
sc = SparkContext(conf=conf)

# create gxdb from a csv file
data_file = 'ECGFiveDays.csv'
db_path = 'gxdb/test_db'
mydb = gxdb.from_csv(data_file, sc=sc, feature_num=2)
mydb.build(similarity_threshold=0.1, loi=slice(110, 135))

# generate the query sets
query_set = generate_query(file_name='ECG_Queries_set.csv', feature_num=2)
# randomly pick a sequence as the query from the query sequence, make sure the picked sequence is in the input list
# this query'id must exist in the database
time_query_bf = []
time_query_gx = []
relative_error_list = []
accuracy_list = []

for i, q in enumerate(query_set):
    start = time.time()
    lst= []
    print(q)
    lst.append(q)
    query_result = mydb.query(query=q, best_k=5)
    time_query_gx.append(time.time() - start)

    #start = time.time()
    print("Running brute force")
    query_result_bf = mydb.query_brute_force(query=query_set[0], best_k=5)
    #print(query_result_bf)
    lst.append(query_result_bf)
    time_query_bf.append(time.time() - start)
    with open('results_bf.csv','a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(lst)
    csvfile.close()
    # relative_errors = [abs(x[0]-y[0]) for x, y in zip(query_result, query_result_bf)]
    # accuracies = [abs(x[0]-y[0])/y[0] for x, y in zip(query_result, query_result_bf)]

    #relative_error_list.append(np.mean(relative_errors))
    #accuracy_list.append(np.mean(accuracies))

# TODO memory optimization: brainstorm memory optimization, encode features (ids), length batches
