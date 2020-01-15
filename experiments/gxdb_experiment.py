import os
import time

import findspark

import genex.database.genex_database as gxdb
from pyspark import SparkContext, SparkConf
import matplotlib.pyplot as plt


spark_location = '/Users/Leo/spark-2.4.3-bin-hadoop2.7' # Set your own
java8_location = '/Library/Java/JavaVirtualMachines/jdk1.8.0_151.jdk/Contents/Home/jre'
os.environ['JAVA_HOME'] = java8_location
findspark.init(spark_home=spark_location)

# create the spark context
num_cores = 16
conf = SparkConf(). \
    setMaster("local[" + str(num_cores) + "]"). \
    setAppName("Genex").set('spark.driver.memory', '32G'). \
    set('spark.driver.maxResultSize', '32G')
sc = SparkContext(conf=conf)

# create gxdb from a csv file
data_file = 'data/ItalyPower.csv'
db_path = 'results/test_db'

mydb = gxdb.from_csv(data_file, sc=sc, feature_num=2)
# mydb.save(path=db_path)
# del mydb  # test saving before building
#
# mydb = gxdb.from_db(path=db_path, sc=sc)
mydb.build(similarity_threshold=0.1)

# mydb.save(path=db_path)
# del mydb  # test saving after building
#
# mydb = gxdb.from_db(path=db_path, sc=sc)

# generate the query
q = mydb.get_random_seq_of_len(20, seed=1)

start = time.time()
query_result = mydb.query(query=q, best_k=5)
duration = time.time() - start
# TODO memory optimization: brainstorm memory optimization, encode features (ids), length batches

# plt.plot(q.fetch_data(mydb.data_normalized), label='query', linewidth=5)
# for qr in query_result:
#     plt.plot(qr[1].data, label=str(qr[0]))
# plt.legend()
# plt.show()
#
# start = time.time()
# query_result_bf = mydb.query_brute_force(query=q, best_k=5)
# duration = time.time() - start
#
# # plot the bf result
# plt.plot(q.fetch_data(mydb.data_normalized), label='query', linewidth=5)
# for qr in query_result_bf:
#     plt.plot(qr[1].data, label=str(qr[0]))
# plt.legend()
# plt.show()
