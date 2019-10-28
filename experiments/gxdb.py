import random

import genex.database.genex_database as gxdb
from pyspark import SparkContext, SparkConf

# create the spark context
num_cores = 12
conf = SparkConf(). \
    setMaster("local[" + str(num_cores) + "]"). \
    setAppName("Genex").set('spark.driver.memory', '31G'). \
    set('spark.driver.maxResultSize', '31G')
sc = SparkContext(conf=conf)

# create gxdb from a csv file
data_file = 'data/ECGFiveDays.csv'
db_path = 'results/test_db'

mydb = gxdb.from_csv(data_file, sc=sc, feature_num=2)
mydb.save(path=db_path)
del mydb  # test saving before building

mydb = gxdb.from_db(path=db_path, sc=sc)
mydb.build(similarity_threshold=0.01, loi=slice(110, 135))
mydb.save(path=db_path)
del mydb  # test saving after building

mydb = gxdb.from_db(path=db_path, sc=sc)

# generate the query sets
random.seed(0)
q = mydb.get_random_seq_of_len(120)

query_result = mydb.query(query=q, best_k=5)

# TODO memory optimization: brainstorm memory optimization, encode features (ids), length batches
