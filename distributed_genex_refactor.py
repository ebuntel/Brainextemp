import math
import csv

import genex.database.genex_database as gxdb
from genex.preprocess import genex_normalize
from genex.utils import normalize_sequence
import heapq
import time
from genex.cluster import sim_between_seq
import matplotlib.pyplot as plt

fn = 'SART2018_HbO.csv'

from pyspark import SparkContext, SparkConf
num_cores = 8

# conf = SparkConf(). \
#     setMaster("local[" + str(num_cores) + "]"). \
#     setAppName("Genex").set('spark.driver.memory', '15G'). \
#     set('spark.driver.maxResultSize', '15G')

# Setting up the number of the workers automatically based on the amount of the cores in the server
# by setting up the local[*] property
conf_a = SparkConf(). \
    setMaster("local[*]"). \
    setAppName("Genex").set('spark.driver.memory', '8G'). \
    set('spark.driver.maxResultSize', '8G')

sc = SparkContext(conf=conf_a)

# mydb = gxdb.from_csv(fn, feature_num=5, sc=sc)
#
# mydb.data_normalized = mydb.data_normalized[:10]
#
# mydb.build(similarity_threshold=0.1, loi=[200])
# mydb.save('new_build')
# result = mydb.data_normalized_clustered
# # TODO create query here
# query = result
# query_bc = sc.broadcast(query)
# mydb.query(query_bc, best_k=5, threshold=0.1)
# query_r = mydb.query_rdd.collect()
# mydb.save('new_build')
new_db = gxdb.from_db(sc, 'new_build')

from genex.parse import generate_query
from genex.utils import normalize_sequence

query_set = generate_query(file_name='queries.csv', feature_num=5)
df_norm_list = new_db.data_normalized.values.tolist()
df_norm_list = map(lambda x: gxdb._row_to_feature_and_data(x), df_norm_list)
# randomly pick a sequence as the query from the query sequence, make sure the picked sequence is in the input list
# query = next((item for item in query_set if item.id in dict(df_norm_list).keys()), None)
# fetch the data for the query
query = query_set[10]
query.set_data(query.fetch_data(df_norm_list))

from genex.preprocess import genex_normalize
z, global_max, global_min = genex_normalize(new_db.data_normalized, True)
normalize_sequence(query, global_max, global_min, z_normalize=True)
query_bc = sc.broadcast(query)

query_r = new_db.query(query_bc, best_k=5, unique_id=True, overlap=1.0)
print(query_r)
