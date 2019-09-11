import math
import csv

import genex.database.genex_database as gxdb
from genex.preprocess import min_max_normalize
from genex.utils import normalize_sequence
import heapq
import time
from genex.cluster import sim_between_seq
import matplotlib.pyplot as plt

fn = 'SART2018_HbO.csv'

mydb = gxdb.from_csv(fn, feature_num=5)

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

mydb.data_normalized = mydb.data_normalized[:10]

mydb.build(sc=sc, similarity_threshold=0.1)
mydb.save(folder_name='test_db')

print()
