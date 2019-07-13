from genex.parse import generate_source
from genex.preprocess import do_gcluster
from pyspark import SparkContext, SparkConf

import datetime

fn = 'SART2018_HbO.csv'
dist = 'eu'

res_list = generate_source(fn, feature_num=5)

# initialize the spark context
conf = SparkConf().\
    setMaster("local[*]").\
    setAppName("Genex").set('spark.driver.memory', '31G').\
    set('spark.driver.maxResultSize', '31G')

sc = SparkContext(conf=conf)

c_50_100_start_time = datetime.datetime.now()
c_50_100 = do_gcluster(input_list=res_list, loi=[50, 62], sc=sc, similarity_threshold=0.1, del_data=True, is_collect=True, dist_type=dist,
                data_slices=1024, log_level=1)
c_50_100_end_time = datetime.datetime.now()