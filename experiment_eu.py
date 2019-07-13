from genex.parse import generate_source
from genex.preprocess import do_gcluster
from pyspark import SparkContext, SparkConf

import datetime

fn = 'SART2018_HbO.csv'

res_list = generate_source(fn, feature_num=5)

# initialize the spark context
conf = SparkConf().\
    setMaster("local[*]").\
    setAppName("Genex").set('spark.driver.memory', '15G').\
    set('spark.driver.maxResultSize', '15G') # .set('spark.driver.memory', '16G')

sc = SparkContext(conf=conf)

eu_50_62_start_time = datetime.datetime.now()
c_eu_50_62 = do_gcluster(input_list=res_list, loi=[50, 62], sc=sc, similarity_threshold=0.1, del_data=True, is_collect=True, dist_type='eu',
                data_slices=1024, log_level=1)
eu_50_62_end_time = datetime.datetime.now()