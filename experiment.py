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
    setAppName("Genex").set('spark.driver.memory', '64G').\
    set('spark.driver.maxResultSize', '64G')

sc = SparkContext(conf=conf)

# Cluster of length 211 to 274
c_211_274_start_time = datetime.datetime.now()
c_211_274 = do_gcluster(input_list=res_list, loi=[211, 274], sc=sc, similarity_threshold=0.1, del_data=True, is_collect=True, dist_type=dist,
                data_slices=1024, log_level=1)
c_211_274_end_time = datetime.datetime.now()

# Cluster of length 147 to 210
c_147_210_start_time = datetime.datetime.now()
c_147_210 = do_gcluster(input_list=res_list, loi=[147, 210], sc=sc, similarity_threshold=0.1, del_data=True, is_collect=True, dist_type=dist,
                data_slices=1024, log_level=1)
c_147_210_end_time = datetime.datetime.now()

# Cluster of length 83 to 146
c_83_146_start_time = datetime.datetime.now()
c_83_146 = do_gcluster(input_list=res_list, loi=[83, 146], sc=sc, similarity_threshold=0.1, del_data=True, is_collect=True, dist_type=dist,
                data_slices=1024, log_level=1)
c_83_146_end_time = datetime.datetime.now()

# Cluster of length 19 to 82
c_19_82_start_time = datetime.datetime.now()
c_19_82 = do_gcluster(input_list=res_list, loi=[19, 82], sc=sc, similarity_threshold=0.1, del_data=True, is_collect=True, dist_type=dist,
                data_slices=1024, log_level=1)
c_19_82_end_time = datetime.datetime.now()