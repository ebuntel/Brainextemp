from genex.parse import generate_source
from genex.preprocess import do_gcluster
from pyspark import SparkContext, SparkConf

import datetime

fn = 'SART2018_HbO.csv'

res_list = generate_source(fn, feature_num=5)

# initialize the spark context
conf = SparkConf(). \
    setMaster("local[*]"). \
    setAppName("Genex").set('spark.driver.memory', '32G'). \
    set('spark.driver.maxResultSize', '32G')

sc = SparkContext(conf=conf)

# Clustering with Euclidean distance ########################################################################
eu_dist = 'eu'

c_211_274_eu_start_time = datetime.datetime.now()
c_211_274_eu = do_gcluster(input_list=res_list, loi=[211, 274], sc=sc, similarity_threshold=0.1, del_data=True,
                           is_collect=True, dist_type=eu_dist,
                           data_slices=1024, log_level=1)
c_211_274_eu_end_time = datetime.datetime.now()

# Cluster of length 147 to 210
c_147_210_eu_start_time = datetime.datetime.now()
c_147_210_eu = do_gcluster(input_list=res_list, loi=[147, 210], sc=sc, similarity_threshold=0.1, del_data=True,
                           is_collect=True, dist_type=eu_dist,
                           data_slices=1024, log_level=1)
c_147_210_eu_end_time = datetime.datetime.now()

# Cluster of length 83 to 146
c_83_146_eu_start_time = datetime.datetime.now()
c_83_146_eu = do_gcluster(input_list=res_list, loi=[83, 146], sc=sc, similarity_threshold=0.1, del_data=True,
                          is_collect=True, dist_type=eu_dist,
                          data_slices=1024, log_level=1)
c_83_146_eu_end_time = datetime.datetime.now()

# Cluster of length 19 to 82
c_19_82_eu_start_time = datetime.datetime.now()
c_19_82_eu = do_gcluster(input_list=res_list, loi=[19, 82], sc=sc, similarity_threshold=0.1, del_data=True,
                         is_collect=True, dist_type=eu_dist,
                         data_slices=1024, log_level=1)
c_19_82_eu_end_time = datetime.datetime.now()

# Clustering with Manhattan distance ########################################################################
ma_dist = 'ma'

c_211_274_ma_start_time = datetime.datetime.now()
c_211_274_ma = do_gcluster(input_list=res_list, loi=[211, 274], sc=sc, similarity_threshold=0.1, del_data=True,
                           is_collect=True, dist_type=ma_dist,
                           data_slices=1024, log_level=1)
c_211_274_ma_end_time = datetime.datetime.now()

# Cluster of length 147 to 210
c_147_210_ma_start_time = datetime.datetime.now()
c_147_210_ma = do_gcluster(input_list=res_list, loi=[147, 210], sc=sc, similarity_threshold=0.1, del_data=True,
                           is_collect=True, dist_type=ma_dist,
                           data_slices=1024, log_level=1)
c_147_210_ma_end_time = datetime.datetime.now()

# Cluster of length 83 to 146
c_83_146_ma_start_time = datetime.datetime.now()
c_83_146_ma = do_gcluster(input_list=res_list, loi=[83, 146], sc=sc, similarity_threshold=0.1, del_data=True,
                          is_collect=True, dist_type=ma_dist,
                          data_slices=1024, log_level=1)
c_83_146_ma_end_time = datetime.datetime.now()

# Cluster of length 19 to 82
c_19_82_ma_start_time = datetime.datetime.now()
c_19_82_ma = do_gcluster(input_list=res_list, loi=[19, 82], sc=sc, similarity_threshold=0.1, del_data=True,
                         is_collect=True, dist_type=ma_dist,
                         data_slices=1024, log_level=1)
c_19_82_ma_end_time = datetime.datetime.now()

# Clustering with Chebyshev distance ########################################################################
ch_dist = 'ch'

c_211_274_ch_start_time = datetime.datetime.now()
c_211_274_ch = do_gcluster(input_list=res_list, loi=[211, 274], sc=sc, similarity_threshold=0.1, del_data=True,
                           is_collect=True, dist_type=ch_dist,
                           data_slices=1024, log_level=1)
c_211_274_ch_end_time = datetime.datetime.now()

# Cluster of length 147 to 210
c_147_210_ch_start_time = datetime.datetime.now()
c_147_210_ch = do_gcluster(input_list=res_list, loi=[147, 210], sc=sc, similarity_threshold=0.1, del_data=True,
                           is_collect=True, dist_type=ch_dist,
                           data_slices=1024, log_level=1)
c_147_210_ch_end_time = datetime.datetime.now()

# Cluster of length 83 to 146
c_83_146_ch_start_time = datetime.datetime.now()
c_83_146_ch = do_gcluster(input_list=res_list, loi=[83, 146], sc=sc, similarity_threshold=0.1, del_data=True,
                          is_collect=True, dist_type=ch_dist,
                          data_slices=1024, log_level=1)
c_83_146_ch_end_time = datetime.datetime.now()

# Cluster of length 19 to 82
c_19_82_ch_start_time = datetime.datetime.now()
c_19_82_ch = do_gcluster(input_list=res_list, loi=[19, 82], sc=sc, similarity_threshold=0.1, del_data=True,
                         is_collect=True, dist_type=ch_dist,
                         data_slices=1024, log_level=1)
c_19_82_ch_end_time = datetime.datetime.now()

# Query ###################################################################################################
from genex.parse import generate_query

query_set = generate_query(file_name='queries.csv', feature_num=5)
query_results_ch = {}

for query in query_set:
    query_results_ch[query] = query_result = c.gquery(query,
                                                      sc=sc, loi=[147, 274],
                                                      k=5, dist_type='ch',
                                                      data_slices=2048,
                                                      ex_sameID=True,
                                                      overlap=0.75)

# Count the labels ########################################################################################
