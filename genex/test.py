import os

from genex.parse import generate_source
from genex.preprocess import do_gcluster
from pyspark import SparkContext, SparkConf
import numpy as np
def normalize_num(num, max, min):
    return (num - min) / (max - min)
def flatten(l):
    flat_list = []
    for sublist in l:
        for item in sublist:
            flat_list.append(item)
    return flat_list
def _normalize(input_list):
    # scaler = MinMaxScaler(feature_range=(0, 1))
    #
    #
    #
    # input_array = np.array(list(map(lambda x: x[1], input_list)), dtype=np.float64)
    #
    # input_array_scaled = scaler.fit_transform(input_array)

    flattened_list = np.array(flatten(list(map(lambda x: x[1], input_list))))
    global_max = flattened_list.max()
    global_min = flattened_list.min()

    normalize_list = map(lambda id_sequence:
                         [id_sequence[0], list(map(lambda num: normalize_num(num, global_max, global_min), id_sequence[1]))]
                         , input_list)


    return list(normalize_list), global_min, global_max



# fn = 'SART2018.csv'
from genex.query import get_most_k_sim, get_query_from_sequence, get_query_from_csv_with_id, gquery

query_id = ('2013e_001','102 2-Back','A-DC8','232876359.3750','232906359.3750')

fn = '2013e_001_2_channels_02backs.csv'
fn1 = 'SART2018_1-100.csv'
# can be optimized here
res_list = generate_source(fn, feature_num=5)
res_list, global_min, global_max = _normalize(res_list)
res_list2 = generate_source(fn1, feature_num=5)
# not worked before
sequence = get_query_from_sequence(query_id, 25, 76, res_list)
# problem: can we use normalized data to calculate dtw?
# good way to normalize?
# also the object can not be distributed
query_list = []
query_set = get_query_from_csv_with_id('query_example.csv')
for cur_query in query_set:
    cur_sequence = [normalize_num(num, global_max, global_min) for num in get_query_from_sequence(cur_query[0], int(cur_query[1]), int(cur_query[2]), res_list)]
    query_list.append(cur_sequence)
# initialize the spark context
conf = SparkConf().setMaster("local[*]").setAppName("Genex").set('spark.driver.memory', '16G')
sc = SparkContext(conf=conf)
# conf2 = SparkConf().setMaster("local").setAppName("Genex2").set('spark.driver.memory', '2G').set('spark.driver.allowMultipleContexts', 'true')
#
# sc2 = SparkContext(conf=conf)
query_sequence = res_list[0][1][20:90]
cluster_result = do_gcluster(input_list=res_list, loi=[50, 58], sc=sc, del_data=False,isCollect=False)
cluster_result_col = cluster_result.data.collect()

query_csv_res = cluster_result.data.map(lambda clusters: gquery(query_list,clusters, [50,51], res_list, 2,
                                                               )).collect()
query_res = cluster_result.data.map(lambda clusters: get_most_k_sim(sequence, [50,51], clusters, 2,
                                                               res_list)).collect()
print("")