
from genex.parse import generate_source
fn = '/home/apocalyvec/PycharmProjects/Genex/SART2018_HbO.csv'
input_list = generate_source(fn, feature_num=5)
input_list = input_list[:64]


from genex.preprocess import min_max_normalize
normalized_input_list, global_max, global_min = min_max_normalize(input_list)

from pyspark import SparkContext, SparkConf

num_cores = 32

conf = SparkConf(). \
    setMaster("local[" + str(num_cores) + "]"). \
    setAppName("Genex").set('spark.driver.memory', '31G'). \
    set('spark.driver.maxResultSize', '31G')
sc = SparkContext(conf=conf)

input_rdd = sc.parallelize(normalized_input_list, numSlices= num_cores)
partition_input = input_rdd.glom().collect()


from genex.preprocess import all_sublists_with_id_length
group_rdd = input_rdd.flatMap(
    lambda x: all_sublists_with_id_length(x, [120]))
partition_group = group_rdd.glom().collect()

from genex.cluster import filter_cluster
import pyspark
# first_partition = filter_cluster(partition_group[0], [120, 274], 0.1, log_level=1)

cluster_rdd = group_rdd.mapPartitions(lambda x: filter_cluster(groups=x, st=0.05, log_level=1), preservesPartitioning=False).persist(storageLevel=pyspark.StorageLevel.MEMORY_ONLY)
cluster_partition = cluster_rdd.glom().collect()