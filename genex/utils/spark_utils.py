from pyspark import SparkContext, SparkConf

from genex.op.query_op import get_dist_query
from genex.op.cluster_op import _cluster_groups, _cluster_to_meta, _cluster_reduce_func
from genex.misc import pr_red
from process_utils import _group_time_series


def _create_sc(num_cores: int, driver_mem: int, max_result_mem: int):
    conf = SparkConf(). \
        setMaster("local[" + str(num_cores) + "]"). \
        setAppName("Genex").set('spark.driver.memory', str(driver_mem) + 'G'). \
        set('spark.driver.maxResultSize', str(max_result_mem) + 'G')
    sc = SparkContext(conf=conf)

    return sc


def _pr_spark_conf(sc: SparkContext):
    pr_red('Number of Works:     ' + str(sc.defaultParallelism))
    pr_red('Driver Memory:       ' + sc.getConf().get("spark.driver.memory"))
    pr_red('Maximum Result Size: ' + sc.getConf().get("spark.driver.maxResultSize"))


def _cluster_with_spark(sc: SparkContext, data_normalized, start, end, st, dist_func, verbose):
    # validate and save the loi to gxdb class fields
    # distribute the data_original
    input_rdd = sc.parallelize(data_normalized, numSlices=sc.defaultParallelism)
    # partition_input = input_rdd.glom().collect() #  for debug purposes
    # Grouping the data_original
    # group = _group_time_series(input_rdd.glom().collect()[0], start, end) # for debug purposes
    group_rdd = input_rdd.mapPartitions(
        lambda x: _group_time_series(time_series=x, start=start, end=end), preservesPartitioning=True)
    # group_partition = group_rdd.glom().collect()  # for debug purposes
    # group = group_rdd.collect()  # for debug purposes
    # Cluster the data_original with Gcluster
    # cluster = _cluster_groups(groups=group_rdd.glom().collect()[0], st=similarity_threshold,
    #                           dist_func=dist_func, verbose=1)  # for debug purposes
    cluster_rdd = group_rdd.mapPartitions(lambda x: _cluster_groups(
        groups=x, st=st, dist_func=dist_func, log_level=verbose)).cache()
    # cluster_partition = cluster_rdd.glom().collect()  # for debug purposes
    cluster_rdd.count()
    # Combining two dictionary using **kwargs concept
    cluster_meta_dict = _cluster_to_meta_spark(cluster_rdd)
    return cluster_rdd, cluster_meta_dict


def _cluster_to_meta_spark(cluster_rdd):
    a = cluster_rdd.map(_cluster_to_meta).collect()
    return dict(cluster_rdd.
                map(_cluster_to_meta).
                reduceByKey(_cluster_reduce_func).collect())


def _query_bf_spark(query, sc: SparkContext, data_normalized: list, start, end, dt_index):
    input_rdd = sc.parallelize(data_normalized, numSlices=sc.defaultParallelism)
    group_rdd = input_rdd.mapPartitions(
        lambda x: _group_time_series(time_series=x, start=start, end=end), preservesPartitioning=True)
    slice_rdd = group_rdd.flatMap(lambda x: x[1])
    # for debug purpose
    # a = slice_rdd.collect()
    dist_rdd = slice_rdd.map(lambda x: get_dist_query(query, x, dt_index=dt_index))
    candidate_list = dist_rdd.collect()
    return candidate_list