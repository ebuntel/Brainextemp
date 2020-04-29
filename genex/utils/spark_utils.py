from pyspark import SparkContext, SparkConf
from tslearn.piecewise import PiecewiseAggregateApproximation

from genex.op.query_op import _get_dist_query, _get_dist_paa
from genex.op.cluster_op import _build_clusters, _cluster_to_meta, _cluster_reduce_func, _build_clusters_dynamic
from genex.misc import pr_red
from genex.utils.process_utils import _group_time_series, dss
from genex.utils.ts_utils import paa_compress
from genex.utils.utils import flatten


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


def _cluster_with_spark(sc: SparkContext, data_normalized, data_normalized_bc,
                        start, end, st, dist_func, pnorm, verbose, group_only, use_dss, _use_dynamic):
    # validate and save the loi to gxdb class fields
    parallelism = sc.defaultParallelism
    # if False:
    if len(data_normalized) == 1 and len(data_normalized[0][1]) > parallelism and use_dss:
        print('_cluster_with_spark: Using DSS')
        # series, we use step-distribution-grouping
        groups = [i for i in range(parallelism)]
        group_rdd = sc.parallelize(groups, numSlices=parallelism)

        # ss = []
        # group_partition = group_rdd.glom().collect()  # for debug purposes
        # for gp in group_partition:
        #     g = _sdg(gp, data_normalized_bc, start, end, parallelism)  # for debug purposes
        #     ss = ss + (flatten([x[1] for x in g]))
        group_rdd = group_rdd.mapPartitions(
            lambda x: dss(x, data_normalized_bc, start, end, parallelism), preservesPartitioning=True).cache()
    else:
        # distribute the data_original
        input_rdd = sc.parallelize(data_normalized, numSlices=parallelism)
        # partition_input = input_rdd.glom().collect()  # for debug purposes
        # Grouping the data_original
        # group = _group_time_series(input_rdd.glom().collect()[0], start, end)  # for debug purposes
        group_rdd = input_rdd.mapPartitions(
            lambda x: _group_time_series(time_series=x, start=start, end=end), preservesPartitioning=True).cache()

    subsequence_rdd = group_rdd.flatMap(lambda x: x[1]).cache()

    # group_partition = group_rdd.glom().collect()  # for debug purposes
    # group = group_rdd.collect()  # for debug purposes
    # all_ss = flatten([x[1] for x in group])

    # Cluster the data_original with Gcluster
    # cluster = _build_clusters(groups=group_rdd.glom().collect()[0], st=similarity_threshold,
    #                           dist_func=dist_func, verbose=1)  # for debug purposes
    if group_only:
        return subsequence_rdd, None, None
    if _use_dynamic:
        cluster_rdd = group_rdd.mapPartitions(lambda x: _build_clusters_dynamic(
            groups=x, st=st, dist_func=dist_func, data_list=data_normalized, log_level=verbose)).cache()
    else:
        cluster_rdd = group_rdd.mapPartitions(lambda x: _build_clusters(
            groups=x, st=st, dist_func=dist_func, data_list=data_normalized, log_level=verbose)).cache()
        # cluster_partition = cluster_rdd.glom().collect()  # for debug purposes
    cluster_rdd.count()

    # Combining two dictionary using **kwargs concept
    cluster_meta_dict = _cluster_to_meta_spark(cluster_rdd)
    return subsequence_rdd, cluster_rdd, cluster_meta_dict


def _cluster_to_meta_spark(cluster_rdd):
    a = cluster_rdd.map(_cluster_to_meta).collect()
    return dict(cluster_rdd.
                map(_cluster_to_meta).
                reduceByKey(_cluster_reduce_func).collect())


def _query_bf_spark(query, subsequence_rdd, dt_index, data_list):
    pp_rdd = subsequence_rdd.map(
        lambda x: _get_dist_query(query, x, dt_index=dt_index, data_list=data_list.value))
    candidate_list = pp_rdd.collect()
    # clear data stored in the candidate list

    return candidate_list


def _query_paa_spark(query, ss_paaKv_rdd, dt_index):
    pp_rdd = ss_paaKv_rdd.map(
        lambda x: (_get_dist_paa(query, x[1], dt_index=dt_index), x[0]))
    candidate_list = pp_rdd.collect()
    return candidate_list


def _broadcast_kwargs(sc: SparkContext, kwargs_dict):
    """
    return the broadcast version of the kwargs values
    :param kwargs_dict:
    """
    rtn = dict(((key, sc.broadcast(value=value)) for key, value in kwargs_dict))
    return rtn


def _destory_kwarg_bc(kwargs_dict: dict):
    """
    destroy the values in the kwarg dictionary
    :param kwargs_dict:
    """
    [value.destroy() for key, value in kwargs_dict]


def _build_paa_spark(subsequences_rdd, paa_c, data_list, _dummy_slicing, _sc: SparkContext = None, _start=None,
                     _end=None):
    if _dummy_slicing:
        print('Simulating Slicing Time')
        parallelism = _sc.defaultParallelism
        input_rdd = _sc.parallelize(data_list.value, numSlices=parallelism)
        # partition_input = input_rdd.glom().collect()  # for debug purposes
        # Grouping the data_original
        # group = _group_time_series(input_rdd.glom().collect()[0], start, end)  # for debug purposes
        group_rdd = input_rdd.mapPartitions(
            lambda x: _group_time_series(time_series=x, start=_start, end=_end), preservesPartitioning=True).cache()
        subsequence_rdd = group_rdd.flatMap(lambda x: x[1]).cache()
        subsequence_rdd.count()
        subsequence_rdd.unpersist()  # remove the dummy subsequence to free resources
        del subsequence_rdd

    ss_paaKv_rdd = subsequences_rdd.map(
        lambda x: (x, paa_compress(x.fetch_data(data_list.value), paa_c))).cache()

    ss_paaKv_rdd.count()
    return ss_paaKv_rdd
