from pyspark import SparkContext, SparkConf
from tslearn.piecewise import PiecewiseAggregateApproximation

from genex.op.query_op import _get_dist_sequence, _get_dist_array, _get_dist_sequence_piecewise
from genex.op.cluster_op import _build_clusters, _cluster_to_meta, _cluster_reduce_func, _build_clusters_dynamic
from genex.misc import pr_red
from genex.utils.process_utils import _group_time_series, dss, dss_multiple
from genex.utils.ts_utils import paa_compress, sax_compress
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

    if use_dss:
        print('_cluster_with_spark: Using Generalized DSS')
        input = [i for i in range(parallelism)]  # creates parallelized slicing indices
        input_rdd = sc.parallelize(input, numSlices=parallelism)

        # a = []# debug
        # input_partitions = input_rdd.glom().collect()  # debug
        # for ip in input_partitions:# debug
        #     a.append(dss_multiple(ip, data_normalized_bc.value, start, end, parallelism))# debug

        group_rdd = input_rdd.mapPartitions(
            lambda x: dss_multiple(x, data_normalized_bc.value, start, end, parallelism),
            preservesPartitioning=True).cache()
        # b = group_rdd.collect()  # debug
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
    # all_ss_num = sum(len(x[1]) * (len(x[1]) + 1) / 2 for x in data_normalized)

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
    return dict(cluster_rdd.
                map(_cluster_to_meta).
                reduceByKey(_cluster_reduce_func).collect())


def _query_bf_spark(query, subsequence_rdd, dt_index, data_list):
    pp_rdd = subsequence_rdd.map(
        lambda x: _get_dist_sequence(query, x, dt_index=dt_index, data_list=data_list.value))
    candidate_list = pp_rdd.collect()
    # clear data stored in the candidate list

    return candidate_list


def _query_piecewise_spark(query, subsequence_rdd, dt_index, data_list, piecewise, n_segment):
    pp_rdd = subsequence_rdd.map(
        lambda x: _get_dist_sequence_piecewise(query, x, dt_index=dt_index, data_list=data_list.value,
                                               piecewise= piecewise, n_segment=n_segment))
    candidate_list = pp_rdd.collect()
    # clear data stored in the candidate list

    return candidate_list


def _query_paa_spark(query, paa_kv_rdd, dt_index, n_segment):
    q_paa_data = paa_compress(query.get_data(), n_segment)
    pp_rdd = paa_kv_rdd.map(
        lambda x: (_get_dist_array(q_paa_data, x[1], dt_index=dt_index), x[0]))
    candidate_list = pp_rdd.collect()
    return candidate_list


def _query_sax_spark(query, sax_kv_rdd, dt_index, n_segment, n_sax_symbols):
    q_sax_data = sax_compress(query.get_data(), n_segment, n_sax_symbols)
    sax_rdd = sax_kv_rdd.map(
        lambda x: (_get_dist_array(q_sax_data, x[1], dt_index=dt_index), x[0]))
    candidate_list = sax_rdd.collect()
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


def _build_piecewise_spark(subsequences_rdd, mode: str, n_segment: int, n_sax_symbols: int, data_list, _dummy_slicing,
                           _sc: SparkContext = None, _start=None, _end=None):
    if _dummy_slicing:
        assert _start and _end  # must include start and end if apply dummy slice
        print('Simulating Slicing Time')
        parallelism = _sc.defaultParallelism
        input_rdd = _sc.parallelize(data_list.value, numSlices=parallelism)
        # partition_input = input_rdd.glom().collect()  # for debug purposes
        # Grouping the data_original
        # group = _group_time_series(input_rdd.glom().collect()[0], start, end)  # for debug purposes
        group_rdd = input_rdd.mapPartitions(
            lambda x: _group_time_series(time_series=x, start=_start, end=_end), preservesPartitioning=True).cache()
        dummy_ss_rdd = group_rdd.flatMap(lambda x: x[1]).cache()
        dummy_ss_rdd.count()
        dummy_ss_rdd.unpersist()  # remove the dummy subsequence to free resources
        del dummy_ss_rdd

    # ss_paaKv = []
    # for ss in subsequences_rdd.collect():
    #     ss_paaKv.append(paa_compress(ss.fetch_data(data_list.value), paa_seg))
    # ss_saxKv = []
    # for ss in subsequences_rdd.collect():
    #     ss_saxKv.append(sax_compress(ss.fetch_data(data_list.value), n_segment, n_sax_symbols))
    if mode == 'paa':
        piecewise_kv_rdd = subsequences_rdd.map(
            lambda x: (x, paa_compress(x.fetch_data(data_list.value), n_segment))).cache()
    elif mode == 'sax':
        piecewise_kv_rdd = subsequences_rdd.map(
            lambda x: (x, sax_compress(x.fetch_data(data_list.value), n_segment, n_sax_symbols))).cache()
    else:
        raise Exception('spark_utils: unrecognized piecewise mode, it must be paa or sax')

    piecewise_kv_rdd.count()
    # a = ss_paaKv_rdd.collect()
    # b = max([len(x[1]) for x in a])
    return piecewise_kv_rdd
