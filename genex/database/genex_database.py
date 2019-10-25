import heapq
import json
import math
import os
import pickle
import random
import uuid

from pyspark import SparkContext
from pyspark.sql import SQLContext
import pandas as pd
import numpy as np
import shutil

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from genex.classes.Sequence import Sequence
from genex.cluster import sim_between_seq, _cluster_groups, lb_kim_sequence, lb_keogh_sequence
from genex.preprocess import get_subsequences, genex_normalize, _group_time_series, _slice_time_series
from genex.utils import scale, _validate_gxdb_build_arguments, _df_to_list, _process_loi, _query_partition, \
    _validate_gxdb_query_arguments, _create_f_uuid_map


def from_csv(file_name, feature_num: int, sc: SparkContext,
             _rows_to_consider: int = None,
             _memory_opt: str = None):
    """
    build a genex_database object from given csv,
    Note: if time series are of different length, shorter sequences will be post padded to the length
    of the longest sequence in the dataset

    :param file_name:
    :param feature_num:
    :param sc: spark context on which the database will run

    :param _rows_to_consider: experiment parameter that takes a iterable of two integers.
            Only rows in between the given interval will be take into the database.
    :param _is_use_uuid: experiment parameter. If set to true, the feature (id) of the time series will be


    :return: a genex_database object that holds the original time series
    """

    df = pd.read_csv(file_name)

    if _memory_opt == 'uuid':
            df.insert(0, 'uuid', [uuid.uuid4() for x in range(len(df))], False)
            feature_uuid_dict = _create_f_uuid_map(df=df, feature_num=feature_num)
            # take off the feature columns
            df = df.drop(df.columns[list(range(1, feature_num+1))], axis=1, inplace=False)
            data_list = _df_to_list(df, feature_num=1)  # now the only feature is the uuid

    elif _memory_opt == 'encoding':
        for i in range(feature_num):
            le = LabelEncoder()
            df.iloc[:, i] = le.fit_transform(df.iloc[:, i])

        data_list = _df_to_list(df, feature_num=feature_num)
    else:
        data_list = _df_to_list(df, feature_num=feature_num)

    if _rows_to_consider is not None:
        data_list = data_list[_rows_to_consider[0]:_rows_to_consider[1]]

    data_norm_list, global_max, global_min = genex_normalize(data_list, z_normalization=True)

    # return Genex_database
    return genex_database(data=data_list, data_normalized=data_norm_list, global_max=global_max, global_min=global_min,
                          spark_context=sc)


def from_db(sc: SparkContext, path: str):
    """
    returns a previously saved gxdb object from its saved path

    :param sc: spark context on which the database will run
    :param path: path of the saved gxdb object

    :return: a genex database object that holds clusters of time series data
    """

    # TODO the input fold_name is not existed
    data = pickle.load(open(os.path.join(path, 'data.gxdb'), 'rb'))
    data_normalized = pickle.load(open(os.path.join(path, 'data_normalized.gxdb'), 'rb'))

    conf = json.load(open(os.path.join(path, 'conf.json'), 'rb'))
    init_params = {'data': data, 'data_normalized': data_normalized, 'spark_context': sc,
                   'global_max': conf['global_max'], 'global_min': conf['global_min']}
    db = genex_database(**init_params)

    db.set_clusters(db.get_sc().pickleFile(os.path.join(path, 'clusters.gxdb/*')))
    db.set_conf(conf)

    return db


class genex_database:
    """
    Genex Database

    Init parameters
    data
    data_normalized
    scale_funct
    """

    def __init__(self, **kwargs):
        """

        :param kwargs:
        """
        self.data = kwargs['data']
        self.data_normalized = kwargs['data_normalized']
        self.sc = kwargs['spark_context']
        self.cluster_rdd = None

        self.conf = {'build_conf': None,
                     'global_max': kwargs['global_max'],
                     'global_min': kwargs['global_min']}

    def set_conf(self, conf):
        self.conf = conf

    def set_clusters(self, clusters):
        self.cluster_rdd = clusters

    def get_sc(self):
        return self.sc

    def build(self, similarity_threshold: float, dist_type: str = 'eu', loi: slice = None, verbose: int = 1,
              _batch_size=None, _is_cluster=True):
        """
        Groups and clusters the time series set

        :param similarity_threshold: The upper bound of the similarity value between two time series (Value must be
                                      between 0 and 1)
        :param dist_type: Distance type used for similarity calculation between sequences
        :param loi: default value is none, otherwise using slice notation [start, stop: step]
        :param verbose: Print logs when grouping and clustering the data
        :param batch_size:
        :param _is_cluster: Decide whether time series data is clustered or not

        """
        _validate_gxdb_build_arguments(locals())
        start, end = _process_loi(loi)
        # update build configuration
        self.conf['build_conf'] = {'similarity_threshold': similarity_threshold,
                                   'dist_type': dist_type,
                                   'loi': (start, end)}

        # exit without clustering
        if not _is_cluster:
            return

        # validate and save the loi to gxdb class fields
        # distribute the data
        input_rdd = self.sc.parallelize(self.data_normalized, numSlices=self.sc.defaultParallelism)
        # partition_input = input_rdd.glom().collect() #  for debug purposes
        # Grouping the data
        # group = _group_time_series(input_rdd.glom().collect()[0], start, end) # for debug purposes
        group_rdd = input_rdd.mapPartitions(
            lambda x: _group_time_series(time_series=x, start=start, end=end), preservesPartitioning=True)
        # group_partition = group_rdd.glom().collect()  # for debug purposes

        # Cluster the data with Gcluster
        # cluster = _cluster_groups(groups=group_rdd.glom().collect()[0], st=similarity_threshold,
        #                           dist_type=dist_type, verbose=1)  # for debug purposes
        cluster_rdd = group_rdd.mapPartitions(lambda x: _cluster_groups(
            groups=x, st=similarity_threshold, dist_type=dist_type, log_level=verbose)).cache()
        # cluster_partition = cluster_rdd.glom().collect()  # for debug purposes

        cluster_rdd.collect()

        self.cluster_rdd = cluster_rdd

    def query_brute_force(self, query: Sequence, best_k: int):
        """
        Retrieve best k matches for query sequence using Brute force method

        :param query: Sequence being queried
        :param best_k: Number of best matches to retrieve for the given query

        :return: a list containing best k matches for given query sequence

        """

        dist_type = self.conf.get('build_conf').get('dist_type')

        query.fetch_and_set_data(self.data_normalized)
        input_rdd = self.sc.parallelize(self.data_normalized, numSlices=self.sc.defaultParallelism)

        start, end = self.conf.get('build_conf').get('loi')
        slice_rdd = input_rdd.mapPartitions(
            lambda x: _slice_time_series(time_series=x, start=start, end=end), preservesPartitioning=True)

        # for debug purpose
        # a = slice_rdd.collect()

        dist_rdd = slice_rdd.map(lambda x: (sim_between_seq(query, x, dist_type=dist_type), x))

        candidate_list = dist_rdd.collect()
        candidate_list.sort(key=lambda x: x[0])

        query_result = candidate_list[:best_k]
        return query_result

    def group_sequences(self):
        """
        helper function to monitor memory usage
        """
        input_rdd = self.sc.parallelize(self.data_normalized, numSlices=self.sc.defaultParallelism)
        # process all possible length
        start, end = _process_loi(None)

        slice_rdd = input_rdd.mapPartitions(
            lambda x: _slice_time_series(time_series=x, start=start, end=end), preservesPartitioning=True)

        return slice_rdd.collect()

    def get_random_seq_of_len(self, sequence_len):
        target = random.choice(self.data_normalized)

        start = random.randint(0, len(target[1]) - sequence_len)
        seq = Sequence(target[0], start, start + sequence_len - 1)

        return seq

    def save(self, path: str):
        """
        The save method saves the database onto the disk.
        :param path: path to save the database to

        """
        if os.path.exists(path):
            print('Path ' + path + ' already exists, overwriting...')
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            os.makedirs(path)

        # save the clusters if the db is built
        if self.cluster_rdd is not None:
            self.cluster_rdd.saveAsPickleFile(os.path.join(path, 'clusters.gxdb'))

        # save data files
        pickle.dump(self.data, open(os.path.join(path, 'data.gxdb'), 'wb'))
        pickle.dump(self.data_normalized, open(os.path.join(path, 'data_normalized.gxdb'), 'wb'))

        # save configs
        with open(path + '/conf.json', 'w') as f:
            json.dump(self.conf, f, indent=4)

    def is_id_exists(self, sequence: Sequence):
        return sequence.seq_id in dict(self.data).keys()

    def _get_data_normalized(self):
        return self.data_normalized

    def query(self, query: Sequence, best_k: int, exclude_same_id: bool = False, overlap: float = 1.0,
              _lb_opt_repr: str = 'none', _repr_kim_rf=0.5, _repr_keogh_rf=0.75,
              _lb_opt_cluster: str = 'none', _cluster_kim_rf=0.5, _cluster_keogh_rf=0.75,
              ):
        """
        Find best k matches for given query sequence using Distributed Genex method

        :param: query: Sequence to be queried
        :param best_k: Number of best matches to retrieve
        :param exclude_same_id: Whether to exclude query sequence in the retrieved matches
        :param overlap: Value for overlapping parameter (Must be between 0 and 1 inclusive)
        :param _lb_opt_repr: Type of optimization used for representatives (lbh or none)
        :param _repr_kim_rf: Value of LB_Kim reduction factor for representatives (0.25 or 0.5 or 0.75)
        :param _repr_keogh_rf: Value of LB_Keogh reduction factor for representatives (0.25 or 0.5 or 0.75)
        :param _lb_opt_cluster: Type of optimization used for clusters (lbh or bsf or lbh_bst or none)
        :param _lb_opt_repr: lbh, none
        :param _cluster_kim_rf: Value of LB_Kim reduction factor for clusters (0.25 or 0.5 or 0.75)
        :param _cluster_keogh_rf: Value of LB_Keogh reduction factor for clusters (0.25 or 0.5 or 0.75)

        :return: a list containing k best matches for given query sequence
        """
        _validate_gxdb_query_arguments(locals())

        query.fetch_and_set_data(self._get_data_normalized())
        query = self.sc.broadcast(query)

        data_normalized = self.sc.broadcast(self._get_data_normalized())

        st = self.conf.get('build_conf').get('similarity_threshold')
        dist_type = self.conf.get('build_conf').get('dist_type')

        # for debug purposes
        # a = _query_partition(cluster=self.cluster_rdd.glom().collect()[0], q=query, k=best_k, data_normalized=data_normalized, dist_type=dist_type,
        #                      _lb_opt_cluster=_lb_opt_cluster, _lb_opt_repr=_lb_opt_repr,
        #                      exclude_same_id=exclude_same_id, overlap=overlap,
        #
        #                      repr_kim_rf=_repr_kim_rf, repr_keogh_rf=_repr_keogh_rf,
        #                      cluster_kim_rf=_cluster_kim_rf, cluster_keogh_rf=_cluster_keogh_rf,
        #                      )
        query_rdd = self.cluster_rdd.mapPartitions(
            lambda x:
            _query_partition(cluster=x, q=query, k=best_k, data_normalized=data_normalized, dist_type=dist_type,
                             _lb_opt_cluster=_lb_opt_cluster, _lb_opt_repr=_lb_opt_repr,
                             exclude_same_id=exclude_same_id, overlap=overlap,

                             repr_kim_rf=_repr_kim_rf, repr_keogh_rf=_repr_keogh_rf,
                             cluster_kim_rf=_cluster_kim_rf, cluster_keogh_rf=_cluster_keogh_rf,
                             )
        )
        aggre_query_result = query_rdd.collect()
        heapq.heapify(aggre_query_result)
        best_matches = []

        for i in range(best_k):
            best_matches.append(heapq.heappop(aggre_query_result))

        return best_matches


def _is_overlap(seq1: Sequence, seq2: Sequence, overlap: float) -> bool:
    """
     Check for overlapping between two time series sequences

    :param seq1: Time series Sequence
    :param seq2: Time series Sequence
    :param overlap: Value for overlap (must be between 0 and 1 inclusive)

    :return: boolean value based on whether two sequences overlap more or less than given overlap parameter
    """
    if seq1.seq_id != seq2.seq_id:  # overlap does NOT matter if two seq have different id
        return True
    else:
        of = _calculate_overlap(seq1, seq2)
        return _calculate_overlap(seq1, seq2) >= overlap


def _calculate_overlap(seq1, seq2) -> float:
    """
    Calculate overlap between two time series sequence

    :param seq1: Time series sequence
    :param seq2: Time series sequence

    :return: overlap value between two sequences
    """
    if seq2.end > seq1.end and seq2.start >= seq1.start:
        return (seq1.end - seq2.start + 1) / (seq2.end - seq1.start + 1)
    elif seq1.end > seq2.end and seq1.start >= seq2.start:
        return (seq2.end - seq1.start + 1) / (seq1.end - seq2.start + 1)
    if seq2.end >= seq1.end and seq2.start > seq1.start:
        return (seq1.end - seq2.start + 1) / (seq2.end - seq1.start + 1)
    elif seq1.end >= seq2.end and seq1.start > seq2.start:
        return (seq2.end - seq1.start + 1) / (seq1.end - seq2.start + 1)

    elif seq1.end > seq2.end and seq2.start >= seq1.start:
        return len(seq2) / len(seq1)
    elif seq2.end > seq1.end and seq1.start >= seq2.start:
        return len(seq1) / len(seq2)
    elif seq1.end >= seq2.end and seq2.start > seq1.start:
        return len(seq2) / len(seq1)
    elif seq2.end >= seq1.end and seq1.start > seq2.start:
        return len(seq1) / len(seq2)

    elif seq2.start > seq1.end or seq1.start > seq2.end:  # does not overlap at all
        return 0.0
    else:
        print(seq1)
        print(seq2)
        raise Exception('FATAL: sequence 100% overlap, please report the bug')
