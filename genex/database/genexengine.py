import heapq
import json
import math
import multiprocessing
import os
import pickle
import random
import uuid

from deprecated import deprecated
from pyspark import SparkContext
import pandas as pd
import numpy as np
import shutil
from scipy.spatial.distance import cityblock
from scipy.spatial.distance import minkowski
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import chebyshev
from sklearn.preprocessing import LabelEncoder

from genex.classes.Sequence import Sequence
from genex.cluster import _randomize, sim_between_seq
from genex.spark_utils import _create_sc, _pr_spark_conf, _cluster_with_spark, _is_using_spark
from genex.utils import _multiprocess_backend, genex_normalize
from genex.utils import _validate_gxdb_build_arguments, _df_to_list, _process_loi, _query_partition, \
    _validate_gxdb_query_arguments, _create_f_uuid_map
from mutiproces_utils import _cluster_multi_process
from process_utils import _group_time_series, _slice_time_series


def eu_norm(x, y):
    euclidean(x, y) / np.sqrt(len(x))


def ma_norm(x, y):
    cityblock(x, y) / len(x)


def ch_norm(x, y):
    chebyshev(x, y)


def min_norm(x, y):
    chebyshev(x, y)


dist_func_index = {'eu': eu_norm,
                   'ma': ma_norm,
                   'ch': ch_norm,
                   'min': min_norm
                   }
dist_type_index = {'eu': 0,
                   'ma': 1,
                   'ch': 2,
                   'min': 2}


def from_csv(file_name, feature_num: int,
             num_worker: int,
             use_spark: bool = True, driver_mem: int = 16, max_result_mem: int = 16,
             is_header=True,
             add_uuid=False,
             _rows_to_consider: int = None,
             _memory_opt: str = None,
             _is_z_normalize=True):
    """
    build a genex_database object from given csv,
    Note: if time series are of different length, shorter sequences will be post padded to the length
    of the longest sequence in the dataset

    :param is_header: a boolean value indicated whether the names of the id columns are provided
    :param add_uuid:
    :param _is_z_normalize:
    :param _memory_opt:
    :param driver_mem:
    :param max_result_mem:
    :param use_spark:
    :param num_worker:
    :param file_name:
    :param feature_num:
    :param sc: spark context on which the database will run

    :param _rows_to_consider: experiment parameter that takes a iterable of two integers.
            Only rows in between the given interval will be take into the database.
    :param _is_use_uuid: experiment parameter. If set to true, the feature (id) of the time series will be

    :return: a genex_database object that holds the original time series
    """
    if is_header is False:
        df = pd.read_csv(file_name, header=None, skiprows=1)
    else:
        df = pd.read_csv(file_name)

    if feature_num == 0:
        add_uuid = True
        print('msg: from_csv, feature num is 0')

    # checking whether the original id is unique
    df_raw_id = df.iloc[:, :feature_num].copy()
    raw_id_num = df_raw_id.groupby(list(df_raw_id.columns[:]))

    if len(raw_id_num) < len(df):
        add_uuid = True
        print('msg: the key for each time series is not unique')

    if add_uuid:
        print('auto-generating uuid')
        feature_num = feature_num + 1
        df.insert(0, 'uuid', [uuid.uuid4() for x in range(len(df))], False)

    if _memory_opt == 'uuid':
        df.insert(0, 'uuid', [uuid.uuid4() for x in range(len(df))], False)
        feature_uuid_dict = _create_f_uuid_map(df=df, feature_num=feature_num)
        # take off the feature columns
        df = df.drop(df.columns[list(range(1, feature_num + 1))], axis=1, inplace=False)
        data_list = _df_to_list(df, feature_num=1)  # now the only feature is the uuid

    elif _memory_opt == 'encoding':
        for i in range(feature_num):
            le = LabelEncoder()
            df.iloc[:, i] = le.fit_transform(df.iloc[:, i])

        data_list = _df_to_list(df, feature_num=feature_num)
    else:
        data_list = _df_to_list(df, feature_num=feature_num)

    if _rows_to_consider is not None:
        if type(_rows_to_consider) == list:
            assert len(_rows_to_consider) == 2
            data_list = data_list[_rows_to_consider[0]:_rows_to_consider[1]]
        elif type(_rows_to_consider) == int:
            data_list = data_list[:_rows_to_consider]
        else:
            raise Exception('_rows_to_consider must be either a list or an integer')

    data_norm_list, global_max, global_min = genex_normalize(data_list, z_normalization=_is_z_normalize)

    mp_context = _multiprocess_backend(use_spark, num_worker, driver_mem=driver_mem, max_result_mem=max_result_mem)

    return GenexEngine(data_raw=df, data_original=data_list, data_normalized=data_norm_list, global_max=global_max,
                       global_min=global_min,
                       mp_context=mp_context)


def from_db(path: str,
            num_worker: int,
            use_spark: bool = True, driver_mem: int = 16, max_result_mem: int = 16,
            ):
    """
    returns a previously saved gxdb object from its saved path

    :param max_result_mem:
    :param driver_mem:
    :param use_spark:
    :param num_worker:
    :param sc: spark context on which the database will run
    :param path: path of the saved gxdb object

    :return: a genex database object that holds clusters of time series data_original
    """

    # TODO the input fold_name is not existed
    if os.path.exists(path) is False:
        raise ValueError('There is no such database, check the path again.')

    data_raw = pd.read_csv(os.path.join(path, 'data_raw.csv'))
    data = pickle.load(open(os.path.join(path, 'data_original.gxdb'), 'rb'))
    data_normalized = pickle.load(open(os.path.join(path, 'data_normalized.gxdb'), 'rb'))

    conf = json.load(open(os.path.join(path, 'conf.json'), 'rb'))
    mp_context = _multiprocess_backend(use_spark, num_worker, driver_mem=driver_mem, max_result_mem=max_result_mem)
    init_params = {'data_raw': data_raw, 'data_original': data, 'data_normalized': data_normalized,
                   'mp_context': mp_context,
                   'global_max': conf['global_max'], 'global_min': conf['global_min']}
    db = GenexEngine(**init_params)
    db.set_conf(conf)

    if os.path.exists(os.path.join(path, 'clusters.gxdb')):
        db.set_clusters(db.get_mp_context().pickleFile(os.path.join(path, 'clusters.gxdb/*')))
        db.set_cluster_meta_dict(pickle.load(open(os.path.join(path, 'cluster_meta_dict.gxdb'), 'rb')))

    return db


class GenexEngine:
    """
    Genex Engine

    Init parameters
    data_original
    data_normalized
    scale_funct
    """

    def __init__(self, **kwargs):
        """

        :param kwargs:
        """
        self.data_raw = kwargs['data_raw']
        self.data_original = kwargs['data_original']
        self.data_normalized = kwargs['data_normalized']
        self.mp_context = kwargs['mp_context']
        self.clusters = None
        self.cluster_meta_dict = None

        self.conf = {'build_conf': None,
                     'global_max': kwargs['global_max'],
                     'global_min': kwargs['global_min']}
        self.bf_query_buffer = dict()

    def set_conf(self, conf):
        self.conf = conf

    def set_clusters(self, clusters):
        self.clusters = clusters

    def get_mp_context(self):
        return self.mp_context

    def get_num_ts(self):
        return len(self.data_original)

    def is_seq_exist(self, seq: Sequence):
        try:
            seq.fetch_data(self.data_normalized)
        except KeyError:
            return False
        return True

    def build(self, st: float, dist_type: str = 'eu', loi: slice = None, verbose: int = 1,
              _batch_size=None, _is_cluster=True):
        """
        Groups and clusters the time series set

        :param st: The upper bound of the similarity value between two time series (Value must be
                                      between 0 and 1)
        :param dist_type: Distance type used for similarity calculation between sequences
        :param loi: default value is none, otherwise using slice notation [start, stop: step]
        :param verbose: Print logs when grouping and clustering the data_original
        :param batch_size:
        :param _is_cluster: Decide whether time series data_original is clustered or not

        """
        _validate_gxdb_build_arguments(locals())
        start, end = _process_loi(loi)
        # update build configuration
        self.conf['build_conf'] = {'similarity_threshold': st,
                                   'dist_type': dist_type,
                                   'loi': [start, end]}
        # No () inside the JSON file

        # exit without clustering
        if not _is_cluster:
            return

        # determine the distance calculation function
        try:
            dist_func = dist_func_index[dist_type]
        except ValueError:
            raise Exception('Unknown distance type: ' + str(dist_type))

        if _is_using_spark(self.mp_context):  # If using Spark backend
            self.clusters, self.cluster_meta_dict = _cluster_with_spark(self.mp_context, self.data_normalized,
                                                                        start, end, st, dist_func, verbose)
        else:
            a = self.get_num_ts()
            b = self.mp_context['num_worker']
            self.clusters, self.cluster_meta_dict = \
                _cluster_multi_process(self.data_normalized,
                                       self.mp_context['num_worker'],
                                       start, end, st, dist_func, verbose)

    def get_cluster(self, rprs: Sequence):
        length = None

        for k, v in self.cluster_meta_dict.items():
            if rprs in v.keys():
                length = k
                break

        if length is None:
            raise ValueError('get_cluster: Couldn\'t find the representative in the cluster, please check the input.')

        target_cluster_rdd = self.clusters.filter(lambda x: rprs in x[1].keys()).collect()
        cluster = target_cluster_rdd[0][1].get(rprs)

        return cluster

    def get_num_subsequences(self):
        try:
            assert self.clusters is not None
        except AssertionError:
            raise Exception('get_num_subsequences: the database must be build before calling this function')

        rtn = 0
        clusters = [x[1] for x in self.cluster_meta_dict.items()]

        for c in clusters:
            for key, value in c.items():
                rtn += value
        return rtn

    def query_brute_force(self, query: Sequence, best_k: int):
        """
        Retrieve best k matches for query sequence using Brute force method

        :param query: Sequence being queried
        :param best_k: Number of best matches to retrieve for the given query

        :return: a list containing best k matches for given query sequence
        """
        dist_type = self.conf.get('build_conf').get('dist_type')

        query.fetch_and_set_data(self.data_normalized)
        input_rdd = self.mp_context.parallelize(self.data_normalized, numSlices=self.mp_context.defaultParallelism)

        start, end = self.conf.get('build_conf').get('loi')

        bf_query_key = (dist_type, query, start, end)

        if bf_query_key not in self.bf_query_buffer.keys():
            group_rdd = input_rdd.mapPartitions(
                lambda x: _group_time_series(time_series=x, start=start, end=end), preservesPartitioning=True)

            slice_rdd = group_rdd.flatMap(lambda x: x[1])
            # for debug purpose
            # a = slice_rdd.collect()
            dist_rdd = slice_rdd.map(lambda x: (sim_between_seq(query, x, dt_index=dist_type_index[dist_type]), x))
            candidate_list = dist_rdd.collect()
            self.bf_query_buffer[bf_query_key] = candidate_list
        else:
            print('bf_query: using buffered bf results, key=' + str([str(x) for x in bf_query_key]))
            candidate_list = self.bf_query_buffer[bf_query_key]  # retrive the buffer candidate list

        candidate_list.sort(key=lambda x: x[0])
        return candidate_list[:best_k]

    def group_sequences(self):
        """
        helper function to monitor memory usage
        """
        input_rdd = self.mp_context.parallelize(self.data_normalized, numSlices=self.mp_context.defaultParallelism)
        # process all possible length
        start, end = _process_loi(None)

        slice_rdd = input_rdd.mapPartitions(
            lambda x: _slice_time_series(time_series=x, start=start, end=end), preservesPartitioning=True)

        return slice_rdd.collect()

    def get_random_seq_of_len(self, sequence_len, seed):
        random.seed(seed)

        target = random.choice(self.data_normalized)

        try:
            start = random.randint(0, len(target[1]) - sequence_len)
            seq = Sequence(target[0], start, start + sequence_len - 1)
        except ValueError:
            raise Exception('get_random_seq_of_len: given length does not exist in the database. If you think this is '
                            'an implementation error, please report to the Repository as an issue.')

        try:
            assert len(seq.fetch_data(self.data_original)) == sequence_len
        except AssertionError:
            raise Exception('get_random_seq_of_len: given length does not exist in the database. If you think this is '
                            'an implementation error, please report to the Repository as an issue.')
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
        if self.clusters is not None:
            self.clusters.saveAsPickleFile(os.path.join(path, 'clusters.gxdb'))
            pickle.dump(self.cluster_meta_dict, open(os.path.join(path, 'cluster_meta_dict.gxdb'), 'wb'))

        # save data_original files
        pickle.dump(self.data_original, open(os.path.join(path, 'data_original.gxdb'), 'wb'))
        pickle.dump(self.data_normalized, open(os.path.join(path, 'data_normalized.gxdb'), 'wb'))
        self.data_raw.to_csv(os.path.join(path, 'data_raw.csv'))

        # save configs
        with open(path + '/conf.json', 'w') as f:
            json.dump(self.conf, f, indent=4)

    def is_id_exists(self, sequence: Sequence):
        return sequence.seq_id in dict(self.data_original).keys()

    def _get_data_normalized(self):
        return self.data_normalized

    def query(self, query: Sequence, best_k: int, exclude_same_id: bool = False, overlap: float = 1.0,
              loi: slice = None,
              _lb_opt_repr: str = 'none', _repr_kim_rf=0.5, _repr_keogh_rf=0.75,
              _lb_opt_cluster: str = 'none', _cluster_kim_rf=0.5, _cluster_keogh_rf=0.75,
              _ke=None, _radius: int = 0):
        """
        Find best k matches for given query sequence using Distributed Genex method

        :param query:
        :param _radius:
        :param _ke:
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
        _ke_factor = 1
        if _ke is None or _ke < best_k:
            _ke = best_k * _ke_factor
        else:
            if _ke > self.get_num_subsequences():
                raise Exception('query: _ke cannot be greater than the number of subsequences in the database.')

        query.fetch_and_set_data(self._get_data_normalized())
        query = self.mp_context.broadcast(query)

        data_normalized = self.mp_context.broadcast(self._get_data_normalized())

        st = self.conf.get('build_conf').get('similarity_threshold')
        dist_type = self.conf.get('build_conf').get('dist_type')

        # for debug purposes
        # a = _query_partition(cluster=self.cluster_rdd.collect(), q=query, k=best_k, ke=_ke,
        #                      data_normalized=data_normalized, loi=loi,
        #                      _lb_opt_cluster=_lb_opt_cluster, _lb_opt_repr=_lb_opt_repr,
        #                      exclude_same_id=exclude_same_id, overlap=overlap,
        #
        #                      repr_kim_rf=_repr_kim_rf, repr_keogh_rf=_repr_keogh_rf,
        #                      cluster_kim_rf=_cluster_kim_rf, cluster_keogh_rf=_cluster_keogh_rf,
        #                      radius=_radius, st=st
        #                      )
        # seq_num = self.get_num_subsequences()

        query_rdd = self.clusters.mapPartitions(
            lambda c:
            _query_partition(cluster=c, q=query, k=best_k, ke=_ke, data_normalized=data_normalized, loi=loi,
                             dt_index=dist_type_index[dist_type],
                             _lb_opt_cluster=_lb_opt_cluster, _lb_opt_repr=_lb_opt_repr,
                             exclude_same_id=exclude_same_id, overlap=overlap,

                             repr_kim_rf=_repr_kim_rf, repr_keogh_rf=_repr_keogh_rf,
                             cluster_kim_rf=_cluster_kim_rf, cluster_keogh_rf=_cluster_keogh_rf,
                             radius=_radius, st=st
                             )
        )

        #### testing distribute query vs. one-core query
        # result_distributed = query_rdd.collect()
        # result_distributed.sort(key=lambda x: x[0])
        # result_distributed = result_distributed[:10]
        # result_one_core = a
        # result_one_core.sort(key=lambda x: x[0])
        # result_one_core = result_one_core[:10]
        # is_same = np.equal(result_distributed, result_one_core)

        aggre_query_result = query_rdd.collect()
        heapq.heapify(aggre_query_result)
        best_matches = []

        for i in range(best_k):
            best_matches.append(heapq.heappop(aggre_query_result))

        return best_matches

    def set_cluster_meta_dict(self, cluster_meta_dict):
        self.cluster_meta_dict = cluster_meta_dict

    def get_max_seq_len(self):
        return max([len(x[1]) for x in self.data_normalized])


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
