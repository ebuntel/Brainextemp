import heapq
import json
import math
import multiprocessing
import os
import pickle
import random

from pyspark import SparkContext
import numpy as np
import shutil
from scipy.spatial.distance import cityblock
from scipy.spatial.distance import minkowski
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import chebyshev

from genex.classes.Sequence import Sequence
from genex.misc import pr_red
from genex.op.query_op import _query_partition
from genex.utils.spark_utils import _cluster_with_spark, _query_bf_spark, _pr_spark_conf, _create_sc
from genex.utils.utils import _validate_gxdb_build_arguments, _process_loi, _validate_gxdb_query_arguments

from mutiproces_utils import _cluster_multi_process, _query_bf_mp, _query_mp
from process_utils import _slice_time_series


def eu_norm(x, y):
    return euclidean(x, y) / np.sqrt(len(x))


def ma_norm(x, y):
    return cityblock(x, y) / len(x)


def ch_norm(x, y):
    return chebyshev(x, y)


def min_norm(x, y):
    return chebyshev(x, y)


dt_func_dict = {'eu': eu_norm,
                'ma': ma_norm,
                'ch': ch_norm,
                'min': min_norm
                }
dt_pnorm_dict = {'eu': 2,
                 'ma': 1,
                 'ch': math.inf,
                 'min': math.inf}


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
        self.conf = {'global_max': kwargs['global_max'],
                     'global_min': kwargs['global_min'],
                     'backend': kwargs['backend']}
        self.build_conf = None
        self.bf_query_buffer = dict()

    def __set_conf(self, conf):
        self.conf = conf

    def _set_clusters(self, clusters):
        self.clusters = clusters

    def get_mp_context(self):
        return self.mp_context

    def get_num_ts(self):
        return len(self.data_original)

    def set_build_conf(self, build_conf: dict):
        self.build_conf = build_conf

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
        self.build_conf = {'similarity_threshold': st,
                           'dist_type': dist_type,
                           'loi': (start, end)}

        # exit without clustering
        if not _is_cluster:
            return

        # determine the distance calculation function
        try:
            dist_func = dt_func_dict[dist_type]
        except ValueError:
            raise Exception('Unknown distance type: ' + str(dist_type))

        if self.is_using_spark():  # If using Spark backend
            self.clusters, self.cluster_meta_dict = _cluster_with_spark(self.mp_context, self.data_normalized,
                                                                        start, end, st, dist_func, verbose)
        else:
            self.clusters, self.cluster_meta_dict = self.clusters, self.cluster_meta_dict = \
                _cluster_multi_process(self.mp_context,
                                       self.data_normalized,
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
        dist_type = self.build_conf.get('dist_type')
        dt_index = dt_pnorm_dict[dist_type]
        start, end = self.build_conf.get('loi')

        query.fetch_and_set_data(self.data_normalized)
        bf_query_key = (dist_type, query, start, end)

        if bf_query_key not in self.bf_query_buffer.keys():
            if self.is_using_spark():
                candidate_list = _query_bf_spark(query, self.mp_context, self.data_normalized, start, end, dt_index)
            else:
                candidate_list = _query_bf_mp(query, self.mp_context, self.data_normalized, start, end, dt_index)

            candidate_list.sort(key=lambda x: x[0])
            self.bf_query_buffer[bf_query_key] = candidate_list
        else:
            print('bf_query: using buffered bf results, key=' + str([str(x) for x in bf_query_key]))
            candidate_list = self.bf_query_buffer[bf_query_key]  # retrieve the buffer candidate list

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
            self._save_cluster(path)
            pickle.dump(self.cluster_meta_dict, open(os.path.join(path, 'cluster_meta_dict.gxdb'), 'wb'))
            with open(path + '/build_conf.json', 'w') as f:
                json.dump(self.build_conf, f, indent=4)

        # save data_original files
        pickle.dump(self.data_original, open(os.path.join(path, 'data_original.gxdb'), 'wb'))
        pickle.dump(self.data_normalized, open(os.path.join(path, 'data_normalized.gxdb'), 'wb'))
        self.data_raw.to_csv(os.path.join(path, 'data_raw.csv'))

        # save configs
        with open(path + '/conf.json', 'w') as f:
            json.dump(self.conf, f, indent=4)

    def _save_cluster(self, path):
        if self.is_using_spark():
            self.clusters.saveAsPickleFile(os.path.join(path, 'clusters.gxdb'))
        else:
            pickle.dump(self.clusters, open(os.path.join(path, 'clusters.gxdb'), 'wb'))

    def load_cluster(self, path):
        if self.is_using_spark():
            self._set_clusters(self.get_mp_context().pickleFile(os.path.join(path, 'clusters.gxdb/*')))
        else:
            self._set_clusters(pickle.load(open(os.path.join(path, 'clusters.gxdb'), 'rb')))

    def is_id_exists(self, sequence: Sequence):
        return sequence.seq_id in dict(self.data_original).keys()

    def _get_data_normalized(self):
        return self.data_normalized

    def is_using_spark(self):
        return self.conf['backend'] == 'spark'

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
        data_normalized = self._get_data_normalized()

        st = self.build_conf.get('similarity_threshold')
        dist_type = self.build_conf.get('dist_type')

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
        if self.is_using_spark():
            query = self.mp_context.broadcast(query)
            data_normalized = self.mp_context.broadcast(data_normalized)

        query_args = {  # order of this kwargs MUST be perserved in accordance to genex.op.query_op._query_partition
            'q': query, 'k': best_k, 'ke': _ke, 'data_normalized': data_normalized,
            'loi': loi, 'pnorm': dt_pnorm_dict[dist_type],
            '_lb_opt_cluster': _lb_opt_cluster, '_lb_opt_repr': _lb_opt_repr,
            'overlap': overlap, 'exclude_same_id': exclude_same_id, 'radius': _radius, 'st': st
        }
        if self.is_using_spark():
            query_rdd = self.clusters.mapPartitions(lambda c: _query_partition(cluster=c, **query_args))
            candidates = query_rdd.collect()
        else:
            candidates = _query_mp(self.mp_context, self.clusters, **query_args)

        #### testing distribute query vs. one-core query
        # result_distributed = query_rdd.collect()
        # result_distributed.sort(key=lambda x: x[0])
        # result_distributed = result_distributed[:10]
        # result_one_core = a
        # result_one_core.sort(key=lambda x: x[0])
        # result_one_core = result_one_core[:10]
        # is_same = np.equal(result_distributed, result_one_core)

        heapq.heapify(candidates)
        best_matches = []

        for i in range(best_k):
            best_matches.append(heapq.heappop(candidates))

        return best_matches

    def set_cluster_meta_dict(self, cluster_meta_dict):
        self.cluster_meta_dict = cluster_meta_dict

    def get_max_seq_len(self):
        return max([len(x[1]) for x in self.data_normalized])

    def stop(self):
        """
        Must be called before removing a gxe object
        """
        if self.is_using_spark():
            self.mp_context.stop()
        else:
            self.mp_context.terminate()
            self.mp_context.close()


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
