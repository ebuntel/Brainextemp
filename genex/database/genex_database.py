import heapq
import json
import os
import pickle
from pyspark import SparkContext
from pyspark.sql import SQLContext
import pandas as pd
import numpy as np
import shutil
from genex.classes.Sequence import Sequence
from genex.cluster import sim_between_seq, filter_cluster, lb_kim_sequence, lb_keogh_sequence
from genex.preprocess import get_subsequences, genex_normalize
from genex.utils import scale, _validate_gxdb_build_arguments, _df_to_list


def from_csv(file_name, feature_num: int, sc: SparkContext):
    """
    build a genex_database object from given csv,
    Note: if time series are of different length, shorter sequences will be post padded to the length
    of the longest sequence in the dataset

    :param file_name:
    :param feature_num:
    :param sc:
    :return:
    """

    df = pd.read_csv(file_name)
    data_list = _df_to_list(df, feature_num=feature_num)

    data_norm_list, global_max, global_min = genex_normalize(data_list, z_normalization=True)

    # return Genex_database
    return genex_database(data=data_list, data_normalized=data_norm_list, global_max=global_max, global_min=global_min,
                          spark_context=sc)


def from_db(sc: SparkContext, path: str):
    """

    :param sc:
    :param path:
    :return:
    """

    # TODO the input fold_name is not existed
    data = pickle.load(open(os.path.join(path, 'data.gxdb'), 'rb'))
    data_normalized = pickle.load(open(os.path.join(path, 'data_normalized.gxdb'), 'rb'))

    conf = json.load(open(os.path.join(path, 'conf.json'), 'rb'))
    init_params = {'data': data, 'data_normalized': data_normalized, 'spark_context': sc,
                   'global_max': conf['global_max'], 'global_min': conf['global_min']}
    db = genex_database(**init_params)

    db.set_clusters(db.get_sc().pickleFile(os.path.join(path, 'clustered_data/*')))

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
        self.clusters = None


        self.conf = {'build_conf': None,
                     'global_max': kwargs['global_max'],
                     'global_min': kwargs['global_min']}

    def set_conf(self, conf):
        self.conf = conf

    def set_clusters(self, clusters):
        self.clusters

    def get_sc(self):
        return self.sc

    def _process_loi(self, loi):
        if not loi:
            loi = slice(1, )  # set the end to the max time series length

        if not loi.start:
            loi.start = 1
        if not loi.end:
            pass

    def build(self, similarity_threshold: float, dist_type: str = 'eu', loi: slice = None, verbose: int = 1):
        """

        :param loi: default value is none, otherwise using slice notation [start, stop: step]
        :param similarity_threshold:
        :param dist_type:
        :param verbose:
        :return:
        """
        _validate_gxdb_build_arguments(self, locals())

        # update build configuration
        self.conf['build_conf'] = {'similarity_threshold': similarity_threshold,
                                   'dist_type': dist_type,
                                   'loi': loi}

        # Transforming pandas dataframe into spark dataframe
        sqlCtx = SQLContext(self.sc)

        data_rdd = sqlCtx.createDataFrame(self.data_normalized).rdd

        input_rdd = data_rdd.map(
            lambda x: _row_to_feature_and_data(x)
        )

        # validate and save the loi to gxdb class fields
        self._process_loi(loi)

        # Grouping the data
        group_rdd = input_rdd.flatMap(
            lambda x: get_subsequences(x, loi)
        )

        # Cluster the data with Gcluster
        cluster_rdd = group_rdd.mapPartitions(lambda x: filter_cluster(groups=x, st=similarity_threshold, log_level=1),
                                              preservesPartitioning=False).cache()

        # TODO check spark action for better strategy
        cluster_rdd.collect()

        self.clusters = cluster_rdd

    def save(self, path: str):
        """
        The save method saves the databse onto the disk.
        :param path: path to save the database to
        :return:
        """
        if os.path.exists(path):
            print('Path ' + path + ' already exists, overwriting...')
            shutil.rmtree(path)
            os.makedirs(path)

        # save the clusters if the db is built
        if self.clusters is not None:
            self.clusters.saveAsPickleFile(os.path.join(path, 'clusters.gxdb'))

        # save data files
        pickle.dump(self.data, open(os.path.join(path, 'data.gxdb'), 'wb'))
        pickle.dump(self.data_normalized, open(os.path.join(path, 'data_normalized.gxdb'), 'wb'))

        # save configs
        with open(path + '/conf.json', 'w') as f:
            json.dump(self.conf, f, indent=4)

    def query(self, query: Sequence, best_k: int, unique_id: bool, overlap: float):
        """

        :param overlap:
        :param query:
        :param best_k:
        :param unique_id:
        :return:
        """
        query_bc = self.sc.broadcast(query)
        df_z_norm_list = self.data_normalized.values.tolist()
        df_z_norm_list_bc = self.sc.broadcast(map(lambda x: _row_to_feature_and_data(x), df_z_norm_list))

        query_result_partition = self.clusters.mapPartitions(
            lambda x:
            _query_cluster_partition(cluster=x, q=query_bc, st=self.conf.get('similarity_threshold'), k=best_k,
                                     normalized_input=df_z_norm_list_bc, dist_type=self.conf.get('dist_type'),
                                     exclude_same_id=unique_id, overlap=overlap)
        ).cache()
        # TODO fetch data values from dataFrame
        aggre_q_rlts = query_result_partition.takeOrdered(best_k, key=lambda x: x[0])

        raw_data = map(lambda x: _row_to_feature_and_data(x), self.data.values.tolist())

        for i in aggre_q_rlts:
            i[1].set_data(i[1].fetch_data(raw_data))

        return aggre_q_rlts

    # TODO
    # mydb.select: get cluster
    # mydb

    def __len__(self):
        try:
            assert self.collected
        except AssertionError:
            raise Exception('Gcluster must be _collected before retrieving items, use gcluster.collect()')
        try:
            return len(self.clusters.keys())
        except AttributeError as error:
            raise Exception('Gcluster clusters not set')

    def __getitem__(self, sliced: slice):
        try:
            assert self.collected
        except AssertionError:
            raise Exception('Gcluster must be _collected before retrieving items, use gcluster.collect()')

        if isinstance(sliced, int):
            try:
                try:
                    assert min(self.clusters.keys()) <= sliced <= max(self.clusters.keys())
                except AssertionError:
                    raise Exception('Slicing Gcluster index out of bound')
                return self.clusters[sliced]
            except KeyError as error:
                raise Exception('This Gcluser does not have cluster of given length')

        try:
            assert slice.step is not None
        except AssertionError:
            raise Exception('Gcluser does not support stepping in slice')

        try:  # making sure that the slice index is within bound
            if sliced.start is not None:
                assert sliced.start >= min(self.clusters.keys())
            if sliced.stop is not None:
                assert sliced.stop <= max(self.clusters.keys())
        except AssertionError as error:
            raise Exception('Slicing Gcluster index out of bound')

        rtn = []

        if sliced.start is not None and sliced.stop is not None:
            for i in range(sliced.start, sliced.stop + 1):
                rtn.append(self.clusters[i])

        elif sliced.start is None and sliced.stop is not None:
            for i in range(min(self.clusters.keys()), sliced.stop + 1):
                rtn.append(self.clusters[i])

        elif sliced.start is not None and sliced.stop is None:
            for i in range(sliced.start, max(self.clusters.keys()) + 1):
                rtn.append(self.clusters[i])

        return rtn

    def __str__(self):
        if not self.collected:
            return 'Gluster at ' + str(hex(id(self))) + ' is NOT collected'
        else:
            return

    def _set_data_dict(self, data_dict: dict):
        self.clusters = data_dict

    def collect(self):
        try:
            assert not self.collected
        except AssertionError:
            raise Exception('Gcluster is already _collected')

        self.clusters = dict(self.clusters.collect())
        self.collected = True

    def gfilter(self, size=None, filter_features=None):
        """

        :param size: length can be an integer or a tuple or a list of two integers
        :param filter_features:
        """
        # check if ther result has been collected
        try:
            assert self.collected
        except AssertionError:
            raise Exception('Gluster at ' + str(hex(id(self))) + ' is NOT collected')

        try:  # validate size parameter
            if size is not None:
                assert isinstance(size, int) or isinstance(size, list) or isinstance(size, tuple)
                if isinstance(size, list) or isinstance(size, tuple):
                    assert len(size) == 2
                    assert size[0] < size[1]
                    for item in size:
                        assert isinstance(item, int)

        except AssertionError:
            raise Exception('Given size for filtering clusters is not valid, check filter under Gcluster in the '
                            'documentation for more details')

        try:  # validate filter_features parameter
            if filter_features is not None:
                assert isinstance(filter_features, list) or isinstance(filter_features, tuple) or isinstance(
                    filter_features, str)
                if isinstance(filter_features, list) or isinstance(filter_features, tuple):
                    assert len(filter_features) != 0
                    for feature in filter_features:
                        assert feature in self.feature_list
                elif isinstance(filter_features, str):
                    assert filter_features in self.feature_list
        except AssertionError:
            raise Exception(
                'Given filter_features(s) for filtering clusters is not valid, filter_features for filtering '
                'must contain at least one '
                'filter_features and provided filter_features must be presented in the dataset')

        self.filters = (size, filter_features)  # update self filters
        self.filtered_clusters = self.clusters

        # filter the clusters by size
        if isinstance(size, int):
            self.filtered_clusters = dict(filter(lambda x: x[0] == size, list(self.clusters.items())))
        elif isinstance(size, list) or isinstance(size, tuple):
            self.filtered_clusters = dict(
                filter(lambda x: x[0] in range(size[0], size[1] + 1), list(self.clusters.items())))

        if isinstance(filter_features, str):
            self.filtered_clusters = dict(map(lambda seq_size_cluster:
                                              (seq_size_cluster[0],
                                               dict(map(lambda repr_cluster:
                                                        (repr_cluster[0],  # the representative of the cluster
                                                         list(filter(
                                                             lambda cluster_seq:
                                                             (filter_features in cluster_seq.id) or repr_cluster[
                                                                 0] == cluster_seq,
                                                             repr_cluster[1]
                                                             # list that contains all the seqence in the cluster

                                                         ))), seq_size_cluster[1].items()))),
                                              self.filtered_clusters.items()))
            # feature filter is applied on clusters that has already been filtered by size
        # todo implement && filter
        elif isinstance(filter_features, list):
            self.filtered_clusters = dict(map(lambda seq_size_cluster:
                                              (seq_size_cluster[0],
                                               dict(map(lambda repr_cluster:
                                                        (repr_cluster[0],  # the representative of the cluster
                                                         list(filter(
                                                             lambda cluster_seq:
                                                             (any([i for i in filter_features if
                                                                   i in cluster_seq.id])) or
                                                             repr_cluster[0] == cluster_seq,
                                                             repr_cluster[1]
                                                             # list that contains all the seqence in the cluster

                                                         ))), seq_size_cluster[1].items()))),
                                              self.filtered_clusters.items()))
            # feature filter is applied on clusters that has already been filtered by size

    def _gfilter(self, size=None, filter_features=None):
        """

        :param size: length can be an integer or a tuple or a list of two integers
        :param filter_features:
        """
        # check if ther result has been collected
        try:
            assert self.collected
        except AssertionError:
            raise Exception('Gluster at ' + str(hex(id(self))) + ' is NOT collected')

        try:  # validate size parameter
            if size is not None:
                assert isinstance(size, int) or isinstance(size, list) or isinstance(size, tuple)
                if isinstance(size, list) or isinstance(size, tuple):
                    assert len(size) == 2
                    assert size[0] < size[1]
                    for item in size:
                        assert isinstance(item, int)

        except AssertionError:
            raise Exception('Given size for filtering clusters is not valid, check filter under Gcluster in the '
                            'documentation for more details')

        try:  # validate filter_features parameter
            if filter_features is not None:
                assert isinstance(filter_features, list) or isinstance(filter_features, tuple) or isinstance(
                    filter_features, str)
                if isinstance(filter_features, list) or isinstance(filter_features, tuple):
                    assert len(filter_features) != 0
                    for feature in filter_features:
                        assert feature in self.feature_list
                elif isinstance(filter_features, str):
                    assert filter_features in self.feature_list
        except AssertionError:
            raise Exception(
                'Given filter_features(s) for filtering clusters is not valid, filter_features for filtering '
                'must contain at least one '
                'filter_features and provided filter_features must be presented in the dataset')

        filter_result = self.clusters;
        # filter the clusters by size
        if isinstance(size, int):
            filter_result = dict(filter(lambda x: x[0] == size, list(filter_result.items())))
        elif isinstance(size, list) or isinstance(size, tuple):
            filter_result = dict(filter(lambda x: x[0] in range(size[0], size[1] + 1), list(filter_result.items())))

        if isinstance(filter_features, str):
            filter_result = dict(map(lambda seq_size_cluster:
                                     (seq_size_cluster[0],
                                      dict(map(lambda repr_cluster:
                                               (repr_cluster[0],  # the representative of the cluster
                                                list(filter(
                                                    lambda cluster_seq:
                                                    (filter_features in cluster_seq.id) or repr_cluster[
                                                        0] == cluster_seq,
                                                    repr_cluster[1]
                                                    # list that contains all the seqence in the cluster

                                                ))), seq_size_cluster[1].items()))), filter_result.items()))
            # feature filter is applied on clusters that has already been filtered by size
        # todo implement && filter
        elif isinstance(filter_features, list):
            filter_result = dict(map(lambda seq_size_cluster:
                                     (seq_size_cluster[0],
                                      dict(map(lambda repr_cluster:
                                               (repr_cluster[0],  # the representative of the cluster
                                                list(filter(
                                                    lambda cluster_seq:
                                                    (any([i for i in filter_features if i in cluster_seq.id])) or
                                                    repr_cluster[0] == cluster_seq,
                                                    repr_cluster[1]
                                                    # list that contains all the seqence in the cluster

                                                ))), seq_size_cluster[1].items()))), filter_result.items()))
            # feature filter is applied on clusters that has already been filtered by size
        return filter_result

    def get_feature_list(self):
        return self.feature_list

    # methods to retrieve the actual clusters
    def get_representatives(self, filter=False):
        d = list(self.filtered_clusters if filter else self.clusters.items())
        e = map(lambda x: [x[0], list(x[1].keys())], d)

        return dict(e)

    def get_cluster(self, rep_seq):
        try:
            lenc = self.clusters[len(rep_seq)]
            return self.clusters[len(rep_seq)][rep_seq]
        except KeyError as e:
            raise Exception("Gcluster: get_cluster: does not have a cluster represented by the given representative")

    def gquery(self, query_sequence: Sequence, sc: SparkContext,
               loi=None, foi=None, k: int = 1, dist_type: str = 'eu', data_slices: int = 32,
               ex_sameID: bool = False, overlap: float = 0.0):
        # TODO update gquery so that it can utilize past query result to do new queries
        # input validation
        try:
            query_sequence.fetch_data(input_list=self.norm_data)
        except KeyError as ke:
            raise Exception('Given query sequence is not present in this Gcluster')

        if overlap != 0.0:
            try:
                assert 0.0 <= overlap <= 1.0
            except AssertionError as e:
                raise Exception('gquery: overlap factor must be a float between 0.0 and 1.0')

        if loi[0] <= 0:
            raise Exception('gquery: first element of loi must be equal to or greater than 1')
        if loi[0] >= loi[1]:
            raise Exception('gquery: Start must be greater than end in the '
                            'Length of Interest')

        r_heap = self._gfilter(size=loi, filter_features=foi)  # retrieve cluster sequences of interests
        r_heap = list(r_heap.items())

        bc_norm_data = sc.broadcast(
            self.norm_data)  # broadcast the normalized data so that the Sequence objects can find data faster
        rheap_rdd = sc.parallelize(r_heap, numSlices=data_slices)
        rheap_rdd = rheap_rdd.flatMap(lambda x: x[1])  # retrieve all the sequences and flatten

        # calculate the distances, create a key-value pair: key = dist from query to the sequence, value = the sequence
        # ready to be heapified!
        rheap_rdd = rheap_rdd.map(lambda x: (
            sim_between_seq(query_sequence.fetch_data(bc_norm_data.value), x.fetch_data(bc_norm_data.value),
                            dist_type=dist_type), x))
        r_heap = rheap_rdd.collect()
        heapq.heapify(r_heap)

        query_result = []

        while len(query_result) < k:
            # create a cluster to query
            querying_cluster = []
            while len(querying_cluster) <= k:
                try:
                    top_rep = heapq.heappop(r_heap)
                except IndexError as ie:
                    print('Warning: R space exhausted, best k not reached, returning all the matches so far')
                    return query_result

                querying_cluster = querying_cluster + self.get_cluster(
                    top_rep[1])  # top_rep: (dist to query, rep sequence)

            query_cluster_rdd = sc.parallelize(querying_cluster, numSlices=data_slices)

            if ex_sameID:  # filter by not same id
                query_cluster_rdd = query_cluster_rdd.filter(lambda x: x.id != query_sequence.id)

            # TODO do not fetch data everytime for the query sequence
            query_cluster_rdd = query_cluster_rdd.map(lambda x: (
                sim_between_seq(query_sequence.fetch_data(bc_norm_data.value), x.fetch_data(bc_norm_data.value),
                                dist_type=dist_type), x))
            qheap = query_cluster_rdd.collect()
            heapq.heapify(qheap)

            while len(query_result) < k and len(qheap) != 0:
                current_match = heapq.heappop(qheap)

                if not any(_isOverlap(current_match[1], prev_match[1], overlap) for prev_match in
                           query_result):  # check for overlap against all the matches so far
                    query_result.append(current_match)

        return query_result


def _query_cluster_partition(cluster, q, st: float, k: int, normalized_input, dist_type: str, loi=None,
                             exclude_same_id: bool = True, overlap: float = 1.0,
                             lb_heuristic=True, reduction_factor_lbkim='half', reduction_factor_lbkeogh=2):
    if reduction_factor_lbkim == 'half':
        lbkim_mult_factor = 0.5
    elif reduction_factor_lbkim == '1quater':
        lbkim_mult_factor = 0.25
    elif reduction_factor_lbkim == '3quater':
        lbkim_mult_factor = 0.75
    else:
        raise Exception('reduction factor must be one of the specified string')

    if reduction_factor_lbkeogh == 'half':
        lbkeogh_mult_factor = 0.5
    elif reduction_factor_lbkeogh == '1quater':
        lbkeogh_mult_factor = 0.25
    elif reduction_factor_lbkeogh == '3quater':
        lbkeogh_mult_factor = 0.75
    else:
        raise Exception('reduction factor must be one of the specified string')

    q_value = q.value
    q_length = len(q_value.data)

    normalized_input = normalized_input.value

    if loi is not None:
        cluster_dict = dict(x for x in cluster if x[0] in range(loi[0], loi[1]))
    else:
        cluster_dict = dict(cluster)
        # get the seq length of the query, start query in the cluster that has the same length as the query

    target_length = len(q_value)

    # get the seq length Range of the partition
    try:
        len_range = (min(cluster_dict.keys()), max(cluster_dict.keys()))
    except ValueError as ve:
        raise Exception('cluster does not have the given query loi!')

    # if given query is longer than the longest cluster sequence,
    # set starting clusterSeq length to be of the same length as the longest sequence in the partition
    target_length = max(min(target_length, len_range[1]), len_range[0])

    query_result = []
    # temperoray variable that decides whether to look up or down when a cluster of a specific length is exhausted

    while len(cluster_dict) > 0:
        target_cluster = cluster_dict[target_length]
        target_cluster_reprs = target_cluster.keys()
        target_cluster_reprs = list(
            map(lambda rpr: [sim_between_seq(rpr.fetch_data(normalized_input), q_value.data, dist_type=dist_type), rpr],
                target_cluster_reprs))
        # add a counter to avoid comparing a Sequence object with another Sequence object
        heapq.heapify(target_cluster_reprs)

        while len(target_cluster_reprs) > 0:
            querying_repr = heapq.heappop(target_cluster_reprs)
            querying_cluster = target_cluster[querying_repr[1]]

            # filter by id
            if exclude_same_id:
                querying_cluster = (x for x in querying_cluster if x.id != q_value.id)

            # fetch data for the target cluster
            querying_cluster = ((x, x.fetch_data(normalized_input)) for x in
                                querying_cluster)  # now entries are (seq, data)
            if lb_heuristic:
                # Sorting sequence using cascading bounds
                # need to make sure that the query and the candidates are of the same length when calculating LB_keogh
                if target_length != q_length:
                    print('interpolating')
                    querying_cluster = (
                        (x[0], np.interp(np.linspace(0, 1, q_length), np.linspace(0, 1, len(x[1])), x[1])) for x in
                        querying_cluster)  # now entries are (seq, interp_data)

                # sorting the sequence using LB_KIM bound
                querying_cluster = [(x[0], x[1], lb_kim_sequence(x[1], q_value.data)) for x in querying_cluster]
                querying_cluster.sort(key=lambda x: x[2])
                # checking how much we reduce the cluster
                if type(reduction_factor_lbkim) == str:
                    querying_cluster = querying_cluster[:int(len(querying_cluster) * lbkim_mult_factor)]
                elif type(reduction_factor_lbkim) == int:
                    querying_cluster = querying_cluster[:reduction_factor_lbkim * k]
                else:
                    raise Exception('Type of reduction factor must be str or int')

                # Sorting the sequence using LB Keogh bound
                querying_cluster = [(x[0], x[1], lb_keogh_sequence(x[1], q_value.data)) for x in
                                    querying_cluster]  # now entries are (seq, data, lb_heuristic)
                querying_cluster.sort(key=lambda x: x[2])  # sort by the lb_heuristic
                # checking how much we reduce the cluster
                if type(reduction_factor_lbkeogh) == str:
                    querying_cluster = querying_cluster[:int(len(querying_cluster) * lbkeogh_mult_factor)]
                elif type(reduction_factor_lbkim) == int:
                    querying_cluster = querying_cluster[:reduction_factor_lbkeogh * k]
                else:
                    raise Exception('Type of reduction factor must be str or int')

            querying_cluster = [(sim_between_seq(x[1], q_value.data, dist_type=dist_type), x[0]) for x in
                                querying_cluster]  # now entries are (dist, seq)

            heapq.heapify(querying_cluster)

            for cur_match in querying_cluster:
                if cur_match[0] < st:
                    if overlap != 1.0:
                        if not any(_isOverlap(cur_match[1], prev_match[1], overlap) for prev_match in
                                   query_result):  # check for overlap against all the matches so far
                            print('Adding to querying result')
                            query_result.append(cur_match[:2])
                        else:
                            print('Overlapped, Not adding to query result')
                    else:
                        print('Not applying overlapping')
                        query_result.append(cur_match[:2])

                    if (len(query_result)) >= k:
                        return query_result

        cluster_dict.pop(target_length)  # remove this len-cluster just queried

        # find the next closest sequence length
        if len(cluster_dict) != 0:
            target_length = min(list(cluster_dict.keys()), key=lambda x: abs(x - target_length))
        else:
            break
    return query_result


# Transforming each row object in a RDD[Row] or DataFrame to a list which contains a tuple as keyID and list as value
# return [(ID), [values]]

def _isOverlap(seq1: Sequence, seq2: Sequence, overlap: float) -> bool:
    if seq1.id != seq2.id:  # overlap does NOT matter if two seq have different id
        return True
    else:
        of = _calculate_overlap(seq1, seq2)
        return _calculate_overlap(seq1, seq2) >= overlap


def _calculate_overlap(seq1, seq2) -> float:
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
