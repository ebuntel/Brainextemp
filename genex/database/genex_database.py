import heapq
import json
import os
import pickle
from pyspark import SparkContext, StorageLevel
from pyspark.sql import SQLContext
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import shutil
from genex.classes.Sequence import Sequence
from genex.cluster import sim_between_seq, filter_cluster
from genex.preprocess import all_sublists_with_id_length



def from_csv(file_name, feature_num: int):
    """
    build a genex_database object from given csv,
    Note: if time series are of different length, shorter sequences will be post padded to the length
    of the longest sequence in the dataset
    :param file_name:
    :return:
    """
    df = pd.read_csv(file_name).fillna(0)

    # prepare to minmax normalize the data columns
    def scale(ts_df):
        time_series = ts_df.iloc[:, feature_num:].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        num_time_series = len(time_series)
        time_series = time_series.reshape(-1, 1)
        time_series = scaler.fit_transform(time_series)
        time_series = time_series.reshape(num_time_series, -1)

        df_normalized = ts_df.copy()
        df_normalized.iloc[:, feature_num:] = time_series

        return df_normalized

    df_normalized = scale(df)

    # return genex_database
    rtn = genex_database(data=df, data_normalized=df_normalized, scale_func=scale)

    return rtn


def _validate_inputs(args):
    print(args)

    return


class genex_database:
    """
    Genex Database

    Init parameters
    data
    data_normalized
    scale_funct
    """

    def __init__(self, **kwargs):
        self.data = kwargs['data']
        self.data_normalized = kwargs['data_normalized']
        self.scale_func = kwargs['scale_func']

    def build(self, sc: SparkContext, similarity_threshold: float, dist_type: str = 'eu', verbose: int = 1):
        _validate_inputs(locals())
        self.conf = {'similarity_threshold':similarity_threshold, 'dist_type':dist_type, 'verbose':verbose}

        # Transforming pandas dataframe into spark dataframe
        sqlCtx = SQLContext(sc)

        # Only for test build functionality, just take the first 20 rows in the original dataframe
        data_rdd = sqlCtx.createDataFrame(self.data_normalized).rdd

        # Transforming the input_rdd RDD[Row] into a collection of lists which contains a tuple and list
        def _row_to_list(row):
            key = tuple([x for x in row[:4]])
            value = [y for y in row[5:]]
            return [key, value]

        input_rdd = data_rdd.map(
            lambda x: _row_to_list(x)
        )

        # Grouping the data
        group_rdd = input_rdd.flatMap(
            lambda x: all_sublists_with_id_length(x, [120])
        )

        # Cluster the data with Gcluster
        cluster_rdd = group_rdd.mapPartitions(lambda x: filter_cluster(groups=x, st=similarity_threshold, log_level=1),
                                              preservesPartitioning=False)

        cluster_rdd.first()

        sl = StorageLevel(True, True, False, False, 1)
        self.data_normalized_clustered = cluster_rdd.persist(storageLevel=sl)


    def save(self, folder_name: str):
        path = 'gxdb/' + folder_name

        if os.path.exists(path):
            shutil.rmtree(path)

        self.data_normalized_clustered.saveAsTextFile(path + '/clustered_data')
        self.data.to_csv(path + '/data.csv')
        self.data_normalized.to_csv(path + '/data_normalized.csv')

        with open(path + '/conf.json', 'w') as f:
            json.dump(self.conf, f, indent=4)

        # with open(path + '/functions.txt', 'w') as t:
        #     pickle.dump(self.scale_func, t)

    def from_db(self):
        pass

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
