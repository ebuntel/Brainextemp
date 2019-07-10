import heapq
from pyspark import SparkContext

from .Sequence import Sequence
from ..cluster import sim_between_seq


class Gcluster:
    """
    Genex Clusters Object

    clusters: dictionary object that holds the cluster information
        key(integer): length of the sequence in the cluster
        value: dict: the clustered sequence of keyed length
            key: Sequence object that is the representative
            value: list of Sequence that are represented by the key
    """

    def __init__(self, feature_list, data, norm_data, cluster_dict=None, collected: bool = None,
                 global_max: float = None,
                 global_min: float = None):
        self.feature_list = feature_list
        self.data = data
        self.norm_data = norm_data

        self.clusters = cluster_dict

        self.filtered_clusters = cluster_dict
        self.filters = None

        self._collected = collected

        self.global_max = global_max
        self.global_min = global_min

        self._qheap = []

    def __len__(self):
        try:
            assert self._collected
        except AssertionError:
            raise Exception('Gcluster must be _collected before retrieving items, use gcluster.collect()')
        try:
            return len(self.clusters.keys())
        except AttributeError as error:
            raise Exception('Gcluster clusters not set')

    def __getitem__(self, sliced: slice):
        try:
            assert self._collected
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
        if not self._collected:
            return 'Gluster at ' + str(hex(id(self))) + ' is NOT collected'
        else:
            return

    def _set_data_dict(self, data_dict: dict):
        self.clusters = data_dict

    def collect(self):
        try:
            assert not self._collected
        except AssertionError:
            raise Exception('Gcluster is already _collected')

        self.clusters = dict(self.clusters.collect())
        self._collected = True

    def gfilter(self, size=None, filter_features=None):
        """

        :param size: length can be an integer or a tuple or a list of two integers
        :param filter_features:
        """
        # check if ther result has been collected
        try:
            assert self._collected
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
            assert self._collected
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
        # TODO query_sequence from external sources
        if loi[0] <= 0:
            raise Exception('gquery: first element of loi must be equal to or greater than 1')
        if loi[0] >= loi[1]:
            raise Exception('gquery: Start must be greater than end in the '
                            'Length of Interest')

        self._qheap = self._gfilter(size=loi, filter_features=foi)  # retrieve cluster sequences of interests
        self._qheap = list(self._qheap.items())

        bc_norm_data = sc.broadcast(
            self.norm_data)  # broadcast the normalized data so that the Sequence objects can find data faster
        qheap_rdd = sc.parallelize(self._qheap, numSlices=data_slices)
        qheap_rdd = qheap_rdd.flatMap(lambda x: x[1])  # retrieve all the sequences and flatten

        # calculate the distances, create a key-value pair: key = dist from query to the sequence, value = the sequence
        # ready to be heapified!
        qheap_rdd = qheap_rdd.map(lambda x: (
            sim_between_seq(query_sequence.fetch_data(bc_norm_data.value), x.fetch_data(bc_norm_data.value), dist_type=dist_type), x))
        self._qheap = qheap_rdd.collect()
        heapq.heapify(self._qheap)

        # create a cluster to query
        querying_cluster = []
        while len(querying_cluster) <= k:
            top_rep = heapq.heappop(self._qheap)
            querying_cluster = querying_cluster + self.get_cluster(top_rep[1])  # top_rep: (dist to query, rep sequence)

        query_cluster_rdd = sc.parallelize(querying_cluster, numSlices=data_slices)

        if ex_sameID:  # filter by not same id
            query_cluster_rdd = query_cluster_rdd.filter(lambda x: x.id != query_sequence.id)

        # TODO do not fetch data everytime for the query sequence
        query_cluster_rdd = query_cluster_rdd.map(lambda x: (
            sim_between_seq(query_sequence.fetch_data(bc_norm_data.value), x.fetch_data(bc_norm_data.value), dist_type=dist_type), x))
        self._qheap = query_cluster_rdd.collect()
        heapq.heapify(self._qheap)

        query_result = []
        if overlap != 0.0:
            try:
                assert 0.0 <= overlap <= 1.0
            except AssertionError as e:
                raise Exception('gquery: overlap factor must be a float between 0.0 and 1.0')

        last_best_match = None
        while len(query_result) < k:
            current_match = heapq.heappop(self._qheap)

            if last_best_match:
                if not _isOverlap(current_match[1], last_best_match[1],
                                  overlap):  # check for overlap against the last best match
                    query_result.append(current_match)  # note this may result in number of query_result is less than k
                    last_best_match = current_match
            else:
                query_result.append(current_match)
                last_best_match = current_match

        return query_result


def _isOverlap(seq1: Sequence, seq2: Sequence, overlap: float) -> bool:
    if seq1.id != seq2.id:  # overlap does NOT matter if two seq have different id
        return True
    else:
        of =  _calculate_overlap(seq1, seq2)
        return _calculate_overlap(seq1, seq2) >= overlap


def _calculate_overlap(seq1, seq2) -> float:
    if seq2.end > seq1.end and seq2.start > seq1.start:
        return (seq1.end - seq2.start + 1) / (seq2.end - seq1.start + 1)
    elif seq1.end > seq2.end and seq1.start > seq2.start:
        return (seq2.end - seq1.start + 1) / (seq1.end - seq2.start + 1)
    elif seq1.end > seq2.end and seq2.start > seq1.start:
        return len(seq2) / len(seq1)
    elif seq2.end > seq1.end and seq1.start > seq2.start:
        return len(seq1) / len(seq2)

    elif seq2.start > seq1.end or seq1.start > seq2.end:  # does not overlap at all
        return 0.0
    else:
        raise Exception('FATAL: sequence 100% overlap, please report the bug')
