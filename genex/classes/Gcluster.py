class Gcluster:
    """
    Genex Clusters Object

    clusters: dictionary object that holds the cluster information
        key(integer): length of the sequence in the cluster
        value: dict: the clustered sequence of keyed length
            key: Sequence object that is the representative
            value: list of Sequence that are represented by the key
    """

    def __init__(self, feature_list, cluster_dict=None, collected: bool = None, global_max: float = None,
                 global_min: float = None):
        self.feature_list = feature_list
        self.clusters = cluster_dict

        self.filtered_clusters = cluster_dict
        self.filters = None

        self._collected = collected

        self.global_max = global_max
        self.global_min = global_min

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
            raise Exception('Given filter_features(s) for filtering clusters is not valid, filter_features for filtering '
                            'must contain at least one '
                            'filter_features and provided filter_features must be presented in the dataset')

        self.filters = (size, filter_features)  # update self filters
        # filter the clusters by size
        if isinstance(size, int):
            self.filtered_clusters = dict(filter(lambda x: x[0] == size, list(self.clusters.items())))
        elif isinstance(size, list) or isinstance(size, tuple):
            self.filtered_clusters = dict(filter(lambda x: x[0] in range(size[0], size[1] + 1), list(self.clusters.items())))

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

                                                     ))), seq_size_cluster[1].items()))), self.filtered_clusters.items()))
            # feature filter is applied on clusters that has already been filtered by size
        # todo implement && filter
        elif isinstance(filter_features, list):
            self.filtered_clusters = dict(map(lambda seq_size_cluster:
                                          (seq_size_cluster[0],
                                           dict(map(lambda repr_cluster:
                                                    (repr_cluster[0],  # the representative of the cluster
                                                     list(filter(
                                                         lambda cluster_seq:
                                                         (any([i for i in filter_features if i in cluster_seq.id])) or
                                                         repr_cluster[0] == cluster_seq,
                                                         repr_cluster[1]
                                                         # list that contains all the seqence in the cluster

                                                     ))), seq_size_cluster[1].items()))), self.filtered_clusters.items()))
            # feature filter is applied on clusters that has already been filtered by size

    def get_feature_list(self):
        return self.feature_list

    # methods to retrieve the actual clusters
    def get_representatives(self, filter=False):
        d = list(self.filtered_clusters if filter else self.clusters.items())
        e = map(lambda x: [x[0], list(x[1].keys())], d)

        return dict(e)
