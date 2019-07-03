class Gcluster:
    """
    Genex Clusters Object

    data: dictionary object that holds the cluster information
        key(integer): length of the sequence in the cluster
        value: dict: the clustered sequence of keyed length
            key: Sequence object that is the representative
            value: list of Sequence that are represented by the key
    """
    def __init__(self, data_dict = None, collected=None):
        self.data = data_dict
        self._collected = collected

    def __len__(self):
        try:
            assert self._collected
        except AssertionError:
            raise Exception('Gcluster must be _collected before retrieving items, use gcluster.collect()')
        try:
            return len(self.data.keys())
        except AttributeError as error:
            raise Exception('Gcluster data not set')

    def __getitem__(self, sliced:slice):
        try:
            assert self._collected
        except AssertionError:
            raise Exception('Gcluster must be _collected before retrieving items, use gcluster.collect()')

        if isinstance(sliced, int):
            try:
                try:
                    assert min(self.data.keys()) <= sliced <= max(self.data.keys())
                except AssertionError:
                    raise Exception('Slicing Gcluster index out of bound')
                return self.data[sliced]
            except KeyError as error:
                raise Exception('This Gcluser does not have cluster of given length')

        try:
            assert slice.step is not None
        except AssertionError:
            raise Exception('Gcluser does not support stepping in slice')

        try:  # making sure that the slice index is within bound
            if sliced.start is not None:
                assert sliced.start >= min(self.data.keys())
            if sliced.stop is not None:
                assert sliced.stop <= max(self.data.keys())
        except AssertionError as error:
            raise Exception('Slicing Gcluster index out of bound')

        rtn = []

        if sliced.start is not None and sliced.stop is not None:
            for i in range(sliced.start, sliced.stop+1):
                rtn.append(self.data[i])

        elif sliced.start is None and sliced.stop is not None:
            for i in range(min(self.data.keys()), sliced.stop + 1):
                rtn.append(self.data[i])

        elif sliced.start is not None and sliced.stop is None:
            for i in range(sliced.start, max(self.data.keys()) + 1):
                rtn.append(self.data[i])

        return rtn

    def _set_data_dict(self, data_dict: dict):
        self.data = data_dict

    def collect(self):
        try:
            assert not self._collected
        except AssertionError:
            raise Exception('Gcluster is already _collected')

        self.data = dict(self.data.collect())
        self._collected = True