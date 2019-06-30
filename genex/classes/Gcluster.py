class Gcluster:
    """
    Genex Clusters Object

    data_dict: dictionary object that holds the cluster information
        key(integer): length of the sequence in the cluster
        value: dict: the clustered sequence of keyed length
            key: Sequence object that is the representative
            value: list of Sequence that are represented by the key
    """
    def __init__(self, data_dict: dict = None):
        self.data_dict = data_dict

    def __len__(self):
        try:
            return len(self.data_dict.keys())
        except AttributeError as error:
            raise Exception('Gcluster data not set')

    def __getitem__(self, sliced:slice):
        if isinstance(sliced, int):
            try:
                return self.data_dict[sliced]
            except KeyError as error:
                raise Exception('This Gcluser does not have cluster of given length')
                return

        try:
            assert slice.step is not None
        except AssertionError:
            raise Exception('Gcluser does not support stepping in slice')

        try:  # making sure that the slice index is within bound
            if sliced.start is not None:
                assert sliced.start >= min(self.data_dict.keys())
            if sliced.stop is not None:
                assert sliced.stop <= max(self.data_dict.keys())
        except AssertionError as error:
            raise Exception('Slicing Gcluster index out of bound')

        rtn = []

        if sliced.start is not None and sliced.stop is not None:
            for i in range(sliced.start, sliced.stop+1):
                rtn.append(self.data_dict[i])

        elif sliced.start is None and sliced.stop is not None:
            for i in range(min(self.data_dict.keys()),  sliced.stop+1):
                rtn.append(self.data_dict[i])

        elif sliced.start is not None and sliced.stop is None:
            for i in range(sliced.start,  max(self.data_dict.keys())+1):
                rtn.append(self.data_dict[i])

        return rtn

    def _set_data_dict(self, data_dict: dict):
        self.data_dict = data_dict
