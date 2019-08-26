wrap_in_parantheses = lambda x: "(" + str(x) + ")"


class Sequence:

    def __init__(self, id: tuple, start: int, end: int, data: list = None):
        self.id = id
        self.start = start
        self.end = end
        self.data = data

    def __str__(self):
        label_features = [wrap_in_parantheses(feature) for feature in self.id]
        id = "_".join(label_features).replace('  ', '-').replace(' ', '-')
        return id + ': (' + str(self.start) + ':' + str(self.end) + ')'

    def __len__(self):
        return self.end - self.start + 1

    def __hash__(self):
        return hash((self.id, self.start, self.end))

    def __eq__(self, other):
        return (self.id, self.start, self.end) == (other.id, other.start, other.end)

    def __le__(self, other):
        return True

    def __ge__(self, other):
        return True

    def __lt__(self, other):
        return True

    # return comparison
    def __ne__(self, other):
        return True

    # return comparison
    def __gt__(self, other):
        return True

    def del_data(self):
        self.data = None

    def get_data(self):
        return self.data

    def set_data(self, data):
        self.data = data

    def fetch_data(self, input_list, save_data: bool = False):
        # TODO not tested
        try:
            input_dict = dict(input_list)  # validate by converting input_list into a dict
        except (TypeError, ValueError):
            raise Exception('sequence: fetch_data: input_list is not key-value pair.')

        try:
            return input_dict[self.id][self.start: self.end]
        except KeyError and IndexError as e:
            print(self)
            if type(e) is KeyError:
                raise Exception('Given data list does not have a sequence with this sequence id')
            elif type(e) is IndexError:
                raise Exception('This sequence is out of bound with given data list')

    def _check_feature(self, features):
        """

        :param features: feature can be a string or a list: features to check id against
        :return: true if given any of the feature in the given features is in self.id
        """
        try:
            assert isinstance(features, str) or isinstance(features, list) or isinstance(features, tuple)
        except AssertionError:
            raise Exception('Invalide features in _check_feature for the Sequence object')

        if isinstance(features, str):
            return features in self.id
        else:  # if features is a list or tuple
            return any([i for i in features if i in self.id])
