wrap_in_parantheses = lambda x: "(" + str(x) + ")"


class Sequence:
    def __init__(self, id:str, start:int, end:int):
        self.id = id
        self.start = start
        self.end = end

    def __init__(self, id:str, start:int, end:int, data:list):
        self.id = id
        self.start = start
        self.end = end
        self.data = data

    def __str__(self):
        label_features = [wrap_in_parantheses(feature) for feature in self.id]
        id = "_".join(label_features).replace('  ', '-').replace(' ', '-')
        return id + ': (' + str(self.start) + ':' + str(self.end) + ')'

    def del_data(self):
        self.data = None

    def get_data(self):
        # TODO what to do after trimming the data
        return self.data

    def set_data(self, data):
        self.data = data

    def fetch_data(self, input_list):
        # TODO not tested
        try:
            input_dict = dict(input_list)  # validate by converting input_list into a dict
        except (TypeError, ValueError):
            raise Exception('sequence: fetch_data: input_list is not key-value pair.')

        return input_dict[self.id][self.start: self.end]
