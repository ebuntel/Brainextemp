class sequence:
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
        return self.id + ': (' + str(self.start) + ':' + str(self.end) + ')'

    def del_data(self):
        self.data = None

    def get_data(self):
        # TODO what to do after trimming the data
        return self.data

    def set_data(self, data):
        self.data = data