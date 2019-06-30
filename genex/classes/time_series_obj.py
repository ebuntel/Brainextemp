class TimeSeriesObj:
    def __init__(self, tid, start_point, end_point, raw_data=None, group_in=None, is_representative=False):
        self.id = tid
        self.start_point = start_point
        self.end_point = end_point
        self.raw_data = raw_data
        # represent which cluster
        self.group_in = group_in
        self.is_representative = is_representative

    def get_length(self):
        return self.end_point - self.start_point

    def set_group_represented(self, group_id):
        self.group_in = group_id

    def get_group_represented(self):
        return self.group_in

    def remove_group_represented(self):
        self.group_in = None

    def set_raw_data(self, new_raw_data):
        self.raw_data = new_raw_data

    def get_raw_data(self):
        return self.raw_data

    def set_representative(self):
        self.is_representative = True

    def remove_representative(self):
        self.is_representative = False

    def to_string(self):
        return self.id + str(self.start_point) + str(self.end_point)
