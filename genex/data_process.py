import math


def get_data(tid, time_series_dict, start=None, end=None):
    """
    This function return sub-sequence indexed by id, start point and end point in clusters

    :param tid: id of time series sequence
    :param start: starting index
    :param end: end index
    :param timeSeries: raw time series clusters

    :return list: return a sub-sequence indexed by id, start point and end point in clusters
    """
    # TODO should we make timeSeries a dic as well as a list (for distributed computing)

    if tid not in time_series_dict.keys():
        raise Exception("data_operations: get_data: subsequnce of ID " + tid + " not found in TimeSeries!")
    else:
        if start is None or end is None:
            return time_series_dict[tid][:]
        else:
            return time_series_dict[tid][start:end]


def get_data_for_timeSeriesObj(time_series_obj, time_series_dict):
    """
    This function returns data_original for given time series object

    :type time_series_obj: TimeSeriesObj
    :param time_series_obj: Time series object
    :param time_series_dict: dictionary of raw time series data_original
    :return: a dictionary containing data_original points for given time series id from start to end point
    """
    tid = time_series_obj.id
    if tid not in time_series_dict.keys():
        raise Exception("data_operations: get_data: subsequnce of ID " + tid + " not found in TimeSeries!")
    else:
        return time_series_dict[tid][time_series_obj.start_point:time_series_obj.end_point]


def normalize_ts_dict(time_series_dict):
    """
    this function is now performed by group_operations: generateSource
    :param time_series_dict:
    :return dict: normalized dataset
    """
    # get the max and min
    global_min = math.inf
    global_max = - math.inf

    for ts_key in time_series_dict.keys():
        if len(time_series_dict[ts_key]) == 0:
            continue
        ts_max = max(time_series_dict[ts_key])
        ts_min = min(time_series_dict[ts_key])

        if ts_max > global_max:
            global_max = ts_max

        if ts_min < global_min:
            global_min = ts_min

    normalized_time_series = dict()

    for ts_key in time_series_dict.keys():
        normalized_time_series[ts_key] = []
        for point in time_series_dict[ts_key]:
            normalized_time_series[ts_key].append((point - global_min) / (global_max - global_min))

    return normalized_time_series
    # print("Global Max is " + str(global_max))
    # print("Global Min is " + str(global_min))


def normalize_ts_with_min_max_legacy(ts_dict, global_min, global_max):
    """
    This function accepts diction as input and outputs a dictionary
    :param ts_dict:
    :param min:
    :param max:
    """
    normalized_time_series = dict()

    for ts_key in ts_dict.keys():
        normalized_time_series[ts_key] = []
        for point in ts_dict[ts_key]:
            normalized_time_series[ts_key].append((point - global_min) / (global_max - global_min))

    return normalized_time_series


def normalize_ts_with_min_max(ts_list, global_min, global_max):
    """
    This function accepts LIST as input and outputs a LIST

    :param time_series_dict:
    :param min:
    :param max:
    """
    norm_ts_list = []

    for id_data in ts_list:
        if len(id_data) != 2:
            raise Exception('data_operations: normalize_ts_with_min_max: Invalid ts_list, not 2 elements per entry; Invalid Entry = ' + id_data)
        id = id_data[0]
        data = id_data[1]

        normalized_data = []
        for point in data:
            normalized_data.append((point - global_min) / (global_max - global_min))
        norm_ts_list.append([id, normalized_data])

    return norm_ts_list


def find_k_smallest(target_cluster, query_sequence, k):
    # TODO
    """
    use quickselect to find the k smallest element
    :param target_cluster:
    :param query_sequence:
    :param k:
    :return:
    """
