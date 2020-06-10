import math
import pandas as pd

from .classes.Sequence import Sequence


def strip_function(x: str):
    """
    strip function to remove empty
    :param x:
    :return:
    """
    if x != '' and x != ' ':
        return x.strip()


def remove_trailing_zeros(input_list: list):
    """
    remove trailing zeros
    :param input_list: list of clusters
    :return: clusters after removing trailing zeros
    """
    index = len(input_list) - 1
    for i in range(len(input_list) - 1, -1, -1):
        if input_list[i] == ',' or input_list[i] == '0' or input_list[i] == '':
            continue
        else:
            index = i
            break
    return input_list[0:index]


def get_subsquences(input_list: list):
    """
    user defined function for mapping for spark
    :param input_list: input list, has two rows, the first row is ID, the second is a list of clusters
    :return: val: a list of a list of value [length, id, start_point, end_point]
    """
    id = input_list[0]
    val = []
    # Length from start+lengthrange1, start + lengthrange2
    # in reality, start is 0, end is len(input_list)

    for i in range(len(input_list)):
        # make sure to get the end
        for j in range(len(input_list[i])):
            if i + j < len(input_list[1]):
                # length, id, start, end
                val.append([j, id, i, i + j])

    return val


def generate_source(file_name, feature_num):
    """
    Not doing sub-sequence here
    get the id of time series and clusters of time series

    :param file_name: path to csv file
    :param feature_num: number of features that makes up the id

    :return: a list of clusters in [id, [list of clusters]] format
    """

    features_to_append = list(i for i in range(feature_num))

    # List of result
    ts_list = []
    # wrap_in_parantheses = lambda x: "(" + str(x) + ")"

    with open(file_name, 'r') as f:
        for i, line in enumerate(f):
            if i != 0:
                # why the last line will be none?
                features = list(map(lambda x: strip_function(x),
                                    line.strip()[:-1].split(',')))
                # print(features)
                if len(features) > len(features_to_append):
                    label_features_index = [features[feature] for feature in features_to_append]

                if line != "" and line != "\n":
                    data = remove_trailing_zeros(line.split(",")[:-1])

                    # Get feature values for label
                    # label_features = [wrap_in_parantheses(clusters[index]) for index in
                    #                   range(0, len(label_features_index))]
                    id_list = []
                    [id_list.append(data[index]) for index in range(0, len(label_features_index))]
                    # series_label = "_".join(label_features).replace('  ', '-').replace(' ', '-')

                    # check if the number of feature correct by catching 'could not convert string to float' errors
                    try:
                        series_data = list(map(float, data[len(label_features_index):]))
                    except ValueError:
                        raise Exception('parse: generate_source: wrong number of features')
                    ts_list.append([tuple(id_list), series_data])

    return ts_list


def generate_query(file_name: str, feature_num: int):
    """
     Generates a list of query sequences (indexed by id, start and end point) from given csv file.
    :param file_name: Source file for the query set
    :param feature_num: Number of features
    :return: a list of query sequences indexed by id, start and end point
    """

    query_set = []
    with open(file_name, 'r') as f:
        for i, line in enumerate(f):
            # first line is the header, do not take
            if i == 0:
                continue
            cur_query = line.strip().split(',')
            try:
                assert len(cur_query) == feature_num + 2
                features = tuple(cur_query[0:feature_num])
                start = int(cur_query[feature_num])
                end = int(cur_query[feature_num+1])
                query_set.append(Sequence(features, start, end))
            except ValueError or AssertionError as e:
                raise Exception('Invalid Query csv, please refer to get_query_csv in README')
    return query_set