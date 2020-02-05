import math
from logging import warning

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from genex.classes.Sequence import Sequence
from genex.misc import prYellow


def normalize_sequence(seq: Sequence, max, min, z_normalize=True):
    """
    Use min max and z normalization to normalize time series data_original

    :param seq: Time series sequence
    :param max: maximum value in sequence
    :param min: minimum value in sequence
    :param z_normalize: whether data_original is z normalized or not

    """

    if seq.data is None:
        raise Exception('Given sequence does not have data_original set, use fetch_data to set its data_original first')
    data = seq.data
    if z_normalize:
        data = [(x - np.mean(data) / np.std(data)) for x in data]

    normalized_data = list(map(lambda num: normalize_num(num, max, min), data))

    seq.set_data(normalized_data)


def scale(ts_df, feature_num):
    time_series = ts_df.iloc[:, feature_num:].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    num_time_series = len(time_series)
    time_series = time_series.reshape(-1, 1)
    time_series = scaler.fit_transform(time_series)
    time_series = time_series.reshape(num_time_series, -1)

    df_normalized = ts_df.copy()
    df_normalized.iloc[:, feature_num:] = time_series

    return df_normalized, scaler


def get_trgt_len(l_list, q_len):
    """
    from the l_list, give the closest value to the q_len
    :param l_list:
    :param q_len:
    :return:
    """
    return min(list(l_list), key=lambda x: abs(x - q_len))


def get_trgt_len_within_r(l_list, q_len, radius):
    """
    UNIT TEST NEEDED
    :param l_list:
    :param q_len:
    :param radius:
    :return:
    """
    return [x for x in l_list if q_len - radius <= x <= q_len + radius]
    # trgt_l_list = []
    # l_list_copy = l_list.copy()
    # for r in range(radius + 1):
    #     if l_list_copy:
    #         trgt_l = get_trgt_len(l_list=l_list_copy, q_len=q_len)
    #         trgt_l_list.append(trgt_l)
    #         l_list_copy.remove(trgt_l)
    #     else:
    #         break
    # return trgt_l_list


def get_sequences_represented(reprs, cluster):
    """

    :param reprs: list of Sequences that are representatives
    :param cluster: repr -> list of sequences represented, mapping from representativs to their clusters
    """
    return flatten([cluster[r] for r in reprs]) if reprs is not None else list()


def flatten(l):
    return [item for sublist in l for item in sublist]


def _validate_gxdb_build_arguments(args: dict):
    """
    sanity check function for the arguments of build as a class method @ genex_databse object
    :param args:
    :return:
    """
    # TODO finish the exception messages

    if 'loi' in args and args['loi'] is not None:
        loi = args['loi']
        try:
            iter(loi)
            assert len(loi) == 1 or len(loi) == 2
        except (AssertionError, TypeError) as ae:
            raise Exception('Build check argument failed: the Length of Interest (loi) must be an iterable of length'
                            '1 or 2')
    try:
        assert 0. < args['st'] < 1.
    except AssertionError as ae:
        raise Exception('Build check argument failed: build st must be between 0. and 1. and not '
                        'equal to 0. and 1.')
    print(args)

    return


def _validate_gxe_query_arguments(args: dict):
    """
    sanity check function for the arguments of query as a class method @ genex_databse object
    :param args:
    :return:
    """
    pass


def _df_to_list(df, feature_num):
    df_list = [_row_to_feature_and_data(x, feature_num) for x in df.values.tolist()]
    return df_list


def _create_f_uuid_map(df, feature_num: int):
    f_uuid_dict = dict()
    for x in df.values.tolist():
        f_uuid_dict[tuple(x[1:feature_num])] = x[0]

    return f_uuid_dict


def _row_to_feature_and_data(row, feature_num):
    # list slicing syntax: ending at the key_num-th element but not include it
    # seq_id = tuple([(name, value) for name, value in zip(feature_head[:feature_num], row[:feature_num])])
    # seq_id = tuple([str(x) for x in row[:feature_num]])
    seq_id = tuple([str(x) for x in row[:feature_num]])
    try:
        data = [x for x in row[feature_num:] if not np.isnan(x)]
    except TypeError as te:
        raise Exception(
            'Genex: this may due to an incorrect feature_num, please check you data_original file for the number '
            'of features\n '
            + 'Exception: ' + str(te))
    return seq_id, data


def _process_loi(loi, max_len):
    """
    Process the length of interest parameter to get the start and end index
    :param loi: length of interest parameter
    :return: start and end index
    """
    if loi is not None:
        start = max(1, loi[0])
        if len(loi) == 2:
            end = min(max_len, loi[1])
            if loi[1] > max_len:
                warning(
                    'Warning: Length of interest must end less or equal to the length of the longest time series in the '
                    'dataset, setting end to that max.')
        else:
            end = max_len
        if loi[0] < 1:
            warning('Warning: Length of interest must start at a value greater than 1, setting start to 1.')
    else:
        start = 1
        end = max_len
    return int(start), int(end)


from functools import reduce
from itertools import groupby


def reduce_by_key(func, iterable):
    """Reduce by key.
    Equivalent to the Spark counterpart
    Inspired by http://stackoverflow.com/q/33648581/554319
    1. Sort by key
    2. Group by key yielding (key, grouper)
    3. For each pair yield (key, reduce(func, last element of each grouper))
    """
    get_first = lambda p: p[0]
    get_second = lambda p: p[1]
    # iterable.groupBy(._1).map(l => (l._1, l._2.map(._2).reduce(func)))
    return map(
        lambda l: (l[0], reduce(func, map(get_second, l[1]))),
        groupby(sorted(iterable, key=get_first), get_first)
    )


def normalize_num(num, global_max, global_min):
    return (num - global_min) / (global_max - global_min)


def genex_normalize(input_list, z_normalization=False):
    # perform z normalization
    if z_normalization:
        z_normalized_input_list = _z_normalize(input_list)
    else:
        print('Not using z-normalization')
    # get a flatten z normalized list so to obtain the global min and max
    flattened_list = flatten([x[1] for x in z_normalized_input_list])
    global_max = np.max(flattened_list)
    global_min = np.min(flattened_list)

    # perform Min-max normalization
    zmm_normalized_list = _min_max_normalize(z_normalized_input_list, global_max=global_max, global_min=global_min)

    normalized_array = np.asarray([x[1] for x in zmm_normalized_list])
    return zmm_normalized_list, global_max, global_min


def _z_normalize(input_list):
    z_normalized_list = [(x[0], (x[1] - np.mean(x[1])) / np.std(x[1])) for x in input_list]
    return z_normalized_list


def _min_max_normalize(input_list, global_max, global_min):
    mm_normalized_list = [(x[0], (x[1] - global_min) / (global_max - global_min)) for x in input_list]
    return mm_normalized_list


def _isOverlap(seq1: Sequence, seq2: Sequence, overlap: float) -> bool:
    if seq1.seq_id != seq2.seq_id:  # overlap does NOT matter if two seq have different id
        return True
    else:
        of =  _calculate_overlap(seq1, seq2)
        return _calculate_overlap(seq1, seq2) >= overlap


def _calculate_overlap(seq1, seq2) -> float:
    if seq2.end > seq1.end and seq2.start >= seq1.start:
        return (seq1.end - seq2.start + 1) / (seq2.end - seq1.start + 1)
    elif seq1.end > seq2.end and seq1.start >= seq2.start:
        return (seq2.end - seq1.start + 1) / (seq1.end - seq2.start + 1)
    if seq2.end >= seq1.end and seq2.start > seq1.start:
        return (seq1.end - seq2.start + 1) / (seq2.end - seq1.start + 1)
    elif seq1.end >= seq2.end and seq1.start > seq2.start:
        return (seq2.end - seq1.start + 1) / (seq1.end - seq2.start + 1)

    elif seq1.end > seq2.end and seq2.start >= seq1.start:
        return len(seq2) / len(seq1)
    elif seq2.end > seq1.end and seq1.start >= seq2.start:
        return len(seq1) / len(seq2)
    elif seq1.end >= seq2.end and seq2.start > seq1.start:
        return len(seq2) / len(seq1)
    elif seq2.end >= seq1.end and seq1.start > seq2.start:
        return len(seq1) / len(seq2)

    elif seq2.start > seq1.end or seq1.start > seq2.end:  # does not overlap at all
        return 0.0
    else:
        print(seq1)
        print(seq2)
        raise Exception('FATAL: sequence 100% overlap, please report the bug')


