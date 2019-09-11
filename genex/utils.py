from sklearn.preprocessing import MinMaxScaler

from genex.classes.Sequence import Sequence
from genex.preprocess import normalize_num


def normalize_sequence(seq: Sequence, max, min):
    if seq.data is None:
        raise Exception('Given sequence does not have data set, use fetch_data to set its data first')

    data = seq.data
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
