from genex.classes.Sequence import Sequence
from genex.preprocess import normalize_num

def normalize_sequence(seq: Sequence, max, min):
    if seq.data is None:
        raise Exception('Given sequence does not have data set, use fetch_data to set its data first')

    data = seq.data
    normalized_data = list(map(lambda num: normalize_num(num, max, min), data))

    seq.set_data(normalized_data)
