from genex.classes.Sequence import Sequence
import numpy as np


class Event:
    def __init__(self, name, startT, endT, data, normalized=False):
        self.name = name
        self.startT = startT
        self.endT = endT
        self.data = data
        self.normalized = normalized

    def __str__(self):
        return self.name + '_(' + str(self.startT) + ')_(' + str(self.endT) + ')'


def resolve_esi(e, data):
    """
    get the event starting index
    :param e:
    :param data:
    """
    e_seq = e.data[:-2]  # offset the seq in case the array can not be found
    return data.tostring().index(e_seq.tostring()) // data.itemsize  # find where the event occured


def plot_event(e, ax, data, marker='.', label=None, use_line=True):
    e_start_index = resolve_esi(e, data)
    if use_line:
        samples = 5
        ax.plot([e_start_index] * samples, np.linspace(-15, 10, samples), label=str(e) if label is None else label)
    else:
        ax.scatter(list(range(e_start_index, e_start_index + len(e.data))), e.data,
                   label=str(e) if label is None else label
                   , marker=marker)


def plot_sequence(s: Sequence, ax, marker='.', label=None):
    ax.scatter(list(range(s.start, s.start + len(s))), s.data, label=str(s) if label is None else label
               , marker=marker)


def extract_query(e: Event, f, woi, data):
    """
    create a query array from the the event
    :param data:
    :param e:
    :param doi:
    """
    assert woi[0] > 0 and woi[1] > 0
    e_start_index = resolve_esi(e, data)
    st = int(e_start_index - woi[0] * f)
    ed = int(e_start_index + woi[0] * f)
    return Event(e.name, e.startT - woi[0], e.endT + woi[1], data[st:ed]), st, ed


def extract_query_normalized(e: Event, f, woi, data, data_normalized):
    query_event, st, ed = extract_query(e, f, woi, data)
    return Event(query_event.name, query_event.startT, query_event.endT, data_normalized[st:ed], normalized=True)


def df_to_event(df):
    return [Event(line[1], float(line[3]) / 1e3, float(line[4]) / 1e3, e_seq) for line, e_seq in
            zip(df.axes[0], df.values)]
