import pandas as pd
import numpy as np
import time
from genex.utils.gxe_utils import from_csv, from_db
import matplotlib.pyplot as plt

cluster_st = 0.1
sampling_rate = 6.6231
doi = [1, 120]  # duration of interest

# events = 'true answer', 'bCurrent Item: nonTarget',
results = ['paa_data incorrect']
data_all = '/home/apocalyvec/data/sart/SART_AS/101-SART-June2018-AS_Ch1HbO.csv'
df_tc = pd.read_csv('/home/apocalyvec/data/sart/SART_AS/101-SART-June2018-AS_Ch1HbO_targetIncorrect(1)(1).csv')

gxe = from_csv(data_all, feature_num=5, header=None, num_worker=12, use_spark=True, driver_mem=12, max_result_mem=12)
# gxe.build(st=cluster_st, loi=[int(sampling_rate)])  # cluster only sequences that are longer than 1 second
start = time.time()
gxe.build(st=cluster_st,
          loi=[int(d * sampling_rate) for d in doi])  # cluster only sequences that are longer than 1 second
print('Build took ' + str(time.time() - start) + ' sec')


def plot_event(d, f, event):
    """

    :param event: tuple: (name, start time, end time, data)
    :param d: the original single time series on which the events occurs
    :param f: the sample rate
    """


def resolve_event_index(event, f, it):
    """
    retrieve the indices of an event given its start time, end time, and f
    :param event: tuple: (name, start time, end time, data)
    :param f:
    """
    start_i, end_i = (event[1]) * f, (event[2]) * f
    return int(start_i), int(end_i)


data = np.array(gxe.data_original[0][1])  # there's only on time series
data_id = gxe.data_original[0][0]
data_e = (data_id[1], float(data_id[3]) / 1e3, float(data_id[4]) / 1e3, data_id[5:])


for line, e_seq in zip(df_tc.axes[0], df_tc.values):
    e = (line[1], float(line[3]) / 1e3, float(line[4]) / 1e3, line[5:])
    # i, j = resolve_event_index(e, sampling_rate, init_time)

    data.tostring().index(e_seq[:-1].tostring()) // data.itemsize  # find where the event occured
    break

plt.plot(np.linspace(data_e[1], data_e[2], num=len(data)), data)
plt.xlabel('Time (sec)')
plt.show()

# # test query speed
# query_timing_list = []
# for i in range(1, 100):
#     print('querying ' + str(i) + ' of ' + str(100))
#     q_size = int(i * sampling_rate)
#     q = np.random.randn(q_size)
#     print(q)
#     q_start = time.time()
#     query_result = gxe.query(q, best_k=15)
#     q_time = time.time() - q_start
#     print('query took ' + str(q_time) + ' sec')
#     query_timing_list.append([i, q_size, q_time])
#
#
# query_timing_list = np.array(query_timing_list)
# plt.plot(query_timing_list[:, 0], query_timing_list[:, 2])
# plt.xlabel('Length of the query (seconds)')
# plt.ylabel('time took to query (sec)')
# plt.show()
