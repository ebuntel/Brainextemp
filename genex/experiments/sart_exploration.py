import pandas as pd
import numpy as np
import time

from genex.experiments.exp_utils import df_to_event, extract_query, plot_event, extract_query_normalized, plot_sequence
from genex.utils.gxe_utils import from_csv, from_db
import matplotlib.pyplot as plt
import matplotlib

cluster_st = 0.1
sampling_rate = 6.6231
doi = [1, 120]  # duration of interest

# events = 'true answer', 'bCurrent Item: nonTarget',
results = ['paa_data incorrect']
data_all = '/home/apocalyvec/data/sart/SART_AS/101-SART-June2018-AS_Ch1HbO.csv'
df_tic = pd.read_csv('/home/apocalyvec/data/sart/SART_AS/101-SART-June2018-AS_Ch1HbO_targetIncorrect(1)(1).csv')
db_path = '/home/apocalyvec/data/sart/SART_AS/gxe_doi(1)(120)_101-SART-June2018-AS_Ch1HbO'
gxe = from_db(db_path, num_worker=12, driver_mem=12, max_result_mem=12)

best_k = 50
e_woi = [5, 5]  # event window of interest
data = np.array(gxe.data_original[0][1])  # there's only on time series
data_normalized = np.array(gxe.data_normalized[0][1])  # there's only on time series

data_id = gxe.data_original[0][0]
data_e = (data_id[1], float(data_id[3]) / 1e3, float(data_id[4]) / 1e3, data_id[5:])

matplotlib.use('TkAgg')
plt.ion()
fig, ax = plt.subplots()
# plt.plot(np.linspace(data_e[1], data_e[2], num=len(data)), data)
ax.plot(data, label=data_all.split('/')[-1])
ax.set_xlabel('Time (sec)')
ax.set_ylabel('Signal Magnitude')

events = df_to_event(df_tic)  # target incorrect events

# plot with un-normalized data, this function also return the indices for the use in extract_query_normalized
q = extract_query(events[0], sampling_rate, e_woi, data)[0]
q_norm = extract_query_normalized(events[0], sampling_rate, e_woi, data, data_normalized)  # query with normalized data
q_matches = gxe.query(q.data, best_k)

plot_event(q, ax, data, marker='x', label='Query ' + str(q), use_line=False)  # plot the query event

for ev in events:  # plot all the same events
    plot_event(ev, ax, data)
    ax.legend(prop={'size': 6})

for i, item in enumerate(q_matches):
    dist, mtc_seq = item
    gxe.set_seq_data(mtc_seq)
    plot_sequence(mtc_seq, ax, marker='X', label=str(i) + 'th match, dist=' + str(dist) + '; ' + str(mtc_seq))
    ax.legend(prop={'size': 6})


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
