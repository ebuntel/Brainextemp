import pandas as pd
import numpy as np
import time
from genex.utils.gxe_utils import from_csv, from_db
import matplotlib.pyplot as plt

cluster_st = 0.1
sampling_rate = 6.6231

events = 'true answer', 'bCurrent Item: nonTarget',

results = ['target incorrect']

data_all = '/home/apocalyvec/data/sart/101-SART-June2018-AS_(all).csv'

gxe = from_csv(data_all, feature_num=5, header=None, num_worker=32, use_spark=True, driver_mem=24, max_result_mem=24)
# gxe.build(st=cluster_st, loi=[int(sampling_rate)])  # cluster only sequences that are longer than 1 second
start = time.time()
gxe.build(st=cluster_st, _dsg=True)  # cluster only sequences that are longer than 1 second
print('Build took ' + str(time.time() - start) + ' sec')

# test query speed
query_timing_list = []
for i in range(1, 100):
    print('querying ' + str(i) + ' of ' + str(100))
    q_size = int(i * sampling_rate)
    q = np.random.randn(q_size)
    print(q)
    q_start = time.time()
    query_result = gxe.query(q, best_k=15)
    q_time = time.time() - q_start
    print('query took ' + str(q_time) + ' sec')
    query_timing_list.append([i, q_size, q_time])


query_timing_list = np.array(query_timing_list)
plt.plot(query_timing_list[:, 0], query_timing_list[:, 2])
plt.xlabel('Length of the query (seconds)')
plt.ylabel('time took to query (sec)')
plt.show()