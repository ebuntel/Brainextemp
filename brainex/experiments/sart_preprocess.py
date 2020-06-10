import os

import pandas as pd
import numpy as np
import time
from brainex.utils.gxe_utils import from_csv, from_db
import matplotlib.pyplot as plt

cluster_st = 0.1
sampling_rate = 6.6231
doi = [1, 60]  # duration of interest

data_root = '/home/apocalyvec/data/sart/'
# dataset_name = '001-SART-August2017-MB'

datasets = [x for x in os.listdir(os.path.join(data_root, 'all')) if x.split('.')[-1] == 'csv']
datasets.sort()
for i, ds in enumerate(datasets):
    ds_path = os.path.join(data_root, 'all', ds)
    print('Processing ' + str(i) + ' of ' + str(len(datasets)) + ': ' + ds_path)
    gxe = from_csv(ds_path, feature_num=5, num_worker=32, use_spark=True, driver_mem=24, max_result_mem=24, header=None)
    start = time.time()
    gxe.build(st=cluster_st,
              loi=[int(d * sampling_rate) for d in doi])  # cluster only sequences that are longer than 1 second
    print('Build took ' + str(time.time() - start) + ' sec')
    gxe.save(os.path.join(data_root, ('gxe' + ds.strip('.csv'))))
    gxe.stop()
    del gxe





# data = np.array(gxe.data_original[0][1])  # there's only on time series
# data_id = gxe.data_original[0][0]
# data_e = (data_id[1], float(data_id[3]) / 1e3, float(data_id[4]) / 1e3, data_id[5:])
#
#
# plt.plot(np.linspace(data_e[1], data_e[2], num=len(data)), data)
# plt.xlabel('Time (sec)')
# plt.show()
