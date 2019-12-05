import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

file_list = [
    'results/rand_query_result/Gun_Point_TRAIN_result',
    'results/rand_query_result/ECGFiveDays_result',
    'results/rand_query_result/ItalyPowerDemand_result',
    'results/rand_query_result/synthetic_control_TRAIN_result'
]

dist_types = ['eu', 'ma', 'ch']
num_sample = 40
num_most_k = 15
gx_time_col_num = 3
bf_time_col_num = 4
offset_start = 2
offset_between_sample = 2
cluster_loc = (0, 1)

for dt in dist_types:
    result_list = [x + '_dist_' + dt + '.csv' for x in file_list]
    for f in result_list:
        df = pd.read_csv(f)
        # first grab the gx times
        bf_time_list = []
        gx_time_list = []
        for i in range(num_sample):
            bf_time_list.append()


x = np.arange(len(dist_types))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()