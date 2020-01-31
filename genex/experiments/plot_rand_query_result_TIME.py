import matplotlib.pyplot as plt
import matplotlib

import pandas as pd
import numpy as np

from genex.experiments.plot_rand_query_result_RMSE import autolabel


date = 'Jan-30-2020-15-N-UseSpark-R1-LBOptNormalDTW_numSample400'
notes = 'UseSpark-R1-LBOptNormalDTW_numSample400'

font = {'family': 'DejaVu Sans',
        'weight': 'bold',
        'size': 12}
matplotlib.rc('font', **font)

file_list = [
    'results/' + date + '/Gun_Point_TRAIN',
    'results/' + date + '/ECGFiveDays',
    'results/' + date + '/ItalyPower',
    'results/' + date + '/synthetic_control_TRAIN'
]

dist_types = ['eu', 'ma', 'ch']
# eu distance ##############################################################################
# result_list = [x + '_dist_' + 'eu' + '.csv' for x in file_list]
# ma distance ##############################################################################
# result_list = [x + '_dist_' + 'ma' + '.csv' for x in file_list]
# ch distance ##############################################################################
# result_list = [x + '_dist_' + 'ch' + '.csv' for x in file_list]

algorithm_dict = {'Naive': [], 'BrainEx Query': [], 'BrainEx Cluster': []}
num_sample = 40
num_most_k = 15
gx_time_col_num = 3
bf_time_col_num = 4
offset_start = 1
offset_between_sample = 2
cluster_loc = (0, 1)

x = np.arange(len(dist_types))  # the label locations
width = 0.20  # the width of the bars
fig, ax = plt.subplots()
fig.set_size_inches(15, 8)

for i, alg in enumerate(algorithm_dict.keys()):

    dt_timing_dict = dict()
    for dist_type in dist_types:
        result_list = [x + '_' + dist_type + '.csv' for x in file_list]
        bf_time_list = []
        gx_time_list = []
        cluster_time_list = []
        for f in result_list:
            df = pd.read_csv(f)
            for j in range(num_sample):
                bf_time_list.append(df.iloc[offset_start + (num_most_k + offset_between_sample) * j, bf_time_col_num])
                gx_time_list.append(df.iloc[offset_start + (num_most_k + offset_between_sample) * j, gx_time_col_num])
        cluster_time_list.append(df.iloc[cluster_loc])

        if alg == 'Naive':
            timing = np.mean(bf_time_list)
        elif alg == 'BrainEx Query':
            timing = np.mean(gx_time_list)
        elif alg == 'BrainEx Cluster':
            timing = np.mean(cluster_time_list)

        algorithm_dict[alg].append(timing)

    rect = ax.bar(x + 4 * i * width / len(algorithm_dict), algorithm_dict[alg], width, label=alg)
    autolabel(rect, ax, algorithm_dict)


ax.set_ylabel('Time (seconds)')
ax.set_xlabel('Distance Type')
ax.set_title('Query Timing Evaluation for Difference Distances \n' + notes)
ax.set_xticks(x)
ax.set_xticklabels(['Euclidean', 'Manhattan', 'Chebyshev'])
ax.legend()
fig.tight_layout()

print('The average query time is ' + str(np.mean(algorithm_dict['BrainEx Query'])))
plt.show()
