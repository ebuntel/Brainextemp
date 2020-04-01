from genex.utils.gxe_utils import from_csv, from_db
import pandas as pd
import numpy as np
import time

import matplotlib.pyplot as plt

data = pd.read_csv('/home/apocalyvec/PycharmProjects/Genex/genex/experiments/data_original/SART2018_HbO.csv')

# # get subject AS
# data_subjectAS = data[data['Subject Name'] == '101-SART-June2018-AS']
# data_subjectAS_ch1 = data_subjectAS[data[' Channel Name'] == 'Channel-1 HbO']
#
# gxe_subjectAS_ch1 = from_csv(data_subjectAS_ch1,
#                              feature_num=5, num_worker=32, use_spark=True, driver_mem=24, max_result_mem=24)
# start = time.time()
# gxe_subjectAS_ch1.build(st=0.1, loi=[int(gxe_subjectAS_ch1.get_max_seq_len() * 0.6)])  # cluster only the longer series
# print('Building took ' + str(time.time() - start) + ' sec')

# # get the most common pattern of all length
# pattern_subjectAS_ch1 = dict()  # seq_len (int) -> [most represented representative (sequence), number of represented sequences]
# for seq_len, clusters in gxe_subjectAS_ch1.cluster_meta_dict.items():
#     cluster_list = list(clusters.items())  # repr (sequence) -> number of represented sequences (int)
#     cluster_list.sort(key=lambda x: x[1], reverse=True)
#     pattern_subjectAS_ch1[seq_len] = cluster_list[0]
#
# # now get the most common pattern among the sequence length
# # [most represented representative (sequence), number of represented sequences]
# pattern_list_subjectAS_ch1 = list(pattern_subjectAS_ch1.values())
# pattern_list_subjectAS_ch1.sort(key=lambda x: x[1], reverse=True)

# pattern_subjectAS_ch1 = gxe_subjectAS_ch1.motif(k=5, absolute=False)

# plot the top five patterns
# fig, ax = plt.subplots()
# fig.set_size_inches(15, 8)
# for i in range(5):
#     seq = pattern_subjectAS_ch1[i][0]
#     ax.plot(gxe_subjectAS_ch1.get_seq_data(seq, normalize=True),
#             label=str(seq) + 'Relative' + ' Representativeness: ' + str(pattern_subjectAS_ch1[i][1]))
# ax.set_ylabel('Normalized Signal Level')
# ax.set_xlabel('Samples in Time')
# ax.set_title('Subject Name: ' + '101-SART-June2018-AS;' + ' Channel Name: ' + 'Channel-1 HbO')
# ax.legend()
# plt.show()

cluster_st = 0.1
loi_offset = 0.6
motif_k = 5

for subj in data['Subject Name'].unique():  # iterate through the subjects
    for chan in data[' Channel Name'].unique():  # iterate through the channels
        # filter by subject name, channel and event
        data_subjChanCor = data[(data['Subject Name'] == subj) &
                                (data[' Channel Name'] == chan) &
                                (data[' Event Name'] == 'target correct')]

        gxe = from_csv(data_subjChanCor, feature_num=5, num_worker=32, use_spark=True, driver_mem=24, max_result_mem=24)
        gxe.build(st=cluster_st, loi=[int(gxe.get_max_seq_len() * loi_offset)])  # cluster only the longer sequences

        # Plot the absolute motif ################################################################################
        pattern_abs = gxe.motif(k=motif_k, absolute=True)
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 8)
        for seq, representativeness in pattern_abs:
            ax.plot(gxe.get_seq_data(seq, normalize=True),
                    label=str(seq) + '; Absolute' + ' Representativeness: ' + str(representativeness))
        # ax.set_ylim(.2, .6)
        ax.set_ylabel('Normalized Signal Level')
        ax.set_xlabel('Samples in Time')
        ax.set_title('Top 5 Most Common Pattern for: Subject Name: ' + subj + ' Channel Name: ' + chan)
        ax.legend()
        plt.show()

        # Plot the relative motif ################################################################################
        pattern_rel = gxe.motif(k=motif_k, absolute=False)
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 8)
        for seq, representativeness in pattern_rel:
            ax.plot(gxe.get_seq_data(seq, normalize=True),
                    label=str(seq) + '; Relative' + ' Representativeness: ' + str(representativeness))
        # ax.set_ylim(.2, .6)
        ax.set_ylabel('Normalized Signal Level')
        ax.set_xlabel('Samples in Time')
        ax.set_title('Top 5 Most Common Pattern for: Subject Name: ' + subj + ' Channel Name: ' + chan)
        ax.legend()
        plt.show()

        gxe.stop()
        break
    break
# Plot the correct and incorrect signal ################################################################################
# data_subjectAS_ch1_COR = data_subjectAS_ch1[data[' Event Name'] == 'target correct']
# data_subjectAS_ch1_INC = data_subjectAS_ch1[data[' Event Name'] == 'target incorrect']

# data_INC = data_subjectAS_ch1_INC.iloc[:, 5:].values
# data_COR = data_subjectAS_ch1_COR.iloc[:, 5:].values

# for seq in data_INC:
#     plt.plot(seq)
# plt.ylim(-10, 10)
# plt.title('Incorrect Target (8 sec before to 15 sec after)')
# plt.ylabel('Level')
# plt.xlabel('Samples in Time')
# plt.show()
#
# for seq in data_COR:
#     plt.plot(seq)
# plt.ylim(-10, 10)
# plt.title('Correct Target (8 sec before to 15 sec after)')
# plt.ylabel('Level')
# plt.xlabel('Samples in Time')
# plt.show()
########################################################################################################################
