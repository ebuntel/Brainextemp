import os

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np


if __name__ == '__main__':
    font = {'family': 'DejaVu Sans',
            'weight': 'bold',
            'size': 12}
    matplotlib.rc('font', **font)

    notes = 'UCR_Small[0-50000] - Incomplete'
    # root = 'results/ucr_experiment/dynamic/Apr-28-2020-3-N-UCR_test_eu_soi_0-to-50000/'
    root = 'results/ucr_experiment/dynamic/Apr-28-2020-21-N-UCR_test_eu_soi_0-to-50000/'

    file_list = os.listdir(root)
    file_list = [os.path.join(root, x) for x in file_list]

    # dt_dict = {'eu': 'Euclidean', 'ma': 'Manhattan', 'ch': 'Chebyshev'}
    dt_dict = {'eu': 'Euclidean'}

    fig_name = 'Query Performance across Dataset Sizes\n'
    k_to_look = [1, 5, 15]

    axis_label_ft = 18
    title_ft = 20

    data_size_coord = (0, 23)

    # End of Initial Variables #########################################################################################
    dt_list = dt_dict.keys()
    title = fig_name + notes
    x = np.arange(len(k_to_look))  # the label locations

    for i, dt in enumerate(dt_list):
        fd = {x: x for x in file_list if '_' + dt + '.csv' in x}  # filter to get the result of a specific dist type
        # plot the clustering time as a heatmap across data length and data rows
        bin_gx_c_time = []
        bin_dynamic_c_time = []

        bin_size = []

        bin_qbf_time = []  # query brute force
        bin_qpaa_time = []  # query paa
        bin_qgx_time = []  # query genex
        bin_qdynamic_time = []  # query genex

        bin_qpaa_error = []
        bin_qgx_error = []
        bin_qdynamic_error = []

        for entry in fd.items():
            dataset_name, result_file = entry
            df = pd.read_csv(result_file)

            # information about this dataset's result
            size = df.iloc[data_size_coord]

            gx_c_time = df.iloc[1, 1]
            dynamic_c_time = df.iloc[1, 2]
            paa_build_time = df.iloc[1, 3]

            qbf_time = [x for x in df.iloc[:, 5].values if not np.isnan(x)]
            qpaa_time = [x for x in df.iloc[:, 6].values if not np.isnan(x)]
            qgx_time = [x for x in df.iloc[:, 7].values if not np.isnan(x)]
            qdynamic_time = [x for x in df.iloc[:, 8].values if not np.isnan(x)]

            qpaa_error = [x for x in df.iloc[:, 9].values if not np.isnan(x)]
            qgx_error = [x for x in df.iloc[:, 10].values if not np.isnan(x)]
            qdynamic_error = [x for x in df.iloc[:, 11].values if not np.isnan(x)]

            bin_size.append(size)

            bin_gx_c_time.append(gx_c_time)
            bin_dynamic_c_time.append(dynamic_c_time)

            bin_qbf_time.append(np.mean(qbf_time))
            bin_qpaa_time.append(np.mean(qpaa_time))
            bin_qgx_time.append(np.mean(qgx_time))
            bin_qdynamic_time.append(np.mean(qdynamic_time))

            bin_qpaa_error.append(np.mean(qpaa_error))
            bin_qgx_error.append(np.mean(qgx_error))
            bin_qdynamic_error.append(np.mean(qdynamic_error))
            pass


        # Plot the Cluster Time
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 8)
        plt.title('Cluster time across Data Size for Distance Type: ' + dt_dict[dt])
        plt.scatter(bin_size, bin_gx_c_time, c='cyan', label='Genex Cluster Time', marker='x', alpha=0.75)
        plt.scatter(bin_size, bin_dynamic_c_time, c='green', label='Dynamic Cluster Time', marker='x', alpha=0.75)
        plt.ylabel('Time (second)')
        plt.xlabel('Time series length (number of data points)')
        # plt.ylim(-1, 25)
        plt.legend()
        plt.show()

        # Plot the Query Time
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 8)
        plt.title('Query Time across Data Size for Distance Type: ' + dt_dict[dt])
        plt.scatter(bin_size, bin_qbf_time, c='red', label='Brute Force Query Time', alpha=0.75)
        plt.scatter(bin_size, bin_qpaa_time, c='orange', label='PAA Query Time')
        plt.scatter(bin_size, bin_qgx_time, c='blue', label='Genex Query Time', alpha=0.75)
        plt.scatter(bin_size, bin_qdynamic_time, c='green', label='Dynamic Query Time', alpha=0.75)
        plt.ylabel('Time (second)')
        plt.xlabel('Time series length (number of data points)')
        # plt.ylim(-1, 25)
        plt.legend()
        plt.show()

        # Plot the accuracy
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 8)
        plt.title('Normalized Error across Data Size for Distance Type: ' + dt_dict[dt])
        plt.scatter(bin_size, bin_qpaa_error, c='orange', label='PAA Query Error')
        plt.scatter(bin_size, bin_qgx_error, c='blue', label='Genex Query Error', alpha=0.75)
        plt.scatter(bin_size, bin_qdynamic_error, c='green', label='Dynamic Query Error', alpha=0.75)
        plt.ylabel('Normalized Error')
        plt.xlabel('Data size (number of data points)')
        # plt.ylim(-0.0025, 0.14)
        plt.legend()
        plt.show()

        avg_dynamic_cluster_time = np.mean(bin_dynamic_c_time)
        avg_gx_cluster_time = np.mean(bin_gx_c_time)

        avg_dynamic_q_time = np.mean(bin_qdynamic_time)
        avg_gx_q_time = np.mean(bin_qgx_time)

        avg_dynamic_error = np.mean(bin_qdynamic_error)
        avg_gx_error = np.mean(bin_qgx_error)