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

    # date = 'Jan-30-2020-12-N-UseSpark-R1-noOptFastDTW_numSample400'
    # notes = 'UseSpark-R1-noOptFastDTW_numSample400'
    notes = 'UCR_Small[0-50000]'
    root = 'results/ucr_experiment/DSS/Apr-29-2020-15-N-UCR_test_eu_soi_0-to-50000'
    file_list = os.listdir(root)
    file_list = [os.path.join(root, x) for x in file_list]

    # dt_dict = {'eu': 'Euclidean', 'ma': 'Manhattan', 'ch': 'Chebyshev'}
    dt_dict = {'eu': 'Euclidean'}

    fig_name = 'Query Performance across Dataset Sizes\n'
    k_to_look = [1, 5, 15]

    axis_label_ft = 18
    title_ft = 20

    # End of Initial Variables #########################################################################################
    dt_list = dt_dict.keys()
    title = fig_name + notes
    x = np.arange(len(k_to_look))  # the label locations

    for i, dt in enumerate(dt_list):
        fd = {x: x for x in file_list if '_' + dt + '.csv' in x}  # filter to get the result of a specific dist type
        # plot the clustering time as a heatmap across data length and data rows
        bin_paa_prep_time = []
        bin_gx_prep_time = []
        bin_dss_prep_time = []

        bin_size = []

        bin_qbf_time = []  # query brute force
        bin_qpaa_time = []  # query paa
        bin_qgx_time = []  # query genex
        bin_qdss_time = []  # query genex

        bin_qpaa_error = []
        bin_qgx_error = []
        bin_qdss_error = []

        for entry in fd.items():
            dataset_name, result_file = entry
            df = pd.read_csv(result_file)

            # information about this dataset's result
            size = df.iloc[(0, 24)]

            paa_prep_time = df.iloc[(0, 1)]
            gx_prep_time = df.iloc[(0, 2)]
            dss_prep_time = df.iloc[(0, 3)]

            qbf_time = [x for x in df.iloc[:, 6].values if not np.isnan(x)]
            qpaa_time = [x for x in df.iloc[:, 7].values if not np.isnan(x)]
            qgx_time = [x for x in df.iloc[:, 8].values if not np.isnan(x)]
            qdss_time = [x for x in df.iloc[:, 9].values if not np.isnan(x)]

            qpaa_error = [x for x in df.iloc[:, 10].values if not np.isnan(x)]
            qgx_error = [x for x in df.iloc[:, 11].values if not np.isnan(x)]
            qdss_error = [x for x in df.iloc[:, 12].values if not np.isnan(x)]

            bin_size.append(size)

            bin_paa_prep_time.append(paa_prep_time)
            bin_gx_prep_time.append(gx_prep_time)
            bin_dss_prep_time.append(dss_prep_time)

            bin_qbf_time.append(np.mean(qbf_time))
            bin_qpaa_time.append(np.mean(qpaa_time))
            bin_qgx_time.append(np.mean(qgx_time))
            bin_qdss_time.append(np.mean(qdss_time))

            bin_qpaa_error.append(np.mean(qpaa_error))
            bin_qgx_error.append(np.mean(qgx_error))
            bin_qdss_error.append(np.mean(qdss_error))
            pass

        note = '\n Each dot represents on dataset'

        # Plot the Cluster Time
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 8)
        plt.title('Cluster time across Data Size for Distance Type: ' + dt_dict[dt] + note)
        plt.scatter(bin_size, bin_paa_prep_time, c='blue', label='PAA Preparation Time')
        plt.scatter(bin_size, bin_gx_prep_time, c='cyan', label='Genex Cluster Time', marker='x')
        plt.scatter(bin_size, bin_dss_prep_time, c='green', label='DSS Cluster Time', marker='x')
        plt.ylabel('Time (second)')
        plt.xlabel('Time series length (number of data points)')
        # plt.ylim(-1, 25)
        plt.legend()
        plt.show()

        # Plot the Query Time
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 8)
        plt.title('Query Time across Data Size for Distance Type: ' + dt_dict[dt] + note)
        plt.scatter(bin_size, bin_qbf_time, c='red', label='Brute Force Query Time')
        plt.scatter(bin_size, bin_qpaa_time, c='orange', label='PAA Query Time')
        plt.scatter(bin_size, bin_qgx_time, c='blue', label='Genex Query Time')
        plt.scatter(bin_size, bin_qdss_time, c='green', label='DSS Query Time')
        plt.ylabel('Time (second)')
        plt.xlabel('Time series length (number of data points)')
        # plt.ylim(-1, 25)
        plt.legend()
        plt.show()

        # Plot the accuracy
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 8)
        plt.title('Normalized Error across Data Size for Distance Type: ' + dt_dict[dt] + note)
        plt.scatter(bin_size, bin_qpaa_error, c='orange', label='PAA Query Error')
        plt.scatter(bin_size, bin_qgx_error, c='blue', label='Genex Query Error')
        plt.scatter(bin_size, bin_qdss_error, c='green', label='DSS Query Error')
        plt.ylabel('Normalized Error')
        plt.xlabel('Data size (number of data points)')
        # plt.ylim(-0.0025, 0.14)
        plt.legend()
        plt.show()

        print('BrainEx Clustering on average took: ' + str(np.mean(bin_gx_prep_time)))
        print('DSS Clustering on average took: ' + str(np.mean(bin_dss_prep_time)))

        print('BrainEx error on average is: ' + str(np.mean(bin_qgx_error)))
        print('DSS error on average is: ' + str(np.mean(bin_qdss_error)))

        print('BrainEx query on average took: ' + str(np.mean(bin_qgx_time)))
        print('DSS query on average took: ' + str(np.mean(bin_qdss_time)))