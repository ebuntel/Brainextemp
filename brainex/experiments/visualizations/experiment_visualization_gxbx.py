import os

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np


def min_max_normalize(l):
    return (l - np.min(l)) / (np.max(l) - np.min(l))

if __name__ == '__main__':
    font = {'family': 'DejaVu Sans',
            'weight': 'bold',
            'size': 12}
    matplotlib.rc('font', **font)

    root_gx_prep = '/home/apocalyvec/data/Genex/exps_results_summary/preprocess.csv'
    root_gx_eu = '/home/apocalyvec/data/Genex/exps_results_summary/sim_euclidean.csv'
    root_gx_ma = '/home/apocalyvec/data/Genex/exps_results_summary/sim_manhattan.csv'
    root_gx_ch = '/home/apocalyvec/data/Genex/exps_results_summary/sim_minkowski.csv'

    root = '/home/apocalyvec/data/Genex/brainex/Jul-10-2020-21'
    file_list = os.listdir(root)
    file_list = [os.path.join(root, x) for x in file_list]

    dt_dict = {'eu': 'Euclidean', 'ma': 'Manhattan', 'ch': 'Chebyshev'}
    dt_gx_path_dict = {'eu': root_gx_eu, 'ma': root_gx_ma, 'ch': root_gx_ch}
    dt_offset_dict = {'eu': 0, 'ma': 1, 'ch': 2}

    # dt_dict = {'eu': 'Euclidean'}

    fig_name = 'Query Performance across Dataset Sizes\n'

    axis_label_ft = 18
    title_ft = 20

    # End of Initial Variables #########################################################################################
    dt_list = dt_dict.keys()
    title = fig_name
    df_genex_prep = pd.read_csv(root_gx_prep)

    for i, dt in enumerate(dt_list):
        fd = [x for x in file_list if dt in x.strip('.csv').split('_')]

        bin_bx_prep_time = []
        bin_gx_prep_time = []

        bin_size = []
        bin_qbx_time = []
        bin_qbx_error = []
        bin_qgx_time = []
        bin_qgx_error = []

        for result_file in fd:
            df = pd.read_csv(result_file)
            dataset_name = result_file.strip(dt + '_' + '.csv').split('/')[-1]

            if not np.any(df_genex_prep['dataset'] == dataset_name):
                continue
            gx_prep_time = df_genex_prep.loc[df_genex_prep['dataset'] == dataset_name].iloc[dt_offset_dict[dt], 1]

            size = df.iloc[(0, -1)]
            bx_prep_time = df.iloc[(0, 1)]
            qbx_time = [x for x in df.iloc[:, 5].values if not np.isnan(x)]
            qbx_error = [x for x in df.iloc[:, 6].values if not np.isnan(x)]

            bin_size.append(size)
            bin_bx_prep_time.append(bx_prep_time)
            bin_gx_prep_time.append(gx_prep_time)

            bin_qbx_time.append(np.mean(qbx_time))
            bin_qbx_error.append(np.mean(qbx_error))

            df_gx_query = pd.read_csv(dt_gx_path_dict[dt])
            df_gx_query_dataset = df_gx_query.loc[(df_gx_query['name'] == dataset_name) & (df_gx_query['method'] == 'genex_0.1')]
            bin_qgx_error.append(df_gx_query_dataset.values[0][3])
            bin_qgx_time.append(df_gx_query_dataset.values[0][4])

        note = '\n Each dot represents on dataset'
        prep_line = np.linspace(min(bin_size), max(bin_size), 100)

        # # Plot the Cluster Time
        # fig, ax = plt.subplots()
        # fig.set_size_inches(15, 8)
        # plt.title('Cluster time across Data Size for Distance Type: ' + dt_dict[dt] + note)
        # # plt.scatter(bin_size, bin_paa_prep_time, c='blue', label='PAA Preparation Time')
        #
        # model = np.poly1d(np.polyfit(bin_size, bin_bx_prep_time, 3))
        # plt.plot(prep_line, model(prep_line), label='BrainEx Cluster Time Fitted', c='blue')
        # plt.scatter(bin_size, bin_bx_prep_time, c='cyan', label='BrainEx Cluster Time', marker='x')
        #
        # model = np.poly1d(np.polyfit(bin_size, bin_gx_prep_time, 3))
        # plt.plot(prep_line, model(prep_line), label='Genex Cluster Time Fitted', c='orange')
        # plt.scatter(bin_size, bin_gx_prep_time, c='orange', label='Genex Cluster Time', marker='x')
        #
        # plt.ylabel('Time (second)')
        # plt.xlabel('Time series length (number of data points)')
        # # plt.ylim(-100, 200)
        # plt.legend()
        # plt.show()
        #
        # print('Distance type: ' + dt_dict[dt])
        # print('Genex average preprocess time is ' + str(np.mean(bin_gx_prep_time)))
        # print('BrainEx average preprocess time is ' + str(np.mean(bin_bx_prep_time)))
        # print('BrainEx is ' + str(np.mean(bin_gx_prep_time) / np.mean(bin_bx_prep_time)) + ' times faster.')

        pass
        # Plot the Query Time
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 8)

        plt.title('Query Time across Data Size for Distance Type: ' + dt_dict[dt])
        # plt.scatter(bin_size, bin_qbf_time, c='red', label='Brute Force Query Time')
        # plt.scatter(bin_size, bin_qpaa_time, c='orange', label='PAA Query Time')

        model = np.poly1d(np.polyfit(bin_size, bin_qgx_time, 3))  # change the Y vector in this line
        plt.plot(prep_line, model(prep_line), label='Genex Query Time Fitted', c='orange')
        plt.scatter(bin_size, bin_qgx_time, c='orange', label='Query Query Time')

        model = np.poly1d(np.polyfit(bin_size, bin_qbx_time, 3))  # change the Y vector in this line
        plt.plot(prep_line, model(prep_line), label='BrainEx Query Time Fitted', c='blue')
        plt.scatter(bin_size, bin_qbx_time, c='blue', label='BrainEx Query Time')

        print('Distance type: ' + dt_dict[dt])
        print('Genex average ERROR is ' + str(np.mean(bin_qgx_error)))
        print('BrainEx average ERROR is ' + str(np.mean(bin_qbx_error)))
        # print('BrainEx is ' + str(np.mean(bin_gx_prep_time) / np.mean(bin_bx_prep_time)) + ' times faster.')

        plt.ylabel('Time (second)')
        plt.xlabel('Time series length (number of data points)')
        # plt.ylim(-1, 25)
        plt.legend()
        plt.show()
