import os

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np



def get_mse(file_name: str, best_k: int, num_sample: int, num_most_k: int) -> float:
    assert best_k <= 15
    offset_start = 4
    diff_col_num = 5
    offset_between_sample = 2
    mse_list = []

    df = pd.read_csv(file_name)
    for i in range(num_sample):
        row_start = offset_start + (num_most_k + offset_between_sample) * i
        row_end = row_start + best_k
        mse_list += list(df.iloc[row_start:row_end, diff_col_num].values)
    rtn = np.mean(mse_list)
    return rtn


def autolabel(rects, ax, dataset):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height, 4)),
                    xy=(rect.get_x() + rect.get_width()/len(dataset), height),
                    xytext=(2, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


if __name__ == '__main__':
    font = {'family': 'DejaVu Sans',
            'weight': 'bold',
            'size': 12}
    matplotlib.rc('font', **font)

    # date = 'Jan-30-2020-12-N-UseSpark-R1-noOptFastDTW_numSample400'
    # notes = 'UseSpark-R1-noOptFastDTW_numSample400'
    notes = 'UCR_Small[0-50000]'
    root = 'results/ucr_experiment/Small'
    file_list = os.listdir(root)
    file_list = [os.path.join(root, x) for x in file_list]

    dt_dict = {'eu': 'Euclidean', 'ma': 'Manhattan', 'ch': 'Chebyshev'}
    # dt_dict = {'eu': 'Euclidean'}

    fig_name = 'Query Performance across Dataset Sizes\n'
    k_to_look = [1, 5, 15]

    axis_label_ft = 18
    title_ft = 20

    c_time_coord = (1, 1)
    num_row_coord = (0, 14)
    num_col_med_coord = (0, 15)
    data_size_coord = (0, 17)

    # End of Initial Variables #########################################################################################
    dt_list = dt_dict.keys()
    title = fig_name + notes
    x = np.arange(len(k_to_look))  # the label locations



    for i, dt in enumerate(dt_list):
        fd = {x: x for x in file_list if '_' + dt + '.csv' in x}  # filter to get the result of a specific dist type
        # plot the clustering time as a heatmap across data length and data rows
        bin_c_time = []
        bin_rows = []
        bin_length = []
        bin_size = []

        bin_qbf_time = []  # query brute force
        bin_qgx_time = []  # query genex
        bin_qpaa_time = []  # query paa

        bin_qgx_error = []
        bin_qpaa_error = []
        for entry in fd.items():
            dataset_name, result_file = entry
            df = pd.read_csv(result_file)

            # information about this dataset's result
            c_time = df.iloc[c_time_coord]
            rows = df.iloc[num_row_coord]
            length = df.iloc[num_col_med_coord]
            size = df.iloc[data_size_coord]
            qbf = [x for x in df.iloc[:, 3].values if not np.isnan(x)]
            qpaa = [x for x in df.iloc[:, 4].values if not np.isnan(x)]
            qgx = [x for x in df.iloc[:, 5].values if not np.isnan(x)]

            qpaa_error = [x for x in df.iloc[:, 6].values if not np.isnan(x)]
            qgx_error = [x for x in df.iloc[:, 7].values if not np.isnan(x)]

            bin_c_time.append(c_time)
            bin_rows.append(rows)
            bin_length.append(length)
            bin_size.append(size)
            bin_qbf_time.append(np.mean(qbf))
            bin_qpaa_time.append(np.mean(qpaa))
            bin_qgx_time.append(np.mean(qgx))

            bin_qpaa_error.append(np.mean(qpaa_error))
            bin_qgx_error.append(np.mean(qgx_error))


        # Plot the Query and Cluster Time
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 8)
        plt.title('Cluster(Gx) and Query Time across Data Size for Distance Type: ' + dt_dict[dt])
        plt.scatter(bin_size, bin_qbf_time, c='red', label='Brute Force Query Time')
        plt.scatter(bin_size, bin_qpaa_time, c='orange', label='PAA Query Time')
        plt.scatter(bin_size, bin_qgx_time, c='blue', label='Genex Query Time')
        plt.scatter(bin_size, bin_c_time, c='cyan', label='Genex Cluster Time', marker='x')
        plt.ylabel('Time (second)')
        plt.xlabel('Data size (number of data points)')
        plt.ylim(-1, 25)
        plt.legend()
        plt.show()

        # Plot the accuracy
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 8)
        plt.title('Normalized Error across Data Size for Distance Type: ' + dt_dict[dt])
        plt.scatter(bin_size, bin_qpaa_error, c='blue', label='Genex Query Error')
        plt.scatter(bin_size, bin_qgx_error, c='orange', label='PAA Query Error')
        plt.ylabel('Normalized Error')
        plt.xlabel('Data size (number of data points)')
        plt.ylim(-0.0025, 0.14)
        plt.legend()
        plt.show()