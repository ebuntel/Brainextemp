import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np



def get_mse(file_name: str, best_k: int, num_sample: int, num_most_k: int) -> float:
    assert best_k <= 15
    offset_start = 2
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
    date = 'Jan-30-2020-15-N-UseSpark-R1-LBOptNormalDTW_numSample400'
    notes = 'UseSpark-R1-LBOptNormalDTW_numSample400'

    dt_dict = {'eu': 'Euclidean', 'ma': 'Manhattan', 'ch': 'Chebyshev'}
    title = 'Query Accuracy across Distance Types \n'
    k_to_look = [1, 3, 5, 9, 15]
    num_sample = 40
    num_most_k = 15
    datasets = [
        'ItalyPower',
        'ECGFiveDays',
        'Gun_Point_TRAIN',
        'synthetic_control_TRAIN'
    ]
    width = 0.20  # the width of the bars
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 8)
    axis_label_ft = 18
    title_ft = 20
    # End of Initial Variables #########################################################################################
    dt_list = dt_dict.keys()
    title = title + notes
    dt_error_dict = {dt: [] for dt in dt_list}
    x = np.arange(len(k_to_look))  # the label locations

    for i, dt in enumerate(dt_list):
        fd = {x: 'results/' + date + '/' + x + '_' + dt + '.csv' for x in datasets}
        overrall_error_list = []

        for k in k_to_look:
            error_of_k = []  # error of k across datasets
            for entry in fd.items():
                dataset_name, result_file = entry
                error_of_k.append(get_mse(file_name=result_file, best_k=k, num_sample=num_sample, num_most_k=num_most_k))
            dt_error_dict[dt].append(np.mean(error_of_k))

        overrall_error_list += dt_error_dict[dt]
        rect = ax.bar(x + 3.4 * i * width/len(dt_list), dt_error_dict[dt], width, label=dt_dict[dt])
        autolabel(rect, ax, dt_error_dict)

    ax.set_ylabel('Percentage Error',  fontsize=axis_label_ft)
    ax.set_xlabel('Best K',  fontsize=axis_label_ft)
    ax.set_title(title, fontsize=title_ft)
    ax.set_xticks(x)
    ax.set_xticklabels(k_to_look)
    ax.legend()

    fig.tight_layout()

    plt.show()
    print(date + ': The overall error for  is ' + str(np.mean(overrall_error_list)))