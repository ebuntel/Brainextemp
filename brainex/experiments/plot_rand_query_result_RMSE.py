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

    date = 'Jan-30-2020-12-N-UseSpark-R1-noOptFastDTW_numSample400'
    notes = 'UseSpark-R1-noOptFastDTW_numSample400'

    # dist_type = 'ma'
    # title = 'Manhattan Query Accuracy across Datasets \n' + notes
    dist_type = 'eu'
    title = 'Euclidean Query Accuracy across Datasets \n' + notes
    # dist_type = 'ch'
    # title = 'Chebyshev Query Accuracy across Datasets \n' + notes

    file_dict = {'Gun Point': 'results/' + date + '/Gun_Point_TRAIN_' + dist_type + '.csv',
                 'ECG Five Days': 'results/' + date + '/ECGFiveDays_' + dist_type + '.csv',
                 'Italy Power Demand': 'results/' + date + '/ItalyPower_' + dist_type + '.csv',
                 'Synthetic Control': 'results/' + date + '/synthetic_control_TRAIN_' + dist_type + '.csv'
                 }

    dataset_mse_dict = {'Gun Point': [],
                        'ECG Five Days': [],
                        'Italy Power Demand': [],
                        'Synthetic Control': []
                        }  # the second dimension is 'k to look
    overall_error_list = []
    k_to_look = [1, 3, 5, 9, 15]

    num_sample = 40
    num_most_k = 15

    x = np.arange(len(k_to_look))  # the label locations
    width = 0.20  # the width of the bars
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 8)

    for i, entry in enumerate(file_dict.items()):
        dataset_name, result_file = entry
        for k in k_to_look:
            mse = get_mse(file_name=result_file, best_k=k, num_sample=num_sample, num_most_k=num_most_k)
            dataset_mse_dict[dataset_name].append(mse)
        overall_error_list += dataset_mse_dict[dataset_name]
        rect = ax.bar(x + 4 * i * width/len(dataset_mse_dict), dataset_mse_dict[dataset_name], width, label=dataset_name)
        autolabel(rect, ax, dataset_mse_dict)

    ax.set_ylabel('Percentage Error')
    ax.set_xlabel('Best K')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(k_to_look)
    ax.legend()

    fig.tight_layout()

    plt.show()
    print(date + ': The overall error for ' + dist_type + ' is ' + str(np.mean(overall_error_list)))