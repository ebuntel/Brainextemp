import time

import pandas as pd
import numpy as np

from genex.utils.gxe_utils import from_csv
from genex.utils.gxe_utils import from_db


data_file_train = 'data_original/SART2018_HbO/SART2018_HbO_train.csv'
data_file_test = 'data_original/SART2018_HbO/SART2018_HbO_test.csv'
feature_num = 6
db_path = 'results/saves/SART2018_HbO_train_gxe-lenTop9e-1_st9e-1_dtCH'
k = 5
label_index = 2
num_worker = 12
dist_type = 'ch'

# Creating new Train GXE ###########################################################
start = time.time()
gxe_train = from_csv(data_file_train, feature_num=feature_num, num_worker=12, use_spark=True)
gxe_train.build(st=0.1, loi=[gxe_train.get_max_seq_len() * 0.9], dist_type=dist_type)
build_time = time.time() - start
gxe_train.save(db_path)
# Load Existing Train GXE #############################################################
# gxe_train = from_db('results/saves/SART2018_HbO_train', num_worker=num_worker)

# don't need to build the test gxe
gxe_test = from_csv(data_file_test, feature_num=feature_num, num_worker=num_worker, use_spark=False)
query_time = []
correct_list = []
test_list = gxe_test.get_norm_ts_list()
for test_seq in test_list:
    label_true = test_seq.seq_id[label_index]
    start = time.time()
    label_pred = gxe_train.predice_label_knn(test_seq, k, label_index=label_index, verbose=1)
    query_time.append(time.time() - start)
    is_correct = label_pred == label_true
    print('The predict was ' + 'correct' if is_correct else 'wrong')
    correct_list.append(is_correct)

avg_q_time = np.mean(query_time)
accuracy = correct_list.count(True) / len(correct_list)
