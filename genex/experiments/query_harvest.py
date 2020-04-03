import math
import os
import random
import shutil
import time
from datetime import datetime
from logging import warning

import numpy as np
import pandas as pd

# spark_location = '/Users/Leo/spark-2.4.3-bin-hadoop2.7'  # Set your own
# java8_location = '/Library/Java/JavaVirtualMachines/jdk1.8.0_151.jdk/Contents/Home/jre'
# os.environ['JAVA_HOME'] = java8_location
# findspark.init(spark_home=spark_location)
from genex.utils.gxe_utils import from_csv

########################################################################################################################
mp_args = {'num_worker': 12,
           'driver_mem': 12,
           'max_result_mem': 12}


# eu_exclude = ['ChlorineConcentration',
#               'ElectricDevices',
#               'Haptics',
#               'InsectEPGRegularTrain',
#               'Lightning2',
#               'Meat',
#               'Trace',
#               ]
########################################################################################################################

def experiment_genex(data, output, feature_num, num_sample, query_split,
                     dist_type, _lb_opt, _radius, use_spark: bool, loi_range: float, st: float, paa_c: float):
    # create gxdb from a csv file

    # set up where to save the results
    result_headers = np.array(
        [['cluster_time', 'query',
          'bf_time', 'paa_time', 'gx_time',
          'dist_diff_btw_paa_bf', 'dist_diff_btw_gx_bf',
          'bf_dist', 'bf_match',
          'paa_dist', 'paa_match',
          'gx_dist', 'gx_match',
          'num_rows', 'num_cols_max', 'num_cols_median', 'data_size', 'num_query']])

    result_df = pd.DataFrame(columns=result_headers[0, :])

    gxe = from_csv(data, num_worker=mp_args['num_worker'], driver_mem=mp_args['driver_mem'],
                   max_result_mem=mp_args['max_result_mem'],
                   feature_num=feature_num, use_spark=use_spark, _rows_to_consider=num_sample,
                   header=None)
    num_rows = len(gxe.data_raw)
    num_query = int(query_split * num_rows)
    try:
        assert num_query > 0
    except AssertionError:
        raise Exception('Number of query with given query_split yields zero query sequence, try increase query_split')
    loi = (int(gxe.get_max_seq_len() * (1 - loi_range)), int(gxe.get_max_seq_len() * (1 + loi_range)))

    print('Number of rows is  ' + str(num_rows))
    print('Max seq len is ' + str(gxe.get_max_seq_len()))

    result_df = result_df.append({'num_rows': num_rows}, ignore_index=True)  # use append to create the first row
    result_df['num_cols_max'] = gxe.get_max_seq_len()
    result_df['num_cols_median'] = np.median(gxe.get_seq_length_list())
    result_df['data_size'] = gxe.get_data_size()
    result_df['num_query'] = num_query

    print('Generating query of max seq len ...')
    # generate the query sets
    query_set = list()
    # get the number of subsequences
    # randomly pick a sequence as the query from the query sequence, make sure the picked sequence is in the input list
    # this query'id must exist in the database
    for i in range(num_query):
        random.seed(i)
        query_len = random.choice(list(range(loi[0], loi[1])))
        this_query = gxe.get_random_seq_of_len(query_len, seed=i)
        query_set.append(this_query)
        print('Adding to query set: ' + str(this_query))

    print('Performing clustering ...')
    print('Using dist_type = ' + str(dist_type) + ', Length of query is ' + str(query_len))
    print('Using loi offset of ' + str(loi_range))
    print('Building length of interest is ' + str(loi))
    print('Building Similarity Threshold is ' + str(st))

    cluster_start_time = time.time()
    gxe.build(st=st, dist_type=dist_type, loi=loi)
    cluster_time = time.time() - cluster_start_time
    result_df = result_df.append({'cluster_time': cluster_time}, ignore_index=True)
    print('Clustering took ' + str(cluster_time) + ' sec')
    # randomly pick a sequence as the query from the query sequence, make sure the picked sequence is in the input list
    # this query'id must exist in the database
    overall_diff_gxbf_list = []
    overall_diff_paabf_list = []

    print('Evaluating ...')
    for i, q in enumerate(query_set):
        print('Dataset: ' + data + ' - dist_type: ' + dist_type + '- Querying #' + str(i) + ' of ' + str(
            len(query_set)) + '; query = ' + str(q))
        start = time.time()
        print('Running Genex Query ...')
        query_result_gx = gxe.query(query=q, best_k=15, _lb_opt=_lb_opt, _radius=_radius)
        # print('...Not Actually running... Simulating results!')
        # query_result_gx = [(0.0, [1, 2, 3])] * 15
        gx_time = time.time() - start
        print('Genex  query took ' + str(gx_time) + ' sec')

        start = time.time()
        print('Running Brute Force Query ...')
        query_result_bf = gxe.query_brute_force(query=q, best_k=15, _use_cache=False)
        # print('...Not Actually running... Simulating results!')
        # query_result_bf = [(0.0, [1, 2, 3])] * 15
        bf_time = time.time() - start
        print('Brute force query took ' + str(bf_time) + ' sec')

        # Pure PAA Query
        start = time.time()
        print('Running Pure PAA Query ...')
        query_result_paa = gxe.query_brute_force(query=q, best_k=15, _use_cache=False, _paa=paa_c)
        paa_time = time.time() - start
        print('Pure PAA query took ' + str(paa_time) + ' sec')

        # save the results
        print('Saving results for query #' + str(i) + ' of ' + str(len(query_set)))
        result_df = result_df.append({'query': str(q), 'bf_time': bf_time, 'paa_time': paa_time, 'gx_time': gx_time},
                                     ignore_index=True)
        diff_gxbf_list = []
        diff_paabf_list = []

        for bf_r, paa_r, gx_r in zip(query_result_bf, query_result_paa, query_result_gx):
            diff_gxbf = abs(gx_r[0] - bf_r[0])
            diff_paabf = abs(paa_r[0] - bf_r[0])

            diff_gxbf_list.append(diff_gxbf)
            diff_paabf_list.append(diff_paabf)

            overall_diff_gxbf_list.append(diff_gxbf)
            overall_diff_paabf_list.append(diff_paabf)

            # 'bf_time', 'paa_time', 'gx_time',
            # 'dist_diff_btw_paa_bf', 'dist_diff_btw_gx_bf',
            # 'bf_dist', 'bf_match',
            # 'paa_dist', 'paa_match',
            # 'gx_dist', 'gx_match',
            result_df = result_df.append({'dist_diff_btw_paa_bf': diff_paabf,
                                          'dist_diff_btw_gx_bf': diff_gxbf,
                                          'bf_dist': bf_r[0], 'bf_match': bf_r[1],
                                          'paa_dist': paa_r[0], 'paa_match': paa_r[1],
                                          'gx_dist': gx_r[0], 'gx_match': gx_r[1],
                                          }, ignore_index=True)
        print('GX error for query ' + str(q) + ' is ' + str(np.mean(diff_gxbf_list)))
        print('PAA error for query ' + str(q) + ' is ' + str(np.mean(diff_paabf_list)))
        result_df = result_df.append({'dist_diff_btw_paa_bf': np.mean(diff_gxbf_list),
                                      'dist_diff_btw_gx_bf': np.mean(diff_paabf_list)}, ignore_index=True)

        result_df.to_csv(output)

        # evict the mp context every 3 queries
        print('Done')
        # if i % 3:
        #     gxe.reset_mp(use_spark=use_spark, **mp_args)
        #     print('     Evicting Multiprocess Context')

    # save the overall difference
    result_df = result_df.append({'dist_diff_btw_paa_bf': np.mean(overall_diff_paabf_list),
                                  'dist_diff_btw_gx_bf': np.mean(overall_diff_gxbf_list)}, ignore_index=True)
    print('Result saved to ' + output)
    result_df.to_csv(output)
    # terminate the spark session
    gxe.stop()

    return gxe


# def generate_exp_set_inplace(dataset_list, dist_type, notes: str):
#     today = datetime.now()
#     dir_name = os.path.join('results', today.strftime("%b-%d-%Y-") + str(today.hour) + '-N-' + notes)
#     if not os.path.exists(dir_name):
#         os.mkdir(dir_name)
#
#     config_list = []
#     for d in dataset_list:
#         d_path = os.path.join('data', d + '.csv')
#         assert os.path.exists(d_path)
#
#         config_list.append({
#             'data': d_path,
#             'output': os.path.join(dir_name, d + '_' + dist_type + '.csv'),
#             'feature_num': 0,
#             'dist_type': dist_type
#         })
#     return config_list


def generate_exp_set_from_root(root, output, exclude_list, dist_type: str, notes: str, soi):
    today = datetime.now()
    output_dir_path = os.path.join(output, today.strftime("%b-%d-%Y-") + str(today.hour) + '-N-' + notes)
    if not os.path.exists(output_dir_path):
        print('Creating output path: ' + output_dir_path)
        os.mkdir(output_dir_path)
    else:
        print('Output folder already exist, overwriting')
        shutil.rmtree(output_dir_path, ignore_errors=False, onerror=None)
        os.mkdir(output_dir_path)

    config_list = []
    dataset_list = get_dataset_train_path(root, exclude_list)
    dataset_list = dataset_list
    for d_name, dataset_path in dataset_list.items():

        # check dataset size
        df = pd.read_csv(dataset_path, sep='\t', header=None)
        if df.size < soi[0] or df.size > soi[1]:
            continue
        print('Adding ' + dataset_path)
        config_list.append({
            'data': dataset_path,
            'output': os.path.join(output_dir_path, d_name + '_' + dist_type + '.csv'),
            'feature_num': 0,
            'dist_type': dist_type
        })
    if len(config_list) < 1:
        raise Exception('No dataset satisfied the given soi')
    print('Added ' + str(len(config_list)) + ' datasets with the given soi')
    return config_list


def run_exp_set(exp_set, num_sample, query_split,
                _lb_opt, radius, use_spark, loi_range, st, paa_c):
    for es in exp_set:
        experiment_genex(**es, num_sample=num_sample, query_split=query_split,
                         _lb_opt=_lb_opt, _radius=radius, use_spark=use_spark, loi_range=loi_range, st=st, paa_c=paa_c)


def get_dataset_train_path(root, exclude_list):
    trailing = '_TRAIN.tsv'
    data_path_list = {}
    for name in os.listdir(root):
        if name in exclude_list:
            continue
        assert os.path.isdir(os.path.join(root, name))
        this_path = os.path.join(root, name, name + trailing)
        try:
            assert os.path.isfile(this_path)
        except AssertionError:
            warning('File not exist: ' + this_path)
        data_path_list[name] = this_path
    return data_path_list

# datasets = [
#     'ItalyPower',
#     'ECGFiveDays',
#     'Gun_Point_TRAIN',
#     'synthetic_control_TRAIN'
# ]
########################################################################################################################
# ex_config_0 = {
#     'num_sample': 40,
#     'num_query': 40,
#     '_lb_opt': False,
#     'radius': 0,
#     'use_spark': True
# }
# start = time.time()
# notes_0 = 'UseSpark-R0-noOpt'
# es_eu_0 = generate_exp_set(datasets, 'eu', notes=notes_0)
# es_ma_0 = generate_exp_set(datasets, 'ma', notes=notes_0)
# es_ch_0 = generate_exp_set(datasets, 'ch', notes=notes_0)
# run_exp_set(es_eu_0, **ex_config_0)
# run_exp_set(es_ma_0, **ex_config_0)
# run_exp_set(es_ch_0, **ex_config_0)
# duration1 = time.time() - start
# print('Finished at')
# print(datetime.now())
# print('The experiment with radius 0 took ' + str(duration1/3600) + ' hrs')
########################################################################################################################
# ex_config_1 = {
#     'num_sample': 40,
#     'num_query': 40,
#     '_lb_opt': False,
#     'radius': 0,
#     'use_spark': False
# }
# start = time.time()
# notes_1 = 'noneSpark-R0-noOpt'
# es_eu_1 = generate_exp_set(datasets, 'eu', notes=notes_1)
# es_ma_1 = generate_exp_set(datasets, 'ma', notes=notes_1)
# es_ch_1 = generate_exp_set(datasets, 'ch', notes=notes_1)
# run_exp_set(es_eu_1, **ex_config_1)
# run_exp_set(es_ma_1, **ex_config_1)
# run_exp_set(es_ch_1, **ex_config_1)
# duration1 = time.time() - start
# print('Finished at')
# print(datetime.now())
# print('The experiment with radius 0 took ' + str(duration1/3600) + ' hrs')

########################################################################################################################
# ex_config_2 = {
#     'num_sample': 40,
#     'num_query': 40,
#     '_lb_opt_repr': 'bsf',
#     '_lb_opt_cluster': 'bsf',
#     'radius': 0
# }
# start = time.time()
# notes_2 = 'UseSpark-R0-bsf'
# es_eu_2 = generate_exp_set(datasets, 'eu', notes=notes_2)
# es_ma_2 = generate_exp_set(datasets, 'ma', notes=notes_2)
# es_ch_2 = generate_exp_set(datasets, 'ch', notes=notes_2)
# run_exp_set(es_eu_2, **ex_config_2)
# run_exp_set(es_ma_2, **ex_config_2)
# run_exp_set(es_ch_2, **ex_config_2)
# duration2 = time.time() - start
# print('Finished at')
# print(datetime.now())
# print('The experiment with radius 0 took ' + str(duration2/3600) + ' hrs')

########################################################################################################################
# ex_config_3 = {
#     'num_sample': 40,
#     'num_query': 40,
#     '_lb_opt': True,
#     'radius': 0,
#     'use_spark': True
# }
# start = time.time()
# notes_3 = 'UseSpark-R0-bsfKimOnly'
# es_eu_3 = generate_exp_set(datasets, 'eu', notes=notes_3)
# es_ma_3 = generate_exp_set(datasets, 'ma', notes=notes_3)
# es_ch_3 = generate_exp_set(datasets, 'ch', notes=notes_3)
# run_exp_set(es_eu_3, **ex_config_3)
# run_exp_set(es_ma_3, **ex_config_3)
# run_exp_set(es_ch_3, **ex_config_3)
# duration3 = time.time() - start
# print('Finished at')
# print(datetime.now())
# print('The experiment took ' + str(duration3/3600) + ' hrs')
# ########################################################################################################################
# ex_config_4 = {
#     'num_sample': 40,
#     'num_query': 40,
#     '_lb_opt_repr': 'none',
#     '_lb_opt_cluster': 'none',
#     'radius': 1,
#     'use_spark': True
# }
# start = time.time()
# notes_4 = 'UseSpark-R1-noOpt'
# es_eu_4 = generate_exp_set(datasets, 'eu', notes=notes_4)
# es_ma_4 = generate_exp_set(datasets, 'ma', notes=notes_4)
# es_ch_4 = generate_exp_set(datasets, 'ch', notes=notes_4)
# run_exp_set(es_eu_4, **ex_config_4)
# run_exp_set(es_ma_4, **ex_config_4)
# run_exp_set(es_ch_4, **ex_config_4)
# duration4 = time.time() - start
# print('Finished at')
# print(datetime.now())
# print('The experiment took ' + str(duration4/3600) + ' hrs')
########################################################################################################################
# num_sample = 200
########################################################################################################################
# ex_config_6 = {
#     'num_sample': num_sample,
#     'num_query': 40,
#     '_lb_opt': False,
#     'radius': 1,
#     'use_spark': True
# }
# start = time.time()
# notes_6 = 'UseSpark-R1-noOpt_numSample400'
# es_eu_6 = generate_exp_set(datasets, 'eu', notes=notes_6)
# es_ma_6 = generate_exp_set(datasets, 'ma', notes=notes_6)
# es_ch_6 = generate_exp_set(datasets, 'ch', notes=notes_6)
# run_exp_set(es_eu_6, **ex_config_6)
# run_exp_set(es_ma_6, **ex_config_6)
# run_exp_set(es_ch_6, **ex_config_6)
# duration6 = time.time() - start
# print('Finished at')
# print(datetime.now())
# print('The experiment took ' + str(duration6 / 3600) + ' hrs')

########################################################################################################################
# ex_config_5 = {
#     'num_sample': num_sample,
#     'num_query': 40,
#     '_lb_opt': True,
#     'radius': 1,
#     'use_spark': True
# }
# start = time.time()
# notes_5 = 'UseSpark-R1-LBOpt_numSample400'
# es_eu_5 = generate_exp_set(datasets, 'eu', notes=notes_5)
# es_ma_5 = generate_exp_set(datasets, 'ma', notes=notes_5)
# es_ch_5 = generate_exp_set(datasets, 'ch', notes=notes_5)
# run_exp_set(es_eu_5, **ex_config_5)
# run_exp_set(es_ma_5, **ex_config_5)
# run_exp_set(es_ch_5, **ex_config_5)
# duration5 = time.time() - start
# print('Finished at')
# print(datetime.now())
# print('The experiment took ' + str(duration5 / 3600) + ' hrs')

# num_sample = 500
# root = '/home/apocalyvec/data/UCRArchive_2018'
#
#
# ex_config_ucr_0 = {
#     'num_sample': num_sample,
#     'num_query': 10,
#     '_lb_opt': False,
#     'radius': 1,
#     'use_spark': True,
#     'loi_range': 0.1,
#     'st': 0.1
# }

# exp_set_args_eu = {
#     'dist_type': 'eu',
#     'notes': 'UCR0_numSampleAll_eu_101-to-128',
#     'start': 101,
#     'end': 128
# }
# exp_set_args_ma_1 = {
#     'dist_type': 'ma',
#     'notes': 'UCR0_numSampleAll_ma_0-to-30',
#     'start': 0,
#     'end': 30
# }
# exp_set_args_ma_2 = {
#     'dist_type': 'ma',
#     'notes': 'UCR0_numSampleAll_ma_101-to-128',
#     'start': 101,
#     'end': 128
# }
# exp_set_args_ch_1 = {
#     'dist_type': 'ch',
#     'notes': 'UCR0_numSampleAll_ch_28-to-30',
#     'start': 28,
#     'end': 30
# }
# exp_set_args_ch_2 = {
#     'dist_type': 'ch',
#     'notes': 'UCR0_numSampleAll_ch_62-to-128',
#     'start': 62,
#     'end': 128
# }
# # es_eu_ucr = generate_exp_set_from_root(root, **exp_set_args_eu)
# # es_ma_ucr_1 = generate_exp_set_from_root(root, **exp_set_args_ma_1)
# # es_ma_ucr_2 = generate_exp_set_from_root(root, **exp_set_args_ma_2)
# es_ch_ucr_1 = generate_exp_set_from_root(root, **exp_set_args_ch_1)
# es_ch_ucr_2 = generate_exp_set_from_root(root, **exp_set_args_ch_2)
#
# # run_exp_set(es_eu_ucr, **ex_config_ucr_0)
# # run_exp_set(es_ma_ucr_1, **ex_config_ucr_0)
# # run_exp_set(es_ma_ucr_2, **ex_config_ucr_0)
# run_exp_set(es_ch_ucr_1, **ex_config_ucr_0)
# run_exp_set(es_ch_ucr_2, **ex_config_ucr_0)
