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
from brainex.utils.gxe_utils import from_csv


def experiment_BrainEX(mp_args, data: str, output: str, feature_num, num_sample, query_split,
                       dist_type, _lb_opt, _radius, use_spark: bool, loi_range: float, st: float,
                       n_segment: float, best_ks):
    # set up where to save the results
    result_headers = np.array(
        [['k',
          'paa_preprocess_time', 'sax_preprocess_time', 'gx_preprocess_time', 'dssgx_preprocess_time',
          # preprocessing times
          'query', 'query_len',  # the query sequence
          'bf_query_time', 'paa_query_time', 'sax_query_time', 'gx_query_time', 'dssgx_query_time',  # query times
          'dist_diff_btw_paa_bf', 'dist_diff_btw_sax_bf', 'dist_diff_btw_gx_bf', 'dist_diff_btw_dssgx_bf',  # errors
          'bf_dist', 'bf_match',  # bf matches
          'paa_dist', 'paa_match',  # paa matches
          'sax_dist', 'sax_match',  # paa matches
          'gx_dist', 'gx_match',  # gx matches
          'dssgx_dist', 'dssgx_match',  # dssgx matches
          'num_rows', 'num_cols_max', 'num_cols_median', 'data_size', 'num_query']])  # meta info about this experiment

    overall_diff_paabf_list = []
    overall_diff_saxbf_list = []
    overall_diff_gxbf_list = []
    overall_diff_dssgxbf_list = []

    q_records = {}

    # load in the test data for getting query sequences outside of the training set, which is the genexengine to query
    test_data = data.replace('TRAIN', 'TEST')
    gxe_test = from_csv(test_data, feature_num=feature_num, num_worker=mp_args['num_worker'],
                        use_spark=False, header=None)

    gxe = from_csv(data, num_worker=mp_args['num_worker'], driver_mem=mp_args['driver_mem'],
                   max_result_mem=mp_args['max_result_mem'],
                   feature_num=feature_num, use_spark=use_spark, _rows_to_consider=num_sample,
                   header=None)  # load in the training set as the main genexengine to query
    num_rows = len(gxe.data_raw)
    num_query = max(1, int(query_split * num_rows))
    loi = (max(1, int(gxe.get_max_seq_len() * (1 - loi_range))), int(gxe.get_max_seq_len()))
    print('Max seq len is ' + str(gxe.get_max_seq_len()))

    print('Generating query set')
    # generate the query sets
    query_set = list()
    # get the number of subsequences randomly pick a sequence as the query from the query sequence, make sure the
    # picked sequence is in the input list this query'id must exist in the database
    random.seed(42)
    sep = 3  # three for query len of: small, medium and large
    for i in range(sep):
        for j in range(math.ceil(num_query / 3)):
            q_range = loi[1] - loi[0]
            qrange_start = math.ceil(loi[0] + i * q_range / sep)
            qrange_end = math.floor(loi[0] + (i + 1) * q_range / sep)

            query_len = random.choice(list(range(qrange_start, qrange_end)))
            query_train = gxe.get_random_seq_of_len(query_len, seed=i * j)

            # normalize the external query on the scale of the train
            # test set may have different max seq len
            query_test = gxe_test.get_random_seq_of_len(min(query_len, gxe_test.get_max_seq_len()), seed=i * j, with_data=True,
                                                        normalize=False).get_data()
            query_test = gxe.normalize(query_test)

            query_set += [query_train, query_test]
            print('Adding to query set from TRAIN: ' + str(query_train))
            print('Adding to query set from TEST: ' + str(query_test))

    gxe_test.stop()  # the test gxe is no longer needed as we already loaded queries from it
    del gxe_test
    print('Using dist_type = ' + str(dist_type))
    print('Using loi offset of ' + str(loi_range))
    print('Building length of interest is ' + str(loi))
    print('Building Similarity Threshold is ' + str(st))

    print('Performing Regular clustering ...')
    cluster_start_time = time.time()
    gxe.build(st=st, dist_type=dist_type, loi=loi, _use_dss=False, _use_dynamic=False)
    cluster_time_gx = time.time() - cluster_start_time
    print('gx_cluster_time took ' + str(cluster_time_gx) + ' sec')

    gxe.set_piecewise_segment(n_segment=n_segment)
    paa_build_time = np.NaN
    sax_build_time = np.NaN

    print('Evaluating Query with BF, PAA, SAX')
    for i, q in enumerate(query_set):
        print('Dataset: ' + data + ' - dist_type: ' + dist_type + '- Querying #' + str(i) + ' of ' + str(
            len(query_set)))

        query_result_bf, bf_time = run_query(gxe, q, best_k=max(best_ks), algo='bf', _lb_opt=_lb_opt, _radius=_radius)
        query_result_paa, paa_time = run_query(gxe, q, best_k=max(best_ks), algo='paa', _lb_opt=_lb_opt, _radius=_radius)
        query_result_sax, sax_time = run_query(gxe, q, best_k=max(best_ks), algo='sax', _lb_opt=_lb_opt, _radius=_radius)

        q_records[str(q)] = {'bf_query_time': bf_time, 'paa_query_time': paa_time, 'sax_query_time': sax_time,
                             'gx_query_time': {}, 'dssgx_query_time': {},
                             'bf_match': query_result_bf, 'paa_match': query_result_paa, 'sax_match': query_result_sax,
                             'gx_match': {}, 'dssgx_match': {}}

    for k in best_ks:
        print('Evaluating Query with Regular BrainEx for k = ' + str(k))
        for i, q in enumerate(query_set):
            print('Dataset: ' + data + ' - dist_type: ' + dist_type + '- Querying #' + str(i) + ' of ' + str(
                len(query_set)) + '; query = ' + str(q))
            query_result_gx, gx_time = run_query(gxe, q, best_k=k, algo='bx', _lb_opt=_lb_opt, _radius=_radius)
            q_records[str(q)]['gx_query_time'][k] = gx_time
            q_records[str(q)]['gx_match'][k] = query_result_gx

    print('Performing clustering with DSS algorithm...')
    gxe.stop()
    del gxe
    gxe = from_csv(data, num_worker=mp_args['num_worker'], driver_mem=mp_args['driver_mem'],
                   max_result_mem=mp_args['max_result_mem'],
                   feature_num=feature_num, use_spark=use_spark, _rows_to_consider=num_sample,
                   header=None)
    cluster_start_time = time.time()
    gxe.build(st=st, dist_type=dist_type, loi=loi, _use_dss=True, _use_dynamic=False)
    cluster_time_dynamicGx = time.time() - cluster_start_time
    print('gx_cluster_time Dynamic took ' + str(cluster_time_dynamicGx) + ' sec')

    for k in best_ks:
        print('Evaluating DSS Bx for k = ' + str(k))
        for i, q in enumerate(query_set):
            print('Dataset: ' + data + ' - dist_type: ' + dist_type + '- Querying #' + str(i) + ' of ' + str(
                len(query_set)))
            query_result_dss, dss_time = run_query(gxe, q, best_k=k, algo='bx', _lb_opt=_lb_opt, _radius=_radius)
            q_records[str(q)]['dssgx_query_time'][k] = dss_time
            q_records[str(q)]['dssgx_match'][k] = query_result_dss

    # add the meta information and preprocessing times
    for k in best_ks:
        result_df = pd.DataFrame(columns=result_headers[0, :])
        result_df = result_df.append({'k': k,
                                      'gx_preprocess_time': cluster_time_gx,
                                      'dssgx_preprocess_time': cluster_time_dynamicGx,
                                      'paa_preprocess_time': paa_build_time,
                                      'sax_preprocess_time': sax_build_time,
                                      'num_rows': num_rows,
                                      'num_cols_max': gxe.get_max_seq_len(),
                                      'num_cols_median': np.median(gxe.get_seq_length_list()),
                                      'data_size': gxe.get_data_size(),
                                      'num_query': len(query_set)}, ignore_index=True)
        # add the accuracies and time performance
        for i, q in enumerate(query_set):
            this_record = q_records[str(q)]
            q_data = str(
                q.tostring() if type(q) == np.ndarray else np.array(gxe.get_seq_data(q, normalize=False)).tostring())
            result_df = result_df.append({'query': str(q) + ':' + q_data,
                                          'query_len': len(q),
                                          'bf_query_time': this_record['bf_query_time'],
                                          'paa_query_time': this_record['paa_query_time'],
                                          'sax_query_time': this_record['sax_query_time'],
                                          'gx_query_time': this_record['gx_query_time'][k],  # k for best k
                                          'dssgx_query_time': this_record['dssgx_query_time'][k]},  # k for best k
                                         ignore_index=True)  # append the query times

            # add the accuracies
            for bf_r, paa_r, sax_r, gx_r, dynamic_r in zip(this_record['bf_match'][:k],
                                                           this_record['paa_match'][:k],
                                                           this_record['sax_match'][:k],
                                                           this_record['gx_match'][k],
                                                           this_record['dssgx_match'][k]):  # resolve the query matches
                diff_saxbf = abs(sax_r[0] - bf_r[0])
                diff_paabf = abs(paa_r[0] - bf_r[0])
                diff_gxbf = abs(gx_r[0] - bf_r[0])
                diff_dssgxbf = abs(dynamic_r[0] - bf_r[0])

                overall_diff_saxbf_list.append(diff_saxbf)
                overall_diff_paabf_list.append(diff_paabf)
                overall_diff_gxbf_list.append(diff_gxbf)
                overall_diff_dssgxbf_list.append(diff_dssgxbf)

                result_df = result_df.append({'dist_diff_btw_paa_bf': diff_paabf,
                                              'dist_diff_btw_sax_bf': diff_saxbf,
                                              'dist_diff_btw_gx_bf': diff_gxbf,
                                              'dist_diff_btw_dssgx_bf': diff_dssgxbf,
                                              'bf_dist': bf_r[0], 'bf_match': bf_r[1],
                                              'paa_dist': paa_r[0], 'paa_match': paa_r[1],
                                              'sax_dist': sax_r[0], 'sax_match': sax_r[1],
                                              'gx_dist': gx_r[0], 'gx_match': gx_r[1],
                                              'dssgx_dist': dynamic_r[0], 'dssgx_match': dynamic_r[1]
                                              }, ignore_index=True)
            print('Current PAA error for query is ' + str(np.mean(overall_diff_paabf_list)))
            print('Current SAX error for query is ' + str(np.mean(overall_diff_saxbf_list)))
            print('Current BX error for query is ' + str(np.mean(overall_diff_gxbf_list)))
            print('Current Dynamic error for query is ' + str(np.mean(overall_diff_dssgxbf_list)))
        result_path = output + '_k=' + str(k) + '.csv'
        print('Result saved to ' + result_path)
        result_df.to_csv(result_path)

    gxe.stop()
    print('Done with ' + data)


def run_query(gxe, q, best_k, algo, _lb_opt, _radius):
    start = time.time()
    if algo == 'bf':
        q_result = gxe.query_brute_force(query=q, best_k=best_k, _use_cache=False)
    elif algo == 'bx':
        q_result = gxe.query(query=q, best_k=15, _lb_opt=_lb_opt, _radius=_radius)
    else:
        q_result = gxe.query_brute_force(query=q, best_k=15, _use_cache=False, _piecewise=algo,
                                         _use_built_piecewise=False)

    q_time = time.time() - start
    print(algo + ' query took ' + str(q_time) + ' sec')
    return q_result, q_time


def experiment_GENEX(mp_args, dataset: str, queryset: str, output: str, feature_num, dist_type, _lb_opt, _radius, use_spark: bool, st: float):
    # set up where to save the results
    result_headers = np.array(
        [['bx_preprocess_time',
          'query', 'query_len',  # the query sequence
          'bf_query_time', 'bx_query_time',
          'dist_diff_btw_bx_bf',
          'bf_dist', 'bf_match',  # bf matches
          'bx_dist', 'bx_match',  # gx matches
          'data_size'
        ]])  # meta info about this experiment

    overall_diff_bxbf_list = []
    q_records = {}

    # load in the test data for getting query sequences outside of the training set, which is the genexengine to query
    bxe = from_csv(dataset, num_worker=mp_args['num_worker'], driver_mem=mp_args['driver_mem'],
                   max_result_mem=mp_args['max_result_mem'],
                   feature_num=feature_num, use_spark=use_spark,
                   header=None)  # load in the training set as the main brainex to query
    # generate the query sets
    query_set = list()
    query_df = pd.read_csv(queryset, header=None)
    # get the number of subsequences randomly pick a sequence as the query from the query sequence, make sure the
    # picked sequence is in the input list this query'id must exist in the database
    for row in query_df.values:
        query_set.append(row[1:])

    print('Using dist_type = ' + str(dist_type))
    print('Building Similarity Threshold is ' + str(st))

    print('Performing Regular clustering ...')
    cluster_start_time = time.time()
    # bxe.build(st=st, dist_type=dist_type, loi=[bxe.get_max_seq_len()], _use_dss=False, _use_dynamic=False)
    bxe.build(st=st, dist_type=dist_type, _use_dss=False, _use_dynamic=False)
    cluster_time_bx = time.time() - cluster_start_time
    print('bx cluster took ' + str(cluster_time_bx) + ' sec')

    print('Evaluating Query with BF and Bx')
    for i, q in enumerate(query_set):
        print('Dataset: ' + dataset + ' - dist_type: ' + dist_type + '- Querying #' + str(i) + ' of ' + str(
            len(query_set)) )
        query_result_bf, bf_time = run_query(bxe, q, best_k=1, algo='bf', _lb_opt=_lb_opt, _radius=_radius)
        query_result_bx, bx_time = run_query(bxe, q, best_k=1, algo='bx', _lb_opt=_lb_opt, _radius=_radius)
        q_records[str(q)] = {'bf_query_time': bf_time, 'bx_query_time': bx_time,
                             'bf_match': query_result_bf, 'bx_match': query_result_bx}

    result_df = pd.DataFrame(columns=result_headers[0, :])
    result_df = result_df.append({'bx_preprocess_time': cluster_time_bx,
                                  'data_size': bxe.get_data_size()}, ignore_index=True)
    # add the accuracies and time performance
    overall_diff_bxbf_list = []
    for i, q in enumerate(query_set):
        this_record = q_records[str(q)]
        q_data = str(
            q.tostring() if type(q) == np.ndarray else np.array(bxe.get_seq_data(q, normalize=False)).tostring())
        result_df = result_df.append({'query': str(q) + ':' + q_data,
                                      'query_len': len(q),
                                      'bf_query_time': this_record['bf_query_time'],
                                      'bx_query_time': this_record['bx_query_time']},
                                     ignore_index=True)  # append the query times

        # add the accuracies
        for bf_r, bx_r in zip(this_record['bf_match'],
                                                       this_record['bx_match']):  # resolve the query matches
            diff_bxbf = abs(bx_r[0] - bf_r[0])
            overall_diff_bxbf_list.append(diff_bxbf)
            result_df = result_df.append({'dist_diff_btw_bx_bf': diff_bxbf,
                                          'bf_dist': bf_r[0], 'bf_match': bf_r[1],
                                          'bx_dist': bx_r[0], 'bx_match': bx_r[1],
                                          }, ignore_index=True)
        print('Current BX error for query is ' + str(np.mean(overall_diff_bxbf_list)))
    result_path = output + '.csv'
    print('Result saved to ' + result_path)
    result_df.to_csv(result_path)

    bxe.stop()
