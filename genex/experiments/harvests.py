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


def experiment_BrainEX(mp_args, data, output, feature_num, num_sample, query_split,
                       dist_type, _lb_opt, _radius, use_spark: bool, loi_range: float, st: float, paa_c: float):
    # set up where to save the results
    result_headers = np.array(
        [['paa_preprocess_time', 'gx_preprocess_time', 'dssgx_preprocess_time',  # preprocessing times
          'query', 'query_len',  # the query sequence
          'bf_query_time', 'paa_query_time', 'gx_query_time', 'dssgx_query_time',  # query times
          'dist_diff_btw_paa_bf', 'dist_diff_btw_gx_bf', 'dist_diff_btw_dssgx_bf',  # errors
          'bf_dist', 'bf_match',  # bf matches
          'paa_dist', 'paa_match',  # paa matches
          'gx_dist', 'gx_match',  # gx matches
          'dssgx_dist', 'dssgx_match',  # dssgx matches
          'num_rows', 'num_cols_max', 'num_cols_median', 'data_size', 'num_query']])  # meta info about this experiment

    result_df = pd.DataFrame(columns=result_headers[0, :])

    overall_diff_dssgxbf_list = []
    overall_diff_paabf_list = []
    overall_diff_gxbf_list = []

    q_records = {}

    gxe = from_csv(data, num_worker=mp_args['num_worker'], driver_mem=mp_args['driver_mem'],
                   max_result_mem=mp_args['max_result_mem'],
                   feature_num=feature_num, use_spark=use_spark, _rows_to_consider=num_sample,
                   header=None)
    num_rows = len(gxe.data_raw)
    num_query = max(1, int(query_split * num_rows))
    loi = (int(gxe.get_max_seq_len() * (1 - loi_range)), int(gxe.get_max_seq_len()))
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
            this_query = gxe.get_random_seq_of_len(query_len, seed=i * j)
            query_set.append(this_query)
            print('Adding to query set: ' + str(this_query))

    print('Using dist_type = ' + str(dist_type))
    print('Using loi offset of ' + str(loi_range))
    print('Building length of interest is ' + str(loi))
    print('Building Similarity Threshold is ' + str(st))

    print('Performing Regular clustering ...')
    cluster_start_time = time.time()
    gxe.build(st=st, dist_type=dist_type, loi=loi, _use_dss=False, _use_dynamic=False)
    cluster_time_gx = time.time() - cluster_start_time
    print('gx_cluster_time took ' + str(cluster_time_gx) + ' sec')

    print('Preparing PAA Subsequences')
    start = time.time()
    gxe.build_paa(paa_c, _dummy_slicing=True)
    paa_build_time = time.time() - start
    print('Prepare PAA subsequences took ' + str(paa_build_time))

    print('Evaluating Query with Regular Genex, BF and PAA')
    for i, q in enumerate(query_set):
        print('Dataset: ' + data + ' - dist_type: ' + dist_type + '- Querying #' + str(i) + ' of ' + str(
            len(query_set)) + '; query = ' + str(q))
        start = time.time()
        print('Running Brute Force Query ...')
        query_result_bf = gxe.query_brute_force(query=q, best_k=15, _use_cache=False)
        bf_time = time.time() - start
        print('Brute force query took ' + str(bf_time) + ' sec')

        start = time.time()
        print('Running Pure PAA Query ...')
        query_result_paa = gxe.query_brute_force(query=q, best_k=15, _use_cache=False, _paa=True)
        paa_time = time.time() - start
        print('Pure PAA query took ' + str(paa_time) + ' sec')

        print('Evaluating Regular Gx')
        start = time.time()
        print('Running Genex Query ...')
        query_result_gx = gxe.query(query=q, best_k=15, _lb_opt=_lb_opt, _radius=_radius)
        gx_time = time.time() - start
        print('Genex  query took ' + str(gx_time) + ' sec')

        q_records[str(q)] = {'bf_query_time': bf_time, 'paa_query_time': paa_time,
                             'gx_query_time': gx_time, 'dssgx_query_time': None,
                             'bf_match': query_result_bf, 'paa_match': query_result_paa,
                             'gx_match': query_result_gx, 'dssgx_match': None}

    print('Performing clustering with DSS algorithm...')
    del gxe
    gxe = from_csv(data, num_worker=mp_args['num_worker'], driver_mem=mp_args['driver_mem'],
                   max_result_mem=mp_args['max_result_mem'],
                   feature_num=feature_num, use_spark=use_spark, _rows_to_consider=num_sample,
                   header=None)
    cluster_start_time = time.time()
    gxe.build(st=st, dist_type=dist_type, loi=loi, _use_dss=True, _use_dynamic=False)
    cluster_time_dynamicGx = time.time() - cluster_start_time
    print('gx_cluster_time Dynamic took ' + str(cluster_time_dynamicGx) + ' sec')

    print('Evaluating DynamicGx')
    for i, q in enumerate(query_set):
        print('Dataset: ' + data + ' - dist_type: ' + dist_type + '- Querying #' + str(i) + ' of ' + str(
            len(query_set)) + '; query = ' + str(q))
        start = time.time()
        qr_dynamic = gxe.query(query=q, best_k=15, _lb_opt=_lb_opt, _radius=_radius)
        dynamic_time = time.time() - start
        print('Dynamic query took ' + str(dynamic_time) + ' sec')
        q_records[str(q)]['dssgx_query_time'] = dynamic_time,
        q_records[str(q)]['dssgx_query_time'] = q_records[str(q)]['dssgx_query_time'][0]  # TODO fix tupling issue,
        q_records[str(q)]['dssgx_match'] = qr_dynamic

    # culminate the result in the result data frame
    result_df = result_df.append({'gx_preprocess_time': cluster_time_gx,
                                  'dssgx_preprocess_time': cluster_time_dynamicGx,
                                  'paa_preprocess_time': paa_build_time,
                                  'num_rows': num_rows,
                                  'num_cols_max': gxe.get_max_seq_len(),
                                  'num_cols_median': np.median(gxe.get_seq_length_list()),
                                  'data_size': gxe.get_data_size(),
                                  'num_query': len(query_set)}, ignore_index=True)
    for i, q in enumerate(query_set):
        this_record = q_records[str(q)]
        result_df = result_df.append({'query': str(q), 'query_len': len(q),
                                      'bf_query_time': this_record['bf_query_time'],
                                      'paa_query_time': this_record['paa_query_time'],
                                      'gx_query_time': this_record['gx_query_time'],
                                      'dssgx_query_time': this_record['dssgx_query_time']},
                                     ignore_index=True)  # append the query times

        for bf_r, paa_r, gx_r, dynamic_r in zip(this_record['bf_match'], this_record['paa_match'],
                                                this_record['gx_match'],
                                                this_record['dssgx_match']):  # resolve the query matches
            diff_paabf = abs(paa_r[0] - bf_r[0])
            diff_gxbf = abs(gx_r[0] - bf_r[0])
            diff_dssgxbf = abs(dynamic_r[0] - bf_r[0])

            overall_diff_paabf_list.append(diff_paabf)
            overall_diff_gxbf_list.append(diff_gxbf)
            overall_diff_dssgxbf_list.append(diff_dssgxbf)

            result_df = result_df.append({'dist_diff_btw_paa_bf': diff_paabf,
                                          'dist_diff_btw_gx_bf': diff_gxbf,
                                          'dist_diff_btw_dssgx_bf': diff_dssgxbf,
                                          'bf_dist': bf_r[0], 'bf_match': bf_r[1],
                                          'paa_dist': paa_r[0], 'paa_match': paa_r[1],
                                          'gx_dist': gx_r[0], 'gx_match': gx_r[1],
                                          'dssgx_dist': dynamic_r[0], 'dssgx_match': dynamic_r[1]
                                          }, ignore_index=True)
        print('Current PAA error for query is ' + str(np.mean(overall_diff_paabf_list)))
        print('Current GX error for query is ' + str(np.mean(overall_diff_gxbf_list)))
        print('Current Dynamic error for query is ' + str(np.mean(overall_diff_dssgxbf_list)))

    print('Result saved to ' + output)
    result_df.to_csv(output)
    gxe.stop()
    print('Done')
