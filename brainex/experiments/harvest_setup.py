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
from brainex.experiments.harvests import experiment_BrainEX
from brainex.utils.gxe_utils import from_csv


########################################################################################################################

# eu_exclude = ['ChlorineConcentration',
#               'ElectricDevices',
#               'Haptics',
#               'InsectEPGRegularTrain',
#               'Lightning2',
#               'Meat',
#               'Trace',
#               ]
########################################################################################################################

def experiment_genex(mp_args, data, output, feature_num, num_sample, query_split,
                     dist_type, _lb_opt, _radius, use_spark: bool, loi_range: float, st: float, paa_seg: float):
    # create gxdb from a csv file

    # set up where to save the results
    result_headers = np.array(
        [['cluster_time', 'paa_build_time',
          'query',
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
    num_query = max(int(query_split * num_rows), 1)
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
    print('Using dist_type = ' + str(dist_type))
    print('Using loi offset of ' + str(loi_range))
    print('Building length of interest is ' + str(loi))
    print('Building Similarity Threshold is ' + str(st))

    cluster_start_time = time.time()
    gxe.build(st=st, dist_type=dist_type, loi=loi)
    cluster_time = time.time() - cluster_start_time
    print('Clustering took ' + str(cluster_time) + ' sec')
    # randomly pick a sequence as the query from the query sequence, make sure the picked sequence is in the input list
    # this query'id must exist in the database

    print('Preparing PAA Subsequences')
    start = time.time()
    gxe.build_piecewise(paa_seg, _dummy_slicing=True)
    paa_build_time = time.time() - start
    print('Prepare PAA subsequences took ' + str(paa_build_time))

    result_df = result_df.append({'cluster_time': cluster_time,
                                  'paa_build_time': paa_build_time
                                  }, ignore_index=True)

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
        query_result_paa = gxe.query_brute_force(query=q, best_k=15, _use_cache=False, _piecewise=True)
        paa_time = time.time() - start
        print('Pure PAA query took ' + str(paa_time) + ' sec')

        # save the results
        print('Saving results for query #' + str(i) + ' of ' + str(len(query_set)))
        result_df = result_df.append({'query': str(q), 'bf_time': bf_time, 'paa_time': paa_time, 'gx_time': gx_time},
                                     ignore_index=True)

        for bf_r, paa_r, gx_r in zip(query_result_bf, query_result_paa, query_result_gx):
            diff_gxbf = abs(gx_r[0] - bf_r[0])
            diff_paabf = abs(paa_r[0] - bf_r[0])

            overall_diff_gxbf_list.append(diff_gxbf)
            overall_diff_paabf_list.append(diff_paabf)

            result_df = result_df.append({'dist_diff_btw_paa_bf': diff_paabf,
                                          'dist_diff_btw_gx_bf': diff_gxbf,
                                          'bf_dist': bf_r[0], 'bf_match': bf_r[1],
                                          'paa_dist': paa_r[0], 'paa_match': paa_r[1],
                                          'gx_dist': gx_r[0], 'gx_match': gx_r[1],
                                          }, ignore_index=True)
        print('Current GX error is ' + str(np.mean(overall_diff_gxbf_list)))
        print('Current PAA error is ' + str(np.mean(overall_diff_paabf_list)))

        result_df.to_csv(output)
        print('Done')

    print('Result saved to ' + output)
    result_df.to_csv(output)
    # terminate the spark session
    gxe.stop()

    return gxe


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
        print('Distance type - ' + dist_type + ', adding ' + dataset_path)
        config_list.append((df.size, {  # record the size of the dataframe later start with smaller ones
            'data': dataset_path,
            'output': os.path.join(output_dir_path, d_name + '_' + dist_type),
            'feature_num': 1,  # IMPORTANT this should be 1 for the UCR archive
            'dist_type': dist_type
        }))

    config_list.sort(key=lambda x: x[0])  # sort by dataset size
    config_list = [x[1] for x in config_list]  # remove the dat size variable

    if len(config_list) < 1:
        raise Exception('No dataset satisfied the given soi')
    print('Added ' + str(len(config_list)) + ' datasets with the given soi')
    return config_list


def run_exp_set(exp_set, mp_args, num_sample, query_split, cases_split,
                _lb_opt, radius, use_spark, loi_range, st, n_segment, test_option='BrainEX'):
    options = ['regular', 'DSS', 'dynamic']
    for i, es in enumerate(exp_set):
        print('$$ Running experiment set: ' + str(i) + ' of ' + str(len(exp_set)))
        if test_option == 'DSS':
            experiment_genex_dss(mp_args, **es, num_sample=num_sample, query_split=query_split, cases_split=cases_split,
                                 _lb_opt=_lb_opt, _radius=radius, use_spark=use_spark, loi_range=loi_range, st=st,
                                 paa_seg=n_segment)
        elif test_option == 'regular':
            experiment_genex(mp_args, **es, num_sample=num_sample, query_split=query_split,
                             _lb_opt=_lb_opt, _radius=radius, use_spark=use_spark, loi_range=loi_range, st=st,
                             paa_seg=n_segment)
        elif test_option == 'dynamic':
            experiment_genex_dynamic(mp_args, **es, num_sample=num_sample, query_split=query_split,
                                     _lb_opt=_lb_opt, _radius=radius, use_spark=use_spark, loi_range=loi_range, st=st,
                                     paa_seg=n_segment)
        elif test_option == 'BrainEX':
            experiment_BrainEX(mp_args, **es, num_sample=num_sample, query_split=query_split,
                               _lb_opt=_lb_opt, _radius=radius, use_spark=use_spark, loi_range=loi_range, st=st,
                               n_segment=n_segment)
        else:
            raise Exception('Unrecognized test option, it must be one of the following: ' + str(options))


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


def experiment_genex_dss(mp_args, data, output, feature_num, num_sample, query_split, cases_split,
                         dist_type, _lb_opt, _radius, use_spark: bool, loi_range: float, st: float, paa_seg: float):
    # set up where to save the results
    result_headers = np.array(
        [['gx_cluster_time', 'dssGx_cluster_time', 'paa_build_time',
          'query',
          'bf_time', 'paa_time', 'gx_time', 'dssGx_time',
          'dist_diff_btw_paa_bf', 'dist_diff_btw_gx_bf', 'dist_diff_btw_dssGx_bf',
          'bf_dist', 'bf_match',
          'paa_dist', 'paa_match',
          'gx_dist', 'gx_match',
          'dssGx_dist', 'dssGx_match',
          'data_size', 'num_query']])

    result_df = pd.DataFrame(columns=result_headers[0, :])

    # only take one time series at a time
    data_df = pd.read_csv(data, sep='\t', header=None)
    cases = max(int(data_df.shape[0] * cases_split), 1)
    sample_indices = random.sample(range(0, data_df.shape[0] - 1), cases)

    overall_diff_dssGxbf_list = []
    overall_diff_paabf_list = []
    overall_diff_gxbf_list = []

    q_records = {}

    for ci in range(cases):
        print('Taking ' + str(ci) + ' of ' + str(cases) + ' as dataset')
        data_single_ts = data_df.iloc[sample_indices[ci]:sample_indices[ci] + 1, :]  # randomly take a row
        gxe = from_csv(data_single_ts, num_worker=mp_args['num_worker'], driver_mem=mp_args['driver_mem'],
                       max_result_mem=mp_args['max_result_mem'],
                       feature_num=feature_num, use_spark=use_spark, _rows_to_consider=num_sample,
                       header=None)
        num_query = max(1, int((query_split * gxe.get_data_size())))
        try:
            assert num_query > 0
        except AssertionError:
            raise Exception(
                'Number of query with given query_split yields zero query sequence, try increase query_split')
        loi = (int(gxe.get_max_seq_len() * (1 - loi_range)), int(gxe.get_max_seq_len() * (1 + loi_range)))
        print('Max seq len is ' + str(gxe.get_max_seq_len()))

        result_df = result_df.append({'data_size': gxe.get_data_size(), 'num_query': num_query}, ignore_index=True)

        print('Generating query of max seq len ...')
        # generate the query sets
        query_set = list()
        # get the number of subsequences randomly pick a sequence as the query from the query sequence, make sure the
        # picked sequence is in the input list this query'id must exist in the database
        for i in range(num_query):
            random.seed(i)
            query_len = random.choice(list(range(loi[0], loi[1])))
            this_query = gxe.get_random_seq_of_len(query_len, seed=i)
            query_set.append(this_query)
            print('Adding to query set: ' + str(this_query))

        print('Using dist_type = ' + str(dist_type))
        print('Using loi offset of ' + str(loi_range))
        print('Building length of interest is ' + str(loi))
        print('Building Similarity Threshold is ' + str(st))

        print('Performing Regular clustering ...')
        cluster_start_time = time.time()
        gxe.build(st=st, dist_type=dist_type, loi=loi, _use_dss=False)
        cluster_time_gx = time.time() - cluster_start_time
        print('gx_cluster_time took ' + str(cluster_time_gx) + ' sec')

        print('Preparing PAA Subsequences')
        start = time.time()
        gxe.build_piecewise(paa_seg, _dummy_slicing=True)
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
            query_result_paa = gxe.query_brute_force(query=q, best_k=15, _use_cache=False, _piecewise=True)
            paa_time = time.time() - start
            print('Pure PAA query took ' + str(paa_time) + ' sec')

            print('Evaluating Regular Gx')
            start = time.time()
            print('Running Genex Query ...')
            query_result_gx = gxe.query(query=q, best_k=15, _lb_opt=_lb_opt, _radius=_radius)
            gx_time = time.time() - start
            print('Genex  query took ' + str(gx_time) + ' sec')

            q_records[str(q)] = {'bf_time': bf_time, 'paa_time': paa_time, 'gx_time': gx_time, 'dssGx_time': None,
                                 'bf_result': query_result_bf, 'paa_result': query_result_paa,
                                 'gx_result': query_result_gx, 'dssGx_result': None}

        print('Performing clustering with DSS...')
        gxe.stop()
        del gxe
        gxe = from_csv(data_single_ts, num_worker=mp_args['num_worker'], driver_mem=mp_args['driver_mem'],
                       max_result_mem=mp_args['max_result_mem'],
                       feature_num=feature_num, use_spark=use_spark, _rows_to_consider=num_sample,
                       header=None)
        cluster_start_time = time.time()
        gxe.build(st=st, dist_type=dist_type, loi=loi, _use_dss=True)
        cluster_time_dssGx = time.time() - cluster_start_time
        print('gx_cluster_time took ' + str(cluster_time_dssGx) + ' sec')

        print('Evaluating dssGx')
        for i, q in enumerate(query_set):
            print('Dataset: ' + data + ' - dist_type: ' + dist_type + '- Querying #' + str(i) + ' of ' + str(
                len(query_set)) + '; query = ' + str(q))
            start = time.time()
            qr_dss = gxe.query(query=q, best_k=15, _lb_opt=_lb_opt, _radius=_radius)
            dss_time = time.time() - start
            print('DSS query took ' + str(dss_time) + ' sec')
            q_records[str(q)]['dssGx_time'] = dss_time,
            q_records[str(q)]['dssGx_time'] = q_records[str(q)]['dssGx_time'][0]  # TODO fix tupling issue,
            q_records[str(q)]['dssGx_result'] = qr_dss

        # culminate the result in the result data frame
        result_df = result_df.append({'gx_cluster_time': cluster_time_gx,
                                      'paa_build_time': paa_build_time,
                                      'dssGx_cluster_time': cluster_time_dssGx},
                                     ignore_index=True)
        for i, q in enumerate(query_set):
            this_record = q_records[str(q)]
            result_df = result_df.append({'query': str(q),
                                          'bf_time': this_record['bf_time'],
                                          'paa_time': this_record['paa_time'],
                                          'gx_time': this_record['gx_time'],
                                          'dssGx_time': this_record['dssGx_time']},
                                         ignore_index=True)  # append the query times

            for bf_r, paa_r, gx_r, dss_r in zip(this_record['bf_result'], this_record['paa_result'],
                                                this_record['gx_result'],
                                                this_record['dssGx_result']):  # resolve the query matches
                diff_paabf = abs(paa_r[0] - bf_r[0])
                diff_gxbf = abs(gx_r[0] - bf_r[0])
                diff_dssGxbf = abs(dss_r[0] - bf_r[0])

                overall_diff_paabf_list.append(diff_paabf)
                overall_diff_gxbf_list.append(diff_gxbf)
                overall_diff_dssGxbf_list.append(diff_dssGxbf)

                result_df = result_df.append({'dist_diff_btw_paa_bf': diff_paabf,
                                              'dist_diff_btw_gx_bf': diff_gxbf,
                                              'dist_diff_btw_dssGx_bf': diff_dssGxbf,
                                              'bf_dist': bf_r[0], 'bf_match': bf_r[1],
                                              'paa_dist': paa_r[0], 'paa_match': paa_r[1],
                                              'gx_dist': gx_r[0], 'gx_match': gx_r[1],
                                              'dssGx_dist': dss_r[0], 'dssGx_match': dss_r[1]
                                              }, ignore_index=True)
            print('Current PAA error for query is ' + str(np.mean(overall_diff_paabf_list)))
            print('Current GX error for query is ' + str(np.mean(overall_diff_gxbf_list)))
            print('Current DSS error for query is ' + str(np.mean(overall_diff_dssGxbf_list)))

        print('Result saved to ' + output)
        result_df.to_csv(output)
        gxe.stop()
    print('Done')


def experiment_genex_dynamic(mp_args, data, output, feature_num, num_sample, query_split,
                             dist_type, _lb_opt, _radius, use_spark: bool, loi_range: float, st: float, paa_seg: float):
    # set up where to save the results
    result_headers = np.array(
        [['gx_cluster_time', 'DynamicGx_cluster_time', 'paa_build_time',
          'query',
          'bf_time', 'paa_time', 'gx_time', 'DynamicGx_time',
          'dist_diff_btw_paa_bf', 'dist_diff_btw_gx_bf', 'dist_diff_btw_DynamicGx_bf',
          'bf_dist', 'bf_match',
          'paa_dist', 'paa_match',
          'gx_dist', 'gx_match',
          'DynamicGx_dist', 'DynamicGx_match',
          'num_rows', 'num_cols_max', 'num_cols_median', 'data_size', 'num_query']])

    result_df = pd.DataFrame(columns=result_headers[0, :])

    overall_diff_dynamicGxbf_list = []
    overall_diff_paabf_list = []
    overall_diff_gxbf_list = []

    q_records = {}

    gxe = from_csv(data, num_worker=mp_args['num_worker'], driver_mem=mp_args['driver_mem'],
                   max_result_mem=mp_args['max_result_mem'],
                   feature_num=feature_num, use_spark=use_spark, _rows_to_consider=num_sample,
                   header=None)
    num_rows = len(gxe.data_raw)
    num_query = max(1, int(query_split * num_rows))

    loi = (int(gxe.get_max_seq_len() * (1 - loi_range)), int(gxe.get_max_seq_len() * (1 + loi_range)))
    print('Max seq len is ' + str(gxe.get_max_seq_len()))

    result_df = result_df.append({'num_rows': num_rows}, ignore_index=True)  # use append to create the first row
    result_df['num_cols_max'] = gxe.get_max_seq_len()
    result_df['num_cols_median'] = np.median(gxe.get_seq_length_list())
    result_df['data_size'] = gxe.get_data_size()
    result_df['num_query'] = num_query

    print('Generating query of max seq len ...')
    # generate the query sets
    query_set = list()
    # get the number of subsequences randomly pick a sequence as the query from the query sequence, make sure the
    # picked sequence is in the input list this query'id must exist in the database
    for i in range(num_query):
        random.seed(i)
        query_len = random.choice(list(range(loi[0], loi[1])))
        this_query = gxe.get_random_seq_of_len(query_len, seed=i)
        query_set.append(this_query)
        print('Adding to query set: ' + str(this_query))

    print('Using dist_type = ' + str(dist_type))
    print('Using loi offset of ' + str(loi_range))
    print('Building length of interest is ' + str(loi))
    print('Building Similarity Threshold is ' + str(st))

    print('Performing Regular clustering ...')
    cluster_start_time = time.time()
    gxe.build(st=st, dist_type=dist_type, loi=loi, _use_dynamic=False)
    cluster_time_gx = time.time() - cluster_start_time
    print('gx_cluster_time took ' + str(cluster_time_gx) + ' sec')

    print('Preparing PAA Subsequences')
    start = time.time()
    gxe.build_piecewise(paa_seg, _dummy_slicing=True)
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
        query_result_paa = gxe.query_brute_force(query=q, best_k=15, _use_cache=False, _piecewise=True)
        paa_time = time.time() - start
        print('Pure PAA query took ' + str(paa_time) + ' sec')

        print('Evaluating Regular Gx')
        start = time.time()
        print('Running Genex Query ...')
        query_result_gx = gxe.query(query=q, best_k=15, _lb_opt=_lb_opt, _radius=_radius)
        gx_time = time.time() - start
        print('Genex  query took ' + str(gx_time) + ' sec')

        q_records[str(q)] = {'bf_time': bf_time, 'paa_time': paa_time, 'gx_time': gx_time, 'DynamicGx_time': None,
                             'bf_result': query_result_bf, 'paa_result': query_result_paa,
                             'gx_result': query_result_gx, 'DynamicGx_result': None}

    print('Performing clustering with Dynamic clustering...')
    gxe.stop()
    gxe = from_csv(data, num_worker=mp_args['num_worker'], driver_mem=mp_args['driver_mem'],
                   max_result_mem=mp_args['max_result_mem'],
                   feature_num=feature_num, use_spark=use_spark, _rows_to_consider=num_sample,
                   header=None)
    cluster_start_time = time.time()
    gxe.build(st=st, dist_type=dist_type, loi=loi, _use_dynamic=True)
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
        q_records[str(q)]['DynamicGx_time'] = dynamic_time,
        q_records[str(q)]['DynamicGx_time'] = q_records[str(q)]['DynamicGx_time'][0]  # TODO fix tupling issue,
        q_records[str(q)]['DynamicGx_result'] = qr_dynamic

    # culminate the result in the result data frame
    result_df = result_df.append({'gx_cluster_time': cluster_time_gx,
                                  'DynamicGx_cluster_time': cluster_time_dynamicGx,
                                  'paa_build_time': paa_build_time},
                                 ignore_index=True)
    for i, q in enumerate(query_set):
        this_record = q_records[str(q)]
        result_df = result_df.append({'query': str(q),
                                      'bf_time': this_record['bf_time'],
                                      'paa_time': this_record['paa_time'],
                                      'gx_time': this_record['gx_time'],
                                      'DynamicGx_time': this_record['DynamicGx_time']},
                                     ignore_index=True)  # append the query times

        for bf_r, paa_r, gx_r, dynamic_r in zip(this_record['bf_result'], this_record['paa_result'],
                                                this_record['gx_result'],
                                                this_record['DynamicGx_result']):  # resolve the query matches
            diff_paabf = abs(paa_r[0] - bf_r[0])
            diff_gxbf = abs(gx_r[0] - bf_r[0])
            diff_dynamicGxbf = abs(dynamic_r[0] - bf_r[0])

            overall_diff_paabf_list.append(diff_paabf)
            overall_diff_gxbf_list.append(diff_gxbf)
            overall_diff_dynamicGxbf_list.append(diff_dynamicGxbf)

            result_df = result_df.append({'dist_diff_btw_paa_bf': diff_paabf,
                                          'dist_diff_btw_gx_bf': diff_gxbf,
                                          'dist_diff_btw_DynamicGx_bf': diff_dynamicGxbf,
                                          'bf_dist': bf_r[0], 'bf_match': bf_r[1],
                                          'paa_dist': paa_r[0], 'paa_match': paa_r[1],
                                          'gx_dist': gx_r[0], 'gx_match': gx_r[1],
                                          'DynamicGx_dist': dynamic_r[0], 'DynamicGx_match': dynamic_r[1]
                                          }, ignore_index=True)
        print('Current PAA error for query is ' + str(np.mean(overall_diff_paabf_list)))
        print('Current GX error for query is ' + str(np.mean(overall_diff_gxbf_list)))
        print('Current Dynamic error for query is ' + str(np.mean(overall_diff_dynamicGxbf_list)))

    print('Result saved to ' + output)
    result_df.to_csv(output)
    gxe.stop()
    print('Done')
