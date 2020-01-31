import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

# spark_location = '/Users/Leo/spark-2.4.3-bin-hadoop2.7'  # Set your own
# java8_location = '/Library/Java/JavaVirtualMachines/jdk1.8.0_151.jdk/Contents/Home/jre'
# os.environ['JAVA_HOME'] = java8_location
# findspark.init(spark_home=spark_location)


# create the spark context
from genex.utils.gxe_utils import from_csv

########################################################################################################################
mp_args = {'num_worker': 12,
           'driver_mem': 25,
           'max_result_mem': 25}


########################################################################################################################

def experiment_genex(data, output, feature_num, num_sample, num_query, add_uuid,
                     dist_type, _lb_opt, _radius, use_spark: bool):
    # create gxdb from a csv file

    # set up where to save the results
    result_headers = np.array(
        [['cluster_time', 'query', 'gx_time', 'bf_time', 'diff', 'gx_dist', 'gx_match', 'bf_dist', 'bf_match']])
    result_df = pd.DataFrame(columns=result_headers[0, :])

    print('Performing clustering ...')
    gxe = from_csv(data, num_worker=mp_args['num_worker'], driver_mem=mp_args['driver_mem'],
                   max_result_mem=mp_args['max_result_mem'],
                   feature_num=feature_num, use_spark=use_spark, add_uuid=add_uuid, _rows_to_consider=num_sample)

    print('Generating query of max seq len ...')
    # generate the query sets
    query_set = list()
    # get the number of subsequences
    # randomly pick a sequence as the query from the query sequence, make sure the picked sequence is in the input list
    # this query'id must exist in the database
    for i in range(num_query):
        query_set.append(gxe.get_random_seq_of_len(int(gxe.get_max_seq_len() / 2), seed=i))

    cluster_start_time = time.time()
    print('Using dist_type = ' + str(dist_type))
    gxe.build(st=0.1, dist_type=dist_type)
    cluster_time = time.time() - cluster_start_time
    result_df = result_df.append({'cluster_time': cluster_time}, ignore_index=True)
    print('Clustering took ' + str(cluster_time) + ' sec')
    # randomly pick a sequence as the query from the query sequence, make sure the picked sequence is in the input list
    # this query'id must exist in the database
    overall_diff_list = []

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

        start = time.time()
        print('Genex  query took ' + str(gx_time) + ' sec')
        print('Running Brute Force Query ...')
        query_result_bf = gxe.query_brute_force(query=q, best_k=15)
        # print('...Not Actually running... Simulating results!')
        # query_result_bf = [(0.0, [1, 2, 3])] * 15
        bf_time = time.time() - start
        print('Brute force query took ' + str(bf_time) + ' sec')
        # save the results
        print('Saving results for query #' + str(i) + ' of ' + str(len(query_set)))
        result_df = result_df.append({'query': str(q), 'gx_time': gx_time, 'bf_time': bf_time}, ignore_index=True)
        diff_list = []
        for gx_r, bf_r in zip(query_result_gx, query_result_bf):
            diff = abs(gx_r[0] - bf_r[0])
            diff_list.append(diff)
            overall_diff_list.append(diff)
            result_df = result_df.append({'diff': diff,
                                          'gx_dist': gx_r[0], 'gx_match': gx_r[1],
                                          'bf_dist': bf_r[0], 'bf_match': bf_r[1]}, ignore_index=True)
        result_df = result_df.append({'diff': np.mean(diff_list)}, ignore_index=True)
        result_df.to_csv(output)

        # evict the mp context every 3 queries
        print('Done')
        # if i % 3:
        #     gxe.reset_mp(use_spark=use_spark, **mp_args)
        #     print('     Evicting Multiprocess Context')

    # save the overall difference
    result_df = result_df.append({'diff': np.mean(overall_diff_list)}, ignore_index=True)
    result_df.to_csv(output)
    # terminate the spark session
    gxe.stop()

    return gxe


def generate_exp_set(dataset_list, dist_type, notes: str):
    today = datetime.now()
    dir_name = os.path.join('results', today.strftime("%b-%d-%Y-") + str(today.hour) + '-N-' + notes)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    config_list = []
    for d in dataset_list:
        d_path = os.path.join('data', d + '.csv')
        assert os.path.exists(d_path)

        config_list.append({
            'data': d_path,
            'output': os.path.join(dir_name, d + '_' + dist_type + '.csv'),
            'feature_num': 0,
            'add_uuid': True,
            'dist_type': dist_type
        })
    return config_list


def run_exp_set(exp_set, num_sample, num_query,
                _lb_opt, radius, use_spark):
    for es in exp_set:
        experiment_genex(**es, num_sample=num_sample, num_query=num_query,
                         _lb_opt=_lb_opt, _radius=radius, use_spark=use_spark)


datasets = [
    'ItalyPower',
    'ECGFiveDays',
    'Gun_Point_TRAIN',
    'synthetic_control_TRAIN'
]
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
num_sample = 200
########################################################################################################################
ex_config_6 = {
    'num_sample': num_sample,
    'num_query': 40,
    '_lb_opt': False,
    'radius': 1,
    'use_spark': True
}
start = time.time()
notes_6 = 'UseSpark-R1-noOpt_numSample400'
es_eu_6 = generate_exp_set(datasets, 'eu', notes=notes_6)
es_ma_6 = generate_exp_set(datasets, 'ma', notes=notes_6)
es_ch_6 = generate_exp_set(datasets, 'ch', notes=notes_6)
run_exp_set(es_eu_6, **ex_config_6)
run_exp_set(es_ma_6, **ex_config_6)
run_exp_set(es_ch_6, **ex_config_6)
duration6 = time.time() - start
print('Finished at')
print(datetime.now())
print('The experiment took ' + str(duration6 / 3600) + ' hrs')

########################################################################################################################
ex_config_5 = {
    'num_sample': num_sample,
    'num_query': 40,
    '_lb_opt': True,
    'radius': 1,
    'use_spark': True
}
start = time.time()
notes_5 = 'UseSpark-R1-LBOpt_numSample400'
es_eu_5 = generate_exp_set(datasets, 'eu', notes=notes_5)
es_ma_5 = generate_exp_set(datasets, 'ma', notes=notes_5)
es_ch_5 = generate_exp_set(datasets, 'ch', notes=notes_5)
run_exp_set(es_eu_5, **ex_config_5)
run_exp_set(es_ma_5, **ex_config_5)
run_exp_set(es_ch_5, **ex_config_5)
duration5 = time.time() - start
print('Finished at')
print(datetime.now())
print('The experiment took ' + str(duration5 / 3600) + ' hrs')