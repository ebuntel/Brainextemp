import csv
import os
import time
from datetime import datetime
import findspark

import genex.database.genex_database as gxdb
from pyspark import SparkContext, SparkConf

import numpy as np
import pandas as pd
from datetime import date


# spark_location = '/Users/Leo/spark-2.4.3-bin-hadoop2.7'  # Set your own
# java8_location = '/Library/Java/JavaVirtualMachines/jdk1.8.0_151.jdk/Contents/Home/jre'
# os.environ['JAVA_HOME'] = java8_location
# findspark.init(spark_home=spark_location)


# create the spark context
def experiment_genex(data, output, feature_num, num_sample, num_query, add_uuid,
                     dist_type, _lb_opt_repr, _lb_opt_cluster, _radius):
    num_cores = 32
    conf = SparkConf(). \
        setMaster("local[" + str(num_cores) + "]"). \
        setAppName("Genex").set('spark.driver.memory', '64G'). \
        set('spark.driver.maxResultSize', '64G')
    sc = SparkContext(conf=conf)

    # create gxdb from a csv file

    # set up where to save the results
    result_headers = np.array(
        [['cluster_time', 'query', 'gx_time', 'bf_time', 'diff', 'gx_dist', 'gx_match', 'bf_dist', 'bf_match']])
    result_df = pd.DataFrame(columns=result_headers[0, :])

    print('Performing clustering ...')
    mydb = gxdb.from_csv(data, sc=sc, feature_num=feature_num, add_uuid=add_uuid, _rows_to_consider=num_sample)

    print('Generating query of max seq len ...')
    # generate the query sets
    query_set = list()
    # get the number of subsequences
    # randomly pick a sequence as the query from the query sequence, make sure the picked sequence is in the input list
    # this query'id must exist in the database
    for i in range(num_query):
        query_set.append(mydb.get_random_seq_of_len(int(mydb.get_max_seq_len() / 2), seed=i))

    cluster_start_time = time.time()
    print('Using dist_type = ' + str(dist_type))
    mydb.build(similarity_threshold=0.1, dist_type=dist_type)
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
        query_result_gx = mydb.query(query=q, best_k=15,
                                     _lb_opt_cluster=_lb_opt_cluster, _lb_opt_repr=_lb_opt_repr,
                                     _radius=_radius)
        gx_time = time.time() - start

        start = time.time()
        print('Genex  query took ' + str(gx_time) + ' sec')
        print('Running Brute Force Query ...')
        query_result_bf = mydb.query_brute_force(query=q, best_k=15)
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

        print('     Done')

    # save the overall difference
    result_df = result_df.append({'diff': np.mean(overall_diff_list)}, ignore_index=True)
    result_df.to_csv(output)
    # terminate the spark session
    sc.stop()

    return mydb


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
                _lb_opt_repr, _lb_opt_cluster, radius):
    for es in exp_set:
        experiment_genex(**es, num_sample=num_sample, num_query=num_query,
                         _lb_opt_repr=_lb_opt_repr, _lb_opt_cluster=_lb_opt_cluster, _radius=radius)


datasets = [
    'ItalyPower',
    'ECGFiveDays',
    'Gun_Point_TRAIN',
    'synthetic_control_TRAIN'
]

# ex_config_1 = {
#     'num_sample': 40,
#     'num_query': 40,
#     '_lb_opt_repr': 'none',
#     '_lb_opt_cluster': 'none',
#     'radius': 0
# }
# start = time.time()
# notes_1 = 'NoBSF'
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
ex_config_2 = {
    'num_sample': 40,
    'num_query': 40,
    '_lb_opt_repr': 'bsf',
    '_lb_opt_cluster': 'bsf',
    'radius': 0
}
start = time.time()
notes_2 = 'BSFon-R0'
es_eu_2 = generate_exp_set(datasets, 'eu', notes=notes_2)
es_ma_2 = generate_exp_set(datasets, 'ma', notes=notes_2)
es_ch_2 = generate_exp_set(datasets, 'ch', notes=notes_2)
run_exp_set(es_eu_2, **ex_config_2)
run_exp_set(es_ma_2, **ex_config_2)
run_exp_set(es_ch_2, **ex_config_2)
duration2 = time.time() - start
print('Finished at')
print(datetime.now())
print('The experiment with radius 0 took ' + str(duration2/3600) + ' hrs')

