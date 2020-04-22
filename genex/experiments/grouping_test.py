import copy

from genex.utils.gxe_utils import from_csv, from_db
import numpy as np
import time
from matplotlib import pyplot as plt

# vals = np.linspace(6, 8, 15)
#
# result = []  # two's power, data size, withoutDSG, withDSG
# for i in vals:
#     ds = int(2 ** i)
#     print('Testing data size: ' + str(ds))
#     data = np.reshape(np.random.randn(ds), (1, ds))
#
#     gxe = from_csv(data, feature_num=0, header=None, num_worker=32, use_spark=True, driver_mem=26, max_result_mem=26)
#
#     start = time.time()
#     gxe.build(st=0.1, _group_only=False, _dsg=False)
#     withoutDSG = time.time() - start
#     print('Grouping took without DSG ' + str(withoutDSG) + ' sec')
#     withoutDSGss = copy.deepcopy(gxe.get_subsequences())
#
#     start = time.time()
#     gxe.build(st=0.1, _group_only=False, _dsg=True)
#     withDSG = time.time() - start
#     print('Grouping took with DSG ' + str(withDSG) + ' sec')
#     withDSGss = copy.deepcopy(gxe.get_subsequences())
#
#     result.append([i, ds, withoutDSG, withDSG])
#
#     # check the elements are the same
#     assert set(withoutDSGss) == set(withDSGss)
#     gxe.stop()

# result = np.array(result)
# plt.plot(result[:, 0], result[:, 2], label='Grouping-only time without DSG')
# plt.plot(result[:, 0], result[:, 3], label='Grouping-only time with DSG')
# plt.xlabel('Sequence length (of 2’s magnitude)')
# plt.ylabel('time took to finish (sec)')
# plt.legend()
# plt.show()


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