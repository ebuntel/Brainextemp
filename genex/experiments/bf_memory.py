import time

from genex.utils.gxe_utils import from_csv

dataset_path = 'genex/experiments/data/ECGFiveDays.csv'
mydb = from_csv(dataset_path, feature_num=0, num_worker=16, use_spark=True, driver_mem=64, max_result_mem=64, _rows_to_consider=400)
mydb.build(st=0.1)

for i in range(20):
    print('Running ' + str(i) + 'th query...')
    query = mydb.get_random_seq_of_len(sequence_len=68, seed=i)
    start = time.time()
    query_result = mydb.query_brute_force(query=query, best_k=5)
    duration = time.time() - start
    print('Done, it took ' + str(duration) + ' sec')
    print('result is ' + str(query_result))
