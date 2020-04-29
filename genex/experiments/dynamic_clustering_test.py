import os
import time

import findspark
import matplotlib.pyplot as plt

from genex.utils.gxe_utils import from_csv, from_db

spark_location = '/Users/Leo/spark-2.4.3-bin-hadoop2.7' # Set your own
java8_location = '/Library/Java/JavaVirtualMachines/jdk1.8.0_151.jdk/Contents/Home/jre'
os.environ['JAVA_HOME'] = java8_location
findspark.init(spark_home=spark_location)

# create gxdb from a csv file
data = 'data/ItalyPower.csv'

mygxe = from_csv(data, feature_num=0, num_worker=12, use_spark=True, driver_mem=10, max_result_mem=10, _rows_to_consider=24)
# mygxe = from_csv(data, feature_num=0, num_worker=3, use_spark=False, _rows_to_consider=6)

start = time.time()
mygxe.build(st=0.2, loi=(5, 6), _use_dynamic=True)
print('Building took ' + str(time.time() - start) + ' sec')

# Save reloading built Genex Engine

subsequence_num = mygxe.get_num_subsequences()
# generate the query sets
q = mygxe.get_random_seq_of_len(15, seed=1)

start = time.time()
query_result_bf = mygxe.query_brute_force(query=q, best_k=5)
duration_bf = time.time() - start

start = time.time()
query_result_0 = mygxe.query(query=q, best_k=5)
duration_withOpt = time.time() - start

start = time.time()
query_result_1 = mygxe.query(query=q, best_k=5, _radius=1, _lb_opt=False)
duration_noOpt = time.time() - start
query_result = mygxe.query(query=q, best_k=5, _lb_opt=True)

# plot the query result
plt.plot(q.fetch_data(mygxe.data_normalized), linewidth=5, color='red')
for qr in query_result_0:
    plt.plot(qr[1].fetch_data(mygxe.data_normalized), color='blue', label=str(qr[0]))
plt.legend()
plt.show()
#
#
# predicted_l0 = mygxe.predice_label_knn([1, 2, 3], 10, 0)
# predicted_l1 = mygxe.predice_label_knn(q, 10, 0, verbose=1)
#
