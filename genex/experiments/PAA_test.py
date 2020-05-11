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
mygxe.build(st=0.1)
print('Building took ' + str(time.time() - start) + ' sec')

start = time.time()
mygxe.build_piecewise(0.5, _dummy_slicing=True)
print('Build PAA took ' + str(time.time() - start) + ' sec')

q = mygxe.get_random_seq_of_len(20, seed=1)

start = time.time()
qr_PAA = mygxe.query_brute_force(query=q, best_k=5, _piecewise=True)
durationPAA = time.time() - start


start = time.time()
qr_bf = mygxe.query_brute_force(query=q, best_k=5, _use_cache=False)
duration_bf = time.time() - start

start = time.time()
qr_gx = mygxe.query(query=q, best_k=5)
duration_gx = time.time() - start

# plot the query result
# plt.plot(q.fetch_data(mygxe.data_normalized), linewidth=5, color='red')
# for qr in qr_PAA:
#     plt.plot(qr[1].fetch_data(mygxe.data_normalized), color='blue', label=str(qr[0]))
# plt.legend()
# plt.title('PAA results')
# plt.show()
#
# plt.plot(q.fetch_data(mygxe.data_normalized), linewidth=5, color='red')
# for qr in qr_bf:
#     plt.plot(qr[1].fetch_data(mygxe.data_normalized), color='blue', label=str(qr[0]))
# plt.legend()
# plt.title('BF results')
# plt.show()
#
# plt.plot(q.fetch_data(mygxe.data_normalized), linewidth=5, color='red')
# for qr in qr_gx:
#     plt.plot(qr[1].fetch_data(mygxe.data_normalized), color='blue', label=str(qr[0]))
# plt.legend()
# plt.title('GX results')
# plt.show()