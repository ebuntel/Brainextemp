import copy
import os
import time

import findspark
import matplotlib.pyplot as plt

from brainex.utils.gxe_utils import from_csv, from_db
from brainex.utils.process_utils import equal_ignore_order

# spark_location = '/Users/Leo/spark-2.4.3-bin-hadoop2.7' # Set your own
# java8_location = '/Library/Java/JavaVirtualMachines/jdk1.8.0_151.jdk/Contents/Home/jre'
# os.environ['JAVA_HOME'] = java8_location
# findspark.init(spark_home=spark_location)

# create gxdb from a csv file
data = 'data/ItalyPower.csv'
# data = 'data/ECGFiveDays.csv'
# data = 'data/101-SART-June2018-AS_(all).csv'

mygxe = from_csv(data, feature_num=0, num_worker=16, use_spark=True, driver_mem=10, max_result_mem=10, _rows_to_consider=32)
start = time.time()
mygxe.build(st=0.1, _use_dss=False, _group_only=True)
print('Grouping Reg took ' + str(time.time() - start) + ' sec')
groups_reg = copy.deepcopy(mygxe.subsequences.collect())
del mygxe

mygxe = from_csv(data, feature_num=0, num_worker=16, use_spark=True, driver_mem=10, max_result_mem=10,  _rows_to_consider=32)
start = time.time()
mygxe.build(st=0.1, _use_dss=True, _group_only=True)
print('Grouping DSS took ' + str(time.time() - start) + ' sec')
groups_dss = copy.deepcopy(mygxe.subsequences.collect())

assert set(groups_dss) == set(groups_reg)

# next, test the query
mygxe.build(st=0.1, _use_dss=True)

subsequence_num = mygxe.get_num_subsequences()
# generate the query
q = mygxe.get_random_seq_of_len(15, seed=1)

start = time.time()
qr_bf = mygxe.query_brute_force(query=q, best_k=5)
duration_bf = time.time() - start

start = time.time()
qr_gx = mygxe.query(query=q, best_k=5)
duration_gx = time.time() - start

# plot the query result
plt.plot(q.fetch_data(mygxe.data_normalized), linewidth=5, color='red')
for qr in qr_bf:
    plt.plot(qr[1].fetch_data(mygxe.data_normalized), color='blue', label=str(qr[0]))
plt.title('BF result')
plt.legend()
plt.show()


plt.plot(q.fetch_data(mygxe.data_normalized), linewidth=5, color='red')
for qr in qr_gx:
    plt.plot(qr[1].fetch_data(mygxe.data_normalized), color='blue', label=str(qr[0]))
plt.title('gx result')
plt.legend()
plt.show()