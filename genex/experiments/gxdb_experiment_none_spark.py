import time
import matplotlib.pyplot as plt

from genex.utils.gxe_utils import from_csv, from_db



# spark_location = '/Users/Leo/spark-2.4.3-bin-hadoop2.7' # Set your own
# java8_location = '/Library/Java/JavaVirtualMachines/jdk1.8.0_151.jdk/Contents/Home/jre'
# os.environ['JAVA_HOME'] = java8_locationcluster_partition
# findspark.init(spark_home=spark_location)

# create gxdb from a csv file
# data_file = 'data_original/ECGFiveDays_altered.csv'
data_file = 'data_original/ItalyPower.csv'
db_path = 'results/test_db'

mygxe = from_csv(data_file, feature_num=1, num_worker=12, use_spark=False, _rows_to_consider=24)

# Save reloading unbuilt Genex Engine
mygxe.save(path=db_path)
mygxe.stop()
del mygxe
mygxe = from_db(path=db_path, num_worker=12)

start = time.time()
mygxe.build(st=0.1, loi=(16, mygxe.get_max_seq_len()))
print('Building took ' + str(time.time() - start) + ' sec')

# Save reloading after built
mygxe.save(path=db_path)
mygxe.stop()
del mygxe
mygxe = from_db(path=db_path, num_worker=12)

subsequence_num = mygxe.get_num_subsequences()
# generate the query sets
q = mygxe.get_random_seq_of_len(15, seed=1)
start = time.time()
query_result = mygxe.query_brute_force(query=q, best_k=5)
duration_bf = time.time() - start
start = time.time()
query_result = mygxe.query(query=q, best_k=5)
duration_genex = time.time() - start

# # TODO memory optimization:  memory optimization, encode features (ids), length batches
plt.plot(q.fetch_data(mygxe.data_normalized), linewidth=5, color='red')
for qr in query_result:
    plt.plot(qr[1].fetch_data(mygxe.data_normalized), color='blue', label=str(qr[0]))
plt.legend()
plt.show()
