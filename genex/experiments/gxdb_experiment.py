import time
import matplotlib.pyplot as plt

from genex.utils.gxe_utils import from_csv, from_db

# spark_location = '/Users/Leo/spark-2.4.3-bin-hadoop2.7' # Set your own
# java8_location = '/Library/Java/JavaVirtualMachines/jdk1.8.0_151.jdk/Contents/Home/jre'
# os.environ['JAVA_HOME'] = java8_location
# findspark.init(spark_home=spark_location)

# create gxdb from a csv file
data_file = 'data_original/ItalyPower.csv'
db_path = 'results/test_db'

mydb = from_csv(data_file, feature_num=1, num_worker=12, use_spark=True, driver_mem=12, max_result_mem=12)

# Save reloading unbuilt Genex database
mydb.save(path=db_path)
mydb.stop()
del mydb
mydb = from_db(path=db_path, num_worker=12)

start = time.time()
mydb.build(st=0.1)
print('Building took ' + str(time.time() - start) + ' sec')

# Save reloading built Genex Engine
mydb.save(path=db_path)
mydb.stop()
del mydb
mydb = from_db(path=db_path, num_worker=12)

# generate the query sets
q = mydb.get_random_seq_of_len(15, seed=1)

start = time.time()
query_result_bf = mydb.query_brute_force(query=q, best_k=5)
duration_bf = time.time() - start

start = time.time()
query_result_0 = mydb.query(query=q, best_k=5)
duration_withOpt = time.time() - start

start = time.time()
query_result_1 = mydb.query(query=q, best_k=5, _radius=1, _lb_opt=False)
duration_noOpt = time.time() - start

# TODO memory optimization:  memory optimization, encode features (ids), length batches
# plot the query result
plt.plot(q.fetch_data(mydb.data_normalized), linewidth=5, color='red')
for qr in query_result_0:
    plt.plot(qr[1].fetch_data(mydb.data_normalized), color='blue', label=str(qr[0]))
plt.legend()
plt.show()


predicted_l0 = mydb.predice_label_knn([1, 2, 3], 10, 0)
predicted_l1 = mydb.predice_label_knn(q, 10, 0, verbose=1)

