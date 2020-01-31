import time
import matplotlib.pyplot as plt

from genex.utils.gxe_utils import from_csv, from_db



# spark_location = '/Users/Leo/spark-2.4.3-bin-hadoop2.7' # Set your own
# java8_location = '/Library/Java/JavaVirtualMachines/jdk1.8.0_151.jdk/Contents/Home/jre'
# os.environ['JAVA_HOME'] = java8_locationcluster_partition
# findspark.init(spark_home=spark_location)

# create gxdb from a csv file
data_file = 'data_original/ItalyPower.csv'
db_path = 'results/archived/test_db'

mydb = from_csv(data_file, feature_num=2, num_worker=32, use_spark=False)

# Save reloading unbuilt Genex Engine
mydb.save(path=db_path)
mydb.stop()
del mydb
mydb = from_db(path=db_path, num_worker=32)

start = time.time()
mydb.build(st=0.1, loi=slice(13, 16))
print('Building took ' + str(time.time() - start) + ' sec')

# Save reloading after built
mydb.save(path=db_path)
mydb.stop()
del mydb
mydb = from_db(path=db_path, num_worker=16)

# generate the query sets
q = mydb.get_random_seq_of_len(15, seed=1)
start = time.time()
query_result = mydb.query_brute_force(query=q, best_k=5)
duration_bf = time.time() - start
query_result = mydb.query(query=q, best_k=5, _radius=1, _lb_opt=False)
duration_noOpt = time.time() - start
start = time.time()
query_result = mydb.query(query=q, best_k=5, _radius=1, _lb_opt=True)
duration_withOpt = time.time() - start
# # TODO memory optimization:  memory optimization, encode features (ids), length batches
plt.plot(q.fetch_data(mydb.data_normalized), linewidth=5, color='red')
for qr in query_result:
    plt.plot(qr[1].fetch_data(mydb.data_normalized), color='blue', label=str(qr[0]))
plt.legend()
plt.show()