import time
import matplotlib.pyplot as plt
from genex.utils.gxe_utils import from_csv, from_db


# create gxdb from a csv file
data = 'data/ItalyPower.csv'
db_path = 'results/test_db'

mygxe = from_csv(data, feature_num=0, num_worker=12, use_spark=True, driver_mem=10, max_result_mem=10, _rows_to_consider=24)

start = time.time()
mygxe.build(st=0.1)
print('Building took ' + str(time.time() - start) + ' sec')

q = mygxe.get_random_seq_of_len(15, seed=1)

start = time.time()
query_result = mygxe.query(query=q, best_k=5)
duration_withOpt = time.time() - start

# plot the query result
plt.plot(q.fetch_data(mygxe.data_normalized), linewidth=5, color='red')
for qr in query_result:
    plt.plot(qr[1].fetch_data(mygxe.data_normalized), label=str(qr[0]))
plt.legend()
plt.show()