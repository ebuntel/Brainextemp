import genex.database.genex_database as gxdb
from pyspark import SparkContext, SparkConf

# create the spark context
num_cores = 4
conf = SparkConf(). \
    setMaster("local[" + str(num_cores) + "]"). \
    setAppName("Genex").set('spark.driver.memory', '12G'). \
    set('spark.driver.maxResultSize', '12G')
sc = SparkContext(conf=conf)

# create gxdb from a csv file
# data_file = 'data/ECGFiveDays.csv'
db_path = 'results/test_db'
#
# mydb = gxdb.from_csv(data_file, sc=sc, feature_num=2)
# mydb.data_normalized = mydb.data_normalized[:10]
# mydb.build(similarity_threshold=0.1, loi=slice(110, 115))
# mydb.save(db_path)

mydb = gxdb.from_db(sc, db_path)
for k, v in mydb.cluster_meta_dict.get(112).items():
    if v > 4:
        test_seq = k
        break

    test_seq = list(mydb.cluster_meta_dict.get(112).keys())[2]

cluster = mydb.get_cluster(test_seq)

import matplotlib.pyplot as plt

fig = plt.figure()
fig.set_size_inches(15.75, 10.5)

for time_series in cluster:
    # plot the sequences represented
    time_series.fetch_and_set_data(mydb.data_normalized)
    plt.plot(time_series.data)
plt.show()

# Presenting filter functionality
fig_filter = plt.figure()
fig_filter.set_size_inches(15.75, 10.5)

cluster_filtered = [x for x in cluster if x.seq_id[0] != 'ECG_1']

for item in cluster_filtered:
    plt.plot(item.data)
plt.show()