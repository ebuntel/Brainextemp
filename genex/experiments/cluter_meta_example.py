import genex.database.genexengine as gxdb
from pyspark import SparkContext, SparkConf

# create the spark context
num_cores = 12
conf = SparkConf(). \
    setMaster("local[" + str(num_cores) + "]"). \
    setAppName("Genex").set('spark.driver.memory', '32G'). \
    set('spark.driver.maxResultSize', '32G')
sc = SparkContext(conf=conf)

# create gxdb from a csv file
data_file = '/home/apocalyvec/PycharmProjects/Genex/experiments/data_original/ItalyPower.csv'
#
mydb = gxdb.from_csv(data_file, sc=sc, feature_num=2)
mydb.build(st=0.5)

for representative, cluster_size in mydb.cluster_meta_dict.get(20).items():
    test_repr = representative
cluster = mydb.get_cluster(test_repr)

import matplotlib.pyplot as plt

fig = plt.figure()
fig.set_size_inches(15.75, 10.5)

for seq in cluster:
    # plot the sequences represented
    seq.fetch_and_set_data(mydb.data_normalized)
    plt.plot(seq.data, label=str(seq))
plt.plot(test_repr.fetch_data(mydb.data_normalized), linewidth=5)
plt.legend()
plt.show()

# Presenting filter functionality
fig_filter = plt.figure()
fig_filter.set_size_inches(15.75, 10.5)

cluster_filtered = [x for x in cluster if x.seq_id[0] != 'Italy_power999']

for item in cluster_filtered:
    plt.plot(item.data, label=str(seq))
plt.plot(test_repr.fetch_data(mydb.data_normalized), linewidth=5)
plt.legend()
plt.show()