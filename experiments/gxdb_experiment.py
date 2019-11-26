import genex.database.genex_database as gxdb
from pyspark import SparkContext, SparkConf

# create the spark context
num_cores = 12
conf = SparkConf(). \
    setMaster("local[" + str(num_cores) + "]"). \
    setAppName("Genex").set('spark.driver.memory', '16G'). \
    set('spark.driver.maxResultSize', '16G')
sc = SparkContext(conf=conf)

# create gxdb from a csv file
data_file = '/Users/Leo/PycharmProjects/Genex/experiments/data/ItalyPower.csv'
db_path = 'results/test_db'

mydb = gxdb.from_csv(data_file, sc=sc, feature_num=2, _rows_to_consider=12)
mydb.save(path=db_path)
del mydb  # test saving before building

mydb = gxdb.from_db(path=db_path, sc=sc)
mydb.build(similarity_threshold=0.01)

mydb.save(path=db_path)
del mydb  # test saving after building

mydb = gxdb.from_db(path=db_path, sc=sc)

# generate the query sets
q = mydb.get_random_seq_of_len(10, seed=1)

query_result = mydb.query(query=q, best_k=5)

# TODO memory optimization: brainstorm memory optimization, encode features (ids), length batches
