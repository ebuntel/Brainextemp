
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
data_file = 'data/ECGFiveDays.csv'
db_path = 'results/test_db'

# The given data file does not include header line for columns
mydb_no_header = gxdb.from_csv(data_file, sc=sc, feature_num=2, is_header=False)
# The first line inside the data file including names of columns
mydb = gxdb.from_csv(data_file, sc=sc, feature_num=2)
mydb.save(path=db_path)
del mydb  # test saving before building

mydb = gxdb.from_db(path=db_path, sc=sc)
mydb.data_normalized = mydb.data_normalized[:10]
mydb.build(similarity_threshold=0.01, loi=slice(110, 115))

mydb.save(path=db_path)
del mydb  # test saving after building

mydb = gxdb.from_db(path=db_path, sc=sc)

# generate the query sets
q = mydb.get_random_seq_of_len(120, seed=1)

query_result = mydb.query(query=q, best_k=5)

# TODO memory optimization: brainstorm memory optimization, encode features (ids), length batches
