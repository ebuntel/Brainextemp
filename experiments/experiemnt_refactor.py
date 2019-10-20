import genex.database.genex_database as gxdb
from pyspark import SparkContext, SparkConf

# create the spark context
num_cores = 12
conf = SparkConf(). \
    setMaster("local[" + str(num_cores) + "]"). \
    setAppName("Genex").set('spark.driver.memory', '31G'). \
    set('spark.driver.maxResultSize', '31G')
sc = SparkContext(conf=conf)

# create gxdb from a csv file
data_file = '/Users/Leo/PycharmProjects/Genex/SART2018_HbO_40.csv'
db_path = '/Users/Leo/PycharmProjects/Genex/gxdb/test_db'
mydb = gxdb.from_csv(data_file, sc=sc, feature_num=5)
mydb.save(path=db_path)
del mydb  # test saving before building

mydb = gxdb.from_db(path=db_path, sc=sc)
mydb.build(similarity_threshold=0.1, loi=slice(200, 205))
mydb.save(path=db_path)
del mydb  # test saving after building

mydb = gxdb.from_db(path=db_path, sc=sc)

# test query
from genex.parse import generate_query

# generate the query sets
query_set = generate_query(file_name='/Users/Leo/PycharmProjects/Genex/queries_test.csv', feature_num=5)
# randomly pick a sequence as the query from the query sequence, make sure the picked sequence is in the input list
# this query'id must exist in the database
# query_seq = next((query_seq for query_seq in query_set if mydb.is_id_exists(query_seq)), None)
query_seq = query_set[0]
# query_result_bf = mydb.query_brute_force(query=query_seq, best_k=5)
query_result = mydb.query(query=query_seq, best_k=5, _lb_opt_repr='lbh')

# TODO memory optimization: brainstorm memory optimization, encode features (ids), length batches
