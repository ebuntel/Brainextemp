import genex.database.genex_database as gxdb
from pyspark import SparkContext, SparkConf

# create the spark context
num_cores = 32
conf = SparkConf(). \
    setMaster("local[" + str(num_cores) + "]"). \
    setAppName("Genex").set('spark.driver.memory', '64G'). \
    set('spark.driver.maxResultSize', '64G')
sc = SparkContext(conf=conf)

# create gxdb from a csv file
data_file = 'SART2018_HbO_altered.csv'
db_path = 'gxdb/test_db'
mydb = gxdb.from_csv(data_file, sc=sc, feature_num=5)
mydb.save(path=db_path)

del mydb

mydb = gxdb.from_db(path=db_path, sc=sc)
mydb.build(similarity_threshold=0.1)

