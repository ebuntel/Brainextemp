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
fn = 'SART2018_HbO.csv'
mydb = gxdb.from_csv(fn, sc=sc, feature_num=5)

mydb.build(similarity_threshold=0.1, loi=slice(120))