import multiprocessing
import sys

from psutil import virtual_memory
from pyspark import SparkConf, SparkContext
from genex.database import genex_database as gxdb


class TestGenex_database:
    num_cores = multiprocessing.cpu_count()
    # mem = virtual_memory()
    # i = mem.total
    conf = SparkConf(). \
        setMaster("local[" + str(num_cores) + "]"). \
        setAppName("Genex").set('spark.driver.memory', '12G'). \
        set('spark.driver.maxResultSize', '12G')
    sc = SparkContext(conf=conf)

    data_file = 'data/ECGFiveDays.csv'
    db_path = 'results/test_db'

    def test_python_version(self):
        assert sys.version_info[0] == 3

    def test_from_csv(self):
        pass

    def test_from_db(self):
        pass

    def test_build_cluster(self):
        pass

    def test_query(self):
        pass