from pyspark import SparkContext, SparkConf

from genex.misc import pr_red


def _create_sc(num_cores: int, driver_mem: int, max_result_mem: int):
    conf = SparkConf(). \
        setMaster("local[" + str(num_cores) + "]"). \
        setAppName("Genex").set('spark.driver.memory', str(driver_mem) + 'G'). \
        set('spark.driver.maxResultSize', str(max_result_mem) + 'G')
    sc = SparkContext(conf=conf)

    return sc


def _pr_spark_conf(sc:SparkContext):
    pr_red('Number of Works:     ' + str(sc.defaultParallelism))
    pr_red('Driver Memory:       ' + sc.getConf().get("spark.driver.memory"))
    pr_red('Maximum Result Size: ' + sc.getConf().get("spark.driver.maxResultSize"))
