from pyspark import SparkContext, SparkConf


def create_sc(num_cores, driver_mem, max_result_mem):
    conf = SparkConf(). \
        setMaster("local[" + str(num_cores) + "]"). \
        setAppName("Genex").set('spark.driver.memory', driver_mem). \
        set('spark.driver.maxResultSize', max_result_mem)
    sc = SparkContext(conf=conf)

    return sc
