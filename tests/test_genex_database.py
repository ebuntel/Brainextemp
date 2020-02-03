import multiprocessing
import pandas as pd
import sys

from psutil import virtual_memory
from pyspark import SparkConf, SparkContext
from genex.database import genexengine as gxdb
from genex.cluster import _cluster_groups


class TestGenex_database:
    num_cores = multiprocessing.cpu_count()

    def test_python_version(self):
        assert sys.version_info[0] == 3

    def test_from_csv(self):
        def _check_unique(x: list):
            seen = set()
            return not any(i in seen or seen.add(i) for i in x)

        data_file = '../experiments/data/Gun_Point_TRAIN.csv'
        feature_num = 1

        df = pd.read_csv(data_file, header=None, skiprows=1, usecols=[0])
        df_raw_id_num = len(df.groupby(list(df.columns[:])))

        test_db = gxdb.from_csv(file_name=data_file, feature_num=feature_num, num_worker=self.num_cores, is_header=False)
        test_db_id_ls = [x[0] for x in test_db.data]

        assert _check_unique(test_db_id_ls)
        assert len(df) == len(test_db_id_ls)

        # Running test for another dataset including column header
        data_file = '../experiments/data/ItalyPower.csv'
        feature_num = 2

        df = pd.read_csv(data_file, index_col=[x for x in range(feature_num)])

        test_db = gxdb.from_csv(file_name=data_file, feature_num=feature_num, num_worker=self.num_cores)
        test_db_id_ls = [x[0] for x in test_db.data]

        assert _check_unique(test_db_id_ls)
        assert len(df) == len(test_db_id_ls)

    def test_from_db(self):
        data_file = '../experiments/data/Gun_Point_TRAIN.csv'
        feature_num = 1
        path = '../experiments/results/test_db'

        # Before executing build method
        db = gxdb.from_csv(data_file, feature_num=feature_num, num_worker=self.num_cores, is_header=False)
        db_attribute = db.conf
        db_data = db.data
        db.save(path=path)

        db_after_save = gxdb.from_db(path=path, num_worker=self.num_cores)
        assert db_attribute == db_after_save.conf
        assert db_data == db_after_save.data
        del db

        # After executing build method
        db = gxdb.from_csv(data_file, feature_num=feature_num, num_worker=self.num_cores, is_header=False)
        db.build(similarity_threshold=0.1, loi=slice(90, 95))
        db_attribute = db.conf
        db_data = db.data
        db.save(path=path)

        db_after_save = gxdb.from_db(path=path, num_worker=self.num_cores)
        assert db_attribute == db_after_save.conf
        assert db_data == db_after_save.data

    def test_build_cluster(self):
        data_file = '../experiments/data/Gun_Point_TRAIN.csv'
        feature_num = 1

        # Checking numbers of subsequences before grouping
        mydb = gxdb.from_csv(data_file, feature_num=feature_num, num_worker=self.num_cores, is_header=False)
        mydb_data_test = mydb.data_normalized[:22]

        input_rdd = self.sc.parallelize(mydb_data_test, numSlices=self.sc.defaultParallelism)
        group_rdd = input_rdd.mapPartitions(
            lambda x: gxdb._group_time_series(time_series=x, start=89, end=94), preservesPartitioning=True).cache()
        sub_sq_count = group_rdd.map(lambda x: len(x[1])).collect()

        sub_sq_num = 0
        for num in sub_sq_count:
            sub_sq_num += num

        print('Number of subsequence before grouping operation')

        # number of sub_sq after grouping
        # cluster_rdd = group_rdd.mapPartitions(lambda x: _cluster_groups(
        #     groups=x, st=0.01, dist_type='eu', log_level=1)).cache()
        #
        # result = cluster_rdd.collect()
        print()

    def test_query(self):
        pass
