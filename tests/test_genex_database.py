import multiprocessing
import os
import pandas as pd
import sys
import random
import pytest as pt
from genex.database import genexengine as gxdb
from genex.utils import gxe_utils as gutils


class TestGenex_database:
    num_cores = multiprocessing.cpu_count()

    def test_python_version(self):
        assert sys.version_info[0] == 3

    def test_load(self):
        # related path vs absolute path
        data_file = '../genex/experiments/data/ItalyPower.csv'
        # Missing a default parameter while loading a data file
        with pt.raises(TypeError) as e:
            gutils.load(data_file, num_worker=self.num_cores)
        assert 'provide a valid feature number' in str(e.value)

        with pt.raises(TypeError) as e:
            gutils.load(data_file, num_worker=self.num_cores, feature_num=1.5)
        assert 'provide a valid feature number' in str(e.value)

        # Loading a dataset from a fake path
        fake_path = '../genex/fake_path'
        with pt.raises(ValueError) as e:
            gutils.load(fake_path, num_worker=self.num_cores)
        assert 'Not a valid file name or directory path' in str(e.value)

    def test_from_csv(self):
        data_file = '../genex/experiments/data/ItalyPower.csv'
        feature_num = 0

        df = pd.read_csv(data_file)

        test_db = gutils.from_csv(data=data_file, feature_num=feature_num, num_worker=self.num_cores,
                                  use_spark=False)
        test_db_id_ls = [x[0] for x in test_db.data_original]

        assert _check_unique(test_db_id_ls)
        assert len(df) == len(test_db_id_ls)
        del test_db

        # Running test for another dataset including column header
        data_file = '../genex/experiments/data_original/SART2018_HbO_altered.csv'
        feature_num = 5
        df = pd.read_csv(data_file, header=0)  # header is only used to avoid duplicate code

        test_db = gutils.from_csv(data=data_file, feature_num=feature_num, num_worker=self.num_cores,
                                  use_spark=False)
        test_db_id_ls = [x[0] for x in test_db.data_original]

        assert _check_unique(test_db_id_ls)
        assert len(df) == len(test_db_id_ls)
        del test_db

    def test_from_csv_2(self):
        # The provided feature number is incorrect
        # 1. feature_num < the real feature number of the dataset
        data_file = '../genex/experiments/data_original/SART2018_HbO_altered.csv'
        feature_num = 2
        df = pd.read_csv(data_file)

        test_db = gutils.from_csv(data=data_file, feature_num=feature_num, num_worker=self.num_cores,
                                  use_spark=False)
        tb_id_ls = [x[0] for x in test_db.data_original]

        assert _check_unique(tb_id_ls)
        assert len(tb_id_ls) == len(df)
        del test_db, tb_id_ls

        # 2. feature_num > the real feature number of the dataset
        feature_num = 6
        test_db = gutils.from_csv(data_file, feature_num, self.num_cores, False)
        id_ls = [x[0] for x in test_db.data_original]

        assert _check_unique(id_ls)
        assert len(id_ls) == len(df)

    def test_from_db(self):
        data_file = '../genex/experiments/data_original/SART2018_HbO_altered.csv'
        feature_num = 5
        path = '../experiments/unittest/test_db'

        # Before executing build method
        db = gutils.load(file_or_path=data_file, num_worker=self.num_cores, feature_num=feature_num, use_spark=False)
        db_attributes = db.conf
        db_data_original = db.data_original
        db.save(path=path)

        db_after_save = gutils.load(file_or_path=path, num_worker=self.num_cores, use_spark=False)

        assert db_attributes == db_after_save.conf
        assert db_data_original == db_after_save.data_original
        del db_after_save

        # After executing build method
        db.build(st=0.05, loi=(db.get_max_seq_len() - 5, db.get_max_seq_len()))
        db_attributes = db.conf
        db_data_original = db.data_original
        db.save(path=path)

        db_after_save = gutils.load(file_or_path=path, num_worker=self.num_cores, use_spark=False)

        assert db_attributes == db_after_save.conf
        assert db_data_original == db_after_save.data_original

    def test_from_db_2(self):
        # Load a dataset from a valid dir path, but the dir itself is empty
        empty_dir_path = os.path.join(os.getcwd(), 'empty_db')

        if os.path.exists(empty_dir_path) is False:
            os.mkdir(empty_dir_path)

        with pt.raises(ValueError) as ve:
            test_db = gutils.from_db(empty_dir_path, num_worker=self.num_cores)
            os.removedirs(empty_dir_path)
        assert 'no such database' in str(ve.value)

    def test_build(self):
        # Test case for the functionality of build method
        # After grouping
        data_file = '../genex/experiments/data/ItalyPower.csv'
        feature_num = 0

        # Checking numbers of subsequences before clustering
        test_db = gutils.from_csv(data_file, feature_num=feature_num, num_worker=12,
                                  use_spark=False, _rows_to_consider=15)

        test_db.build(st=0.05, loi=(test_db.get_max_seq_len() - 5, test_db.get_max_seq_len()))

        seq_num_group = test_db.get_num_subsequences()
        seq_dict_cluster = test_db.cluster_meta_dict
        count = 0

        for v in seq_dict_cluster.values():  # {len of Sequences: {seq repre: number}}
            for num in v.values():
                count += num

        assert seq_num_group == count

    def test_build_2(self):
        # Test cases for parameters
        data_file = '../genex/experiments/data/ItalyPower.csv'
        tdb = gutils.from_csv(data_file, feature_num=0, num_worker=self.num_cores, use_spark=False)

        # Test case for the similarity threshold
        with pt.raises(Exception) as e:
            tdb.build(st=float('inf'))
        assert 'build st must be between 0. and 1.' in str(e.value)

        # Test case for the distance type parameter
        with pt.raises(Exception) as e:
            tdb.build(st=0.6, dist_type='ue')
        assert 'Unknown distance type' in str(e.value)

    def test_build_3(self):
        # Test cases for the loi parameter
        data_file = '../genex/experiments/data/ItalyPower.csv'
        db = gutils.from_csv(data_file, feature_num=0, num_worker=self.num_cores, use_spark=False)

        with pt.raises(Exception) as e:
            db.build(st=0.5, dist_type='eu', loi=40)  # int type loi is not acceptable right now
        assert 'must be an iterable of length 1 or 2' in str(e.value)

        with pt.raises(Exception) as e:
            db.build(st=0.5, loi=(500, ))  # the start point is great than the len of the dataset
        assert 'value type of the loi should be integer ' in str(e.value)

        with pt.raises(Exception) as e:
            db.build(st=0.5, loi=(10.4, 15.3))
        assert 'value type of the loi should be integer ' in str(e.value)

        with pt.raises(Exception) as e:
            db.build(st=0.5, loi=(-4, 32))
        assert 'value type of the loi should be positive integers' in str(e.value)

        with pt.raises(Exception) as e:
            db.build(st=0.5, loi=(-7, -5))
        assert 'value type of the loi should be positive integers' in str(e.value)

    def test_query(self):
        """
        TODO
        4. best_k type
        5. sequence type issue
        :return:
        """
        data_file = '../genex/experiments/data_original/SART2018_HbO_altered.csv'
        feature_num = 5
        test_db = gutils.load(data_file, num_worker=self.num_cores, use_spark=False, feature_num=feature_num)
        test_db.build(0.05, loi=(test_db.get_max_seq_len() - 5, test_db.get_max_seq_len()))

        seed = random.seed(test_db.get_max_seq_len())
        seq_len = random.randint(test_db.get_max_seq_len() - 5, test_db.get_max_seq_len())
        query_seq = test_db.get_random_seq_of_len(sequence_len=seq_len, seed=seed)

        best_k = random.randint(1, 10)
        gq_rlt = test_db.query(query_seq, best_k=best_k)  # new query sequence normalized issue
        assert len(gq_rlt) == best_k

        with pt.raises(Exception) as e:
            gq_rlt = test_db.query('fake_sequence', best_k)
        assert 'Unsupported query type' in str(e.value)

        with pt.raises(Exception) as e:
            gq_rlt = test_db.query(query_seq, best_k, overlap=2)
        assert 'overlap must be between 0. and 1. ' in str(e.value)


def _check_unique(x: list):
    seen = set()
    return not any(i in seen or seen.add(i) for i in x)
