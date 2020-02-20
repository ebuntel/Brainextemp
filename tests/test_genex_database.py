import multiprocessing
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
        data_file = '../genex/experiments/data_original/ItalyPower.csv'
        # Missing a default parameter while loading a data file
        with pt.raises(Exception) as e:
            gutils.load(data_file, num_worker=self.num_cores)
        assert 'provide a valid feature number' in str(e.value)

        fake_path = '../genex/fake_path'
        # Loading a dataset from a fake path
        with pt.raises(ValueError) as e:
            gutils.load(fake_path, num_worker=self.num_cores)
        assert 'Not a valid file name or directory path' in str(e.value)

    def test_from_csv(self):

        def _check_unique(x: list):
            seen = set()
            return not any(i in seen or seen.add(i) for i in x)

        data_file = '../genex/experiments/data_original/ItalyPower.csv'
        feature_num = 2

        df = pd.read_csv(data_file)

        test_db = gutils.from_csv(file_name=data_file, feature_num=feature_num, num_worker=self.num_cores, use_spark=False)
        test_db_id_ls = [x[0] for x in test_db.data_original]

        assert _check_unique(test_db_id_ls)
        assert len(df) == len(test_db_id_ls)
        del test_db

        # Running test for another dataset including column header
        data_file = '../genex/experiments/data/ECGFiveDays.csv'
        feature_num = 2

        df = pd.read_csv(data_file, index_col=[x for x in range(feature_num)])

        test_db = gutils.from_csv(file_name=data_file, feature_num=feature_num, num_worker=self.num_cores, use_spark=False)
        test_db_id_ls = [x[0] for x in test_db.data_original]

        assert _check_unique(test_db_id_ls)
        assert len(df) == len(test_db_id_ls)
        del test_db

    def test_from_db(self):
        data_file = '../genex/experiments/data_original/ItalyPower.csv'
        feature_num = 2
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
        db.build(st=0.05, loi=(db.get_max_seq_len()-5, db.get_max_seq_len()))
        db_attributes = db.conf
        db_data_original = db.data_original
        db.save(path=path)

        db_after_save = gutils.load(file_or_path=path, num_worker=self.num_cores, use_spark=False)

        assert db_attributes == db_after_save.conf
        assert db_data_original == db_after_save.data_original

    def test_build(self):
        data_file = '../genex/experiments/data_original/ItalyPower.csv'
        feature_num = 2

        # Checking numbers of subsequences before grouping
        test_db = gutils.from_csv(data_file, feature_num=feature_num, num_worker=12,
                                  use_spark=False, _rows_to_consider=15)

        test_db.build(st=0.05, loi=(test_db.get_max_seq_len()-5, test_db.get_max_seq_len()))

        seq_num_group = test_db.get_num_subsequences()
        d = test_db.cluster_meta_dict
        count = 0

        for v in d.values():
            for num in v.values():
                count += num

        assert seq_num_group == count

    def test_query(self):
        data_file = '../genex/experiments/data_original/ItalyPower.csv'
        feature_num = 5
        test_db = gutils.load(data_file, num_worker=self.num_cores, use_spark=False, feature_num=feature_num)
        test_db.build(0.05, loi=(test_db.get_max_seq_len()-5, test_db.get_max_seq_len()))

        seed = random.seed(test_db.get_max_seq_len())
        seq_len = random.randint(test_db.get_max_seq_len()-5, test_db.get_max_seq_len())
        query_seq = test_db.get_random_seq_of_len(sequence_len=seq_len, seed=seed)

        gq_rlt = test_db.query(query_seq, best_k=3)
