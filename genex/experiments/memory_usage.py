from memory_profiler import profile
import genex.database.genexengine as gxdb
from genex.utils.spark_utils import _create_sc
import pickle

@profile
def memory_test(data_file_path, feature_num, _memory_opt):
    sc = _create_sc(num_cores=12, driver_mem='31G', max_result_mem='31G')
    mydb = gxdb.from_csv(data_file_path, sc=sc, feature_num=feature_num, _memory_opt=_memory_opt)

    grouped_sequences = mydb.group_sequences()

    return grouped_sequences

if __name__ == '__main__':
    data_file_path = 'data_original/SART2018_HbO_altered.csv'
    feature_num = 20
    result = memory_test(data_file_path=data_file_path, feature_num=feature_num, _memory_opt='encoding')

    '''
    note:
    in _row_to_feature_and_data:
        for sequence id
            do not cast x in using uuid
            cast x to int when using encoding
            cast x to str when not using optimization
    '''
    # result[0].del_data()
    # pickle.dump(result[0], open('results/mem_test/sart20/s_encoding.p', 'wb'))
