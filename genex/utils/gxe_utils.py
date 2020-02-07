import json
import os
import pickle
import uuid
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from genex import GenexEngine
from genex.misc import allUnique
from genex.utils.utils import _create_f_uuid_map, _df_to_list, genex_normalize
from genex.utils.context_utils import _multiprocess_backend


def from_csv(file_name, feature_num: int,
             num_worker: int,
             use_spark: bool,
             header=0,
             driver_mem: int = 16, max_result_mem: int = 16,
             _ts_dim: int = 1,
             _rows_to_consider: int = None,
             _memory_opt: str = None,
             _is_z_normalize=True):
    """
    build a genex_database object from given csv,
    Note: if time series are of different length, shorter sequences will be post padded to the length
    of the longest sequence in the dataset

    :param header:
    :param _ts_dim:
    :param _is_z_normalize:
    :param _memory_opt:
    :param driver_mem:
    :param max_result_mem:
    :param use_spark:
    :param num_worker:
    :param file_name:
    :param feature_num:
    :param sc: spark context on which the database will run

    :param _rows_to_consider: experiment parameter that takes a iterable of two integers.
            Only rows in between the given interval will be take into the database.
    :param _is_use_uuid: experiment parameter. If set to true, the feature (id) of the time series will be

    :return: a genex_database object that holds the original time series
    """
    if file_name.endswith('.csv'):
        df = pd.read_csv(file_name, header=header)
    elif file_name.endswith('.tsv'):
        df = pd.read_csv(file_name, sep='\t', header=header)
    else:
        raise Exception('Unrecognized file type, make sure that the data file extension is either csv or tsv.')

    add_uuid = need_uuid(df, feature_num)
    if add_uuid:
        print('msg: from_csv, feature num is 0, auto-generating uuid')
        feature_num = feature_num + 1
        df.insert(0, 'uuid', [uuid.uuid4() for x in range(len(df))], False)

    # if _memory_opt == 'uuid':
    #     df.insert(0, 'uuid', [uuid.uuid4() for x in range(len(df))], False)
    #     feature_uuid_dict = _create_f_uuid_map(df=df, feature_num=feature_num)
    #     # take off the feature columns
    #     df = df.drop(df.columns[list(range(1, feature_num + 1))], axis=1, inplace=False)
    #     data_list = _df_to_list(df, feature_num=1)  # now the only feature is the uuid

    # elif _memory_opt == 'encoding':
    #     for i in range(feature_num):
    #         le = LabelEncoder()
    #         df.iloc[:, i] = le.fit_transform(df.iloc[:, i])
    #
    #     data_list = _df_to_list(df, feature_num=feature_num)
    # else:
    data_list = _df_to_list(df, feature_num=feature_num)

    if _rows_to_consider is not None:
        if type(_rows_to_consider) == list:
            assert len(_rows_to_consider) == 2
            data_list = data_list[_rows_to_consider[0]:_rows_to_consider[1]]
        elif type(_rows_to_consider) == int:
            data_list = data_list[:_rows_to_consider]
        else:
            raise Exception('_rows_to_consider must be either a list or an integer')

    data_norm_list, global_max, global_min = genex_normalize(data_list, z_normalization=_is_z_normalize)
    mp_context = _multiprocess_backend(use_spark, num_worker=num_worker, driver_mem=driver_mem,
                                       max_result_mem=max_result_mem)
    return GenexEngine(data_raw=df, data_original=data_list, data_normalized=data_norm_list, global_max=global_max,
                       global_min=global_min, has_uuid=add_uuid,
                       mp_context=mp_context, backend='multiprocess' if not use_spark else 'spark',
                       seq_dim=_ts_dim)


def need_uuid(df, feature_num):
    feature_col = df.iloc[:, 0:feature_num].values.tolist()
    return feature_num == 0 or not allUnique(feature_col)


def from_db(path: str,
            num_worker: int,
            driver_mem: int = 16, max_result_mem: int = 16,
            ):
    """
    returns a previously saved gxe object from its saved path

    :param max_result_mem:
    :param driver_mem:
    :param use_spark:
    :param num_worker:
    :param sc: spark context on which the database will run
    :param path: path of the saved gxe object

    :return: a genex database object that holds clusters of time series data_original
    """

    # TODO the input fold_name is not existed
    if os.path.exists(path) is False:
        raise ValueError('There is no such database, check the path again.')

    data_raw = pd.read_csv(os.path.join(path, 'data_raw.csv'))
    data = pickle.load(open(os.path.join(path, 'data_original.gxe'), 'rb'))
    data_normalized = pickle.load(open(os.path.join(path, 'data_normalized.gxe'), 'rb'))

    conf = json.load(open(os.path.join(path, 'conf.json'), 'rb'))

    mp_context = _multiprocess_backend(is_conf_using_spark(conf), num_worker=num_worker, driver_mem=driver_mem,
                                       max_result_mem=max_result_mem)
    init_params = {'data_raw': data_raw, 'data_original': data, 'data_normalized': data_normalized,
                   'mp_context': mp_context, 'conf': conf}
    engine: GenexEngine = GenexEngine(**init_params)

    if os.path.exists(os.path.join(path, 'clusters.gxe')):
        engine.load_cluster(path)
        engine.set_cluster_meta_dict(pickle.load(open(os.path.join(path, 'cluster_meta_dict.gxe'), 'rb')))
        build_conf = json.load(open(os.path.join(path, 'build_conf.json'), 'rb'))
        engine.set_build_conf(build_conf)
        if engine.is_using_spark():
            engine._data_normalized_bc = engine.mp_context.broadcast(engine.data_normalized)
    return engine


def is_conf_using_spark(conf):
    return conf['backend'] == 'spark'
