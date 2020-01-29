import multiprocessing

from genex.misc import pr_red
from genex.utils.spark_utils import _create_sc, _pr_spark_conf


def _multiprocess_backend(use_spark, num_worker, driver_mem, max_result_mem):
    """
    :return None if not using spark
    """
    if use_spark:
        pr_red('Genex Engine: Using PySpark Backend')
        mp_context = _create_sc(num_cores=num_worker, driver_mem=driver_mem, max_result_mem=max_result_mem)
        _pr_spark_conf(mp_context)
    else:
        pr_red('Genex Engine: Using Python Native Multiprocessing')
        mp_context = multiprocessing.Pool(num_worker, maxtasksperchild=1)

    return mp_context