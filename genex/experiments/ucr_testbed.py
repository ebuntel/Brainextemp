import math
import os
import numpy as np

from genex.experiments.query_harvest import generate_exp_set_from_root, run_exp_set


def run_ucr_test(dataset_path, dataset_soi, output_dir, exclude_list, dist_types, ex_config, mp_args):
    """
    The start and end parameter together make an interval that contains the datasets to be included in this experiment
    :param mp_args: the configuration of the multiprocess backend, you should set the number of workers to be the,
            IMPORTANT: go to this site https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-spark-configure.html for
            the correct Spark configuration with AWS; you only need to worry the configs that are exposed to you ->
            that is: the number of workers, the max driver memory and the max result size
    :param dataset_path: the path to the archive datasets
    :param dataset_soi: (soi: size of interest) a iterable of two integers for binning the experiment
    :param output_dir: the path to which the result csv's will be saved
    :param exclude_list: the list of dataset names to be excluded from the experiment archive
    :param dist_types: a list of strings, must contain at least one item. Items must be ones of the following: eu,ch,ma
    :param ex_config: a dict contains hyper-parameters for the experiment. They are
        'num_sample': int, number of samples to consider in each dataset, set this to math.inf for complete experiment
        'query_split': float, a fraction of the dataset to be taken as queries, use 0.2 for the time being
        '_lb_opt': bool, whether to turn of lower-bounding optimization for DTW, leave it False in not otherwise specified
        'radius': int, the length radius for Genex Query, leave it being 1 if not otherwise specified
        'use_spark': bool, whether to use the Spark backend, leave it being True if not otherwise specified
        'loi_range': float, only consider sequences within a percentage length of the longest sequence, use 0.1 for the time being
        'st': float, hyper-parameters that determines the cluster boundary in genex.build, leave it being True if not otherwise specified
        'paa_c': the compression ratio of PAA method, use 0.1 for now
    """
    valid_dt = ['eu', 'ch', 'ma']
    try:
        assert os.path.isdir(dataset_path)
        assert os.path.isdir(output_dir)
        assert dataset_soi[0] >= 0
        assert dataset_soi[0] <= dataset_soi[1]
        assert 0 < len(dist_types) <= 3
        assert np.all([x in valid_dt for x in dist_types])
    except AssertionError:
        raise Exception('Assertion failed in checking parameters')

    exp_arg_list = [{
        'dist_type': dt,
        'notes': 'UCR_test_' + dt + '_soi_' + str(dataset_soi[0]) + '-to-' + str(dataset_soi[1]),
        'soi': dataset_soi,
    } for dt in dist_types]

    exp_set_list = [generate_exp_set_from_root(dataset_path, output_dir, exclude_list, **ea) for ea in exp_arg_list]
    return [run_exp_set(es, mp_args, **ex_config) for es in exp_set_list]


'''
Start of the experiment script
'''
if __name__ == "__main__":
    # Start of Config Parameters #########################
    '''
    check the docstring of the above function - run_ucr_test for details regarding the parameters
    '''
    dataset = '/home/apocalyvec/data/UCRArchive_2018'
    ds_soi = [0, 50000]  # dataset size of interest, see the docstring of run_ucr_test for details
    output_reg = '/home/apocalyvec/data/UCR_test'
    output_dss = '/home/apocalyvec/data/UCR_DSS_test'

    exclude_dataset = ['Missing_value_and_variable_length_datasets_adjusted']

    dist_types_to_test = ['eu', 'ma', 'ch']

    ex_config_test = {
        'num_sample': math.inf,
        'query_split': 0.2,
        '_lb_opt': False,
        'radius': 1,
        'use_spark': True,
        'loi_range': 0.1,
        'st': 0.1,
        'paa_c': 0.6,
    }
    mp_args = {'num_worker': 32,
               'driver_mem': 24,
               'max_result_mem': 24}

    # End of Config Parameters, Experiment starts here ################################################################
    run_ucr_test(dataset, ds_soi, output_reg, exclude_dataset, dist_types=dist_types_to_test, ex_config=ex_config_test, mp_args=mp_args)

    # for test grouping, keep this part commented for the pure UCR testing ############################################
    ex_config_test['_test_dss'] = True
    ex_config_test['loi_range'] = 1.0  # set to cluster the full length
    run_ucr_test(dataset, ds_soi, output_dss, exclude_dataset, dist_types=dist_types_to_test, ex_config=ex_config_test, mp_args=mp_args)
