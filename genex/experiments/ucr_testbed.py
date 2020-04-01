import math
import os
import numpy as np

from genex.experiments.harvest_rand_query import generate_exp_set_from_root, run_exp_set


def run_ucr_test(dataset_path, output_dir, exclude_list, dist_types, start, end, ex_config):
    """
    The start and end parameter together make an interval that contains the datasets to be included in this experiment
    :param dataset_path: the path to the archive datasets
    :param output_dir: the path to which the result csv's will be saved
    :param exclude_list: the list of dataset names to be excluded from the experiment archive
    :param dist_types: a list of strings, must contain at least one item. Items must be ones of the following: eu,ch,ma
    :param start: int specifying the index of the dataset to start with, , works together with parameter end
    :param end: int specifying the index of the dataset to end with, works together with parameter start
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
        assert start >= 0
        assert end <= len(os.listdir(dataset_path))
        assert 0 < len(dist_types) <= 3
        assert np.all([x in valid_dt for x in dist_types])
    except AssertionError:
        raise Exception('Assertion failed in checking parameters')

    exp_arg_list = [{
        'dist_type': dt,
        'notes': 'UCR_test_' + dt + '_' + str(start) + '-to-' + str(end),
        'start': start,
        'end': end
    } for dt in dist_types]

    exp_set_list = [generate_exp_set_from_root(dataset_path, output_dir, exclude_list, **ea) for ea in exp_arg_list]
    return [run_exp_set(es, **ex_config) for es in exp_set_list]


'''
Start of the experiment script
'''
if __name__ == "__main__":

    # Start of Config Parameters #########################
    '''
    check the docstring of the above function - run_ucr_test for details regarding the parameters
    '''
    dataset = '/home/apocalyvec/data/UCRArchive_2018'
    output = '/home/apocalyvec/PycharmProjects/Genex/genex/experiments/results/ucrtest'
    exclude_dataset = ['Missing_value_and_variable_length_datasets_adjusted']

    dist_types_to_test = ['eu', 'ma', 'ch']
    starting_dataset_index = 0
    ending_dataset_index = 16

    ex_config_test = {
        'num_sample': math.inf,
        'query_split': 0.2,
        '_lb_opt': False,
        'radius': 1,
        'use_spark': True,
        'loi_range': 0.1,
        'st': 0.1,
        'paa_c': 0.6
    }
    # End of Config Parameters #########################

    run_ucr_test(dataset, output, exclude_dataset, dist_types=dist_types_to_test, start=starting_dataset_index, end=ending_dataset_index,
                 ex_config=ex_config_test)  # TODO test the test for UCR dataset
