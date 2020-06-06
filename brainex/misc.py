import numpy as np

def pr_red(skk):
    print("\033[91m {}\033[00m" .format(skk))


def prYellow(skk):
    print("\033[93m {}\033[00m" .format(skk))


def merge_dict(dicts: list):
    merged_dict = dict()
    merged_len = 0
    for d in dicts:
        merged_len += len(d)
        merged_dict = {**merged_dict, **d}  # make sure there is no replacement of elements
    try:
        assert merged_len == len(merged_dict)
    except AssertionError as ae:
        print(str(ae))
        raise Exception('duplicate dict keys: dict item replaced!')
    return merged_dict


def fd_workaround():
    pr_red('Unable to install FastDTW, please run this command in the terminal to install the required package: ')
    pr_red('pip install git+git://github.com/ApocalyVec/fastdtw.git')


def allUnique(x):
    seen = list()
    return not any(i in seen or seen.append(i) for i in x)