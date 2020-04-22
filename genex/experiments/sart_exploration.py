import pandas as pd
import numpy as np
import time
from genex.utils.gxe_utils import from_csv, from_db
import matplotlib.pyplot as plt

cluster_st = 0.1
sampling_rate = 6.6231

events = 'true answer', 'bCurrent Item: nonTarget',

results = ['target incorrect']

# data_all = '/home/apocalyvec/data/sart/101-SART-June2018-AS_(all).csv'
data_all = '/home/apocalyvec/data/sart/test.csv'


gxe = from_csv(data_all, feature_num=0, header=None, num_worker=4, use_spark=True, driver_mem=24, max_result_mem=24)
# gxe.build(st=cluster_st, loi=[int(sampling_rate)])  # cluster only sequences that are longer than 1 second
gxe.build(st=cluster_st)  # cluster only sequences that are longer than 1 second
