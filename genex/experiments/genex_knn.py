import time

from genex.utils.gxe_utils import from_csv

data_file = 'data_original/SART2018_HbO.csv'
db_path = 'results/saves/SART2018_HbO_gxe'

start = time.time()
gxe = from_csv(data_file, feature_num=5, num_worker=12, use_spark=True, max_result_mem=12, driver_mem=12)
gxe.build(st=0.1, loi=[gxe.get_max_seq_len() * 0.9])
build_time = time.time() - start

gxe.save(db_path)