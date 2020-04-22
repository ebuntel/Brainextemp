from genex.utils.gxe_utils import from_csv, from_db
import numpy as np
import time

result = []  # data size, withoutDSG, withDSG
for i in range(6, 15):
    ds = 2 ** i
    print('Testing data size: ' + str(ds))
    data = np.reshape(np.random.randn(ds), (1, ds))

    gxe = from_csv(data, feature_num=0, header=None, num_worker=32, use_spark=True, driver_mem=26, max_result_mem=26)

    start = time.time()
    gxe.build(st=0.1, _group_only=True, _dsg=False)  # cluster only sequences that are longer than 1 second
    withoutDSG = time.time() - start
    print('Grouping took without DSG ' + str(withoutDSG) + ' sec')

    start = time.time()
    gxe.build(st=0.1, _group_only=True, _dsg=True)  # cluster only sequences that are longer than 1 second
    withDSG = time.time() - start
    print('Grouping took with DSG ' + str(withDSG) + ' sec')

    result.append([ds, withoutDSG, withDSG])
