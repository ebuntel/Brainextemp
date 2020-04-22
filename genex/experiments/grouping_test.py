import copy

from genex.utils.gxe_utils import from_csv, from_db
import numpy as np
import time
from matplotlib import pyplot as plt

vals = np.linspace(6, 8, 15)

result = []  # two's power, data size, withoutDSG, withDSG
for i in vals:
    ds = int(2 ** i)
    print('Testing data size: ' + str(ds))
    data = np.reshape(np.random.randn(ds), (1, ds))

    gxe = from_csv(data, feature_num=0, header=None, num_worker=32, use_spark=True, driver_mem=26, max_result_mem=26)

    start = time.time()
    gxe.build(st=0.1, _group_only=False, _dsg=False)  # cluster only sequences that are longer than 1 second
    withoutDSG = time.time() - start
    print('Grouping took without DSG ' + str(withoutDSG) + ' sec')
    withoutDSGss = copy.deepcopy(gxe.get_subsequences())

    start = time.time()
    gxe.build(st=0.1, _group_only=False, _dsg=True)  # cluster only sequences that are longer than 1 second
    withDSG = time.time() - start
    print('Grouping took with DSG ' + str(withDSG) + ' sec')
    withDSGss = copy.deepcopy(gxe.get_subsequences())

    result.append([i, ds, withoutDSG, withDSG])

    # check the elements are the same
    assert set(withoutDSGss) == set(withDSGss)
    gxe.stop()

# result = np.array(result)
# plt.plot(result[:, 0], result[:, 2], label='Grouping-only time without DSG')
# plt.plot(result[:, 0], result[:, 3], label='Grouping-only time with DSG')
# plt.xlabel('Sequence length (of 2’s magnitude)')
# plt.ylabel('time took to finish (sec)')
# plt.legend()
# plt.show()