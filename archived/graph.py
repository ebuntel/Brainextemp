# for key, value in experiment_set_dist_ma.items():
#     mydb = experiment_genex(**value, num_sample=40, num_query=40, _dist_type='ma', _lb_opt_repr=_lb_opt_repr,
#                             _lb_opt_cluster=_lb_opt_cluster, _radius=radius)
#
# for key, value in experiment_set_dist_ch.items():
#     mydb = experiment_genex(**value, num_sample=40, num_query=40, _dist_type='ch', _lb_opt_repr=_lb_opt_repr,
#                             _lb_opt_cluster=_lb_opt_cluster, _radius=radius)
# test_result for radius = 0

# data_file = 'data_original/ItalyPower.csv'
# result_file = 'results/ipd/ItalyPowerDemand_result'
# feature_num = 2
# add_uuid = False
# k_to_test = [15, 9, 1]
# ke_result_dict = dict()
# for k in k_to_test:
# ke_result_dict[k] = experiment_genex_ke(data_file, num_sample=40, num_query=40, best_k=k, add_uuid=add_uuid,
#                                         feature_num=feature_num)

# q = Sequence(seq_id=('Italy_power25', '2'), start=7, end=18)
# seq1 = Sequence(seq_id=('Italy_power25', '2'), start=6, end=18)
# seq2 = Sequence(seq_id=('Italy_power25', '2'), start=7, end=17)
# q.fetch_and_set_data(mydb.data_normalized)
# seq1.fetch_and_set_data(mydb.data_normalized)
# seq2.fetch_and_set_data(mydb.data_normalized)
# from dtw import dtw
# import matplotlib.pyplot as plt
# plt.plot(q.data_original, label='query')
# plt.plot(seq1.data_original, label='gx')
# plt.plot(seq2.data_original, label='bf')
# plt.legend()
# plt.show()
# euclidean_norm = lambda x, y: np.abs(x - y)
# x_dist1, cost_matrix1, acc_cost_matrix1, path1 = dtw(q.data_original, seq1.data_original, dist=euclidean_norm)
# x_dist2, cost_matrix2, acc_cost_matrix2, path2 = dtw(q.data_original, seq2.data_original, dist=euclidean_norm)
# plt.imshow(acc_cost_matrix1.T, origin='lower', cmap='gray', interpolation='nearest')
# plt.plot(path1[0], path1[1], 'w')
# plt.title('query and gx')
# plt.show()
#
# plt.imshow(acc_cost_matrix2.T, origin='lower', cmap='gray', interpolation='nearest')
# plt.plot(path2[0], path2[1], 'w')
# plt.title('query and bf')
# plt.show()
# print('distance between query and gx ' + str(x_dist1))
# print('distance between query and bf ' + str(x_dist2))
# dist1 = sim_between_seq(q, seq1, dist_type='eu')
# dist2 = sim_between_seq(q, seq2, dist_type='eu')