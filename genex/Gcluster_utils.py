from .classes.Gcluster import Gcluster


def merge_gclusters(gclusters):
    # gclusters validation
    try:
        iterator = iter(gclusters)
    except TypeError as te:
        raise Exception('Given Gclusters is not iterable.')

    try:
        for gc in gclusters:
            assert type(gc) is Gcluster
            assert gc.collected is True
    except AssertionError as ae:
        raise Exception('Object in the given list must all be Gclusters and have been collected.')

    # validate if the clusters are from the same original data
    try:
        data_list = list(map(lambda x: x.data, gclusters))
        norm_data_list = list(map(lambda x: x.norm_data, gclusters))
        all_feature_lists = list(map(lambda x: x.feature_list, gclusters))
        global_max_list = list(map(lambda x: x.global_max, gclusters))
        global_min_list = list(map(lambda x: x.global_min, gclusters))
        st_list = list(map(lambda x: x.st, gclusters))

        assert all(x == data_list[0] for x in data_list) and \
               all(x == norm_data_list[0] for x in norm_data_list) and \
               all(x == all_feature_lists[0] for x in all_feature_lists) and \
               all(x == global_max_list[0] for x in global_max_list) and \
               all(x == global_min_list[0] for x in global_min_list) and \
               all(x == st_list[0] for x in st_list)
    except AssertionError as ae:
        raise Exception('Gclusters to merge must be from the same original data and have the same Similarity Threshold')

    # merge the clusters
    merged_clusters = {}

    for gc in gclusters:
        merged_clusters.update(gc.clusters)

    return Gcluster(feature_list=all_feature_lists[0],
                    data=data_list[0], norm_data=norm_data_list[0], st=st_list[0],
                    cluster_dict=merged_clusters, collected=True,
                    # this two attribute are different based on is_collect set to true or false
                    global_max=global_max_list[0], global_min=global_min_list[0])