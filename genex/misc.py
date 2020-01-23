def pr_red(skk):
    print("\033[91m {}\033[00m" .format(skk))


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
