import cc3d
import numpy as np

def sort_region(x, num=2):
    """x:3D, select six the most large region"""
    max_label = x.max()
    sum_list = [(x == index).sum() for index in range(1, int(max_label.item())+1)]
    # print(sum_list)
    # sort sum_list, return index, from large to small
    index_list = np.argsort(sum_list)[::-1]
    region_reserved = x == (index_list[0] + 1)
    for index in index_list[1:num]:
        region_reserved = region_reserved | (x == (index+1))
    return np.array(region_reserved, dtype=bool)


def select_two_biggest_connected_region(region, num=2):
    region_mask = cc3d.connected_components(region > 0, connectivity=6)
    region_two = region * sort_region(region_mask, num=num)
    return region_two

