import nrrd
import numpy as np


def get_space_full_name(space):
    assert len(space) == 3, "space should be a 3-letter string"
    full_name = []
    for s in space:
        if s == "L":
            full_name.append("left")
        elif s == "R":
            full_name.append("right")
        elif s == "A":
            full_name.append("anterior")
        elif s == "P":
            full_name.append("posterior")
        elif s == "S":
            full_name.append("superior")
        elif s == "I":
            full_name.append("inferior")
    # link with _ to form full name
    return "-".join(full_name)


def get_direction(space):
    """3D-slicer use default RAS space, so we need to convert the space to RAS"""
    direction = [1, 1, 1]
    if space[0] == "L":
        direction[0] = -1
    if space[1] == "P":
        direction[1] = -1
    if space[2] == "I":
        direction[2] = -1
    return direction


def save_mvf(mvf, save_path, affine, space, scale_factor=10):
    offset = np.array((affine[:3, 3]))
    direction = np.array((affine[:3, :3])).tolist()
    """Seems like mvf in 3D slicer do not consider the direction of the space, 
    so we need to multiply the mvf with the direction matrix to make it consistent with the space"""
    mvf_direction = get_direction(space)
    mvf[:, :, :, 0] = mvf[:, :, :, 0] * mvf_direction[0]
    mvf[:, :, :, 1] = mvf[:, :, :, 1] * mvf_direction[1]
    mvf[:, :, :, 2] = mvf[:, :, :, 2] * mvf_direction[2]
    header = {
        'endian': 'little',
        'encoding': 'raw',
        'space': get_space_full_name(space),
        'space directions': direction + [None],
        'space origin': offset,
        'kinds': ['domain', 'domain', 'domain', 'vector'],
    }
    nrrd.write(save_path, mvf * scale_factor, header=header)


def save_image(image, save_path, affine, space):
    offset = np.array((affine[:3, 3]))
    direction = np.array((affine[:3, :3])).tolist()
    header = {
        'endian': 'little',
        'encoding': 'raw',
        'space': get_space_full_name(space),
        'space directions': direction,
        'space origin': offset,
        'kinds': ['domain', 'domain', 'domain'],
    }
    nrrd.write(save_path, image, header=header)


