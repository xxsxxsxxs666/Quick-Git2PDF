import numpy as np

def fast_index(img, point):
    """point: n * 3"""
    H, W, D = img.shape
    point = point[:, 2] + D * point[:, 1] + W * D * point[:, 0]
    batch_crop = img.reshape(-1)[point]
    return batch_crop


def fast_crop(img, point):
    """
    point: n * h * w * d * 3
    return: n, h, w, d
    """
    n, h, w, d, _ = point.shape
    point = point.reshape(n * h * w * d, 3)
    batch_crop = fast_index(img, point)
    batch_crop = batch_crop.reshape(n, h, w, d)

    return batch_crop


def generate_cube(point, h, w, d):
    """
    point: n * 3
    return: n * h * w * d * 3
    """

    """"""
    n, _ = point.shape
    x = np.arange(-h, h+1)
    y = np.arange(-w, w+1)
    z = np.arange(-d, d+1)

    [Y, X, Z] = np.meshgrid(y, x, z)
    crop_region_index = np.zeros((x.shape[0] * y.shape[0] * z.shape[0], 3))
    crop_region_index[:, 0],  crop_region_index[:, 1], crop_region_index[:, 2] = \
        X.reshape(-1), Y.reshape(-1), Z.reshape(-1)
    region_index = point[:, None, :] + crop_region_index[None, :, :]
    region_index = region_index.reshape(n, x.shape[0], y.shape[0], z.shape[0], 3)
    return region_index.astype(np.uint16)


def fancy_indexing(imgs, centers, pw, ph):
    n = imgs.shape[0]
    img_i, RGB, x, y = np.ogrid[:n, :3, :pw, :ph]
    corners = centers - [pw//2, ph//2]
    x_i = x + corners[:,0,None,None,None]
    y_i = y + corners[:,1,None,None,None]
    return imgs[img_i, RGB, x_i, y_i]


if __name__ == '__main__':
    H = 8
    W = 8
    D = 8
    img = np.arange(H * W * D).reshape(H, W, D)
    point = np.array([[3, 3, 3],
                      [2, 2, 2],
                      [1, 1, 1], ])

    cube_point = generate_cube(point, h=1, w=1, d=1)

    crop = fast_crop(img, point=cube_point)

    print(img)
    print(crop)


