import json
import os
import glob
import random
import numpy as np
import pathlib


def five_fold_generator(L):
    """
    :param L: a list
    :return: five_fold: shape 5*1
    """
    length = len(L)
    index = list(range(length))
    random.shuffle(index)
    d = length // 5
    five_fold = []
    print(f"rand_index:{index}")
    for i in range(5):
        train_index = index[:(4 - i) * d] + index[(5 - i) * d:] if i > 0 else index[:(4 - i) * d]
        val_index = index[(4 - i) * d:(5 - i) * d] if i > 0 else index[(4 - i) * d:]
        five_fold.append({"train_files": [L[k] for k in train_index],
                                "val_files": [L[k] for k in val_index]})
    return five_fold
        # print(f"{i}: train: {five_fold_index[i]['train']}, val: {five_fold_index[i]['val']}")


def shuffle_generator(L):
    """
    :param L: a list
    :return: shuffle_fold: 20% randomly selected sample form origin dataset
    """
    length = len(L)
    index = list(range(length))
    random.shuffle(index)
    train_index, val_index = index[:int(0.8*length)], index[int(0.8*length):]
    shuffle_fold = [{"train_files": [L[k] for k in train_index],
                     "val_files": [L[k] for k in val_index]}]

    return shuffle_fold


def generator_multi_dataset(L1, L2, split_mode='shuffle'):
    """
    :spilt_mode:
    :param L1, L2: a list, diast, syst
    :return: shuffle_fold: 20% randomly selected sample form origin dataset
    """
    length = len(L1)
    assert len(L1) == len(L2), 'length of two list are different '
    train_index, val_index = index_generator(length, split_mode=split_mode)
    fold = []
    for i in range(len(train_index)):
        fold_l1 = {"train_files": [L1[k] for k in train_index[i]],
                   "val_files": [L1[k] for k in val_index[i]]}
        fold_l2 = {"train_files": [L2[k] for k in train_index[i]],
                   "val_files": [L2[k] for k in val_index[i]]}
        fold.append({"train_files": fold_l1["train_files"] + fold_l2["train_files"],
                     "val_files": fold_l1["val_files"] + fold_l2["val_files"], })
    return fold


def prepare_datalist(image_file="images", label_file="label", split_mode='order'):
    """
    :param image_file: the name of image file
    :param label_file: the name of label file
    :return: train_files, val_files. Now we separate them directly. It needs to be modified for 5-fold validation.
    """

    image_path = pathlib.Path(image_file)
    label_path = pathlib.Path(label_file)
    assert image_path.is_dir(), f"img path not exist: {image_path}"
    assert label_path.is_dir(), f"label path not exist: {label_path}"
    # get the image name list in image_path using pathlib
    train_images = [path.name for path in image_path.glob("*.nii.gz")]
    data_dicts = [
        {"image": str(pathlib.Path(image_path, image_name)), "label": str(pathlib.Path(label_path, image_name))}
        for image_name in train_images
    ]
    if split_mode == 'five_fold':
        return five_fold_generator(data_dicts)
    elif split_mode == 'order':
        length = len(data_dicts)
        index = int(0.2 * length)
        train_files, val_files = data_dicts[:-index], data_dicts[-index:]
        return [{"train_files": train_files, "val_files": val_files}]
    elif split_mode == 'shuffle':
        length = len(data_dicts)
        index = int(0.2 * length)
        train_files, val_files = data_dicts[:-index], data_dicts[-index:]
        return [{"train_files": train_files, "val_files": val_files}]
    elif split_mode == 'all':
        return data_dicts
    else:
        raise RuntimeError(f"{split_mode} is not supported")


def prepare_datalist_with_file(image_file="images", label_file="label",img_name="", ):
    """
    :param image_file: the name of image file
    :param label_file: the name of label file
    :return: train_files, val_files. Now we separate them directly. It needs to be modified for 5-fold validation.
    """
    # use json load image name list form img_name, which is a json file
    with open(img_name, 'r') as f:
        name_list = json.load(f)
    image_path = pathlib.Path(image_file)
    label_path = pathlib.Path(label_file)
    assert os.path.isdir(image_path), f"img path not exist: {image_path}"
    assert os.path.isdir(label_path), f"label path not exist: {label_path}"
    data_dicts = [
        {"image": str(pathlib.Path(image_path, name)), "label": str(pathlib.Path(label_path, name))}
        for name in name_list
    ]
    return data_dicts


def prepare_multi_datalist_with_file(main_file, auxiliary_file, label_file, broken_file, broken_gt_file, img_name=None, select_file=None):
    """
    :param main_file:  the root of data file.
    :param auxiliary_file: the name of image file
    :param label_file: the name of label file
    :param broken_file: the name of broken image file
    :param broken_gt_file: the name of broken gt file
    :param select_file: the name of selected file, contains the name of selected image and the number broken part
    :return: train_files, val_files. Now we separate them directly. It needs to be modified for 5-fold validation.
    """
    # use json load image name list form img_name, which is a json file
    if img_name is not None:
        with open(img_name, 'r') as f:
            name_list = json.load(f)
    else:
        name_list = pathlib.Path(main_file).glob("*.nii.gz")
        name_list = [path.name for path in name_list]
        # sort the name list
        name_list.sort()
    if select_file is not None:
        with open(select_file, 'r') as f:
            select_list = json.load(f)
        select_list = [path["name"] for path in select_list if path["num"] > 0]
        name_list = [name for name in name_list if name.replace(".nii.gz", "") in select_list]

    print(f"num:{len(name_list)}")
    main_path = pathlib.Path(main_file)
    auxiliary_path = pathlib.Path(auxiliary_file)

    label_path = pathlib.Path(label_file)
    assert os.path.isdir(main_path), f"img path not exist: {main_path}"
    assert os.path.isdir(label_path), f"label path not exist: {label_path}"
    assert os.path.isdir(auxiliary_path), f"img path not exist: {auxiliary_path}"
    assert os.path.isdir(broken_file), f"img path not exist: {broken_file}"
    assert os.path.isdir(broken_gt_file), f"img path not exist: {broken_gt_file}"
    # because we do not have full data
    data_dicts = [
        {"main": str(pathlib.Path(main_path, name)),
         "auxiliary": str(pathlib.Path(auxiliary_path, name)),
         "label": str(pathlib.Path(label_path, name)),
         "broken": str(pathlib.Path(broken_file, name)),
         "broken_gt": str(pathlib.Path(broken_gt_file, name)),}
        for name in name_list if pathlib.Path(main_path, name).exists() and pathlib.Path(auxiliary_path, name).exists()
    ]
    return data_dicts


def prepare_main_auxiliary_with_img_datalist_with_file(main_file,
                                                       auxiliary_file,
                                                       main_img_file,
                                                       auxiliary_img_file,
                                                       broken_file,
                                                       broken_gt_file,
                                                       label_file=None,
                                                       img_name="",
                                                       select_file=None):
    """
    :param main_file: the path of coarse segmentation in main phase.
    :param auxiliary_file: the path of refined segmentation in auxiliary phase.
    :param main_img_file: the path of image in main phase.
    :param auxiliary_img_file: the path of image in auxiliary phase.
    :param broken_file: the path of broken spheres in main phase.
    :param broken_gt_file: the path of broken spheres generated with gt in main phase.
    :param label_file: the path of label file.
    :param img_name: the name of image file.
    :param select_file: record the number of discontinuity sphere.
    """
    # use json load image name list form img_name, which is a json file
    if img_name is not None:
        with open(img_name, 'r') as f:
            name_list = json.load(f)
    else:
        name_list = pathlib.Path(main_file).glob("*.nii.gz")
        name_list = [path.name for path in name_list]
        # sort the name list
        name_list.sort()

    if select_file is not None:
        with open(select_file, 'r') as f:
            select_list = json.load(f)
        select_list = [path["name"] for path in select_list if path["num"] > 0]
        name_list = [name for name in name_list if name.replace(".nii.gz", "") in select_list]

    main_path = pathlib.Path(main_file)
    auxiliary_path = pathlib.Path(auxiliary_file)
    main_image_path = pathlib.Path(main_img_file)
    auxiliary_image_path = pathlib.Path(auxiliary_img_file)
    assert os.path.isdir(main_path), f"main_path not exist: {main_path}"
    assert os.path.isdir(main_image_path), f"main_image_path not exist: {main_image_path}"
    assert os.path.isdir(auxiliary_image_path), f"auxiliary_image_path not exist: {auxiliary_image_path}"
    assert os.path.isdir(auxiliary_path), f"auxiliary_path not exist: {auxiliary_path}"
    assert os.path.isdir(broken_file), f"broken_file not exist: {broken_file}"
    assert os.path.isdir(broken_gt_file), f"broken_gt_file not exist: {broken_gt_file}"
    if label_file is None:
        data_dicts = [
            {"CS_M": str(pathlib.Path(main_path, name)),
             "CS_A": str(pathlib.Path(auxiliary_path, name)),
             "I_M": str(pathlib.Path(main_image_path, name)),
             "I_A": str(pathlib.Path(auxiliary_image_path, name)),
             "CS_DL": str(pathlib.Path(broken_file, name)),
             "CS_DLGT": str(pathlib.Path(broken_gt_file, name)), }
            for name in name_list
        ]
        return data_dicts
    else:
        label_path = pathlib.Path(label_file)
        assert os.path.isdir(label_path), f"label path not exist: {label_path}"
        data_dicts = [
            {"CS_M": str(pathlib.Path(main_path, name)),
             "CS_A": str(pathlib.Path(auxiliary_path, name)),
             "I_M": str(pathlib.Path(main_image_path, name)),
             "I_A": str(pathlib.Path(auxiliary_image_path, name)),
             "label": str(pathlib.Path(label_path, name)),
             "CS_DL": str(pathlib.Path(broken_file, name)),
             "CS_DLGT": str(pathlib.Path(broken_gt_file, name)), }
            for name in name_list
        ]
        return data_dicts


def prepare_main_with_img_datalist_with_file(main_file,
                                             main_img_file,
                                             broken_file,
                                             broken_gt_file,
                                             label_file=None,
                                             img_name="",
                                             select_file=None):
    """
    :param main_file: the path of coarse segmentation in main phase.
    :param auxiliary_file: the path of refined segmentation in auxiliary phase.
    :param main_img_file: the path of image in main phase.
    :param auxiliary_img_file: the path of image in auxiliary phase.
    :param broken_file: the path of broken spheres in main phase.
    :param broken_gt_file: the path of broken spheres generated with gt in main phase.
    :param label_file: the path of label file.
    :param img_name: the name of image file.
    :param select_file: record the number of discontinuity sphere.
    """
    # use json load image name list form img_name, which is a json file
    if img_name is not None:
        with open(img_name, 'r') as f:
            name_list = json.load(f)
    else:
        name_list = pathlib.Path(main_file).glob("*.nii.gz")
        name_list = [path.name for path in name_list]
        # sort the name list
        name_list.sort()

    if select_file is not None:
        with open(select_file, 'r') as f:
            select_list = json.load(f)
        select_list = [path["name"] for path in select_list if path["num"] > 0]
        name_list = [name for name in name_list if name.replace(".nii.gz", "") in select_list]

    main_path = pathlib.Path(main_file)
    main_image_path = pathlib.Path(main_img_file)
    assert os.path.isdir(main_path), f"main_path not exist: {main_path}"
    assert os.path.isdir(main_image_path), f"main_image_path not exist: {main_image_path}"
    assert os.path.isdir(broken_file), f"broken_file not exist: {broken_file}"
    assert os.path.isdir(broken_gt_file), f"broken_gt_file not exist: {broken_gt_file}"
    if label_file is None:
        data_dicts = [
            {"CS_M": str(pathlib.Path(main_path, name)),
             "I_M": str(pathlib.Path(main_image_path, name)),
             "CS_DL": str(pathlib.Path(broken_file, name)),
             "CS_DLGT": str(pathlib.Path(broken_gt_file, name)), }
            for name in name_list
        ]
        return data_dicts
    else:
        label_path = pathlib.Path(label_file)
        assert os.path.isdir(label_path), f"label path not exist: {label_path}"
        data_dicts = [
            {"CS_M": str(pathlib.Path(main_path, name)),
             "I_M": str(pathlib.Path(main_image_path, name)),
             "label": str(pathlib.Path(label_path, name)),
             "CS_DL": str(pathlib.Path(broken_file, name)),
             "CS_DLGT": str(pathlib.Path(broken_gt_file, name)), }
            for name in name_list
        ]
        return data_dicts



def index_generator(length, split_mode='shuffle'):
    """
    :length: list length
    """
    train_index = []
    val_index = []
    index = list(range(length))
    if split_mode == 'shuffle':
        random.shuffle(index)
        train_index.append(index[:int(0.8 * length)])

        val_index.append(index[int(0.8 * length):])

    elif split_mode == 'five_fold':
        # random.shuffle(index)
        d = length // 5
        for i in range(5):
            train_index.append(index[:(4 - i) * d] + index[(5 - i) * d:] if i > 0 else index[:(4 - i) * d])
            val_index.append(index[(4 - i) * d:(5 - i) * d] if i > 0 else index[(4 - i) * d:])

    elif split_mode == 'order':
        train_index.append(index[:int(0.8 * length)])
        val_index.append(index[int(0.8 * length):])
    else:
        raise RuntimeError(f"{split_mode} is not supported")
    return train_index, val_index


def prepare_datalist_with_heart_label(data_dir, image_file="images", label_file="label", heart_file="heart"):
    """
    :param data_dir:  the root of data file.
    :param image_file: the name of image file
    :param label_file: the name of label file
    :return: train_files, val_files. Now we separate them directly. It needs to be modified for 5-fold validation.
    """

    image_path = os.path.join(data_dir, image_file)
    label_path = os.path.join(data_dir, label_file)
    heart_path = os.path.join(data_dir, heart_file)
    assert os.path.isdir(image_path), "img path not exist"
    assert os.path.isdir(label_path), "label path not exist"
    assert os.path.isdir(heart_path), "label path not exist"

    train_images = sorted(glob.glob(os.path.join(image_path, "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(label_path, "*.nii.gz")))
    train_heart = sorted(glob.glob(os.path.join(heart_path, "*.nii.gz")))

    data_dicts = [
        {"image": image_name, "label": label_name, "heart": heart_seg}
        for image_name, label_name, heart_seg in zip(train_images, train_labels, train_heart)
    ]
    return data_dicts


def write_data_reference(L, save_path):
    "a list contain val_files and train_files"
    with open(os.path.join(save_path, 'train_set.txt'), 'w') as f:
        for dic in L['train_files']:
            for key, file in dic.items():
                f.write(os.path.split(file)[-1] + '\n')
                break

    with open(os.path.join(save_path, 'val_set.txt'), 'w') as f:
        for dic in L['val_files']:
            for key, file in dic.items():
                f.write(os.path.split(file)[-1] + '\n')
                break


def save_json(L, save_path):
    with open(save_path, 'w') as f:
        json.dump(L, f)


def load_json(path):
    with open(path, 'r') as f:
        x = json.load(f)
    return x


def prepare_image_list(image_path):
    """
    generate image list in a dir
    """
    assert os.path.isdir(image_path), "img path not exist"
    train_images = sorted(glob.glob(os.path.join(image_path, "*.nii.gz")))
    data_dicts = [
        {"image": image_name}
        for image_name in train_images
    ]
    return data_dicts


if __name__ == '__main__':
    L = list(range(10))
    print(shuffle_generator(L))

