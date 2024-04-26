import torch
import matplotlib.pyplot as plt
from monai.inferers import sliding_window_inference
import os
import SimpleITK as sitk
from monai.data import Dataset, DataLoader, CacheDataset, decollate_batch, MetaTensor
import numpy as np
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    SaveImaged,
    Invertd,
    Invert,
)


def mirror_sliding_window_inference(inputs, roi_size, sw_batch_size, predictor, mirror_axes=(0, 1, 2),
                                    mode="gaussian", overlap=0.25):
    """
    inputs: torch tensor
    roi_size: tuple
    sw_batch_size: int
    model: model
    mirror: sequence of int, flip on the axis
    """
    prediction = sliding_window_inference(inputs=inputs, roi_size=roi_size, sw_batch_size=sw_batch_size,
                                          predictor=predictor, overlap=overlap)
    if mirror_axes is not None and len(mirror_axes)>0:
        num_predictons = 2 ** len(mirror_axes)
    else:
        num_predictons = 1
    if mirror_axes is not None:
        if 0 in mirror_axes:
            x = torch.flip(inputs, (2,))
            prediction += torch.flip(sliding_window_inference(inputs=x, roi_size=roi_size, sw_batch_size=sw_batch_size,
                                                              predictor=predictor, mode=mode, overlap=overlap), (2,))
        if 1 in mirror_axes:
            x = torch.flip(inputs, (3,))
            prediction += torch.flip(sliding_window_inference(inputs=x, roi_size=roi_size, sw_batch_size=sw_batch_size,
                                                              predictor=predictor, mode=mode, overlap=overlap), (3,))
        if 2 in mirror_axes:
            x = torch.flip(inputs, (4,))
            prediction += torch.flip(sliding_window_inference(inputs=x, roi_size=roi_size, sw_batch_size=sw_batch_size,
                                                              predictor=predictor, mode=mode, overlap=overlap), (4,))
        if 0 in mirror_axes and 1 in mirror_axes:
            x = torch.flip(inputs, (2, 3))
            prediction += torch.flip(sliding_window_inference(inputs=x, roi_size=roi_size, sw_batch_size=sw_batch_size,
                                                              predictor=predictor, mode=mode, overlap=overlap), (2, 3))
        if 0 in mirror_axes and 2 in mirror_axes:
            x = torch.flip(inputs, (2, 4))
            prediction += torch.flip(sliding_window_inference(inputs=x, roi_size=roi_size, sw_batch_size=sw_batch_size,
                                                              predictor=predictor, mode=mode, overlap=overlap), (2, 4))
        if 1 in mirror_axes and 2 in mirror_axes:
            x = torch.flip(inputs, (3, 4))
            prediction += torch.flip(sliding_window_inference(inputs=x, roi_size=roi_size, sw_batch_size=sw_batch_size,
                                                              predictor=predictor, mode=mode, overlap=overlap), (3, 4))
        if 0 in mirror_axes and 1 in mirror_axes and 2 in mirror_axes:
            x = torch.flip(inputs, (2, 3, 4))
            prediction += torch.flip(sliding_window_inference(inputs=x, roi_size=roi_size, sw_batch_size=sw_batch_size,
                                                              predictor=predictor, mode=mode, overlap=overlap), (2, 3, 4))
    prediction /= num_predictons
    return prediction


def mirror_predictor(network, x, mirror_axes):
    prediction = network(x)
    if mirror_axes is not None:
        # check for invalid numbers in mirror_axes
        # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
        assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'

        num_predictons = 2 ** len(mirror_axes)
        if 0 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (2,))), (2,))
        if 1 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (3,))), (3,))
        if 2 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (4,))), (4,))
        if 0 in mirror_axes and 1 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (2, 3))), (2, 3))
        if 0 in mirror_axes and 2 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (2, 4))), (2, 4))
        if 1 in mirror_axes and 2 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (3, 4))), (3, 4))
        if 0 in mirror_axes and 1 in mirror_axes and 2 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (2, 3, 4))), (2, 3, 4))
        prediction /= num_predictons
    return prediction


def cardio_vessel_segmentation_infer(model, val_dataset, device, output_path, window_size,
                                     origin_transforms=None, mirror_axes=(0, 1, 2), overlap=0.25,
                                     sw_batch_size=4, mode="gaussian", post_transforms_device="cpu"):
    """
    only save image in output path
    window size: tuple
    """
    if post_transforms_device is None:
        post_transforms_device = device
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=4)
    model.to(device)
    model.eval()
    post_transforms = Compose([
        Invertd(
            keys="pred",
            transform=origin_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=True, to_onehot=None),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=output_path, output_postfix="", resample=False,
                   separate_folder=False, print_log=True, output_dtype=np.uint16),
    ])
    with torch.no_grad():
        for i, val_data in enumerate(val_loader):
            roi_size = window_size
            val_inputs = val_data["image"].to(device)
            val_data["pred"] = mirror_sliding_window_inference(inputs=val_inputs,
                                                               roi_size=roi_size,
                                                               sw_batch_size=sw_batch_size,
                                                               predictor=model,
                                                               mirror_axes=mirror_axes,
                                                               overlap=overlap,
                                                               mode=mode).to(post_transforms_device)
            for one_data in decollate_batch(val_data):
                post_transforms(one_data)


def cardio_vessel_segmentation_multi_phase_with_image_infer(model, key, val_dataset, device, output_path, window_size,
                                                            origin_transforms=None, mirror_axes=(0, 1, 2), sw_batch_size=4,
                                                            mode="gaussian", post_transforms_device="cpu", overlap=0.25):
    """
    only save image in output path
    window size: tuple or list
    model: model
    key: which information to use, list or tuple
    val_dataset: dataset
    device: device, cpu or gpu
    output_path: the path save infer result
    """
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=4)
    model.to(device)
    model.eval()
    post_transforms = Compose([
        Invertd(
            keys="pred",
            transform=origin_transforms,
            orig_keys="I_M",
            meta_keys="pred_meta_dict",
            orig_meta_keys="I_M_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=True, to_onehot=None),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=output_path, output_postfix="", resample=False,
                   separate_folder=False, print_log=True, output_dtype=np.uint8),
    ])
    with torch.no_grad():
        print(key)
        for i, val_data in enumerate(val_loader):
            roi_size = window_size
            # concat the input by key
            cs_ms = torch.zeros((val_data[key[0]].shape[0], len(key), val_data[key[0]].shape[2],
                                 val_data[key[0]].shape[3], val_data[key[0]].shape[4]), device=device)
            for j in range(len(key)):
                cs_ms[:, j, :, :, :] = val_data[key[j]]
            val_data["pred"] = mirror_sliding_window_inference(inputs=cs_ms, roi_size=roi_size,
                                                               sw_batch_size=sw_batch_size, predictor=model, overlap=overlap,
                                                               mirror_axes=mirror_axes, mode=mode).to(post_transforms_device)
            if not isinstance(val_data["pred"], MetaTensor):
                val_data["pred"] = MetaTensor(val_data["pred"], meta=val_data["I_M"].meta)
            for one_data in decollate_batch(val_data):
                post_transforms(one_data)


