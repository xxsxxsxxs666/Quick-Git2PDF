import numpy as np
import torch
from monai.utils import TransformBackends
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Union
from monai.config import IndexSelection, KeysCollection, SequenceStr, NdarrayOrTensor, DtypeLike, KeysCollection
from monai.data.meta_obj import get_track_meta
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type, convert_to_tensor, get_equivalent_dtype
from monai.transforms.utils_pytorch_numpy_unification import clip
from monai.data.meta_tensor import MetaTensor
from monai.metrics.utils import get_surface_distance, get_mask_edges


from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    RandAffined,
    MapTransform,
    RandFlipd,
    RandRotated,
)
from openpyxl import Workbook, load_workbook
import os
from scipy import ndimage
import numpy as np

from transform.IntensityTransformD import CTNormalizeD, BrightnessMultiplicativeD, ContrastAugmentationD, \
    SimulateLowResolutionD, GammaD
from transform.NoiseTransformD import GaussianNoiseD, GaussianBlurD


class UseHeartsegDeleteInformationd(MapTransform):

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        keys: KeysCollection,
        dilation_key: str,
        dilation_time: int = 1,
        dilation_struct: int = 1,
        test_key: str = None,
        caculate_surface_disdance: bool = False,
        vessel_key: str = None,
        chamber_key: str = None,
        save_path: str = None,
        allow_missing_keys: bool = False,
        **pad_kwargs,
    ) -> None:
        """
        test_key: show whether your mask is reserved totally.
        """
        self.dilation_key = dilation_key
        self.dilation_time = dilation_time
        self.dilation_struct = dilation_struct
        self.test_key = test_key
        self.caculate_surface_disdance = caculate_surface_disdance
        self.vessel_key = vessel_key
        self.chamber_key = chamber_key
        self.save_path = save_path
        self.error_list = []
        self.wb = None

        if self.caculate_surface_disdance:
            row = ('image_name', 'surface disdance', 'error')
            self.wb = Workbook()
            self.excel_writer(row=row, file_name=self.save_path)

        super().__init__(keys, allow_missing_keys=allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        region = convert_to_tensor(data=d[self.dilation_key])
        if self.dilation_time > 1:
            region = region.clip(min=0, max=1) > 0
            region = self.dilation(x=region)  # bool
            distance = -1
            error = 0
            for key in self.key_iterator(d):
                if self.caculate_surface_disdance:
                    vessel_edge, chamber_edge = get_mask_edges(d[self.vessel_key], d[self.chamber_key] > 0)
                    distance = get_surface_distance(vessel_edge, chamber_edge)
                    distance = distance.max()

                if key == self.dilation_key:
                    d[key] = MetaTensor(region, meta=d[key].meta)
                else:
                    # record the img information if label has some voxels out of dilation region
                    if key == self.test_key and (d[key] * (~region)).sum() > 0:
                        self.error_list.append(d[key].meta['filename_or_obj'])
                        error = 1
                        print("---------------error crop----------------")
                        print(d[key].meta['filename_or_obj'])

                    d[key] = d[key] * region

            # record image dilation information in Excel
            if self.wb:
                row = (os.path.split(d[self.test_key].meta['filename_or_obj'])[-1], distance, error)
                self.excel_writer(row=row, file_name=self.save_path)


        else:
            # if you choose to set dilation before training process
            region = region.clip(min=0, max=1) > 0
            for key in self.key_iterator(d):
                if key != self.dilation_key:
                    d[key] = d[key] * region
        return d

    def excel_writer(self, row, file_name):
        ws = self.wb.active
        ws.append(row)
        self.wb.save(file_name)

    def dilation(self, x: torch.Tensor):
        shape = x.shape
        if len(shape) > 3:
            x = x.squeeze()
        x = np.array(x.squeeze())
        struct1 = ndimage.generate_binary_structure(3, self.dilation_struct)
        x = ndimage.binary_dilation(x, structure=struct1, iterations=self.dilation_time).astype(x.dtype)
        if len(shape) > 3:
            return torch.tensor(x).unsqueeze(dim=0)
        else:
            return torch.tensor(x)


class Dilationd(MapTransform):

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        keys: KeysCollection,
        dilation_time: int = 1,
        dilation_struct: int = 1,
        allow_missing_keys: bool = False,
        **pad_kwargs,
    ) -> None:
        """
        test_key: show whether your mask is reserved totally.
        """
        self.dilation_time = dilation_time
        self.dilation_struct = dilation_struct
        super().__init__(keys, allow_missing_keys=allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.keys:
            region = convert_to_tensor(data=d[key])
            region = self.dilation(region)
            d[key] = MetaTensor(region, meta=d[key].meta)
        return d

    def dilation(self, x: torch.Tensor):
        shape = x.shape
        if len(shape) > 3:
            x = x.squeeze()
        x = np.array(x.squeeze())
        struct1 = ndimage.generate_binary_structure(3, self.dilation_struct)
        x = ndimage.binary_dilation(x, structure=struct1, iterations=self.dilation_time).astype(x.dtype)
        if len(shape) > 3:
            return torch.tensor(x).unsqueeze(dim=0)
        else:
            return torch.tensor(x)


def get_transform(transform_dic, mode='train'):
    """dic is transform block of config.yaml"""
    """
    only patch_size and normalization is controlled by config.yaml
    """
    if transform_dic.get('use_config') is True:
        return get_config_transform(transform_dic, mode=mode)
    else:
        return get_default_transform(transform_dic, mode=mode)


def get_default_transform(transform_dic, mode='train'):
    """dic is transform block of config.yaml"""
    """
    only patch_size and normalization is controlled by config.yaml
    """
    if mode == 'train':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"], image_only=False),
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-57, a_max=400,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CTNormalizeD(keys=["image"],
                             mean_intensity=transform_dic["normalize"]["mean"],
                             std_intensity=transform_dic["normalize"]["std"],
                             lower_bound=transform_dic["normalize"]["min"],
                             upper_bound=transform_dic["normalize"]["max"], ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                Orientationd(keys=["image", "label"], axcodes="RAI"),
                # 记得改
                Spacingd(keys=["image", "label"], pixdim=transform_dic["spacing"]
                         , mode=("bilinear", "nearest")),
                RandRotated(
                    keys=["image", "label"],
                    range_x=np.pi / 6,
                    range_y=np.pi / 6,
                    range_z=np.pi / 6,
                    prob=1,
                    padding_mode="zeros",
                    mode=("bilinear", "nearest")
                ),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=tuple(transform_dic["patch_size"]),
                    pos=3,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                GaussianNoiseD(keys=["image"], noise_variance=(0, 0.1), prob=0.1,),
                GaussianBlurD(keys=["image"], blur_sigma=(0.5, 1.15), prob=0.1,),
                BrightnessMultiplicativeD(keys=["image"], prob=0.15, multiplier_range=(0.7, 1.25)),
                ContrastAugmentationD(keys=["image"], prob=0.15),
                SimulateLowResolutionD(keys=["image"], zoom_range=(0.5, 1), prob=0.20,
                                       order_upsample=3, order_downsample=0, ignore_axes=None),
                GammaD(keys=["image"], gamma_range=(0.7, 1.5), invert_image=True,
                       per_channel=True, retain_stats=True, prob=0.1),
                GammaD(keys=["image"], gamma_range=(0.7, 1.5), invert_image=True,
                       per_channel=True, retain_stats=True, prob=0.3),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"], image_only=False),
                EnsureChannelFirstd(keys=["image", "label"]),
                # ScaleIntensityRanged(
                #     keys=["image"], a_min=-57, a_max=400,
                #     b_min=0.0, b_max=1.0, clip=True,
                # ),
                CTNormalizeD(keys=["image"],
                             mean_intensity=transform_dic["normalize"]["mean"],
                             std_intensity=transform_dic["normalize"]["std"],
                             lower_bound=transform_dic["normalize"]["min"],
                             upper_bound=transform_dic["normalize"]["max"], ),
                CropForegroundd(keys=["image", "label"], source_key="image"),  # default: value>0
                Orientationd(keys=["image", "label"], axcodes="RAI"),
                Spacingd(keys=["image", "label"], pixdim=transform_dic["spacing"]
                         , mode=("bilinear", "nearest")),  # Now only Sequential_str, How to add spline
            ]
        )
        save_transform = Compose(
            [
                LoadImaged(keys=["image", "label"], image_only=False),
                EnsureChannelFirstd(keys=["image", "label"]),
                SaveImaged(keys=["image"], output_dir='./transform/test_transform', output_postfix='origin_image',
                           print_log=False),
                SaveImaged(keys=["label"], output_dir='./transform/test_transform', output_postfix='origin_label',
                           print_log=False),
                CTNormalizeD(keys=["image"],
                             mean_intensity=transform_dic["normalize"]["mean"],
                             std_intensity=transform_dic["normalize"]["std"],
                             lower_bound=transform_dic["normalize"]["min"],
                             upper_bound=transform_dic["normalize"]["max"], ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                Orientationd(keys=["image", "label"], axcodes="RAI"),
                Spacingd(keys=["image", "label"], pixdim=transform_dic["spacing"]
                         , mode=("bilinear", "nearest")),
                GaussianNoiseD(keys=["image"], noise_variance=(0, 0.1), prob=1.0,),
                GaussianBlurD(keys=["image"], blur_sigma=(0.5, 1.15), prob=1.0,),
                SaveImaged(keys=["image"], output_dir='./transform/test_transform', output_postfix='tran_image',
                           print_log=False),
                SaveImaged(keys=["label"], output_dir='./transform/test_transform', output_postfix='tran_label',
                           print_log=False),
            ]
        )
        return train_transforms, val_transforms, save_transform
    elif mode == 'infer':
        infer_transforms = Compose(
            [
                LoadImaged(keys=["image"], image_only=False),
                EnsureChannelFirstd(keys=["image"]),
                # ScaleIntensityRanged(
                #     keys=["image"], a_min=-57, a_max=400,
                #     b_min=0.0, b_max=1.0, clip=True,
                # ),
                CTNormalizeD(keys=["image"],
                             mean_intensity=transform_dic["normalize"]["mean"],
                             std_intensity=transform_dic["normalize"]["std"],
                             lower_bound=transform_dic["normalize"]["min"],
                             upper_bound=transform_dic["normalize"]["max"], ),
                # CropForegroundd(keys=["image"], source_key="image"),  # default: value>0
                Orientationd(keys=["image"], axcodes="RAI"),
                Spacingd(keys=["image"], pixdim=transform_dic["spacing"], mode=("bilinear")),  # Now only Sequential_str, How to add spline
            ]
        )
        return infer_transforms
    else:
        raise RuntimeError(f"{mode} is not supported yet")


def get_config_transform(transform_dic, mode='train'):
    """dic is transform block of config.yaml"""
    """
    you can further control your random transformation 
    """
    if transform_dic.get("image_resample") is not None:
        image_resample_mode = transform_dic["image_resample"]["mode"]
        image_resample_padding_mode = transform_dic["image_resample"]["padding_mode"]
    else:
        image_resample_mode = "bilinear"
        image_resample_padding_mode = "border"

    if mode == 'train':
        fixed_transforms = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            CTNormalizeD(keys=["image"],
                         mean_intensity=transform_dic["normalize"]["mean"],
                         std_intensity=transform_dic["normalize"]["std"],
                         lower_bound=transform_dic["normalize"]["min"],
                         upper_bound=transform_dic["normalize"]["max"], ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAI"),
            # 记得改
            Spacingd(keys=["image", "label"], pixdim=transform_dic["spacing"]
                     , mode=(image_resample_mode, "nearest"), padding_mode=(image_resample_padding_mode, "border")),
        ]
        print("----------------------training_fixed_transform-------------------------")
        if transform_dic.get("fixed") is not None and len(transform_dic.get("fixed")) > 0:
            for d in transform_dic["fixed"]:
                key = d.get('name')
                value = d.get('parameter')
                assert key is not None, "name of fixed transform is not defined in config.yaml"
                assert value is not None, "parameter of fixed transform is not defined in config.yaml"
                print(f"name: {key}, parameter: {value}")
                trans_class = eval(key)
                trans_module = trans_class(**value)
                fixed_transforms.append(trans_module)

        random_transforms = []
        # assert transform_dic.get("random") is not None, "random transform is not defined in config.yaml"
        if transform_dic.get("random") is None or len(transform_dic.get("random")) == 0:
            print("warning: random transform is not defined in config.yaml")
        else:
            print("----------------------training_random_transform-------------------------")
            for d in transform_dic["random"]:
                key = d.get('name')
                value = d.get('parameter')
                assert key is not None, "name of random transform is not defined in config.yaml"
                assert value is not None, "parameter of random transform is not defined in config.yaml"
                print(f"name: {key}, parameter: {value}")
                trans_class = eval(key)
                trans_module = trans_class(**value)
                random_transforms.append(trans_module)

        train_transforms = Compose(fixed_transforms + random_transforms)
        val_transforms = Compose(fixed_transforms)
        save_transform = None
        return train_transforms, val_transforms, save_transform
    elif mode == 'infer':
        infer_transforms = [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            CTNormalizeD(keys=["image"],
                         mean_intensity=transform_dic["normalize"]["mean"],
                         std_intensity=transform_dic["normalize"]["std"],
                         lower_bound=transform_dic["normalize"]["min"],
                         upper_bound=transform_dic["normalize"]["max"], ),
            CropForegroundd(keys=["image"], source_key="image"),  # default: value>0
            Orientationd(keys=["image"], axcodes="RAI"),
            Spacingd(keys=["image"], pixdim=transform_dic["spacing"], mode=image_resample_mode,
                     padding_mode=image_resample_padding_mode),
            ]
        if transform_dic.get("infer") is not None and len(transform_dic.get("infer")) > 0:
            for d in transform_dic["infer"]:
                key = d.get('name')
                value = d.get('parameter')
                assert key is not None, "name of infer transform is not defined in config.yaml"
                assert value is not None, "parameter of infer transform is not defined in config.yaml"
                trans_class = eval(key)
                trans_module = trans_class(**value)
                infer_transforms.append(trans_module)

        infer_transforms = Compose(infer_transforms)
        return infer_transforms
    else:
        raise RuntimeError(f"{mode} is not supported yet")


def print_config_transforms(transforms_list):
    import inspect
    for transforms in transforms_list:
        print(f"name: {inspect.getmodule(transforms).__name__}, parameter: {inspect.getmembers(transforms)}")


def get_multi_phase_transform_with_image(transform_dic, mode='train'):
    """dic is transform block of config.yaml"""
    """
    only patch_size and normalization is controlled by config.yaml
    """
    if transform_dic.get('use_config') is True:
        return get_multi_phase_transform_with_image_config(transform_dic, mode=mode)
    else:
        return get_multi_phase_transform_with_image_default(transform_dic, mode=mode)


def get_multi_phase_transform_with_image_default(transform_dic, mode='train'):
    """dic is transform block of config.yaml"""
    if mode == 'train':
        train_transforms = Compose(
            [
                LoadImaged(keys=["CS_M", "label", "CS_A", "I_M", "I_A", "CS_DL", "CS_DLGT"]),
                EnsureChannelFirstd(keys=["CS_M", "label", "CS_A", "I_M", "I_A", "CS_DL", "CS_DLGT"]),
                Orientationd(keys=["CS_M", "label", "CS_A", "I_M", "I_A", "CS_DL", "CS_DLGT"], axcodes="RAI"),

                CTNormalizeD(keys=["I_M", "I_A"],
                             mean_intensity=transform_dic["normalize"]["mean"],
                             std_intensity=transform_dic["normalize"]["std"],
                             lower_bound=transform_dic["normalize"]["min"],
                             upper_bound=transform_dic["normalize"]["max"], ),
                Spacingd(keys=["CS_M", "label", "CS_A", "I_M", "I_A", "CS_DL", "CS_DLGT"], pixdim=transform_dic['spacing']
                         , mode=("nearest", "nearest", "nearest", "bilinear", "bilinear", "nearest", "nearest")),
                RandCropByPosNegLabeld(
                    keys=["CS_M", "label", "CS_A", "I_M", "I_A", "CS_DL", "CS_DLGT"],
                    label_key="CS_DLGT",
                    spatial_size=tuple(transform_dic['patch_size']),
                    pos=1,
                    neg=0,
                    num_samples=2,
                    image_key="I_M",
                    image_threshold=0,
                    allow_smaller=True,
                ),
                # RandFlipd(keys=["CS_M", "label", "CS_A", "I_M", "I_A", "CS_DL", "CS_DLGT"], prob=0.5, spatial_axis=0),
                # RandFlipd(keys=["CS_M", "label", "CS_A", "I_M", "I_A", "CS_DL", "CS_DLGT"], prob=0.5, spatial_axis=1),
                # RandFlipd(keys=["CS_M", "label", "CS_A", "I_M", "I_A", "CS_DL", "CS_DLGT"], prob=0.5, spatial_axis=2),
                # # user can also add other random transforms
                # RandAffined(
                #     keys=["CS_M", "label", "CS_A", "I_M", "I_A", "CS_DL", "CS_DLGT"],
                #     mode=('nearest', 'nearest', 'nearest', 'bilinear', "bilinear", "nearest", "nearest"),
                #     prob=0.5,
                #     rotate_range=(0, 0, np.pi / 15),
                #     scale_range=(0.1, 0.1, 0.1),
                #     padding_mode="zeros"
                # ),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["CS_M", "label", "CS_A", "I_M", "I_A", "CS_DL", "CS_DLGT"]),
                EnsureChannelFirstd(keys=["CS_M", "label", "CS_A", "I_M", "I_A", "CS_DL", "CS_DLGT"]),
                Orientationd(keys=["CS_M", "label", "CS_A", "I_M", "I_A", "CS_DL", "CS_DLGT"], axcodes="RAI"),
                CTNormalizeD(keys=["I_M", "I_A"],
                             mean_intensity=transform_dic["normalize"]["mean"],
                             std_intensity=transform_dic["normalize"]["std"],
                             lower_bound=transform_dic["normalize"]["min"],
                             upper_bound=transform_dic["normalize"]["max"], ),
                Spacingd(keys=["CS_M", "label", "CS_A", "I_M", "I_A", "CS_DL", "CS_DLGT"], pixdim=transform_dic["spacing"],
                         mode=("nearest", "nearest", "nearest", "bilinear", "bilinear", "nearest", "nearest")),
            ]
        )
        save_transform = Compose(
            [
            ]
        )
        return train_transforms, val_transforms, save_transform
    elif mode == 'infer':
        infer_transforms = Compose(
            [
                LoadImaged(keys=["CS_M", "CS_A", "I_M", "I_A", "CS_DL", "CS_DLGT"]),
                EnsureChannelFirstd(keys=["CS_M", "CS_A", "I_M", "I_A", "CS_DL", "CS_DLGT"]),
                Orientationd(keys=["CS_M", "CS_A", "I_M", "I_A", "CS_DL", "CS_DLGT"], axcodes="RAI"),
                CTNormalizeD(keys=["I_M", "I_A"],
                             mean_intensity=transform_dic["normalize"]["mean"],
                             std_intensity=transform_dic["normalize"]["std"],
                             lower_bound=transform_dic["normalize"]["min"],
                             upper_bound=transform_dic["normalize"]["max"], ),
                Spacingd(keys=["CS_M", "CS_A", "I_M", "I_A", "CS_DL", "CS_DLGT"], pixdim=transform_dic["spacing"],
                         mode=("nearest", "nearest", "bilinear", "bilinear", "nearest", "nearest")),
            ]
        )
        return infer_transforms
    else:
        raise RuntimeError(f"{mode} is not supported yet")


def get_multi_phase_transform_with_image_config(transform_dic, mode='train'):
    """dic is transform block of config.yaml"""
    if transform_dic.get("image_resample") is not None:
        image_resample_mode = transform_dic["image_resample"]["mode"]
        image_resample_padding_mode = transform_dic["image_resample"]["padding_mode"]
    else:
        image_resample_mode = "bilinear"
        image_resample_padding_mode = "border"
    if mode == 'train':
        fixed_transforms = [
                LoadImaged(keys=["CS_M", "label", "CS_A", "I_M", "I_A", "CS_DL", "CS_DLGT"]),
                EnsureChannelFirstd(keys=["CS_M", "label", "CS_A", "I_M", "I_A", "CS_DL", "CS_DLGT"]),
                Orientationd(keys=["CS_M", "label", "CS_A", "I_M", "I_A", "CS_DL", "CS_DLGT"], axcodes="RAI"),
                CTNormalizeD(keys=["I_M", "I_A"],
                             mean_intensity=transform_dic["normalize"]["mean"],
                             std_intensity=transform_dic["normalize"]["std"],
                             lower_bound=transform_dic["normalize"]["min"],
                             upper_bound=transform_dic["normalize"]["max"], ),
                Spacingd(keys=["CS_M", "label", "CS_A", "I_M", "I_A", "CS_DL", "CS_DLGT"], pixdim=transform_dic['spacing']
                         , mode=("nearest", "nearest", "nearest", image_resample_mode, image_resample_mode, "nearest", "nearest")),
            ]
        print("----------------------second-stage fixed_transform-------------------------")
        if transform_dic.get("fixed") is not None and len(transform_dic.get("fixed")) > 0:
            for d in transform_dic["fixed"]:
                key = d.get('name')
                value = d.get('parameter')
                assert key is not None, "name of fixed transform is not defined in config.yaml"
                assert value is not None, "parameter of fixed transform is not defined in config.yaml"
                print(f"name: {key}, parameter: {value}")
                trans_class = eval(key)
                trans_module = trans_class(**value)
                fixed_transforms.append(trans_module)

        random_transforms = []
        # assert transform_dic.get("random") is not None, "random transform is not defined in config.yaml"
        if transform_dic.get("random") is None or len(transform_dic.get("random")) == 0:
            print("warning: random transform is not defined in config.yaml")
        else:
            print("----------------------second-stage random_transform-------------------------")
            for d in transform_dic["random"]:
                key = d.get('name')
                value = d.get('parameter')
                assert key is not None, "name of random transform is not defined in config.yaml"
                assert value is not None, "parameter of random transform is not defined in config.yaml"
                print(f"name: {key}, parameter: {value}")
                trans_class = eval(key)
                trans_module = trans_class(**value)
                random_transforms.append(trans_module)

        train_transforms = Compose(fixed_transforms + random_transforms)
        val_transforms = Compose(fixed_transforms)
        save_transform = None
        return train_transforms, val_transforms, save_transform
    elif mode == 'infer':
        infer_transforms = [
                LoadImaged(keys=["CS_M", "CS_A", "I_M", "I_A", "CS_DL", "CS_DLGT"]),
                EnsureChannelFirstd(keys=["CS_M", "CS_A", "I_M", "I_A", "CS_DL", "CS_DLGT"]),
                Orientationd(keys=["CS_M", "CS_A", "I_M", "I_A", "CS_DL", "CS_DLGT"], axcodes="RAI"),
                CTNormalizeD(keys=["I_M", "I_A"],
                             mean_intensity=transform_dic["normalize"]["mean"],
                             std_intensity=transform_dic["normalize"]["std"],
                             lower_bound=transform_dic["normalize"]["min"],
                             upper_bound=transform_dic["normalize"]["max"], ),
                Spacingd(keys=["CS_M", "CS_A", "I_M", "I_A", "CS_DL", "CS_DLGT"], pixdim=transform_dic["spacing"],
                         mode=("nearest", "nearest", image_resample_mode, image_resample_mode, "nearest", "nearest")),
            ]
        print("----------------------second-stage infer_transform-------------------------")
        if transform_dic.get("infer") is not None and len(transform_dic.get("infer")) > 0:
            for d in transform_dic["infer"]:
                key = d.get('name')
                value = d.get('parameter')
                assert key is not None, "name of infer transform is not defined in config.yaml"
                assert value is not None, "parameter of infer transform is not defined in config.yaml"
                print(f"name: {key}, parameter: {value}")
                trans_class = eval(key)
                trans_module = trans_class(**value)
                infer_transforms.append(trans_module)

        infer_transforms = Compose(infer_transforms)
        return infer_transforms
    else:
        raise RuntimeError(f"{mode} is not supported yet")


def get_second_stage_only_one_phase(transform_dic, mode='train'):
    """dic is transform block of config.yaml"""
    """
    only patch_size and normalization is controlled by config.yaml
    """
    if transform_dic.get('use_config') is True:
        return get_second_stage_only_one_phase_default(transform_dic, mode=mode)
    else:
        return get_second_stage_only_one_phase_config(transform_dic, mode=mode)


def get_second_stage_only_one_phase_default(transform_dic, mode='train'):
    """dic is transform block of config.yaml"""
    if mode == 'train':
        train_transforms = Compose(
            [
                LoadImaged(keys=["CS_M", "label", "I_M", "CS_DL", "CS_DLGT"]),
                EnsureChannelFirstd(keys=["CS_M", "label", "I_M", "CS_DL", "CS_DLGT"]),
                Orientationd(keys=["CS_M", "label", "I_M", "CS_DL", "CS_DLGT"], axcodes="RAI"),

                CTNormalizeD(keys=["I_M"],
                             mean_intensity=transform_dic["normalize"]["mean"],
                             std_intensity=transform_dic["normalize"]["std"],
                             lower_bound=transform_dic["normalize"]["min"],
                             upper_bound=transform_dic["normalize"]["max"], ),
                Spacingd(keys=["CS_M", "label", "I_M", "CS_DL", "CS_DLGT"], pixdim=transform_dic['spacing']
                         , mode=("nearest", "nearest", "bilinear", "nearest", "nearest")),
                RandCropByPosNegLabeld(
                    keys=["CS_M", "label", "I_M", "CS_DL", "CS_DLGT"],
                    label_key="CS_DLGT",
                    spatial_size=tuple(transform_dic['patch_size']),
                    pos=1,
                    neg=0,
                    num_samples=2,
                    image_key="I_M",
                    image_threshold=0,
                    allow_smaller=True,
                ),
                # RandFlipd(keys=["CS_M", "label", "CS_A", "I_M", "I_A", "CS_DL", "CS_DLGT"], prob=0.5, spatial_axis=0),
                # RandFlipd(keys=["CS_M", "label", "CS_A", "I_M", "I_A", "CS_DL", "CS_DLGT"], prob=0.5, spatial_axis=1),
                # RandFlipd(keys=["CS_M", "label", "CS_A", "I_M", "I_A", "CS_DL", "CS_DLGT"], prob=0.5, spatial_axis=2),
                # # user can also add other random transforms
                # RandAffined(
                #     keys=["CS_M", "label", "CS_A", "I_M", "I_A", "CS_DL", "CS_DLGT"],
                #     mode=('nearest', 'nearest', 'nearest', 'bilinear', "bilinear", "nearest", "nearest"),
                #     prob=0.5,
                #     rotate_range=(0, 0, np.pi / 15),
                #     scale_range=(0.1, 0.1, 0.1),
                #     padding_mode="zeros"
                # ),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["CS_M", "label", "I_M", "CS_DL", "CS_DLGT"]),
                EnsureChannelFirstd(keys=["CS_M", "label", "I_M", "CS_DL", "CS_DLGT"]),
                Orientationd(keys=["CS_M", "label", "I_M", "CS_DL", "CS_DLGT"], axcodes="RAI"),
                CTNormalizeD(keys=["I_M"],
                             mean_intensity=transform_dic["normalize"]["mean"],
                             std_intensity=transform_dic["normalize"]["std"],
                             lower_bound=transform_dic["normalize"]["min"],
                             upper_bound=transform_dic["normalize"]["max"], ),
                Spacingd(keys=["CS_M", "label", "I_M", "CS_DL", "CS_DLGT"], pixdim=transform_dic["spacing"],
                         mode=("nearest", "nearest", "bilinear", "nearest", "nearest")),
            ]
        )
        save_transform = Compose(
            [
            ]
        )
        return train_transforms, val_transforms, save_transform
    elif mode == 'infer':
        infer_transforms = Compose(
            [
                LoadImaged(keys=["CS_M", "I_M", "CS_DL", "CS_DLGT"]),
                EnsureChannelFirstd(keys=["CS_M", "I_M", "CS_DL", "CS_DLGT"]),
                Orientationd(keys=["CS_M", "I_M", "CS_DL", "CS_DLGT"], axcodes="RAI"),
                CTNormalizeD(keys=["I_M"],
                             mean_intensity=transform_dic["normalize"]["mean"],
                             std_intensity=transform_dic["normalize"]["std"],
                             lower_bound=transform_dic["normalize"]["min"],
                             upper_bound=transform_dic["normalize"]["max"], ),
                Spacingd(keys=["CS_M", "I_M", "CS_DL", "CS_DLGT"], pixdim=transform_dic["spacing"],
                         mode=("nearest", "bilinear", "nearest", "nearest")),
            ]
        )
        return infer_transforms
    else:
        raise RuntimeError(f"{mode} is not supported yet")


def get_second_stage_only_one_phase_config(transform_dic, mode='train'):
    """dic is transform block of config.yaml"""
    if transform_dic.get("image_resample") is not None:
        image_resample_mode = transform_dic["image_resample"]["mode"]
        image_resample_padding_mode = transform_dic["image_resample"]["padding_mode"]
    else:
        image_resample_mode = "bilinear"
        image_resample_padding_mode = "border"
    if mode == 'train':
        fixed_transforms = [
                LoadImaged(keys=["CS_M", "label", "I_M", "CS_DL", "CS_DLGT"]),
                EnsureChannelFirstd(keys=["CS_M", "label", "I_M", "CS_DL", "CS_DLGT"]),
                Orientationd(keys=["CS_M", "label", "I_M", "CS_DL", "CS_DLGT"], axcodes="RAI"),
                CTNormalizeD(keys=["I_M"],
                             mean_intensity=transform_dic["normalize"]["mean"],
                             std_intensity=transform_dic["normalize"]["std"],
                             lower_bound=transform_dic["normalize"]["min"],
                             upper_bound=transform_dic["normalize"]["max"], ),
                Spacingd(keys=["CS_M", "label", "I_M", "CS_DL", "CS_DLGT"], pixdim=transform_dic['spacing']
                         , mode=("nearest", "nearest", image_resample_mode, "nearest", "nearest")),
            ]
        print("----------------------second-stage fixed_transform-------------------------")
        if transform_dic.get("fixed") is not None and len(transform_dic.get("fixed")) > 0:
            for d in transform_dic["fixed"]:
                key = d.get('name')
                value = d.get('parameter')
                assert key is not None, "name of fixed transform is not defined in config.yaml"
                assert value is not None, "parameter of fixed transform is not defined in config.yaml"
                print(f"name: {key}, parameter: {value}")
                trans_class = eval(key)
                trans_module = trans_class(**value)
                fixed_transforms.append(trans_module)

        random_transforms = []
        # assert transform_dic.get("random") is not None, "random transform is not defined in config.yaml"
        if transform_dic.get("random") is None or len(transform_dic.get("random")) == 0:
            print("warning: random transform is not defined in config.yaml")
        else:
            print("----------------------second-stage random_transform-------------------------")
            for d in transform_dic["random"]:
                key = d.get('name')
                value = d.get('parameter')
                assert key is not None, "name of random transform is not defined in config.yaml"
                assert value is not None, "parameter of random transform is not defined in config.yaml"
                print(f"name: {key}, parameter: {value}")
                trans_class = eval(key)
                trans_module = trans_class(**value)
                random_transforms.append(trans_module)

        train_transforms = Compose(fixed_transforms + random_transforms)
        val_transforms = Compose(fixed_transforms)
        save_transform = None
        return train_transforms, val_transforms, save_transform
    elif mode == 'infer':
        infer_transforms = [
                LoadImaged(keys=["CS_M", "I_M", "CS_DL", "CS_DLGT"]),
                EnsureChannelFirstd(keys=["CS_M", "I_M", "CS_DL", "CS_DLGT"]),
                Orientationd(keys=["CS_M", "I_M", "CS_DL", "CS_DLGT"], axcodes="RAI"),
                CTNormalizeD(keys=["I_M"],
                             mean_intensity=transform_dic["normalize"]["mean"],
                             std_intensity=transform_dic["normalize"]["std"],
                             lower_bound=transform_dic["normalize"]["min"],
                             upper_bound=transform_dic["normalize"]["max"], ),
                Spacingd(keys=["CS_M", "I_M", "CS_DL", "CS_DLGT"], pixdim=transform_dic["spacing"],
                         mode=("nearest", image_resample_mode, "nearest")),
            ]
        print("----------------------second-stage infer_transform-------------------------")
        if transform_dic.get("infer") is not None and len(transform_dic.get("infer")) > 0:
            for d in transform_dic["infer"]:
                key = d.get('name')
                value = d.get('parameter')
                assert key is not None, "name of infer transform is not defined in config.yaml"
                assert value is not None, "parameter of infer transform is not defined in config.yaml"
                print(f"name: {key}, parameter: {value}")
                trans_class = eval(key)
                trans_module = trans_class(**value)
                infer_transforms.append(trans_module)

        infer_transforms = Compose(infer_transforms)
        return infer_transforms
    else:
        raise RuntimeError(f"{mode} is not supported yet")


