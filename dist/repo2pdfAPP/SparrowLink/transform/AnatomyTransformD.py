import numpy as np
import torch
from monai.utils import TransformBackends, convert_to_tensor, ensure_tuple
from monai.data.meta_obj import get_track_meta
from monai.config import KeysCollection
from monai.transforms import MapTransform, RandomizableTransform, Randomizable, SpatialCrop, TraceableTransform

from typing import Dict, Hashable, List, Mapping, Optional, Sequence, Union, Tuple, Any
from batchgenerators.augmentations.utils import create_zero_centered_coordinate_mesh, elastic_deform_coordinates, \
    interpolate_img, \
    rotate_coords_2d, rotate_coords_3d, scale_coords, resize_segmentation, resize_multichannel_image, \
    elastic_deform_coordinates_2
import warnings
from monai.transforms.utils_pytorch_numpy_unification import unravel_index
from monai.transforms.utils import correct_crop_centers, map_binary_to_indices, convert_to_dst_type, create_translate, \
     ensure_tuple_rep, ensure_tuple
from monai.data.meta_tensor import MetaTensor
from copy import deepcopy
import time
from torch.nn.functional import grid_sample
from scipy.ndimage.filters import gaussian_filter
from skimage.morphology import skeletonize
from post_processing.fracture_detection import get_point_orientation
import cc3d


def get_mvf_by_gaussian_gradient(mask, spacing, blur, dil_magnitude, directions_of_trans, anisotropy_safety):
    shape = mask.shape
    coords = create_zero_centered_coordinate_mesh(shape)
    t, u, v = get_organ_gradient_field(mask,
                                       spacing=spacing,
                                       blur=blur)
    n_factor = np.sqrt(2 * np.pi)
    if directions_of_trans[0]:
        coords[0, :, :, :] = coords[0, :, :, :] + t * blur * dil_magnitude * n_factor
    if directions_of_trans[1]:
        coords[1, :, :, :] = coords[1, :, :, :] + u * blur * dil_magnitude * n_factor
    if directions_of_trans[2]:
        coords[2, :, :, :] = coords[2, :, :, :] + v * blur * dil_magnitude * n_factor
    deformation_record = (t * dil_magnitude * n_factor, u * dil_magnitude * n_factor, v * dil_magnitude * n_factor)
    for d in range(3):
        ctr = shape[d] / 2  # !!!
        coords[d] += ctr - 0.5  # !!!

    if anisotropy_safety:
        coords[0, 0, :, :][coords[0, 0, :, :] < 0] = 0.0
        coords[0, 1, :, :][coords[0, 1, :, :] < 0] = 0.0
        coords[0, -1, :, :][coords[0, -1, :, :] > (shape[-2] - 1)] = shape[-2] - 1
        coords[0, -2, :, :][coords[0, -2, :, :] > (shape[-2] - 1)] = shape[-2] - 1

    return coords, deformation_record


def find_random_one_numpy(image, patch_size=None):
    # 找到值为1的所有点的坐标
    # copy the image to avoid changing the original image
    if patch_size is not None:
        # set image to 0 in the corner
        # copy the image to avoid changing the original image
        image_copy = image.copy()
        image_copy[:patch_size[0] // 2, :, :] = 0
        image_copy[-patch_size[0] // 2:, :, :] = 0
        image_copy[:, :patch_size[1] // 2, :] = 0
        image_copy[:, -patch_size[1] // 2:, :] = 0
        image_copy[:, :, :patch_size[2] // 2] = 0
        image_copy[:, :, -patch_size[2] // 2:] = 0
        ones_indices = np.where(image_copy > 0)
    else:
        ones_indices = np.where(image == 1)
    # 转换坐标为列表形式 [(x1, y1), (x2, y2), ...]
    ones_list = list(zip(ones_indices[0], ones_indices[1], ones_indices[2]))

    if not ones_list:
        return None  # 如果没有找到值为1的点，则返回None

    return ones_list[np.random.randint(len(ones_list))]


def generate_slice_by_center_and_patch_size(center, patch_size):
    x_slice = slice(center[0] - patch_size[0] // 2, center[0] + patch_size[0] // 2)
    y_slice = slice(center[1] - patch_size[1] // 2, center[1] + patch_size[1] // 2)
    z_slice = slice(center[2] - patch_size[2] // 2, center[2] + patch_size[2] // 2)
    return (x_slice, y_slice, z_slice)


def get_organ_gradient_field(organ, spacing=(1, 1, 1), blur=32):
    """
    from batchgenerators.augmentations.utils, but data shape is (H, W, D) instead of (D, H, W)
    The returned MVF is in image coordinates.
    """
    u_ratio = spacing[0] / spacing[1]
    v_ratio = spacing[0] / spacing[2]

    organ_blurred = gaussian_filter(organ.astype(float),
                                    sigma=(blur, blur * u_ratio, blur * v_ratio),
                                    order=0,
                                    mode='nearest')
    t, u, v = np.gradient(organ_blurred)
    t = t
    u = u * u_ratio
    v = v * v_ratio

    return t, u, v


def interpolator(data, coords, mode, border_mode=None, border_cval=None, keep_meta=True):
    data = convert_to_tensor(data, track_meta=get_track_meta())
    meta = data.meta.copy()
    data_result = np.zeros_like(data)
    if isinstance(mode, int):
        for channel_id in range(data.shape[0]):
            data_result[channel_id] = interpolate_img(np.array(data[channel_id]), coords, mode,
                                                      border_mode, cval=border_cval)
        data_result = MetaTensor(data_result, meta=meta) if keep_meta else data_result
    else:
        # import time
        # tic = time.time()
        data_batch = data.unsqueeze(0)
        data_result = \
            grid_sample(data_batch, coords, mode=mode, padding_mode="zeros", align_corners=False)[0]
        data_result = MetaTensor(data_result, meta=meta) if keep_meta else data_result
        # toc = time.time()
        # print(f"interpolator time: {toc - tic}")
    return data_result


def coords_numpy2torch(coords, shape, change_coords=True):
    """Prepare for torch's grid-sampling"""
    h, w, d = shape
    if change_coords:
        coords_torch = torch.from_numpy(coords)
    else:
        coords_torch = torch.from_numpy(coords).clone()
    coords_permute = torch.flip(coords_torch, dims=[0]).unsqueeze(0).permute(0, 2, 3, 4, 1).to(torch.float32)
    coords_permute[:, :, :, :, 0] /= (d - 1)
    coords_permute[:, :, :, :, 1] /= (w - 1)
    coords_permute[:, :, :, :, 2] /= (h - 1)
    coords_permute -= 0.5
    coords_permute *= 2
    return coords_permute


def mesh_generator_tensor(patch_size, rand_state, p_el_per_sample: float = 1, p_rot_per_sample: float = 1,
                          p_scale_per_sample: float = 1, do_elastic_deform=True,
                          alpha=(0., 1000.), sigma=(10., 13.), do_rotation=True,
                          angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                          p_rot_per_axis: float = 1, do_scale=True, scale=(0.75, 1.25),
                          independent_scale_for_each_axis=False, p_independent_scale_per_axis: int = 1,
                          num_samples: int = 1):
    """TODO: change numpy to torch"""
    dim = len(patch_size)
    coords_list = []
    modified_coords_list = []
    for _ in range(num_samples):
        coords = create_zero_centered_coordinate_mesh(patch_size)
        modified_coords = False
        if do_elastic_deform and rand_state.uniform() < p_el_per_sample:
            a = np.random.uniform(alpha[0], alpha[1])
            s = np.random.uniform(sigma[0], sigma[1])
            coords = elastic_deform_coordinates(coords, a, s)
            modified_coords = True

        if do_rotation and rand_state.uniform() < p_rot_per_sample:

            if np.random.uniform() <= p_rot_per_axis:
                a_x = rand_state.uniform(angle_x[0], angle_x[1])
            else:
                a_x = 0

            if dim == 3:
                if np.random.uniform() <= p_rot_per_axis:
                    a_y = rand_state.uniform(angle_y[0], angle_y[1])
                else:
                    a_y = 0

                if np.random.uniform() <= p_rot_per_axis:
                    a_z = rand_state.uniform(angle_z[0], angle_z[1])
                else:
                    a_z = 0
                coords = rotate_coords_3d(coords, a_x, a_y, a_z)
            else:
                coords = rotate_coords_2d(coords, a_x)
            modified_coords = True
        if do_scale and rand_state.uniform() < p_scale_per_sample:
            if independent_scale_for_each_axis and rand_state.uniform() < p_independent_scale_per_axis:
                sc = []
                for _ in range(dim):
                    if rand_state.random() < 0.5 and scale[0] < 1:
                        sc.append(rand_state.uniform(scale[0], 1))
                    else:
                        sc.append(rand_state.uniform(max(scale[0], 1), scale[1]))
            else:
                if rand_state.random() < 0.5 and scale[0] < 1:
                    sc = rand_state.uniform(scale[0], 1)
                else:
                    sc = rand_state.uniform(max(scale[0], 1), scale[1])

            coords = scale_coords(coords, sc)
            modified_coords = True
        coords_list.append(coords)
        modified_coords_list.append(modified_coords)

    return coords_list, modified_coords_list


def generate_pos_neg_label_crop_centers(
        label,
        spatial_size: Union[Sequence[int], int],
        num_samples: int,
        pos_ratio: float,
        label_spatial_shape: Sequence[int],
        fg_indices = None,
        bg_indices = None,
        rand_state: Optional[np.random.RandomState] = None,
        allow_smaller: bool = False,
) -> List[List[int]]:
    """
    Generate valid sample locations based on the label with option for specifying foreground ratio
    Valid: samples sitting entirely within image, expected input shape: [C, H, W, D] or [C, H, W]

    Args:
        label: used gof generating coordinates
        image: for checking correction of cropping
        spatial_size: spatial size of the ROIs to be sampled.
        num_samples: total sample centers to be generated.
        pos_ratio: ratio of total locations generated that have center being foreground.
        label_spatial_shape: spatial shape of the original label data to unravel selected centers.
        fg_indices: pre-computed foreground indices in 1 dimension.
        bg_indices: pre-computed background indices in 1 dimension.
        rand_state: numpy randomState object to align with other modules.
        allow_smaller: if `False`, an exception will be raised if the image is smaller than
            the requested ROI in any dimension. If `True`, any smaller dimensions will be set to
            match the cropped size (i.e., no cropping in that dimension).

    Raises:
        ValueError: When the proposed roi is larger than the image.
        ValueError: When the foreground and background indices lengths are 0.

    """
    if rand_state is None:
        rand_state = np.random.random.__self__  # type: ignore

    if fg_indices is None or bg_indices is None:
        fg_indices, bg_indices = map_binary_to_indices(label, image=None)

    centers = []
    fg_indices = np.asarray(fg_indices) if isinstance(fg_indices, Sequence) else fg_indices
    bg_indices = np.asarray(bg_indices) if isinstance(bg_indices, Sequence) else bg_indices
    if len(fg_indices) == 0 and len(bg_indices) == 0:
        raise ValueError("No sampling location available.")

    if len(fg_indices) == 0 or len(bg_indices) == 0:
        pos_ratio = 0 if len(fg_indices) == 0 else 1
        warnings.warn(
            f"Num foregrounds {len(fg_indices)}, Num backgrounds {len(bg_indices)}, "
            f"unable to generate class balanced samples, setting `pos_ratio` to {pos_ratio}."
        )

    for _ in range(num_samples):
        indices_to_use = fg_indices if rand_state.rand() < pos_ratio else bg_indices
        random_int = rand_state.randint(len(indices_to_use))
        idx = indices_to_use[random_int]
        center = unravel_index(idx, label_spatial_shape).tolist()
        # shift center to range of valid centers
        centers.append(correct_crop_centers(center, spatial_size, label_spatial_shape, allow_smaller))

    return centers


class CASTransformD(Randomizable, MapTransform):
    """
    Add three components in deformable transformation
    1. Muscle squeezing and muscle relaxation -> based on vessel segmentation, using normal vector of the surface.
    2. Heart motion -> based on chambers segmentation
    3. Cardiac Motion or curve deformation -> based on centerline of segmentation
    Args:
    `dil_ranges`: dilation range per organs
    `modalities`: on which input channels should the transformation be applied
    `directions_of_trans`: to which directions should the organs be dilated per organs
    `p_per_sample`: probability of the transformation per organs
    `spacing_ratio`: ratio of the transversal plane spacing and the slice thickness, in our case it was 0.3125/3
    `blur`: Gaussian kernel parameter, we used the value 32 for 0.3125mm transversal plane spacing
    `anisotropy_safety`: it provides a certain protection against transformation artifacts in 2 slices from the image border
    `max_annotation_value`: the value that should be still relevant for the main task
    `replace_value`: segmentation values larger than the `max_annotation_value` will be replaced with
    """
    backend = [TransformBackends.NUMPY, TransformBackends.TORCH]

    def __init__(self,
                 keys: KeysCollection,
                 label_key: str = 'label',
                 heart_key: str = 'heart',
                 p_anatomy_per_sample: float = 0.5,
                 dil_ranges: Tuple[Tuple[int, int], Tuple[int, int]] = ((0, 0), (0, 0)),
                 directions_of_trans: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = ((1, 1, 1), (1, 1, 1)),
                 spacing_ratio: float = 0.334/0.5,
                 blur: List = [32, 32],
                 anisotropy_safety: bool = True,
                 max_annotation_value: int = 1,
                 allow_missing_keys: bool = False,
                 mode: Union[Sequence[int], int, Sequence[str], str] = 1,
                 border_mode: str = 'constant',
                 cval: float = 0.0,):
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.label_key = label_key
        self.heart_key = heart_key
        self.p_anatomy = p_anatomy_per_sample
        self.dilation_ranges = dil_ranges
        self.directions_of_trans = directions_of_trans
        self.spacing_ratio = spacing_ratio
        self.blur = blur
        self.anisotropy_safety = anisotropy_safety
        self.max_annotation_value = max_annotation_value
        self.border_mode = ensure_tuple_rep(border_mode, len(self.keys))
        self.border_cval = ensure_tuple_rep(cval, len(self.keys))
        self.mode = ensure_tuple_rep(mode, len(self.keys))

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        """TODO: if num_samples > 1, this function uses for loop to interpolate the data, which is not efficient."""
        d: Dict = dict(data)
        mask = [d[self.label_key],]
        if self.heart_key is not None:
            mask.append(d[self.heart_key])

        m, deformation_record = self.randomize(mask=mask)

        deformation_shape = list(d[self.label_key].shape)
        deformation_shape[0] = 3
        meta = d[self.label_key].meta.copy()
        label_deformation_field = torch.zeros(size=deformation_shape, dtype=torch.float32)
        label_deformation_field[0, :, :, :] = torch.tensor(deformation_record[0][0])
        label_deformation_field[1, :, :, :] = torch.tensor(deformation_record[0][1])
        label_deformation_field[2, :, :, :] = torch.tensor(deformation_record[0][2])
        d['label_df'] = MetaTensor(label_deformation_field, meta=meta)
        if self.heart_key is not None:
            heart_deformation_field = torch.zeros(size=deformation_shape, dtype=torch.float32)
            heart_deformation_field[0, :, :, :] = torch.tensor(deformation_record[1][0])
            heart_deformation_field[1, :, :, :] = torch.tensor(deformation_record[1][1])
            heart_deformation_field[2, :, :, :] = torch.tensor(deformation_record[1][2])
            d['heart_df'] = MetaTensor(heart_deformation_field, meta=meta)
            heart_deformation_field = torch.zeros(size=deformation_shape, dtype=torch.float32)
            heart_deformation_field[0, :, :, :] = torch.tensor(deformation_record[2][0])
            heart_deformation_field[1, :, :, :] = torch.tensor(deformation_record[2][1])
            heart_deformation_field[2, :, :, :] = torch.tensor(deformation_record[2][2])
            d['heart_df1'] = MetaTensor(heart_deformation_field, meta=meta)
            heart_deformation_field = torch.zeros(size=deformation_shape, dtype=torch.float32)
            heart_deformation_field[0, :, :, :] = torch.tensor(deformation_record[3][0])
            heart_deformation_field[1, :, :, :] = torch.tensor(deformation_record[3][1])
            heart_deformation_field[2, :, :, :] = torch.tensor(deformation_record[3][2])
            d['heart_df2'] = MetaTensor(heart_deformation_field, meta=meta)
            heart_deformation_field = torch.zeros(size=deformation_shape, dtype=torch.float32)
            heart_deformation_field[0, :, :, :] = torch.tensor(deformation_record[4][0])
            heart_deformation_field[1, :, :, :] = torch.tensor(deformation_record[4][1])
            heart_deformation_field[2, :, :, :] = torch.tensor(deformation_record[4][2])
            d['heart_df3'] = MetaTensor(heart_deformation_field, meta=meta)
            heart_deformation_field = torch.zeros(size=deformation_shape, dtype=torch.float32)
            heart_deformation_field[0, :, :, :] = torch.tensor(deformation_record[5][0])
            heart_deformation_field[1, :, :, :] = torch.tensor(deformation_record[5][1])
            heart_deformation_field[2, :, :, :] = torch.tensor(deformation_record[5][2])
            d['heart_df4'] = MetaTensor(heart_deformation_field, meta=meta)
            heart_deformation_field = torch.zeros(size=deformation_shape, dtype=torch.float32)
            heart_deformation_field[0, :, :, :] = torch.tensor(deformation_record[6][0])
            heart_deformation_field[1, :, :, :] = torch.tensor(deformation_record[6][1])
            heart_deformation_field[2, :, :, :] = torch.tensor(deformation_record[6][2])
            d['heart_df5'] = MetaTensor(heart_deformation_field, meta=meta)
            heart_deformation_field = torch.zeros(size=deformation_shape, dtype=torch.float32)
            heart_deformation_field[0, :, :, :] = torch.tensor(deformation_record[7][0])
            heart_deformation_field[1, :, :, :] = torch.tensor(deformation_record[7][1])
            heart_deformation_field[2, :, :, :] = torch.tensor(deformation_record[7][2])
            d['heart_df6'] = MetaTensor(heart_deformation_field, meta=meta)

        for key, border_mode, border_cval, order in self.key_iterator(d, self.border_mode, self.border_cval, self.mode):
            d[key] = self.interpolator(data=d[key], coords=m, border_mode=border_mode, border_cval=border_cval,
                                       mode=order,)

        return d

    def randomize(self,
                  mask: List = None,
                  spacing=None,):
        """
        return coordinate mesh and dilation magnitude
        """
        deformation_record = []
        if self.R.uniform() < self.p_anatomy:
            shape = mask[0].shape
            coords = create_zero_centered_coordinate_mesh(shape[1:])
            if spacing is not None:
                # TODO: spacing in 1, 2 dimension may be different
                spacing_ratio = spacing[0] / spacing[2]
            else:
                spacing_ratio = self.spacing_ratio

            for i in range(len(mask)):
                dil_magnitude = np.random.uniform(low=self.dilation_ranges[i][0], high=self.dilation_ranges[i][1])
                if i == 0:
                    t, u, v = get_organ_gradient_field(mask[i][0] > 0,
                                                       spacing_ratio=spacing_ratio,
                                                       blur=self.blur[i])
                elif i == 1:
                    t, u, v = get_organ_gradient_field(mask[i][0] > 0,
                                                       spacing_ratio=spacing_ratio,
                                                       blur=self.blur[i])

                    t_1, u_1, v_1 = get_organ_gradient_field(mask[i][0] == 1,
                                                             spacing_ratio=spacing_ratio,
                                                             blur=self.blur[i])
                    t_2, u_2, v_2 = get_organ_gradient_field(mask[i][0] == 2,
                                                                spacing_ratio=spacing_ratio,
                                                                blur=self.blur[i])
                    t_3, u_3, v_3 = get_organ_gradient_field(mask[i][0] == 3,
                                                                spacing_ratio=spacing_ratio,
                                                                blur=self.blur[i])
                    t_4, u_4, v_4 = get_organ_gradient_field(mask[i][0] == 4,
                                                                spacing_ratio=spacing_ratio,
                                                                blur=self.blur[i])
                    t_5, u_5, v_5 = get_organ_gradient_field(mask[i][0] == 5,
                                                             spacing_ratio=spacing_ratio,
                                                             blur=self.blur[i])
                    t_6, u_6, v_6 = get_organ_gradient_field(mask[i][0] == 6,
                                                             spacing_ratio=spacing_ratio,
                                                             blur=self.blur[i])


                sigma = self.blur[i]
                if self.directions_of_trans[i][0]:
                    coords[0, :, :, :] = coords[0, :, :, :] + t * (sigma ** 2) * dil_magnitude
                if self.directions_of_trans[i][1]:
                    coords[1, :, :, :] = coords[1, :, :, :] + u * (sigma ** 2) * dil_magnitude
                if self.directions_of_trans[i][2]:
                    coords[2, :, :, :] = coords[2, :, :, :] + v * (sigma ** 2) * dil_magnitude

                deformation_record.append((t * dil_magnitude, u * dil_magnitude, v * dil_magnitude)) #  * spacing_ratio))
                if i == 1:
                    deformation_record.append((t_1 * dil_magnitude, u_1 * dil_magnitude, v_1 * dil_magnitude))
                    deformation_record.append((t_2 * dil_magnitude, u_2 * dil_magnitude, v_2 * dil_magnitude))
                    deformation_record.append((t_3 * dil_magnitude, u_3 * dil_magnitude, v_3 * dil_magnitude))
                    deformation_record.append((t_4 * dil_magnitude, u_4 * dil_magnitude, v_4 * dil_magnitude))
                    deformation_record.append((t_5 * dil_magnitude, u_5 * dil_magnitude, v_5 * dil_magnitude))
                    deformation_record.append((t_6 * dil_magnitude, u_6 * dil_magnitude, v_6 * dil_magnitude))

            for d in range(3):
                ctr = shape[d + 1] / 2  # !!!
                coords[d] += ctr - 0.5  # !!!

            if self.anisotropy_safety:
                coords[0, 0, :, :][coords[0, 0, :, :] < 0] = 0.0
                coords[0, 1, :, :][coords[0, 1, :, :] < 0] = 0.0
                coords[0, -1, :, :][coords[0, -1, :, :] > (shape[-3] - 1)] = shape[-3] - 1
                coords[0, -2, :, :][coords[0, -2, :, :] > (shape[-3] - 1)] = shape[-3] - 1
        else:
            coords = None

        return coords, deformation_record

    def interpolator(self, data, coords, mode, border_mode=None, border_cval=None):
        data = convert_to_tensor(data, track_meta=get_track_meta())
        meta = data.meta.copy()
        data_result = np.zeros_like(data)
        if isinstance(mode, int):
            for channel_id in range(data.shape[0]):
                data_result[channel_id] = interpolate_img(np.array(data[channel_id]), coords, mode,
                                                          border_mode, cval=border_cval)
            data_result = MetaTensor(data_result, meta=meta)

        else:
            h, w, d = data.shape[1], data.shape[2], data.shape[3]
            coords = coords[::-1, :, :, :].copy()  # xyz -> zyx
            coords_permute = torch.from_numpy(coords).unsqueeze(0).permute(0, 2, 3, 4, 1).to(torch.float32)
            data_batch = data.unsqueeze(0)
            # coords_norm = (coords_permute + 1) / torch.tensor([h, w, d], dtype=torch.float32).reshape(1, 1, 1, 1, 3)
            coords_norm = torch.zeros_like(coords_permute)
            coords_norm[:, :, :, :, 0], coords_norm[:, :, :, :, 1], coords_norm[:, :, :, :, 2] = \
                (coords_permute[:, :, :, :, 0] + 1) / d, (coords_permute[:, :, :, :, 1] + 1) / w, \
                (coords_permute[:, :, :, :, 2] + 1) / h
            coords_norm = (coords_norm - 0.5) * 2
            data_result = \
                grid_sample(data_batch, coords_norm, mode=mode, padding_mode="zeros", align_corners=False)[0]
            data_result = MetaTensor(data_result, meta=meta)
        return data_result


class HeartTransformD(Randomizable, MapTransform):
    """
    Add three components in deformable transformation
    1. Cardiac muscle squeezing and muscle relaxation -> based on vessel segmentation, using normal vector of the surface.
    Args:
    `dil_ranges`: dilation range per organs
    `modalities`: on which input channels should the transformation be applied
    `directions_of_trans`: to which directions should the organs be dilated per organs
    `p_per_sample`: probability of the transformation per organs
    `spacing_ratio`: ratio of the transversal plane spacing and the slice thickness, in our case it was 0.3125/3
    `blur`: Gaussian kernel parameter, we used the value 32 for 0.3125mm transversal plane spacing
    `anisotropy_safety`: it provides a certain protection against transformation artifacts in 2 slices from the image border
    `max_annotation_value`: the value that should be still relevant for the main task
    `replace_value`: segmentation values larger than the `max_annotation_value` will be replaced with
    'heart_select' : select which chamber to do dilation and shrink, In this work [1, 2, 3, 4, 5, 6] represent
    [inner left-aorta, left-ventricular, right-aorta, right-ventricular, main-artery, left-aorta]
    'heart_key' : heart segmentation
    'label_key' : coronary artery segmentation
    """
    backend = [TransformBackends.NUMPY, TransformBackends.TORCH]

    def __init__(self,
                 keys: KeysCollection,
                 heart_key: str = 'heart',
                 artery_key: str = None,
                 heart_select: Tuple[Union[int, Tuple[int]]] = ((5, ), (2, 4), (3, 6)),
                 p_anatomy_heart: float = 0.5,
                 p_anatomy_artery: float = 0.5,
                 dil_ranges: Tuple[Tuple[float, float], Tuple[float, float]] = ((0, 0), (0, 0)),
                 directions_of_trans: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = ((1, 1, 1), (1, 1, 1)),
                 blur: Tuple = (32, 32),
                 anisotropy_safety: bool = True,
                 max_annotation_value: int = 1,
                 allow_missing_keys: bool = False,
                 mode: Union[Sequence[int], int, Sequence[str], str] = 1,
                 batch_interpolate: bool = False,
                 threshold: Tuple = None,
                 border_mode: str = 'constant',
                 cval: float = 0.0,
                 visualize: bool = False,
                 del_heart: bool = True,
                 ):
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.heart_key = heart_key
        self.heart_select = heart_select
        self.artery_key = artery_key
        self.p_anatomy_heart = p_anatomy_heart
        self.p_anatomy_artery = p_anatomy_artery
        self.dilation_ranges = dil_ranges
        self.directions_of_trans = directions_of_trans
        self.blur = blur
        self.anisotropy_safety = anisotropy_safety
        self.max_annotation_value = max_annotation_value
        self.border_mode = ensure_tuple_rep(border_mode, len(self.keys))
        self.border_cval = ensure_tuple_rep(cval, len(self.keys))
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.visualize = visualize

        self.do_heart_transformation = False
        self.do_artery_transformation = False
        self.dil_magnitude_heart = 0
        self.dil_magnitude_artery = 0
        self.random_index = None  # random index to select chambers
        self.threshold = ensure_tuple_rep(threshold, len(self.keys))
        self.batch_interpolate = batch_interpolate

        self.del_heart = del_heart

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        """TODO: if num_samples > 1, this function uses for loop to interpolate the data, which is not efficient."""
        d: Dict = dict(data)
        spacing = d[self.heart_key].meta['pixdim'][1:4]
        shape = d[self.heart_key].shape
        self.randomize()
        if self.do_heart_transformation:
            mask = [d[self.heart_key]]
            mask_dilation = torch.zeros_like(mask[0][0])
            for idx in self.random_index:
                mask_dilation += mask[0][0] == idx

            m, deformation_heart = get_mvf_by_gaussian_gradient(mask=mask_dilation > 0,
                                                                spacing=spacing,
                                                                blur=self.blur[0],
                                                                dil_magnitude=self.dil_magnitude_heart,
                                                                directions_of_trans=self.directions_of_trans[0],
                                                                anisotropy_safety=self.anisotropy_safety)
            if not isinstance(self.mode[0], int):  # assert all modes are the same, either int or str
                m = coords_numpy2torch(m, shape[1:], change_coords=True)
            if self.batch_interpolate:
                # only support torch
                self.batch_interpolator(d=d, m=m, shape=shape[1:])
            else:
                for key, border_mode, border_cval, order in \
                        self.key_iterator(d, self.border_mode, self.border_cval, self.mode):
                    # use torch's grid-sampling to interpolate the data
                    d[key] = interpolator(data=d[key], coords=m, border_mode=border_mode, border_cval=border_cval,
                                          mode=order, )

            if self.visualize:
                deformation_shape = list(d[self.heart_key].shape)
                deformation_shape[0] = 3
                meta = d[self.heart_key].meta.copy()
                heart_deformation_field = torch.zeros(size=deformation_shape, dtype=torch.float32)
                #  change to world coordinates
                heart_deformation_field[0, :, :, :] = torch.tensor(deformation_heart[0]) * spacing[0]
                heart_deformation_field[1, :, :, :] = torch.tensor(deformation_heart[1]) * spacing[1]
                heart_deformation_field[2, :, :, :] = torch.tensor(deformation_heart[2]) * spacing[2]
                d['heart_df'] = MetaTensor(heart_deformation_field, meta=meta)

        if self.do_artery_transformation:
            m, deformation_artery = get_mvf_by_gaussian_gradient(mask=d[self.artery_key][0] > 0,
                                                                 spacing=spacing,
                                                                 blur=self.blur[1],
                                                                 dil_magnitude=self.dil_magnitude_artery,
                                                                 directions_of_trans=self.directions_of_trans[1],
                                                                 anisotropy_safety=self.anisotropy_safety)
            if not isinstance(self.mode[0], int):  # assert all modes are the same, either int or str
                m = coords_numpy2torch(m, shape[1:], change_coords=True)

            if self.batch_interpolate:
                self.batch_interpolator(d=d, m=m, shape=shape[1:])
            else:
                for key, border_mode, border_cval, order in \
                        self.key_iterator(d, self.border_mode, self.border_cval, self.mode):
                    d[key] = interpolator(data=d[key], coords=m, border_mode=border_mode, border_cval=border_cval,
                                          mode=order, )

            if self.visualize:
                deformation_shape = list(d[self.artery_key].shape)
                deformation_shape[0] = 3
                meta = d[self.artery_key].meta.copy()
                label_deformation_field = torch.zeros(size=deformation_shape, dtype=torch.float32)
                label_deformation_field[0, :, :, :] = torch.tensor(deformation_artery[0]) * spacing[0]
                label_deformation_field[1, :, :, :] = torch.tensor(deformation_artery[1]) * spacing[1]
                label_deformation_field[2, :, :, :] = torch.tensor(deformation_artery[2]) * spacing[2]
                d['label_df'] = MetaTensor(label_deformation_field, meta=meta)

        if self.del_heart:
            d.pop(self.heart_key)  # delete heart segmentation to save memory
        return d

    def batch_interpolator(self, d, m, shape, device="cpu"):
        i_h, i_w, i_d = shape
        c = len(self.mode)
        batch_data = torch.zeros((1, c, i_h, i_w, i_d))
        meta_dict = {}

        for i, key in enumerate(self.key_iterator(d)):
            batch_data[0][i] = d[key]
            meta_dict[key] = d[key].meta

        with torch.no_grad():
            data_result = \
                grid_sample(batch_data.to(device), m.to(device), mode=self.mode[0], padding_mode="zeros",
                            align_corners=False)[0]

        for key, t, i in self.key_iterator(d, self.threshold, range(c)):
            if t > 0:
                d[key] = MetaTensor((data_result[i] > t).unsqueeze(0),
                                    meta=meta_dict[key])
            else:
                d[key] = MetaTensor(data_result[i].unsqueeze(0), meta=meta_dict[key])


    def randomize(self, data: Any = None):
        """ return coordinate mesh and dilation magnitude"""
        if self.R.uniform() < self.p_anatomy_heart:
            self.do_heart_transformation = True
            self.dil_magnitude_heart = self.R.uniform(low=9, high=self.dilation_ranges[0][1])
            self.random_index = self.heart_select[self.R.randint(0, len(self.heart_select))]
        else:
            self.do_heart_transformation = False

        if self.artery_key is not None:
            if self.R.uniform() < self.p_anatomy_artery:
                self.do_artery_transformation = True
                self.dil_magnitude_artery = self.R.uniform(low=self.dilation_ranges[1][0],
                                                           high=self.dilation_ranges[1][1])
                print(f"artery dilation magnitude: {self.dil_magnitude_artery}")
        else:
            self.do_artery_transformation = False

        #     shape = mask[0].shape
        #     coords = create_zero_centered_coordinate_mesh(shape[1:])
        #     random_index = self.heart_select[np.random.randint(0, len(self.heart_select))]
        #     mask_dilation = torch.zeros_like(mask[0][0])
        #     for idx in random_index:
        #         mask_dilation += mask[0][0] == idx
        #     t_h, u_h, v_h = get_organ_gradient_field(mask_dilation > 0,
        #                                              spacing=spacing,
        #                                              blur=self.blur[0])
        #     dil_magnitude_heart = np.random.uniform(low=self.dilation_ranges[0][0], high=self.dilation_ranges[0][1])
        #     dil_magnitude_artery = None
        #     t_a, u_a, v_a = None, None, None
        #
        #     if self.artery_key is not None:
        #         assert len(mask) > 0 and len(self.blur) > 0 and len(self.dilation_ranges) > 0, \
        #             "artery label is not available or parameters are not set"
        #         dil_magnitude_artery = np.random.uniform(low=self.dilation_ranges[0][0], high=self.dilation_ranges[0][1])
        #         t_a, u_a, v_a = get_organ_gradient_field(mask[1][0] > 0,
        #                                                  spacing=spacing,
        #                                                  blur=self.blur[1])
        #
        #     n_factor = np.sqrt(2 * np.pi)
        #     if self.directions_of_trans[0][0]:
        #         coords[0, :, :, :] = coords[0, :, :, :] + t_h * self.blur[0] * dil_magnitude_heart * n_factor
        #     if self.directions_of_trans[0][1]:
        #         coords[1, :, :, :] = coords[1, :, :, :] + u_h * self.blur[0] * dil_magnitude_heart * n_factor
        #     if self.directions_of_trans[0][2]:
        #         coords[2, :, :, :] = coords[2, :, :, :] + v_h * self.blur[0] * dil_magnitude_heart * n_factor
        #
        #     deformation_record.append((t_h * dil_magnitude_heart * n_factor, u_h * dil_magnitude_heart * n_factor,
        #                                v_h * dil_magnitude_heart * n_factor))
        #     if self.artery_key is not None:
        #         if self.directions_of_trans[0][0]:
        #             coords[0, :, :, :] = coords[0, :, :, :] + t_a * self.blur[1] * dil_magnitude_artery * n_factor
        #         if self.directions_of_trans[0][1]:
        #             coords[1, :, :, :] = coords[1, :, :, :] + u_a * self.blur[1] * dil_magnitude_artery * n_factor
        #         if self.directions_of_trans[0][2]:
        #             coords[2, :, :, :] = coords[2, :, :, :] + v_a * self.blur[1] * dil_magnitude_artery * n_factor
        #         deformation_record.append(
        #             (t_a * dil_magnitude_artery * n_factor, u_a * dil_magnitude_artery * n_factor,
        #              v_a * dil_magnitude_artery * n_factor))
        #
        #     for d in range(3):
        #         ctr = shape[d + 1] / 2  # !!!
        #         coords[d] += ctr - 0.5  # !!!
        #
        #     if self.anisotropy_safety:
        #         coords[0, 0, :, :][coords[0, 0, :, :] < 0] = 0.0
        #         coords[0, 1, :, :][coords[0, 1, :, :] < 0] = 0.0
        #         coords[0, -1, :, :][coords[0, -1, :, :] > (shape[-3] - 1)] = shape[-3] - 1
        #         coords[0, -2, :, :][coords[0, -2, :, :] > (shape[-3] - 1)] = shape[-3] - 1
        # else:
        #     coords = None
        #
        # return coords, deformation_record


def adjust_contrast(patch_image, contrast_reduction_factor, patch_patch_slice, mask_blur=4):
    # generate weighted mask
    patch_patch_mask = np.ones_like(patch_image)
    patch_patch_mask[patch_patch_slice] = contrast_reduction_factor
    patch_patch_mask_gaussian = gaussian_filter(patch_patch_mask, sigma=mask_blur)
    min_max = patch_patch_mask_gaussian.max() - patch_patch_mask_gaussian.min()
    assert min_max > 0.00001, f"min_max should be larger than 0, {patch_patch_mask_gaussian.max(), patch_patch_mask_gaussian.min()}" \
                              f"{patch_patch_mask.max(), patch_patch_mask.min()}" \
                              f"{contrast_reduction_factor}" \
                              f"{patch_patch_slice}"
    patch_min = patch_patch_mask_gaussian.min()
    patch_patch_mask_gaussian -= patch_min
    patch_patch_mask_gaussian *= ((1 - contrast_reduction_factor) / min_max)
    patch_patch_mask_gaussian += contrast_reduction_factor
    # patch_patch_mask_gaussian = (patch_patch_mask_gaussian - patch_min) * (1 - contrast_reduction_factor) / min_max \
    #                             + contrast_reduction_factor

    patch_mean = patch_image.mean()
    patch_image = (patch_image - patch_mean) * patch_patch_mask_gaussian + patch_mean
    return torch.from_numpy(patch_image)


def check_shape(a, b):
    flag = True
    for i, j in zip(a, b):
        if i > j:
            flag = False
    return flag


def scale_tuple(a, t):
    return tuple([a[i]*t for i in range(len(a))])


def generate_random_vector_perpendicular_to(n):
    n = n / np.linalg.norm(n)

    # 生成一个随机向量
    r = np.random.rand(3)

    # 计算垂直于n的向量
    v = np.cross(n, r)

    # 如果v是零向量（非常罕见的情况，但理论上可能如果r和n平行），重新生成r
    while np.linalg.norm(v) == 0:
        r = np.random.rand(3)
        v = np.cross(n, r)
    # 标准化v
    v = v / np.linalg.norm(v)

    return v


def get_sphere_by_center_and_radius(center, radius, spacing, shape):
    """
    center: sphere center (h, w, d)
    radius: sphere radius， mm
    spacing: voxel spacing, mm
    shape: image shape or patch shape
    """
    h, w, d = shape
    tmp = tuple([np.arange(i) for i in (h, w, d)])
    coords = np.array(np.meshgrid(*tmp, indexing='ij')).astype(float)
    coords -= np.array(center)[:, None, None, None]
    coords = coords * np.array(spacing)[:, None, None, None]
    distance = np.linalg.norm(coords, axis=0)
    sphere = np.zeros((h, w, d))
    sphere[distance <= radius] = 1.0
    return sphere


def generate_mvf_vector(center, artery, centerline, direction_image, spacing, shape, scale_factor, radius, sigma=4):
    orientation_centerline_spacing = \
        get_point_orientation(point=center, centerline=centerline, spacing=spacing, size=3,
                              is_end_point=False)[0]
    if orientation_centerline_spacing is None:
        return None
    direction_image_com = np.array([direction_image[0], direction_image[4], direction_image[8]])
    orientation_centerline_world = orientation_centerline_spacing * direction_image_com
    random_direction_3d = generate_random_vector_perpendicular_to(orientation_centerline_world)
    h, w, d = shape
    mvf_world = np.zeros((h, w, d, 3))
    mvf_world += random_direction_3d
    center_point = center[0]

    mvf_mask = get_sphere_by_center_and_radius(center=center_point, radius=radius, spacing=spacing, shape=shape)
    mvf_mask = mvf_mask * (artery > 0)
    connected_region = cc3d.connected_components(mvf_mask)  # only select the region where center locates
    area_id = connected_region[center_point[0], center_point[1], center_point[2]]
    mvf_mask = mvf_mask * (connected_region == area_id)
    mvf_weight = gaussian_filter(mvf_mask, sigma=sigma) * scale_factor * sigma

    mvf_world = mvf_world * mvf_weight[..., None]
    mvf_image = (mvf_world / np.array(direction_image_com) / np.array(spacing))
    h, w, d = shape
    tmp = tuple([np.arange(i) for i in (h, w, d)])
    coords = np.array(np.meshgrid(*tmp, indexing='ij')).astype(float)
    return coords + mvf_image.transpose(3, 0, 1, 2)


class ArteryTransformD(Randomizable, MapTransform):
    """
    1. local contrast change
    2. artery shift
    3. local vessel shrink or dilation
    """
    backend = [TransformBackends.NUMPY, TransformBackends.TORCH]

    def __init__(self,
                 keys: KeysCollection,
                 image_key: str = None,
                 contrast_patch_patch_size_range: Union[Tuple, Tuple[Tuple]] = ((10, 30), (10, 30), (10, 30)),
                 deform_patch_patch_size_range: Tuple = (5.0, 10.0),
                 contrast_reduction_factor_range: Tuple = (0.6, 1),
                 mvf_scale_factor_range: Tuple = (-2, 2),
                 mask_blur_range: Tuple = (3, 6),
                 artery_key: str = None,
                 centerline_key: str = None,
                 allow_missing_keys: bool = False,
                 p_anatomy_per_sample: float = 0.0,
                 p_contrast_per_sample: float = 0.0,
                 mode: Union[Sequence[int], int, Sequence[str], str] = 1,
                 border_mode: str = 'constant',
                 cval: float = 0.0,
                 visualize: bool = False):
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.artery_key = artery_key
        self.image_key = image_key
        self.centerline_key = centerline_key
        self.p_anatomy = p_anatomy_per_sample
        self.p_contrast = p_contrast_per_sample
        self.border_mode = ensure_tuple_rep(border_mode, len(self.keys))
        self.border_cval = ensure_tuple_rep(cval, len(self.keys))
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.visualize = visualize
        self.do_artery_deformation = False
        self.do_local_contrast_change = False
        self.contrast_reduction_factor_range = contrast_reduction_factor_range
        self.contrast_reduction_factor = None
        self.mask_blur_range = mask_blur_range
        self.mask_blur = None
        self.contrast_patch_patch_size_range = ensure_tuple_rep(contrast_patch_patch_size_range, 3)
        self.deform_patch_patch_size_range = deform_patch_patch_size_range
        self.mvf_mask_radius = None
        self.patch_patch_size_x_1, self.patch_patch_size_y_1, self.patch_patch_size_z_1 = None, None, None
        self.mvf_scale_factor_range = mvf_scale_factor_range
        self.mvf_scale_factor = None

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        """TODO: if num_samples > 1, this function uses for loop to interpolate the data, which is not efficient."""
        d: Dict = dict(data)
        # check whether this patch has label
        self.randomize()
        if self.do_artery_deformation or self.do_local_contrast_change:
            artery = d[self.artery_key]
            shape = artery.shape[1:]
            original_affine = artery.meta.get('original_affine')
            direction_image = np.sign(original_affine[:3, :3].reshape(-1, )).tolist()
            # spacing = artery.meta.get('pixdim')[1:4]
            r_matrix = artery.meta.get('affine')[:3, :3].reshape(-1, ).tolist()
            spacing = np.array([np.abs(r_matrix[0]), np.abs(r_matrix[4]), np.abs(r_matrix[8])], dtype=np.float32)
            origin = original_affine[:3, 3]
            if artery.sum() < 100:  # if no artery, return directly. We assume the artery should be larger than 100
                return d
            if self.centerline_key is not None:
                """get centerline"""
                centerline = data[self.centerline_key]
            else:
                centerline = skeletonize(d[self.artery_key][0] > 0)[None, ]
        else:
            return d
        if self.do_local_contrast_change:
            patch_patch_size = (self.patch_patch_size_x_1, self.patch_patch_size_y_1, self.patch_patch_size_z_1)
            center = np.array(find_random_one_numpy(centerline[0], patch_size=patch_patch_size))[None,]
            if center[0] is not None:
                patch_slice = generate_slice_by_center_and_patch_size(center[0], patch_size=patch_patch_size)
                assert patch_slice[-1].start >= 0, f"patch_slice is None, {center[0]}, {patch_patch_size}"
                d[self.image_key][0] = adjust_contrast(patch_image=d[self.image_key][0],
                                                       patch_patch_slice=patch_slice,
                                                       mask_blur=self.mask_blur,
                                                       contrast_reduction_factor=self.contrast_reduction_factor)
        if self.do_artery_deformation:
            patch_patch_size = (int(self.mvf_mask_radius / spacing[0]), int(self.mvf_mask_radius / spacing[1]),
                                int(self.mvf_mask_radius / spacing[2]))
            # check the patch_patch_size is suitable
            if check_shape(a=scale_tuple(patch_patch_size, 3), b=shape):
                center = np.array(find_random_one_numpy(centerline[0],
                                                        patch_size=scale_tuple(patch_patch_size, 3)))[None,]
                # times 3 here to reduce deformation effect in the edge
                if center[0] is not None:
                    mvf = generate_mvf_vector(center=center, artery=artery[0], centerline=centerline[0] > 0,
                                              scale_factor=self.mvf_scale_factor, radius=self.mvf_mask_radius,
                                              direction_image=direction_image, spacing=spacing, shape=shape)
                    if mvf is None:
                        RuntimeWarning("MVF is None, probably because the center is too close to the edge or isolated")
                        return d
                    if not isinstance(self.mode[0], int):  # assert all modes are the same, either int or str
                        mvf = coords_numpy2torch(mvf, shape, change_coords=True)
                    for key, border_mode, border_cval, order in \
                            self.key_iterator(d, self.border_mode, self.border_cval, self.mode):
                        # use torch's grid-sampling to interpolate the data
                        d[key] = interpolator(data=d[key], coords=mvf, border_mode=border_mode, border_cval=border_cval,
                                              mode=order, )
        return d

    def randomize(self, data: Any = None):
        if self.R.uniform() < self.p_anatomy:
            self.do_artery_deformation = True
            self.mvf_mask_radius = self.R.uniform(self.deform_patch_patch_size_range[0],
                                                  self.deform_patch_patch_size_range[1])
            self.mvf_scale_factor = self.R.uniform(self.mvf_scale_factor_range[0],
                                                   self.mvf_scale_factor_range[1])
        else:
            self.do_artery_deformation = False

        if self.R.uniform() < self.p_contrast:
            self.do_local_contrast_change = True
            self.contrast_reduction_factor = self.R.uniform(self.contrast_reduction_factor_range[0],
                                                            self.contrast_reduction_factor_range[1])
            self.patch_patch_size_x_1 = self.R.randint(self.contrast_patch_patch_size_range[0][0],
                                                       self.contrast_patch_patch_size_range[0][1])
            self.patch_patch_size_y_1 = self.R.randint(self.contrast_patch_patch_size_range[1][0],
                                                       self.contrast_patch_patch_size_range[1][1])
            self.patch_patch_size_z_1 = self.R.randint(self.contrast_patch_patch_size_range[2][0],
                                                       self.contrast_patch_patch_size_range[2][1])
            self.mask_blur = self.R.uniform(self.mask_blur_range[0], self.mask_blur_range[1])
        else:
            self.do_local_contrast_change = False


