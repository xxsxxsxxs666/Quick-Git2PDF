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
from monai.transforms import MapTransform, RandomizableTransform, Randomizable

from batchgenerators.augmentations.noise_augmentations import augment_gaussian_blur, augment_gaussian_noise, \
    augment_rician_noise
from batchgenerators.augmentations.color_augmentations import augment_contrast, augment_brightness_additive, \
    augment_brightness_multiplicative, augment_gamma, augment_illumination, augment_PCA_shift
from batchgenerators.augmentations.resample_augmentations import augment_linear_downsampling_scipy
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Union, Tuple


class CTNormalizeD(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.NormalizeIntensity`.
    This transform can normalize only non-zero values or entire image, and can also calculate
    mean and std on each channel separately.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        keys: KeysCollection,
        mean_intensity: float = None,
        std_intensity: float = None,
        lower_bound: float = None,
        upper_bound: float = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mean_intensity = mean_intensity
        self.std_intensity = std_intensity
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    @staticmethod
    def normalize_intensity(
            img: NdarrayOrTensor,
            mean_intensity: float = None,
            std_intensity: float = None,
            lower_bound: float = None,
            upper_bound: float = None,
            dtype: DtypeLike = np.float32,
    ):
        img = convert_to_tensor(img, track_meta=get_track_meta())
        dtype = dtype or img.dtype
        img = convert_to_tensor(img, track_meta=get_track_meta())
        img = clip(img, a_min=lower_bound, a_max=upper_bound)
        img = (img - mean_intensity) / max(std_intensity, 1e-8)
        # img = MetaTensor(img, meta=img.meta)
        ret: NdarrayOrTensor = convert_data_type(img, dtype=dtype)[0]
        return ret

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.normalize_intensity(img=d[key],
                                              mean_intensity=self.mean_intensity,
                                              std_intensity=self.std_intensity,
                                              lower_bound=self.lower_bound,
                                              upper_bound=self.upper_bound,
                                              )
        return d


class BrightnessMultiplicativeD(RandomizableTransform, MapTransform):
    """
    Adds additive Gaussian Noise
    :param keys: selecting the keys to be transformed
    :param prob: probability of the noise being added, per sample
    :param prob_per_channel: probability of the noise being added, per channel
    CAREFUL: This transform will modify the value range of your data!
    """
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
            self,
            keys: KeysCollection,
            prob: float = 0.1,
            prob_per_channel: float = 1.0,
            per_channel: bool = True,
            multiplier_range: Tuple[float, float] = (0.5, 2),
            allow_missing_keys: bool = False,
    ) -> None:

        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob=prob)
        self.keys = keys
        self.prob = prob
        self.multiplier_range = multiplier_range
        self.per_channel = per_channel
        self.prob_per_channel = prob_per_channel

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "BrightnessMultiplicativeD":
        super().set_random_state(seed=seed, state=state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        for key in self.keys:
            if self._do_transform:
                meta = None
                if get_track_meta():
                    meta = d[key].meta
                dtype = d[key].dtype
                shape = d[key].shape
                assert len(shape) == 4, "img should be 4D array, (c, w, h, d)"

                img = np.array(d[key])
                img = augment_brightness_multiplicative(
                    data_sample=img,
                    multiplier_range=self.multiplier_range,
                    per_channel=self.per_channel,
                )
                if meta:
                    img = MetaTensor(img, meta=meta)
                d[key] = convert_data_type(img, dtype=dtype)[0]

        return d


class ContrastAugmentationD(RandomizableTransform, MapTransform):
    """
    Adds additive Gaussian Noise
    :param keys: selecting the keys to be transformed
    :param prob: probability of the noise being added, per sample
    :param prob_per_channel: probability of the noise being added, per channel
    CAREFUL: This transform will modify the value range of your data!
    """
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
            self,
            keys: KeysCollection,
            prob: float = 0.1,
            prob_per_channel: float = 1.0,
            per_channel: bool = True,
            contrast_range: Tuple[float, float] = (0.75, 1.25),
            preserve_range: bool = True,
            allow_missing_keys: bool = False,
    ) -> None:

        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob=prob)
        self.keys = keys
        self.prob = prob
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range
        self.per_channel = per_channel
        self.prob_per_channel = prob_per_channel

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "ContrastAugmentationD":
        super().set_random_state(seed=seed, state=state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        for key in self.keys:
            if self._do_transform:
                meta = None
                if get_track_meta():
                    meta = d[key].meta
                dtype = d[key].dtype
                shape = d[key].shape
                assert len(shape) == 4, "img should be 4D array, (c, w, h, d)"

                img = np.array(d[key])
                img = augment_contrast(
                    data_sample=img,
                    contrast_range=self.contrast_range,
                    preserve_range=self.preserve_range,
                    per_channel=self.per_channel,
                )
                if meta:
                    img = MetaTensor(img, meta=meta)
                d[key] = convert_data_type(img, dtype=dtype)[0]

        return d


class SimulateLowResolutionD(RandomizableTransform, MapTransform):
    """
    Adds additive Gaussian Noise
    :param keys: selecting the keys to be transformed
    :param prob: probability of the noise being added, per sample
    :param prob_per_channel: probability of the noise being added, per channel
    CAREFUL: This transform will modify the value range of your data!
    """
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
            self,
            keys: KeysCollection,
            prob: float = 0.1,
            prob_per_channel: float = 1.0,
            per_channel: bool = True,
            zoom_range: Tuple[float, float] = (0.5, 1),
            order_downsample: int = 0,
            order_upsample: int = 3,
            ignore_axes: bool = None,
            allow_missing_keys: bool = False,
    ) -> None:

        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob=prob)
        self.keys = keys
        self.prob = prob
        self.zoom_range = zoom_range
        self.order_downsample = order_downsample
        self.order_upsample = order_upsample
        self.ignore_axes = ignore_axes
        self.per_channel = per_channel
        self.prob_per_channel = prob_per_channel

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "SimulateLowResolutionD":
        super().set_random_state(seed=seed, state=state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        for key in self.keys:
            if self._do_transform:
                meta = None
                if get_track_meta():
                    meta = d[key].meta
                dtype = d[key].dtype
                shape = d[key].shape
                assert len(shape) == 4, "img should be 4D array, (c, w, h, d)"

                img = np.array(d[key])
                img = augment_linear_downsampling_scipy(
                    data_sample=img,
                    zoom_range=self.zoom_range,
                    order_downsample=self.order_downsample,
                    order_upsample=self.order_upsample,
                    ignore_axes=self.ignore_axes,
                    per_channel=self.per_channel,
                )

                if meta:
                    img = MetaTensor(img, meta=meta)
                d[key] = convert_data_type(img, dtype=dtype)[0]
        return d


class GammaD(RandomizableTransform, MapTransform):
    """
    Adds additive Gaussian Noise
    :param keys: selecting the keys to be transformed
    :param prob: probability of the noise being added, per sample
    :param prob_per_channel: probability of the noise being added, per channel
    CAREFUL: This transform will modify the value range of your data!
    """
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
            self,
            keys: KeysCollection,
            prob: float = 0.1,
            prob_per_channel: float = 1.0,
            per_channel: bool = True,
            gamma_range: Tuple[float, float] = (0.5, 2),
            invert_image: bool = False,
            retain_stats: bool = False,
            allow_missing_keys: bool = False,
    ) -> None:

        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob=prob)
        self.keys = keys
        self.prob = prob
        self.gamma_range = gamma_range
        self.invert_image = invert_image
        self.retain_stats = retain_stats
        self.per_channel = per_channel
        self.prob_per_channel = prob_per_channel

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "GammaD":
        super().set_random_state(seed=seed, state=state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        for key in self.keys:
            if self._do_transform:
                meta = None
                if get_track_meta():
                    meta = d[key].meta
                dtype = d[key].dtype
                shape = d[key].shape
                assert len(shape) == 4, "img should be 4D array, (c, w, h, d)"

                img = np.array(d[key])
                img = augment_gamma(
                    data_sample=img,
                    gamma_range=self.gamma_range,
                    invert_image=self.invert_image,
                    retain_stats=self.retain_stats,
                    per_channel=self.per_channel,
                )
                if meta:
                    img = MetaTensor(img, meta=meta)
                d[key] = convert_data_type(img, dtype=dtype)[0]

        return d

