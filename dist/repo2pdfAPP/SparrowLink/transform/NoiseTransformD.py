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
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Union, Tuple


class GaussianNoiseD(RandomizableTransform, MapTransform):
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
            noise_variance: Tuple[float, float] = (0.0, 0.1),
            allow_missing_keys: bool = False,
    ) -> None:

        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob=prob)
        self.keys = keys
        self.prob = prob
        self.noise_variance = noise_variance
        self.per_channel = per_channel
        self.prob_per_channel = prob_per_channel

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "GaussianNoiseD":
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
                img = augment_gaussian_noise(data_sample=img,
                                             noise_variance=self.noise_variance,
                                             p_per_channel=self.prob_per_channel,
                                             per_channel=self.per_channel,
                                             )
                if meta:
                    img = MetaTensor(img, meta=meta)
                d[key] = convert_data_type(img, dtype=dtype)[0]

        return d


class GaussianBlurD(RandomizableTransform, MapTransform):
    """
    Adds additive Gaussian Noise
    :param keys: selecting the keys to be transformed
    :param prob: probability of the noise being added
    CAREFUL: This transform will modify the value range of your data!
    """
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
            self,
            keys: KeysCollection,
            blur_sigma: Tuple[float, float] = (0.0, 0.1),
            prob: float = 0.2,
            prob_per_channel: float = 1.0,
            per_channel: bool = True,
            allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob=prob)
        self.prob = prob
        self.blur_sigma = blur_sigma
        self.per_channel = per_channel
        self.prob_per_channel = prob_per_channel

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "GaussianBlurD":
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

                img = augment_gaussian_blur(data_sample=img,
                                            sigma_range=self.blur_sigma,
                                            per_channel=self.per_channel,
                                            p_per_channel=self.prob_per_channel,
                                            )
                if meta:
                    img = MetaTensor(img, meta=meta)
                d[key] = convert_data_type(img, dtype=dtype)[0]
        return d




