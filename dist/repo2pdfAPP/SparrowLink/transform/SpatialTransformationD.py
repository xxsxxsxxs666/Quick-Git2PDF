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
from monai.transforms import MapTransform, RandomizableTransform, Randomizable, InvertibleTransform

from batchgenerators.augmentations.noise_augmentations import augment_gaussian_blur, augment_gaussian_noise, \
    augment_rician_noise
from batchgenerators.augmentations.color_augmentations import augment_contrast, augment_brightness_additive, \
    augment_brightness_multiplicative, augment_gamma, augment_illumination, augment_PCA_shift
from batchgenerators.augmentations.resample_augmentations import augment_linear_downsampling_scipy
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Union, Tuple


class PermuteD(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Orientation`.

    This transform assumes the channel-first input format.
    In the case of using this transform for normalizing the orientations of images,
    it should be used before any anisotropic spatial transforms.
    """

    backend = [TransformBackends.NUMPY, TransformBackends.TORCH]

    def __init__(
        self,
        keys: KeysCollection,
        axcodes: Optional[str] = None,
        as_closest_canonical: bool = False,
        labels: Optional[Sequence[Tuple[str, str]]] = (("L", "R"), ("P", "A"), ("I", "S")),
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = "meta_dict",
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            axcodes: N elements sequence for spatial ND input's orientation.
                e.g. axcodes='RAS' represents 3D orientation:
                (Left, Right), (Posterior, Anterior), (Inferior, Superior).
                default orientation labels options are: 'L' and 'R' for the first dimension,
                'P' and 'A' for the second, 'I' and 'S' for the third.
            as_closest_canonical: if True, load the image as closest to canonical axis format.
            labels: optional, None or sequence of (2,) sequences
                (2,) sequences are labels for (beginning, end) of output axis.
                Defaults to ``(('L', 'R'), ('P', 'A'), ('I', 'S'))``.
            allow_missing_keys: don't raise exception if key is missing.

        See Also:
            `nibabel.orientations.ornt2axcodes`.

        """
        super().__init__(keys, allow_missing_keys)
        self.ornt_transform = Orientation(axcodes=axcodes, as_closest_canonical=as_closest_canonical, labels=labels)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.ornt_transform(d[key])
        return d

    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.ornt_transform.inverse(d[key])
        return d