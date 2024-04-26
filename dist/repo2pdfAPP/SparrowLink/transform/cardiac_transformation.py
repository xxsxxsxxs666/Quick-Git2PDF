import numpy as np
import torch
from monai.utils import TransformBackends
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Union
from monai.config import IndexSelection, KeysCollection, SequenceStr, NdarrayOrTensor, DtypeLike, KeysCollection
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type, convert_to_tensor, get_equivalent_dtype
from monai.data.meta_tensor import MetaTensor
from monai.metrics.utils import get_surface_distance, get_mask_edges


from monai.transforms import (
    MapTransform,
)
from openpyxl import Workbook, load_workbook
import os
from scipy import ndimage
import numpy as np


class UseHeartsegDeleteInformationd(MapTransform):

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        keys: KeysCollection,
        heart_key: str = 'heart',
        vessel_key: str = 'vessel',
        heart_dilation_time: int = 1,
        vessel_dilation_time: int = 1,
        dilation_struct: int = 1,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        test_key: show whether your mask is reserved totally.
        """
        self.heart_dilation_time = heart_dilation_time
        self.vessel_dilation_time = vessel_dilation_time
        self.vessel_key = vessel_key
        self.heart_key = heart_key
        self.dilation_struct = dilation_struct

        super().__init__(keys, allow_missing_keys=allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        heart_region = convert_to_tensor(data=d[self.heart_key])
        vessel_region = convert_to_tensor(data=d[self.vessel_key])
        heart_region = heart_region.clip(min=0, max=1) > 0
        vessel_region = vessel_region.clip(min=0, max=1) > 0
        if self.heart_dilation_time > 1:
            heart_region = self.dilation(x=heart_region, dilation_time=self.heart_dilation_time)
        if self.vessel_dilation_time > 1:
            vessel_region = self.dilation(x=vessel_region, dilation_time=self.vessel_dilation_time)
        region = (heart_region + vessel_region) > 0
        d[self.heart_key] = MetaTensor(region, meta=d[self.heart_key].meta)
        for key in self.key_iterator(d):
            if key not in [self.heart_key, self.vessel_key]:
                d[key] = d[key] * region
        return d

    def dilation(self, x: torch.Tensor, dilation_time: int = 1):
        shape = x.shape
        if len(shape) > 3:
            x = x.squeeze()
        x = np.array(x.squeeze())
        struct1 = ndimage.generate_binary_structure(3, self.dilation_struct)
        x = ndimage.binary_dilation(x, structure=struct1, iterations=dilation_time).astype(x.dtype)
        if len(shape) > 3:
            return torch.tensor(x).unsqueeze(dim=0)
        else:
            return torch.tensor(x)
