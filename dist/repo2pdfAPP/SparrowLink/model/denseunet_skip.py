import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F


from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
from monai.utils import alias, deprecated_arg, export
from monai.networks.layers.convutils import same_padding

@export("monai.networks.nets")
@alias("Unet")
class SkipDenseUNet(nn.Module):

    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            channels: Sequence[int],
            strides: Sequence[int],
            kernel_size: Union[Sequence[int], int] = 3,
            up_kernel_size: Union[Sequence[int], int] = 3,
            num_res_units: int = 0,
            act: Union[Tuple, str] = Act.PRELU,
            norm: Union[Tuple, str] = Norm.INSTANCE,
            dropout: float = 0.0,
            bias: bool = True,
            adn_ordering: str = "NDA",
            dimensions: Optional[int] = None,
    ) -> None:

        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if dimensions is not None:
            spatial_dims = dimensions
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != spatial_dims:
                raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence):
            if len(up_kernel_size) != spatial_dims:
                raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        def _create_block(
                inc: int, outc: int, channels: Sequence[int], strides: Sequence[int], is_top: bool
        ) -> nn.Module:
            """
            Builds the UNet structure from the bottom up by recursing down to the bottom block, then creating sequential
            blocks containing the downsample path, a skip connection around the previous block, and the upsample path.

            Args:
                inc: number of input channels.
                outc: number of output channels.
                channels: sequence of channels. Top block first.
                strides: convolution stride.
                is_top: True if this is the top block.
            """
            c = channels[0]
            s = strides[0]

            subblock: nn.Module

            if len(channels) > 2:
                subblock = _create_block(c, c, channels[1:], strides[1:], False)  # continue recursion down
                upc = c * 2
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = self._get_bottom_layer(c, channels[1])
                upc = c + channels[1]

            down = self._get_down_layer(inc, c, s, is_top)  # create layer in downsampling path
            up = self._get_up_layer(upc, outc, s, is_top)  # create layer in upsampling path

            return self._get_connection_block(down, up, subblock)

        self.model = _create_block(in_channels, out_channels, self.channels, self.strides, True)

    def _get_connection_block(self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module) -> nn.Module:
        """
        Returns the block object defining a layer of the UNet structure including the implementation of the skip
        between encoding (down) and decoding (up) sides of the network.

        Args:
            down_path: encoding half of the layer
            up_path: decoding half of the layer
            subblock: block defining the next layer in the network.
        Returns: block for this layer: `nn.Sequential(down_path, SkipConnection(subblock), up_path)`
        """
        return nn.Sequential(down_path, SkipConnection(subblock), up_path)

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the encoding (down) part of a layer of the network. This typically will downsample data at some point
        in its structure. Its output is used as input to the next layer down and is concatenated with output from the
        next layer to form the input for the decode (up) part of the layer.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        mod: nn.Module
        mod = DenseUnit(
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
        )
        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """
        Returns the bottom or bottleneck layer at the bottom of the network linking encode to decode halves.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
        """
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the decoding (up) part of a layer of the network. This typically will upsample data at some point
        in its structure. Its output is used as input to the next layer up.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        conv: Union[Convolution, nn.Sequential]

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)

        return conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x


class DenseUnit(nn.Module):
    """
    Residual module with multiple convolutions and a residual connection.

    For example:

    .. code-block:: python

        from monai.networks.blocks import ResidualUnit

        convs = ResidualUnit(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            adn_ordering="AN",
            act=("prelu", {"init": 0.2}),
            norm=("layer", {"normalized_shape": (10, 10, 10)}),
        )
        print(convs)

    output::

        DenseUnit(
          (conv): Sequential(
            (unit0): Convolution(
              (conv): Conv3d(1, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
              (adn): ADN(
                (A): PReLU(num_parameters=1)
                (N): LayerNorm((10, 10, 10), eps=1e-05, elementwise_affine=True)
              )
            )
            (unit1): Convolution(
              (conv): Conv3d(1, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
              (adn): ADN(
                (A): PReLU(num_parameters=1)
                (N): LayerNorm((10, 10, 10), eps=1e-05, elementwise_affine=True)
              )
            )
          )
          (residual): Identity()
        )

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        strides: Union[Sequence[int], int] = 1,
        kernel_size: Union[Sequence[int], int] = 3,
        subunits: int = 2,
        adn_ordering: str = "NDA",
        act: Optional[Union[Tuple, str]] = "PRELU",
        norm: Optional[Union[Tuple, str]] = "INSTANCE",
        dropout: Optional[Union[Tuple, str, float]] = None,
        dropout_dim: Optional[int] = 1,
        dilation: Union[Sequence[int], int] = 1,
        bias: bool = True,
        last_conv_only: bool = False,
        padding: Optional[Union[Sequence[int], int]] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strides = strides
        self.dense_blocks = DenseBlock(in_channels=self.in_channels,
                                       out_channels=self.out_channels,
                                       nb_layers=subunits)
        self.transition = TransitionLayer(in_channels=self.out_channels,
                                          out_channels=self.out_channels,
                                          stride=self.strides)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_blocks(x)
        x = self.transition(x)
        return x


class BottleneckLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BottleneckLayer, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        return out


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(TransitionLayer, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_layers=4):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(nb_layers):
            self.layers.append(BottleneckLayer(in_channels + i * out_channels, out_channels))
        self.merge = nn.Sequential(
            nn.Conv3d(in_channels + nb_layers * out_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        layers_concat = [x]
        for layer in self.layers:
            out = layer(Concatenation(layers_concat))
            layers_concat.append(out)
        return self.merge(Concatenation(layers_concat))


def Concatenation(layers):
    return torch.cat(layers, dim=1)


if __name__ == "__main__":
    import torch

    model1 = SkipDenseUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,)

    model2 = SkipDenseUNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH, )

    from torch.distributions.normal import Normal
    d1 = torch.load('H:/Graduate_project/segment_server/experiments/Graduate_project/two_stage2/first_stage/DenseUnet/checkpoint/best_metric_model.pth')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1.to(device)
    model1.load_state_dict(d1)
    d2 = model2.state_dict()
    in_channels = 4
    print("-----------------------------load checkpoint and modify-------------------------------------------")
    for k2, v2 in d2.items():
        if d1[k2].shape != v2.shape:
            assert d1[k2].shape[1] != v2.shape[1] and len(d1[k2].shape) > 1 and len(v2.shape) > 1
            noise_shape = torch.tensor(v2.shape)
            noise_shape[1] = v2.shape[1] - d1[k2].shape[1]
            noise = nn.Parameter(Normal(0, 1e-10).sample(noise_shape)).to(d1[k2].device)
            print(f"key: {k2}, shape1: {d1[k2].shape}, mean1: {d1[k2].mean()}, shape2: {v2.shape}, mean2: {v2.mean()}")
            d1[k2] = torch.cat((d1[k2][:, :1, ...], noise, d1[k2][:, 1:, ...]), dim=1)
    model2.to(device)
    model2.load_state_dict(d1)
    x1 = torch.ones(1, 1, 64, 64, 64).to(device)
    x2 = torch.ones(1, 4, 64, 64, 64).to(device)
    y1 = model1(x1)
    y2 = model2(x2)
    print(((y1 - y2)**2).max(), ((y1 - y2)**2).mean())
    pass
