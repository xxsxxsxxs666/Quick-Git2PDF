"""
3D Channel and Spatial Attention Network (CSA-Net 3D).
"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


def downsample():
    return nn.MaxPool3d(kernel_size=2, stride=2)


def deconv(in_channels, out_channels):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ResEncoder3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResEncoder3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = residual + out
        out = self.relu(out)
        return out


class Decoder3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class SpatialAttentionBlock3d(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionBlock3d, self).__init__()
        self.query = nn.Conv3d(in_channels, in_channels // 8, kernel_size=(1, 3, 1), padding=(0, 1, 0))
        self.key = nn.Conv3d(in_channels, in_channels // 8, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.judge = nn.Conv3d(in_channels, in_channels // 8, kernel_size=(1, 1, 3), padding=(0, 0, 1))
        self.value = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxWxZ )
        :return: affinity value + x
        B: batch size
        C: channels
        H: height
        W: width
        D: slice number (depth)
        """
        B, C, H, W, D = x.size()
        # compress x: [B,C,H,W,Z]-->[B,H*W*Z,C], make a matrix transpose
        proj_query = self.query(x).view(B, -1, W * H * D).permute(0, 2, 1)  # -> [B,W*H*D,C]
        proj_key = self.key(x).view(B, -1, W * H * D)  # -> [B,H*W*D,C]
        proj_judge = self.judge(x).view(B, -1, W * H * D).permute(0, 2, 1)  # -> [B,C,H*W*D]

        affinity1 = torch.matmul(proj_query, proj_key)
        affinity2 = torch.matmul(proj_judge, proj_key)
        affinity = torch.matmul(affinity1, affinity2)
        affinity = self.softmax(affinity)

        proj_value = self.value(x).view(B, -1, H * W * D)  # -> C*N
        weights = torch.matmul(proj_value, affinity)
        weights = weights.view(B, C, H, W, D)
        out = self.gamma * weights + x
        return out


class ChannelAttentionBlock3d(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionBlock3d, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxWxD )
        :return: affinity value + x
        """
        B, C, H, W, D = x.size()
        proj_query = x.view(B, C, -1).permute(0, 2, 1)
        proj_key = x.view(B, C, -1)
        proj_judge = x.view(B, C, -1).permute(0, 2, 1)
        affinity1 = torch.matmul(proj_key, proj_query)
        affinity2 = torch.matmul(proj_key, proj_judge)
        affinity = torch.matmul(affinity1, affinity2)
        affinity_new = torch.max(affinity, -1, keepdim=True)[0].expand_as(affinity) - affinity
        affinity_new = self.softmax(affinity_new)
        proj_value = x.view(B, C, -1)
        weights = torch.matmul(affinity_new, proj_value)
        weights = weights.view(B, C, H, W, D)
        out = self.gamma * weights + x
        return out


class AffinityAttention3d(nn.Module):
    """ Affinity attention module """

    def __init__(self, in_channels):
        super(AffinityAttention3d, self).__init__()
        self.sab = SpatialAttentionBlock3d(in_channels)
        self.cab = ChannelAttentionBlock3d(in_channels)
        # self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x):
        """
        sab: spatial attention block
        cab: channel attention block
        :param x: input tensor
        :return: sab + cab
        """
        sab = self.sab(x)
        cab = self.cab(x)
        out = sab + cab + x
        return out


class CSNet3D(nn.Module):
    def __init__(self, out_channels, in_channels):
        """
        :param classes: the object classes number.
        :param channels: the channels of the input image.
        """
        super(CSNet3D, self).__init__()
        self.out_channels = out_channels
        self.enc_input = ResEncoder3d(in_channels, 16)
        self.encoder1 = ResEncoder3d(16, 32)
        self.encoder2 = ResEncoder3d(32, 64)
        self.encoder3 = ResEncoder3d(64, 128)
        self.encoder4 = ResEncoder3d(128, 256)
        self.downsample = downsample()
        self.affinity_attention = AffinityAttention3d(256)
        self.attention_fuse = nn.Conv3d(256 * 2, 256, kernel_size=1)
        self.decoder4 = Decoder3d(256, 128)
        self.decoder3 = Decoder3d(128, 64)
        self.decoder2 = Decoder3d(64, 32)
        self.decoder1 = Decoder3d(32, 16)
        self.deconv4 = deconv(256, 128)
        self.deconv3 = deconv(128, 64)
        self.deconv2 = deconv(64, 32)
        self.deconv1 = deconv(32, 16)
        self.final = nn.Conv3d(16, out_channels, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        enc_input = self.enc_input(x)
        down1 = self.downsample(enc_input)

        enc1 = self.encoder1(down1)
        down2 = self.downsample(enc1)

        enc2 = self.encoder2(down2)
        down3 = self.downsample(enc2)

        enc3 = self.encoder3(down3)
        down4 = self.downsample(enc3)

        input_feature = self.encoder4(down4)

        # Do Attenttion operations here
        attention = self.affinity_attention(input_feature)
        attention_fuse = input_feature + attention

        # Do decoder operations here
        up4 = self.deconv4(attention_fuse)
        up4 = torch.cat((enc3, up4), dim=1)
        dec4 = self.decoder4(up4)

        up3 = self.deconv3(dec4)
        up3 = torch.cat((enc2, up3), dim=1)
        dec3 = self.decoder3(up3)

        up2 = self.deconv2(dec3)
        up2 = torch.cat((enc1, up2), dim=1)
        dec2 = self.decoder2(up2)

        up1 = self.deconv1(dec2)
        up1 = torch.cat((enc_input, up1), dim=1)
        dec1 = self.decoder1(up1)

        final = self.final(dec1)
        final = F.sigmoid(final)
        return final


if __name__ == "__main__":
    from torch.distributions.normal import Normal
    model1 = CSNet3D(2, 1)
    model2 = CSNet3D(2, 4)


    # from torch.distributions.normal import Normal
    # d1 = model1.state_dict()
    # d2 = model2.state_dict()
    # for k2, v2 in d2.items():
    #     if d1[k2].shape != v2.shape:
    #         assert d1[k2].shape[1] != v2.shape[1] and len(d1[k2].shape) > 1 and len(v2.shape) > 1
    #         noise_shape = torch.tensor(v2.shape)
    #         noise_shape[1] = v2.shape[1] - d1[k2].shape[1]
    #         noise = nn.Parameter(Normal(0, 1e-5).sample(noise_shape)).to(d1[k2].device)
    #         d1[k2] = torch.cat((d1[k2], noise), dim=1)
    #         print(f"key: {k2}, shape1: {d1[k2].shape}, mean1: {d1[k2].mean()}, shape2: {v2.shape}, mean2: {v2.mean()}")

    d1 = torch.load('H:/Graduate_project/segment_server/experiments/Graduate_project/two_stage2/first_stage/CS2net/checkpoint/best_metric_model.pth')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1.to(device)
    model1.load_state_dict(d1)
    d2 = model2.state_dict()
    print("-----------------------------load checkpoint and modify-------------------------------------------")
    for k2, v2 in d2.items():
        if d1[k2].shape != v2.shape:
            assert d1[k2].shape[1] != v2.shape[1] and len(d1[k2].shape) > 1 and len(v2.shape) > 1
            noise_shape = torch.tensor(v2.shape)
            noise_shape[1] = v2.shape[1] - d1[k2].shape[1]
            noise = nn.Parameter(Normal(0, 1e-7).sample(noise_shape)).to(d1[k2].device)
            print(f"key: {k2}, shape1: {d1[k2].shape}, mean1: {d1[k2].mean()}, shape2: {v2.shape}, mean2: {v2.mean()}")
            d1[k2] = torch.cat((d1[k2], noise), dim=1)
    model2.to(device)
    model2.load_state_dict(d1)
    x1 = torch.ones(1, 1, 64, 64, 64).to(device)
    x2 = torch.ones(1, 4, 64, 64, 64).to(device)
    y1 = model1(x1)
    y2 = model2(x2)
    print(((y1 - y2)**2).max(), ((y1 - y2)**2).mean())
    pass
