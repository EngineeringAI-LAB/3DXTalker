import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TCN(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.first_net = SeqTranslator1D(
            in_dim, out_dim, min_layers_num=3, residual=True, norm='ln')
        self.dropout = nn.Dropout(0.1)

    def forward(self, spectrogram, time_steps=None):
        spectrogram = spectrogram
        # spectrogram = self.dropout(spectrogram)
        spectrogram = spectrogram.permute(0, 2, 1)
        x1 = self.first_net(spectrogram)  # .permute(0, 2, 1)
        if time_steps is not None:
            x1 = F.interpolate(
                x1, size=time_steps, align_corners=False, mode='linear')
        x1 = x1.permute(0, 2, 1)
        return self.dropout(x1)


class SeqTranslator1D(nn.Module):
    """(B, C, T)->(B, C_out, T)"""

    def __init__(self,
                 C_in,
                 C_out,
                 kernel_size=None,
                 stride=None,
                 min_layers_num=None,
                 residual=True,
                 norm='bn'):
        super(SeqTranslator1D, self).__init__()

        conv_layers = nn.ModuleList([])
        conv_layers.append(
            ConvNormRelu(
                in_channels=C_in,
                out_channels=C_out,
                type='1d',
                kernel_size=kernel_size,
                stride=stride,
                residual=residual,
                norm=norm))
        self.num_layers = 1
        if min_layers_num is not None and self.num_layers < min_layers_num:
            while self.num_layers < min_layers_num:
                conv_layers.append(
                    ConvNormRelu(
                        in_channels=C_out,
                        out_channels=C_out,
                        type='1d',
                        kernel_size=kernel_size,
                        stride=stride,
                        residual=residual,
                        norm=norm))
                self.num_layers += 1
        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        return self.conv_layers(x)


class ConvNormRelu(nn.Module):
    """(B,C_in,H,W) -> (B, C_out, H, W) there exist some kernel size that makes
    the result is not H/s.

    #TODO: there might some problems with residual
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 type='1d',
                 leaky=False,
                 downsample=False,
                 kernel_size=None,
                 stride=None,
                 padding=None,
                 p=0,
                 groups=1,
                 residual=False,
                 norm='bn'):
        """conv-bn-relu."""
        super(ConvNormRelu, self).__init__()
        self.residual = residual
        self.norm_type = norm
        # kernel_size = k
        # stride = s

        if kernel_size is None and stride is None:
            if not downsample:
                kernel_size = 3
                stride = 1
            else:
                kernel_size = 4
                stride = 2

        if padding is None:
            if isinstance(kernel_size, int) and isinstance(stride, tuple):
                padding = tuple(int((kernel_size - st) / 2) for st in stride)
            elif isinstance(kernel_size, tuple) and isinstance(stride, int):
                padding = tuple(int((ks - stride) / 2) for ks in kernel_size)
            elif isinstance(kernel_size, tuple) and isinstance(stride, tuple):
                padding = tuple(
                    int((ks - st) / 2) for ks, st in zip(kernel_size, stride))
            else:
                padding = int((kernel_size - stride) / 2)

        if self.residual:
            if downsample:
                if type == '1d':
                    self.residual_layer = nn.Sequential(
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding))
                elif type == '2d':
                    self.residual_layer = nn.Sequential(
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding))
            else:
                if in_channels == out_channels:
                    self.residual_layer = nn.Identity()
                else:
                    if type == '1d':
                        self.residual_layer = nn.Sequential(
                            nn.Conv1d(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding))
                    elif type == '2d':
                        self.residual_layer = nn.Sequential(
                            nn.Conv2d(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding))

        in_channels = in_channels * groups
        out_channels = out_channels * groups
        if type == '1d':
            self.conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups)
            self.norm = nn.BatchNorm1d(out_channels)
            self.dropout = nn.Dropout(p=p)
        elif type == '2d':
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups)
            self.norm = nn.BatchNorm2d(out_channels)
            self.dropout = nn.Dropout2d(p=p)
        if norm == 'gn':
            self.norm = nn.GroupNorm(2, out_channels)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(out_channels)
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.relu = nn.ReLU()

    def forward(self, x, **kwargs):
        if self.norm_type == 'ln':
            out = self.dropout(self.conv(x))
            out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        else:
            out = self.norm(self.dropout(self.conv(x)))
        if self.residual:
            residual = self.residual_layer(x)
            out += residual
        return self.relu(out)