import torch
import torch.nn as nn
import torch.nn.functional as F


class LUConv(nn.Module):
    def __init__(self, nchan: int, act: nn.Module = nn.ELU, bias: bool = False):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(nchan, nchan, kernel_size=5, padding=2, bias=bias),
            nn.BatchNorm2d(nchan),
            act(inplace=True),
        )

    def forward(self, x):
        out = self.layers(x)
        return out


def _make_nconv(nchan: int, depth: int, act: nn.Module = nn.ELU, bias: bool = False):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan=nchan, act=act, bias=bias))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 16,
        act: nn.Module = nn.ELU,
        bias: bool = False,
    ):
        super().__init__()

        if out_channels % in_channels != 0:
            raise ValueError(
                f"16 should be divisible by in_channels, got in_channels={in_channels}."
            )

        self.in_channels = in_channels
        self.act_function = act(inplace=True)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, bias=bias),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.conv_block(x)
        x16 = x.repeat(1, 16 // self.in_channels, 1, 1)
        out = self.act_function(out + x16)
        return out


class DownTransition(nn.Module):
    def __init__(
        self,
        in_channels: int,
        nconvs: int,
        act: nn.Module = nn.ELU,
        dropout_prob: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        out_channels = 2 * in_channels
        self.down_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=2, stride=2, bias=bias
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act_function1 = act(inplace=True)
        self.act_function2 = act(inplace=True)
        self.ops = _make_nconv(out_channels, nconvs, act, bias)
        self.dropout = nn.Dropout2d(dropout_prob) if dropout_prob > 0.0 else None

    def forward(self, x):
        down = self.act_function1(self.bn1(self.down_conv(x)))
        if self.dropout is not None:
            out = self.dropout(down)
        else:
            out = down
        out = self.ops(out)
        out = self.act_function2(out + down)
        return out


class UpTransition(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        nconvs: int,
        act: nn.Module = nn.ELU,
        dropout_prob: float = 0.0,
    ):
        super().__init__()

        self.up_conv = nn.ConvTranspose2d(
            in_channels, out_channels // 2, kernel_size=2, stride=2
        )
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.dropout = nn.Dropout2d(dropout_prob) if dropout_prob > 0.0 else None
        self.dropout2 = nn.Dropout2d(0.5)
        self.act_function1 = act(inplace=True)
        self.act_function2 = act(inplace=True)
        self.ops = _make_nconv(out_channels, nconvs, act)

    def forward(self, x, skipx):
        if self.dropout is not None:
            out = self.dropout(x)
        else:
            out = x
        skipxdo = self.dropout2(skipx)
        out = self.act_function1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.act_function2(out + xcat)
        return out


class OutputTransition(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act: nn.Module = nn.ELU,
        bias: bool = False,
    ):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, bias=bias),
            nn.BatchNorm2d(out_channels),
            act(inplace=True),
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.conv_block(x)
        out = self.conv2(out)
        return out


class Vnet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        act: nn.Module = nn.ELU,
        dropout_prob: float = 0.5,
        bias: bool = False,
    ):
        super().__init__()

        self.in_tr = InputTransition(in_channels, 16, act, bias=bias)
        self.down_tr32 = DownTransition(16, 1, act, bias=bias)
        self.down_tr64 = DownTransition(32, 2, act, bias=bias)
        self.down_tr128 = DownTransition(
            64, 3, act, dropout_prob=dropout_prob, bias=bias
        )
        self.down_tr256 = DownTransition(
            128, 2, act, dropout_prob=dropout_prob, bias=bias
        )
        self.up_tr256 = UpTransition(256, 256, 2, act, dropout_prob=dropout_prob)
        self.up_tr128 = UpTransition(256, 128, 2, act, dropout_prob=dropout_prob)
        self.up_tr64 = UpTransition(128, 64, 1, act)
        self.up_tr32 = UpTransition(64, 32, 1, act)
        self.out_tr = OutputTransition(32, out_channels, act, bias=bias)

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        x = self.up_tr256(out256, out128)
        x = self.up_tr128(x, out64)
        x = self.up_tr64(x, out32)
        x = self.up_tr32(x, out16)
        x = self.out_tr(x)
        return x
