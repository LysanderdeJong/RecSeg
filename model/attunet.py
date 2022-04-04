import torch
from torch import nn
from torch.nn import functional as F

from model.unet import ConvBlock, TransposeConvBlock


class AttentionGate(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans_x: int, in_chans_g: int, out_chans: int):
        """
        Args:
            in_chans-_x: Number of channels in the input.
            in_chans-_x: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans_g = in_chans_g
        self.in_chans_x = in_chans_x
        self.out_chans = out_chans

        self.W_g = nn.Sequential(
            nn.Conv2d(in_chans_g, out_chans, kernel_size=1, padding=0, bias=True)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(
                in_chans_x, out_chans, kernel_size=2, padding=0, stride=2, bias=False
            )
        )

        self.psi = nn.Sequential(
            nn.Conv2d(out_chans, 1, kernel_size=1, padding=0, bias=True)
        )

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input 4D tensor of shape `(N, in_chans_x, H, W)`.
            g: Input 4D tensor of shape `(N, in_chans_g, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        # print(x.shape, g.shape)
        print(x.shape, g.shape)
        W_x = self.W_x(x)
        w_g = self.W_g(g)
        print(x.shape, g.shape)
        W_g = F.interpolate(
            w_g,
            size=(W_x.shape[-2], W_x.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        print(x.shape, g.shape)
        f = F.relu(W_x + W_g, inplace=True)
        a = torch.sigmoid(self.psi(f))
        a = F.interpolate(
            a, size=(x.shape[-2], x.shape[-1]), mode="bilinear", align_corners=False
        )
        return a * x


class AttUnet(nn.Module):
    """
    PyTorch implementation of a Attention U-Net model.
    https://arxiv.org/pdf/1804.03999v3.pdf.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
        block=ConvBlock,
        **kwargs
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(block(ch, ch * 2, drop_prob, **kwargs))
            ch *= 2
        self.conv = block(ch, ch * 2, drop_prob, **kwargs)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        self.up_attention_gates = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            self.up_attention_gates.append(AttentionGate(ch, ch * 2, ch))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob, **kwargs),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )
        self.up_attention_gates.append(AttentionGate(ch, ch * 2, ch))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv, attention_gate in zip(
            self.up_transpose_conv, self.up_conv, self.up_attention_gates
        ):
            downsample_layer = stack.pop()
            downsample_layer = attention_gate(downsample_layer, output)
            output = transpose_conv(output)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output
