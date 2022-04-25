import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange

from losses import DiceLoss


class LUConv(nn.Module):
    def __init__(self, nchan: int, act: nn.Module = nn.ELU, bias: bool = False):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(nchan, nchan, kernel_size=5, padding=2, bias=bias),
            nn.BatchNorm2d(nchan),
            act(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


def _make_nconv(nchan: int, depth: int, act: nn.Module = nn.ELU, bias: bool = False):
    layers = [LUConv(nchan=nchan, act=act, bias=bias) for _ in range(depth)]
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
        out = self.dropout(down) if self.dropout is not None else down
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
        out = self.dropout(x) if self.dropout is not None else x
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


class VnetModule(pl.LightningModule):
    def __init__(
        self,
        in_chans=1,
        out_chans=1,
        drop_prob=0.1,
        lr=0.001,
        weight_decay=0.0,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        self.lr = lr
        self.weight_decay = weight_decay

        self.model = Vnet(
            in_channels=self.hparams.in_chans,
            out_channels=self.hparams.out_chans,
            dropout_prob=self.hparams.drop_prob,
        )
        self.example_input_array = torch.rand(
            1, self.hparams.in_chans, 256, 256, device=self.device
        )

        self.dice_loss = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x):
        with torch.no_grad():
            x = F.group_norm(x, num_groups=1)
        x = self.model(x)
        return x

    def step(self, batch, batch_indx=None):
        input, target = batch

        if len(input.shape) == 5:
            input = rearrange(input, "b t c h w -> (b t) c h w")
        if len(target.shape) == 5:
            target = rearrange(target, "b t c h w -> (b t) c h w")

        output = self(input)

        if torch.any(torch.isnan(output)):
            print(output)
            raise ValueError

        loss_dict = {
            "cross_entropy": self.cross_entropy(
                output, torch.argmax(target, dim=1).long()
            )
        }

        dice_loss, dice_score = self.dice_loss(output, target)
        loss_dict["dice_loss"] = dice_loss.mean()
        loss_dict["dice_score"] = dice_score.detach()
        loss_dict["loss"] = loss_dict["cross_entropy"] + loss_dict["dice_loss"]
        return loss_dict, output

    def training_step(self, batch, batch_idx):
        loss_dict, output = self.step(batch, batch_idx)
        for metric, value in loss_dict.items():
            self.log(f"train_{metric}", value.mean().detach())
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        loss_dict, output = self.step(batch, batch_idx)
        for metric, value in loss_dict.items():
            self.log(f"val_{metric}", value.mean().detach(), sync_dist=True)
        return loss_dict, output

    def test_step(self, batch, batch_idx):
        loss_dict, output = self.step(batch, batch_idx)
        for metric, value in loss_dict.items():
            self.log(f"test_{metric}", value.mean().detach(), sync_dist=True)
        return loss_dict

    def predict_step(self, batch, batch_idx, dataloader_idx):
        input, _ = batch
        return self(input)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim, mode="min", factor=0.5, patience=5, cooldown=1
            ),
            "monitor": "val_loss",
        }

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("UnetModel")

        # network params
        parser.add_argument(
            "--in_chans", default=2, type=int, help="Number of V-Net input channels"
        )
        parser.add_argument(
            "--out_chans", default=7, type=int, help="Number of V-Net output chanenls"
        )
        parser.add_argument(
            "--drop_prob", default=0.5, type=float, help="U-Net dropout probability"
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=1e-3, type=float, help="Optimizer learning rate"
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )

        return parent_parser
