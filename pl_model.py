from collections import defaultdict
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics import functional as FM
from argparse import ArgumentParser
from einops import rearrange

from model.unet import Unet
from model.lambda_layer import LambdaBlock
from model.vnet import Vnet
from model.deeplab import DeepLab
from model.unet3d import Unet3d
from model.attunet import AttUnet
from model.cirim import CIRIM
from losses import DiceLoss, ContourLoss


class UnetModule(pl.LightningModule):
    def __init__(
        self,
        in_chans=1,
        out_chans=1,
        chans=32,
        num_pool_layers=4,
        drop_prob=0.1,
        lr=0.001,
        lr_step_size=40,
        lr_gamma=0.0,
        weight_decay=0.0,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.lr = lr
        self.weight_decay = weight_decay

        self.model = Unet(
            in_chans=self.hparams.in_chans,
            out_chans=self.hparams.out_chans,
            chans=self.hparams.chans,
            num_pool_layers=self.hparams.num_pool_layers,
            drop_prob=self.hparams.drop_prob,
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

        loss_dict = {}
        loss_dict["cross_entropy"] = self.cross_entropy(
            output, torch.argmax(target, dim=1).long()
        )
        dice_loss, dice_score = self.dice_loss(output, target)
        loss_dict["dice_loss"] = dice_loss.mean()
        loss_dict["dice_score"] = dice_score.detach()
        loss_dict["loss"] = loss_dict["cross_entropy"] + loss_dict["dice_loss"]
        return loss_dict, output

    def training_step(self, batch, batch_idx):
        loss_dict, output = self.step(batch, batch_idx)
        for metric, value in zip(loss_dict.keys(), loss_dict.values()):
            self.log(f"train_{metric}", value.mean().detach())
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        loss_dict, output = self.step(batch, batch_idx)
        for metric, value in zip(loss_dict.keys(), loss_dict.values()):
            self.log(f"val_{metric}", value.mean().detach(), sync_dist=True)
        return output, loss_dict

    def test_step(self, batch, batch_idx):
        loss_dict, output = self.step(batch, batch_idx)
        for metric, value in zip(loss_dict.keys(), loss_dict.values()):
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
                optim, mode="min", factor=0.1, patience=7, cooldown=1
            ),
            "monitor": "val_loss",
        }

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("UnetModel")

        # network params
        parser.add_argument(
            "--in_chans", default=2, type=int, help="Number of U-Net input channels"
        )
        parser.add_argument(
            "--out_chans", default=7, type=int, help="Number of U-Net output chanenls"
        )
        parser.add_argument(
            "--chans", default=32, type=int, help="Number of top-level U-Net filters."
        )
        parser.add_argument(
            "--num_pool_layers",
            default=4,
            type=int,
            help="Number of U-Net pooling layers.",
        )
        parser.add_argument(
            "--drop_prob", default=0.1, type=float, help="U-Net dropout probability"
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


class LamdaUnetModule(pl.LightningModule):
    def __init__(self, in_chans=1, out_chans=1, chans=32, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = Unet(
            in_chans=self.hparams.in_chans,
            out_chans=self.hparams.out_chans,
            chans=self.hparams.chans,
            num_pool_layers=self.hparams.num_pool_layers,
            drop_prob=self.hparams.drop_prob,
            block=LambdaBlock,
            temporal_kernel=self.hparams.tr,
            num_slices=self.hparams.num_slices,
        )
        self.example_input_array = torch.rand(
            3, self.hparams.in_chans, 256, 256, device=self.device
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

        loss_dict = {}
        loss_dict["cross_entropy"] = self.cross_entropy(
            output, torch.argmax(target, dim=1).long()
        )
        dice_loss, dice_score = self.dice_loss(output, target)
        loss_dict["dice_loss"] = dice_loss.mean()
        loss_dict["dice_score"] = dice_score.detach()
        loss_dict["loss"] = loss_dict["cross_entropy"] + loss_dict["dice_loss"]
        return loss_dict, output

    def training_step(self, batch, batch_idx):
        loss_dict, output = self.step(batch, batch_idx)
        for metric, value in zip(loss_dict.keys(), loss_dict.values()):
            self.log(f"train_{metric}", value.mean().detach())
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        loss_dict, output = self.step(batch, batch_idx)
        for metric, value in zip(loss_dict.keys(), loss_dict.values()):
            self.log(f"val_{metric}", value.mean().detach(), sync_dist=True)
        return output, loss_dict

    def test_step(self, batch, batch_idx):
        loss_dict, output = self.step(batch, batch_idx)
        for metric, value in zip(loss_dict.keys(), loss_dict.values()):
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
                optim, mode="min", factor=0.1, patience=7, cooldown=1
            ),
            "monitor": "val_loss",
        }

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LambdaUnetModel")

        # network params
        parser.add_argument(
            "--in_chans", default=2, type=int, help="Number of U-Net input channels"
        )
        parser.add_argument(
            "--out_chans", default=7, type=int, help="Number of U-Net output chanenls"
        )
        parser.add_argument(
            "--chans", default=32, type=int, help="Number of top-level U-Net filters."
        )
        parser.add_argument(
            "--num_pool_layers",
            default=4,
            type=int,
            help="Number of U-Net pooling layers.",
        )
        parser.add_argument(
            "--drop_prob", default=0.1, type=float, help="U-Net dropout probability"
        )

        parser.add_argument("--tr", default=3, type=int, help="Size of temporal kernel")
        parser.add_argument(
            "--num_slices",
            default=3,
            type=int,
            help="Numer of slices to process simultaneously.",
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

        loss_dict = {}
        loss_dict["cross_entropy"] = self.cross_entropy(
            output, torch.argmax(target, dim=1).long()
        )
        dice_loss, dice_score = self.dice_loss(output, target)
        loss_dict["dice_loss"] = dice_loss.mean()
        loss_dict["dice_score"] = dice_score.detach()
        loss_dict["loss"] = loss_dict["cross_entropy"] + loss_dict["dice_loss"]
        return loss_dict, output

    def training_step(self, batch, batch_idx):
        loss_dict, output = self.step(batch, batch_idx)
        for metric, value in zip(loss_dict.keys(), loss_dict.values()):
            self.log(f"train_{metric}", value.mean().detach())
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        loss_dict, output = self.step(batch, batch_idx)
        for metric, value in zip(loss_dict.keys(), loss_dict.values()):
            self.log(f"val_{metric}", value.mean().detach(), sync_dist=True)
        return output, loss_dict

    def test_step(self, batch, batch_idx):
        loss_dict, output = self.step(batch, batch_idx)
        for metric, value in zip(loss_dict.keys(), loss_dict.values()):
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
                optim, mode="min", factor=0.1, patience=7, cooldown=1
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


class DeepLabModule(pl.LightningModule):
    def __init__(
        self, in_chans=1, out_chans=1, lr=0.001, weight_decay=0.0, **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.lr = lr
        self.weight_decay = weight_decay

        self.model = DeepLab(
            num_input_chans=self.hparams.in_chans, num_classes=self.hparams.out_chans
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

        output = self(input)["out"]

        if torch.any(torch.isnan(output)):
            print(output)
            raise ValueError

        loss_dict = {}
        loss_dict["cross_entropy"] = self.cross_entropy(
            output, torch.argmax(target, dim=1).long()
        )
        dice_loss, dice_score = self.dice_loss(output, target)
        loss_dict["dice_loss"] = dice_loss.mean()
        loss_dict["dice_score"] = dice_score.detach()
        loss_dict["loss"] = loss_dict["cross_entropy"] + loss_dict["dice_loss"]
        return loss_dict, output

    def training_step(self, batch, batch_idx):
        loss_dict, output = self.step(batch, batch_idx)
        for metric, value in zip(loss_dict.keys(), loss_dict.values()):
            self.log(f"train_{metric}", value.mean().detach())
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        loss_dict, output = self.step(batch, batch_idx)
        for metric, value in zip(loss_dict.keys(), loss_dict.values()):
            self.log(f"val_{metric}", value.mean().detach(), sync_dist=True)
        return output, loss_dict

    def test_step(self, batch, batch_idx):
        loss_dict, output = self.step(batch, batch_idx)
        for metric, value in zip(loss_dict.keys(), loss_dict.values()):
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
                optim, mode="min", factor=0.1, patience=5, cooldown=0
            ),
            "monitor": "val_loss",
        }

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("UnetModel")

        # network params
        parser.add_argument(
            "--in_chans", default=2, type=int, help="Number of input channels"
        )
        parser.add_argument(
            "--out_chans", default=7, type=int, help="Number of output chanenls"
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


class Unet3dModule(pl.LightningModule):
    def __init__(
        self,
        in_chans=1,
        out_chans=1,
        chans=32,
        num_pool_layers=4,
        drop_prob=0.1,
        lr=0.001,
        weight_decay=0.0,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.lr = lr
        self.weight_decay = weight_decay

        self.model = Unet3d(
            in_chans=self.hparams.in_chans,
            out_chans=self.hparams.out_chans,
            chans=self.hparams.chans,
            num_pool_layers=self.hparams.num_pool_layers,
            drop_prob=self.hparams.drop_prob,
        )
        self.example_input_array = torch.rand(
            1, self.hparams.in_chans, 3, 256, 256, device=self.device
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
            input = rearrange(input, "b t c h w -> b c t h w")
        if len(target.shape) == 5:
            target = rearrange(target, "b t c h w -> b c t h w")

        output = self(input)

        if torch.any(torch.isnan(output)):
            print(output)
            raise ValueError

        if len(output.shape) == 5:
            output = rearrange(output, "b t c h w -> (b t) c h w")
        if len(target.shape) == 5:
            target = rearrange(target, "b t c h w -> (b t) c h w")

        loss_dict = {}
        loss_dict["cross_entropy"] = self.cross_entropy(
            output, torch.argmax(target, dim=1).long()
        )
        dice_loss, dice_score = self.dice_loss(output, target)
        loss_dict["dice_loss"] = dice_loss.mean()
        loss_dict["dice_score"] = dice_score.detach()
        loss_dict["loss"] = loss_dict["cross_entropy"] + loss_dict["dice_loss"]
        return loss_dict, output

    def training_step(self, batch, batch_idx):
        loss_dict, output = self.step(batch, batch_idx)
        for metric, value in zip(loss_dict.keys(), loss_dict.values()):
            self.log(f"train_{metric}", value.mean().detach())
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        loss_dict, output = self.step(batch, batch_idx)
        for metric, value in zip(loss_dict.keys(), loss_dict.values()):
            self.log(f"val_{metric}", value.mean().detach(), sync_dist=True)
        return output, loss_dict

    def test_step(self, batch, batch_idx):
        loss_dict, output = self.step(batch, batch_idx)
        for metric, value in zip(loss_dict.keys(), loss_dict.values()):
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
                optim, mode="min", factor=0.1, patience=7, cooldown=1
            ),
            "monitor": "val_loss",
        }

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Unet3dModel")

        # network params
        parser.add_argument(
            "--in_chans", default=2, type=int, help="Number of U-Net input channels"
        )
        parser.add_argument(
            "--out_chans", default=7, type=int, help="Number of U-Net output chanenls"
        )
        parser.add_argument(
            "--chans", default=32, type=int, help="Number of top-level U-Net filters."
        )
        parser.add_argument(
            "--num_pool_layers",
            default=4,
            type=int,
            help="Number of U-Net pooling layers.",
        )
        parser.add_argument(
            "--drop_prob", default=0.1, type=float, help="U-Net dropout probability"
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


# Needs to be redone.
class CIRIMModule(pl.LightningModule):
    def __init__(
        self,
        recurrent_layer: str = "IndRNN",
        conv_filters=None,
        conv_kernels=None,
        conv_dilations=None,
        conv_bias=None,
        recurrent_filters=None,
        recurrent_kernels=None,
        recurrent_dilations=None,
        recurrent_bias=None,
        depth: int = 2,
        time_steps: int = 8,
        conv_dim: int = 2,
        loss_fn: str = F.l1_loss,
        num_cascades: int = 1,
        no_dc: bool = False,
        keep_eta: bool = False,
        use_sens_net: bool = False,
        sens_chans: int = 8,
        sens_pools: int = 4,
        sens_mask_type: str = "2D",
        fft_type: str = "orthogonal",
        output_type: str = "SENSE",
        lr=0.001,
        weight_decay=0.0,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay

        self.model = CIRIM(
            recurrent_layer=self.hparams.recurrent_layer,
            conv_filters=self.hparams.conv_filters,
            conv_kernels=self.hparams.conv_kernels,
            conv_dilations=self.hparams.conv_dilations,
            conv_bias=self.hparams.conv_bias,
            recurrent_filters=self.hparams.recurrent_filters,
            recurrent_kernels=self.hparams.recurrent_kernels,
            recurrent_dilations=self.hparams.recurrent_dilations,
            recurrent_bias=self.hparams.recurrent_bias,
            depth=self.hparams.depth,
            time_steps=self.hparams.time_steps,
            conv_dim=self.hparams.conv_dim,
            num_cascades=self.hparams.num_cascades,
            no_dc=self.hparams.no_dc,
            keep_eta=self.hparams.keep_eta,
            use_sens_net=self.hparams.use_sens_net,
            sens_chans=self.hparams.sens_chans,
            sens_pools=self.hparams.sens_pools,
            sens_mask_type=self.hparams.sens_mask_type,
            fft_type=self.hparams.fft_type,
            output_type=self.hparams.output_type,
        )
        self.example_input_array = [
            torch.rand(1, 32, 320, 320, 2),  # kspace
            torch.rand(1, 32, 320, 320, 2),  # sesitivity maps
            torch.rand(1, 1, 320, 320, 1),  # mask
            torch.rand(1, 320, 320, 2),  # initial prediction
            torch.rand(1, 320, 320, 2),  # target
        ]

    def forward(
        self, y, sensitivity_maps, mask, init_pred, target,
    ):
        return self.model(y, sensitivity_maps, mask, init_pred, target)

    def step(self, batch, batch_indx=None):
        y, sensitivity_maps, mask, init_pred, target, fname, slice_num, _ = batch
        y, mask, _ = self.model.process_input(y, mask)
        preds = self.forward(y, sensitivity_maps, mask, init_pred, target)

        loss = self.model.calculate_loss(preds, target, _loss_fn=self.hparams.loss_fn)

        if torch.any(torch.isnan(preds[-1][-1])):
            print(preds[-1][-1])
            raise ValueError

        output = torch.abs(preds[-1][-1])
        output = output / output.amax()
        target = torch.abs(target) / torch.abs(target).amax()

        loss_dict = {"loss": loss}
        loss_dict["psnr"] = FM.psnr(output.unsqueeze(0), target.unsqueeze(0))
        loss_dict["ssim"] = FM.ssim(output.unsqueeze(0), target.unsqueeze(0))
        return loss_dict, preds

    def training_step(self, batch, batch_idx):
        loss_dict, output = self.step(batch, batch_idx)
        for metric, value in zip(loss_dict.keys(), loss_dict.values()):
            self.log(f"train_{metric}", value.mean().detach())
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        loss_dict, output = self.step(batch, batch_idx)
        for metric, value in zip(loss_dict.keys(), loss_dict.values()):
            self.log(f"val_{metric}", value.mean().detach(), sync_dist=True)
        return output, loss_dict

    def test_step(self, batch, batch_idx):
        fname = batch[-3]
        slice_num = batch[-2]
        loss_dict, output = self.step(batch, batch_idx)
        if isinstance(output, list):
            output = output[-1]
        if isinstance(output, list):
            output = output[-1]
        for metric, value in zip(loss_dict.keys(), loss_dict.values()):
            self.log(f"test_{metric}", value.mean().detach(), sync_dist=True)
        return loss_dict, (fname, slice_num, output.detach().cpu().numpy())

    # def test_epoch_end(self, outputs):
    #     reconstructions = defaultdict(list)
    #     for fname, slice_num, output in outputs[1]:
    #         reconstructions[fname].append((slice_num, output))

    #     for fname in reconstructions:
    #         reconstructions[fname] = np.stack([out for _, out in sorted(reconstructions[fname])])  # type: ignore

    #     out_dir = Path(os.path.join(self.logger.log_dir, "reconstructions"))
    #     out_dir.mkdir(exist_ok=True, parents=True)
    #     for fname, recons in reconstructions.items():
    #         with h5py.File(out_dir / fname, "w") as hf:
    #             hf.create_dataset("reconstruction", data=recons)

    def predict_step(self, batch, batch_idx, dataloader_idx):
        y, sensitivity_maps, mask, init_pred, target, fname, slice_num, _ = batch
        y, mask, _ = self.model.process_input(y, mask)
        preds = self.forward(y, sensitivity_maps, mask, init_pred, target)
        output = torch.abs(preds[-1][-1])
        output = output / output.max()
        return output

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim, mode="min", factor=0.5, patience=4, cooldown=1
            ),
            "monitor": "val_loss",
        }

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("CIRIMModel")

        # network params
        parser.add_argument(
            "--num_cascades",
            type=int,
            default=1,
            help="Number of cascades for the model",
        )
        parser.add_argument(
            "--time_steps", type=int, default=8, help="Number of RIM steps"
        )
        parser.add_argument(
            "--recurrent_layer", type=str, default="IndRNN", help="Recurrent layer type"
        )
        parser.add_argument(
            "--conv_filters",
            nargs="+",
            default=[64, 64, 2],
            type=int,
            help="Number of filters the convolutional layers of for the model",
        )
        parser.add_argument(
            "--conv_kernels",
            nargs="+",
            default=[5, 3, 3],
            type=int,
            help="Kernel size for the convolutional layers of the model",
        )
        parser.add_argument(
            "--conv_dilations",
            nargs="+",
            type=int,
            default=[1, 2, 1],
            help="Dilations for the convolutional layers of the model",
        )
        parser.add_argument(
            "--conv_bias",
            nargs="+",
            type=bool,
            default=[True, True, False],
            help="Bias for the convolutional layers of the model",
        )
        parser.add_argument(
            "--recurrent_filters",
            nargs="+",
            type=int,
            default=[64, 64, 0],
            help="Number of filters the recurrent layers of for the model",
        )
        parser.add_argument(
            "--recurrent_kernels",
            nargs="+",
            type=int,
            default=[1, 1, 0],
            help="Kernel size for the recurrent layers of the model",
        )
        parser.add_argument(
            "--recurrent_dilations",
            nargs="+",
            type=int,
            default=[1, 1, 0],
            help="Dilations for the recurrent layers of the model",
        )
        parser.add_argument(
            "--recurrent_bias",
            nargs="+",
            type=bool,
            default=[True, True, False],
            help="Bias for the recurrent layers of the model",
        )
        parser.add_argument("--depth", type=int, default=2, help="Depth of the model")
        parser.add_argument(
            "--conv_dim",
            type=int,
            default=2,
            help="Dimension of the convolutional layers",
        )
        parser.add_argument(
            "--no_dc",
            action="store_false",
            default=True,
            help="Do not use DC component",
        )
        parser.add_argument(
            "--keep_eta", action="store_false", default=True, help="Keep eta constant"
        )
        parser.add_argument(
            "--use_sens_net",
            action="store_true",
            default=False,
            help="Use sensitivity net",
        )
        parser.add_argument(
            "--sens_pools",
            type=int,
            default=4,
            help="Number of pools for the sensitivity net",
        )
        parser.add_argument(
            "--sens_chans",
            type=int,
            default=8,
            help="Number of channels for the sensitivity net",
        )
        parser.add_argument(
            "--sens_mask_type",
            choices=["1D", "2D"],
            default="2D",
            help="Type of mask to use for the sensitivity net",
        )
        parser.add_argument(
            "--output_type",
            choices=["SENSE", "RSS"],
            default="SENSE",
            help="Type of output to use",
        )
        parser.add_argument(
            "--fft_type", type=str, default="backward", help="Type of FFT to use"
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


class AttUnetModule(pl.LightningModule):
    def __init__(
        self,
        in_chans=1,
        out_chans=1,
        chans=32,
        num_pool_layers=4,
        drop_prob=0.1,
        lr=0.001,
        lr_step_size=40,
        lr_gamma=0.0,
        weight_decay=0.0,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.lr = lr
        self.weight_decay = weight_decay

        self.model = AttUnet(
            in_chans=self.hparams.in_chans,
            out_chans=self.hparams.out_chans,
            chans=self.hparams.chans,
            num_pool_layers=self.hparams.num_pool_layers,
            drop_prob=self.hparams.drop_prob,
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

        loss_dict = {}
        loss_dict["cross_entropy"] = self.cross_entropy(
            output, torch.argmax(target, dim=1).long()
        )
        dice_loss, dice_score = self.dice_loss(output, target)
        loss_dict["dice_loss"] = dice_loss.mean()
        loss_dict["dice_score"] = dice_score.detach()
        loss_dict["loss"] = loss_dict["cross_entropy"] + loss_dict["dice_loss"]
        return loss_dict, output

    def training_step(self, batch, batch_idx):
        loss_dict, output = self.step(batch, batch_idx)
        for metric, value in zip(loss_dict.keys(), loss_dict.values()):
            self.log(f"train_{metric}", value.mean().detach())
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        loss_dict, output = self.step(batch, batch_idx)
        for metric, value in zip(loss_dict.keys(), loss_dict.values()):
            self.log(f"val_{metric}", value.mean().detach(), sync_dist=True)
        return output, loss_dict

    def test_step(self, batch, batch_idx):
        loss_dict, output = self.step(batch, batch_idx)
        for metric, value in zip(loss_dict.keys(), loss_dict.values()):
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
                optim, mode="min", factor=0.1, patience=7, cooldown=1
            ),
            "monitor": "val_loss",
        }

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("UnetModel")

        # network params
        parser.add_argument(
            "--in_chans", default=2, type=int, help="Number of U-Net input channels"
        )
        parser.add_argument(
            "--out_chans", default=7, type=int, help="Number of U-Net output chanenls"
        )
        parser.add_argument(
            "--chans", default=32, type=int, help="Number of top-level U-Net filters."
        )
        parser.add_argument(
            "--num_pool_layers",
            default=4,
            type=int,
            help="Number of U-Net pooling layers.",
        )
        parser.add_argument(
            "--drop_prob", default=0.1, type=float, help="U-Net dropout probability"
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
