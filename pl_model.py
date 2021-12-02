import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from argparse import ArgumentParser
from einops import rearrange

from model.unet import Unet
from model.lambda_layer import LambdaBlock
from losses import DiceLoss


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

        self.unet = Unet(
            in_chans=self.hparams.in_chans,
            out_chans=self.hparams.out_chans,
            chans=self.hparams.chans,
            num_pool_layers=self.hparams.num_pool_layers,
            drop_prob=self.hparams.drop_prob,
        )
        
        self.dice_loss = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x):
        x = F.group_norm(x, num_groups=1)
        x = self.unet(x)
        return x
    
    def step(self, batch, batch_indx=None):
        input, target = batch
        output = self(input)
        if torch.any(torch.isnan(output)):
            print(output)
            raise ValueError
        loss_dict = {}
        loss_dict["cross_entropy"] = self.cross_entropy(output, torch.argmax(target, dim=1).long())
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
            self.log(f"val_{metric}", value.mean().detach())
        return output, loss_dict

    def test_step(self, batch, batch_idx):
        loss_dict, output = self.step(batch, batch_idx)
        for metric, value in zip(loss_dict.keys(), loss_dict.values()):
            self.log(f"test_{metric}", value.mean().detach())
        return loss_dict
    
    def predict_step(self, batch, batch_idx, dataloader_idx):
        input, _ = batch
        return self(input)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optim, mode='min', factor=0.1, patience=5, cooldown=0),
                    "monitor": "val_loss"}

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
    def __init__(
        self,
        in_chans=1,
        out_chans=1,
        chans=32,
        **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.unet = Unet(
            in_chans=self.hparams.in_chans,
            out_chans=self.hparams.out_chans,
            chans=self.hparams.chans,
            num_pool_layers=self.hparams.num_pool_layers,
            drop_prob=self.hparams.drop_prob,
            block=LambdaBlock,
            temporal_kernel=self.hparams.tr, 
            num_slices=self.hparams.num_slices
        )
        
        self.dice_loss = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x):
        x = F.group_norm(x, num_groups=1)
        x = self.unet(x)
        return x
    
    def step(self, batch, batch_indx=None):
        input, target = batch
        
        if len(input.shape) == 5:
            input = rearrange(input, 'b t c h w -> (b t) c h w')
            target = rearrange(target, 'b t c h w -> (b t) c h w')
            
        output = self(input)

        if torch.any(torch.isnan(output)):
            print(output)
            raise ValueError
            
        loss_dict = {}
        loss_dict["cross_entropy"] = self.cross_entropy(output, torch.argmax(target, dim=1).long())
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
            self.log(f"val_{metric}", value.mean().detach())
        return output, loss_dict

    def test_step(self, batch, batch_idx):
        loss_dict, output = self.step(batch, batch_idx)
        for metric, value in zip(loss_dict.keys(), loss_dict.values()):
            self.log(f"test_{metric}", value.mean().detach())
        return loss_dict
    
    def predict_step(self, batch, batch_idx, dataloader_idx):
        input, _ = batch
        return self(input)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optim, mode='min', factor=0.1, patience=5, cooldown=0),
                    "monitor": "val_loss"}

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
        
        parser.add_argument(
            "--tr", default=3, type=int, help="Size of temporal kernel"
        )
        parser.add_argument(
            "--num_slices", default=3, type=int, help="Numer of slices to process simultaneously."
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