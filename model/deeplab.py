import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from einops import rearrange
from torchvision.models.segmentation import deeplabv3_resnet50

from losses import DiceLoss, MC_CrossEntropy


class DeepLab(nn.Module):
    def __init__(
        self,
        pretrained: bool = False,
        progress: bool = False,
        num_input_chans: int = 3,
        num_classes: int = 21,
        aux_loss: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.deeplab = deeplabv3_resnet50(
            pretrained, progress, num_classes, aux_loss, **kwargs
        )

        shape = list(self.deeplab.backbone.conv1.weight.shape)
        if shape[1] != num_input_chans:
            shape[1] = num_input_chans
            self.deeplab.backbone.conv1.weight = torch.nn.Parameter(
                torch.rand(shape, requires_grad=True)
            )
            self.deeplab.backbone.conv1.reset_parameters()

    def forward(self, x):
        return self.deeplab(x)


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
            1, self.hparams.in_chans, 200, 200, device=self.device
        )

        self.dice_loss = DiceLoss(include_background=False)
        self.cross_entropy = MC_CrossEntropy(
            weight=torch.tensor([0.05558904, 0.29847416, 0.31283098, 0.33310577])
            if self.hparams.dataset in ["tecfidera", "techfideramri"]
            else None,
        )

    def forward(self, x):
        with torch.no_grad():
            x = F.group_norm(x, num_groups=1)
        x = self.model(x)
        return x

    def step(self, batch, batch_indx=None):
        fname, input, target = batch

        if len(input.shape) == 5:
            input = rearrange(input, "b t c h w -> (b t) c h w")
        if len(target.shape) == 5:
            target = rearrange(target, "b t c h w -> (b t) c h w")

        output = self(input)["out"]

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
        return loss_dict, fname, output

    def training_step(self, batch, batch_idx):
        loss_dict, fname, output = self.step(batch, batch_idx)
        for metric, value in loss_dict.items():
            self.log(f"train_{metric}", value.mean().detach())
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        loss_dict, fname, output = self.step(batch, batch_idx)
        for metric, value in loss_dict.items():
            self.log(f"val_{metric}", value.mean().detach(), sync_dist=True)
        return loss_dict, output

    def test_step(self, batch, batch_idx):
        loss_dict, fname, output = self.step(batch, batch_idx)
        for metric, value in loss_dict.items():
            self.log(f"test_{metric}", value.mean().detach(), sync_dist=True)
        return loss_dict

    def predict_step(self, batch, batch_idx, dataloader_idx):
        fname, input, _ = batch
        return self(input)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim, mode="min", factor=0.5, patience=5, cooldown=0
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
