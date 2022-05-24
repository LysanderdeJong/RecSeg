import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics import functional as FM
from einops import rearrange

from losses import (
    DiceLoss,
    MC_CrossEntropy,
    hausdorf_distance,
    average_surface_distance,
)

from model.lambda_layer import LambdaBlock


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float, **kwargs):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


class Unet(nn.Module):
    """
    PyTorch implementation of a U-Net model.
    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
        block=ConvBlock,
        **kwargs,
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
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob, **kwargs),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

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
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
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


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)


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

        output = self(input)

        if torch.any(torch.isnan(output)):
            print(output)
            raise ValueError

        loss_dict = self.calculate_metrics(output, target, important_only=False)
        return loss_dict, fname, output

    def calculate_metrics(self, preds, target, important_only=True):
        pred_label = torch.softmax(preds, dim=1).argmax(1)
        target_label = target.argmax(1)

        loss_dict = {"cross_entropy": self.cross_entropy(preds, target_label)}
        dice_loss, dice_score = self.dice_loss(preds, target)
        loss_dict["dice_loss"] = dice_loss.mean()
        loss_dict["dice_score"] = dice_score.detach()

        if not important_only:
            dice_per_class = FM.dice_score(
                torch.softmax(preds, dim=1), target_label, bg=True, reduction="none"
            )
            for i, label in enumerate(
                ["background", "graymatter", "whitematter", "lesion"]
            ):
                loss_dict[f"dice_{label}"] = dice_per_class[i]

            loss_dict["f1_micro"] = FM.fbeta(
                pred_label, target_label, mdmc_average="samplewise", ignore_index=0
            )
            loss_dict["f1_macro"] = FM.fbeta(
                pred_label,
                target_label,
                average="macro",
                mdmc_average="samplewise",
                num_classes=preds.shape[1],
                ignore_index=0,
            )
            loss_dict["f1_weighted"] = FM.fbeta(
                pred_label,
                target_label,
                average="weighted",
                mdmc_average="samplewise",
                num_classes=preds.shape[1],
                ignore_index=0,
            )
            f1_per_class = FM.fbeta(
                pred_label,
                target_label,
                average="none",
                mdmc_average="samplewise",
                num_classes=preds.shape[1],
            )
            for i, label in enumerate(
                ["background", "graymatter", "whitematter", "lesion"]
            ):
                loss_dict[f"f1_{label}"] = f1_per_class[i]

            loss_dict["precision_micro"] = FM.precision(
                pred_label, target_label, mdmc_average="samplewise", ignore_index=0
            )
            loss_dict["precision_macro"] = FM.precision(
                pred_label,
                target_label,
                average="macro",
                mdmc_average="samplewise",
                num_classes=preds.shape[1],
                ignore_index=0,
            )
            loss_dict["precision_weighted"] = FM.precision(
                pred_label,
                target_label,
                average="weighted",
                mdmc_average="samplewise",
                num_classes=preds.shape[1],
                ignore_index=0,
            )
            precision_per_class = FM.precision(
                pred_label,
                target_label,
                average="none",
                mdmc_average="samplewise",
                num_classes=preds.shape[1],
            )
            for i, label in enumerate(
                ["background", "graymatter", "whitematter", "lesion"]
            ):
                loss_dict[f"precision_{label}"] = precision_per_class[i]

            loss_dict["recall_micro"] = FM.recall(
                pred_label, target_label, mdmc_average="samplewise", ignore_index=0
            )
            loss_dict["recall_macro"] = FM.recall(
                pred_label,
                target_label,
                average="macro",
                mdmc_average="samplewise",
                num_classes=preds.shape[1],
                ignore_index=0,
            )
            loss_dict["recall_weighted"] = FM.recall(
                pred_label,
                target_label,
                average="weighted",
                mdmc_average="samplewise",
                num_classes=preds.shape[1],
                ignore_index=0,
            )
            recall_per_class = FM.recall(
                pred_label,
                target_label,
                average="none",
                mdmc_average="samplewise",
                num_classes=preds.shape[1],
            )
            for i, label in enumerate(
                ["background", "graymatter", "whitematter", "lesion"]
            ):
                loss_dict[f"recall_{label}"] = recall_per_class[i]

            loss_dict["hausdorff_distance"] = hausdorf_distance(
                preds, target, include_background=False
            )
            loss_dict["average_surface_distance"] = average_surface_distance(
                preds, target, include_background=False
            )

        loss_dict["loss"] = loss_dict["cross_entropy"] + loss_dict["dice_loss"]
        return loss_dict

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

    def predict_step(self, batch, batch_idx):
        loss_dict, fname, segmentation = self.step(batch, batch_idx)
        return loss_dict, fname, segmentation.detach().cpu()

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

        self.dice_loss = DiceLoss(include_background=False)
        self.cross_entropy = MC_CrossEntropy(
            self.hparams.aleatoric_samples,
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

        output = self(input)

        if torch.any(torch.isnan(output)):
            print(output)
            raise ValueError

        loss_dict = {}
        if output.shape[1] == target.shape[1] * 2:
            mean, log_var = output.chunk(2, 1)
            dice_loss, dice_score = self.dice_loss(mean, target)
            loss_dict["dice_loss"] = dice_loss.mean()
            loss_dict["dice_score"] = dice_score.detach()
            loss_dict["mc_cross_entropy"] = self.cross_entropy(
                mean, torch.argmax(target, dim=1).long(), log_var
            )
            loss_dict["loss"] = loss_dict["mc_cross_entropy"] + loss_dict["dice_loss"]
        else:
            loss_dict = self.calculate_metrics(output, target, important_only=False)
        return loss_dict, fname, output

    def calculate_metrics(self, preds, target, important_only=True):
        pred_label = torch.softmax(preds, dim=1).argmax(1)
        target_label = target.argmax(1)

        loss_dict = {"cross_entropy": self.cross_entropy(preds, target_label)}
        dice_loss, dice_score = self.dice_loss(preds, target)
        loss_dict["dice_loss"] = dice_loss.mean()
        loss_dict["dice_score"] = dice_score.detach()

        if not important_only:
            dice_per_class = FM.dice_score(
                torch.softmax(preds, dim=1), target_label, bg=True, reduction="none"
            )
            for i, label in enumerate(
                ["background", "graymatter", "whitematter", "lesion"]
            ):
                loss_dict[f"dice_{label}"] = dice_per_class[i]

            loss_dict["f1_micro"] = FM.fbeta(
                pred_label, target_label, mdmc_average="samplewise", ignore_index=0
            )
            loss_dict["f1_macro"] = FM.fbeta(
                pred_label,
                target_label,
                average="macro",
                mdmc_average="samplewise",
                num_classes=preds.shape[1],
                ignore_index=0,
            )
            loss_dict["f1_weighted"] = FM.fbeta(
                pred_label,
                target_label,
                average="weighted",
                mdmc_average="samplewise",
                num_classes=preds.shape[1],
                ignore_index=0,
            )
            f1_per_class = FM.fbeta(
                pred_label,
                target_label,
                average="none",
                mdmc_average="samplewise",
                num_classes=preds.shape[1],
            )
            for i, label in enumerate(
                ["background", "graymatter", "whitematter", "lesion"]
            ):
                loss_dict[f"f1_{label}"] = f1_per_class[i]

            loss_dict["precision_micro"] = FM.precision(
                pred_label, target_label, mdmc_average="samplewise", ignore_index=0
            )
            loss_dict["precision_macro"] = FM.precision(
                pred_label,
                target_label,
                average="macro",
                mdmc_average="samplewise",
                num_classes=preds.shape[1],
                ignore_index=0,
            )
            loss_dict["precision_weighted"] = FM.precision(
                pred_label,
                target_label,
                average="weighted",
                mdmc_average="samplewise",
                num_classes=preds.shape[1],
                ignore_index=0,
            )
            precision_per_class = FM.precision(
                pred_label,
                target_label,
                average="none",
                mdmc_average="samplewise",
                num_classes=preds.shape[1],
            )
            for i, label in enumerate(
                ["background", "graymatter", "whitematter", "lesion"]
            ):
                loss_dict[f"precision_{label}"] = precision_per_class[i]

            loss_dict["recall_micro"] = FM.recall(
                pred_label, target_label, mdmc_average="samplewise", ignore_index=0
            )
            loss_dict["recall_macro"] = FM.recall(
                pred_label,
                target_label,
                average="macro",
                mdmc_average="samplewise",
                num_classes=preds.shape[1],
                ignore_index=0,
            )
            loss_dict["recall_weighted"] = FM.recall(
                pred_label,
                target_label,
                average="weighted",
                mdmc_average="samplewise",
                num_classes=preds.shape[1],
                ignore_index=0,
            )
            recall_per_class = FM.recall(
                pred_label,
                target_label,
                average="none",
                mdmc_average="samplewise",
                num_classes=preds.shape[1],
            )
            for i, label in enumerate(
                ["background", "graymatter", "whitematter", "lesion"]
            ):
                loss_dict[f"recall_{label}"] = recall_per_class[i]

            loss_dict["hausdorff_distance"] = hausdorf_distance(
                preds, target, include_background=False
            )
            loss_dict["average_surface_distance"] = average_surface_distance(
                preds, target, include_background=False
            )

        loss_dict["loss"] = loss_dict["cross_entropy"] + loss_dict["dice_loss"]
        return loss_dict

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

    def predict_step(self, batch, batch_idx):
        loss_dict, fname, segmentation = self.step(batch, batch_idx)
        return loss_dict, fname, segmentation[0].unsqueeze(0).detach().cpu()

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

        parser.add_argument(
            "--aleatoric_samples",
            default=1,
            type=int,
            help="Number MC samples to take.",
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
