import math
from multiprocessing.sharedctypes import RawArray
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import pytorch_lightning as pl
from torchmetrics import functional as FM
from einops import rearrange

from mridc.collections.common.parts.fft import fft2c, ifft2c
from mridc.collections.common.parts.utils import rss_complex, complex_mul, complex_conj
from losses import (
    DiceLoss,
    MC_CrossEntropy,
    average_surface_distance,
    hausdorf_distance,
)

from model.unet import ConvBlock, TransposeConvBlock


class DC(nn.Module):
    def __init__(self, soft_dc: bool = False) -> None:
        super().__init__()

        self.soft_dc = soft_dc

        if self.soft_dc:
            self.dc_weight = nn.Parameter(torch.ones(1))

    def forward(self, kspace, og_kspace, mask=None):
        if mask is not None:
            zero = torch.zeros_like(kspace, device=kspace.device)
            dc = torch.where(mask.bool(), kspace - og_kspace, zero)

            if self.soft_dc:
                dc *= self.dc_weight

            return kspace - dc
        else:
            return kspace


class UnetEncoder(nn.Module):
    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        drop_prob: float = 0.0,
        padding_size: int = 15,
        normalize: bool = True,
        norm_groups: int = 2,
    ) -> None:
        super().__init__()

        self.in_chans = in_chans
        self.chans = chans
        self.num_pools = num_pools
        self.drop_prob = drop_prob
        self.padding_size = padding_size
        self.normalize = normalize
        self.norm_groups = norm_groups

        self.down_sample_layers = torch.nn.ModuleList(
            [ConvBlock(in_chans, chans, drop_prob)]
        )
        ch = chans
        for _ in range(num_pools - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

    @staticmethod
    def complex_to_chan_dim(x):
        b, c, h, w, two = x.shape
        if two != 2:
            raise AssertionError
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def norm(self, x):
        # group norm
        b, c, h, w = x.shape
        x = x.reshape(b, self.norm_groups, -1)
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        x = (x - mean) / std
        x = x.reshape(b, c, h, w)
        return x, mean, std

    def pad(self, x):
        _, _, h, w = x.shape
        w_mult = ((w - 1) | self.padding_size) + 1
        h_mult = ((h - 1) | self.padding_size) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = torch.nn.functional.pad(x, w_pad + h_pad)
        return x, (h_pad, w_pad, h_mult, w_mult)

    def forward(self, x):
        iscomplex = False
        if x.shape[-1] == 2:
            x = self.complex_to_chan_dim(x)
            iscomplex = True

        mean = 1.0
        std = 1.0

        if self.normalize:
            x, mean, std = self.norm(x)

        x, pad_sizes = self.pad(x)

        stack = []
        output = x

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = torch.nn.functional.avg_pool2d(
                output, kernel_size=2, stride=2, padding=0
            )

        output = self.conv(output)
        stack.append(output)

        if self.normalize:
            return stack, iscomplex, pad_sizes, mean, std
        else:
            return stack, iscomplex, pad_sizes


class UnetDecoder(nn.Module):
    def __init__(
        self,
        chans: int,
        num_pools: int,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        padding_size: int = 15,
        normalize: bool = True,
        norm_groups: int = 2,
    ) -> None:
        super().__init__()

        self.out_chans = out_chans
        self.chans = chans
        self.num_pools = num_pools
        self.drop_prob = drop_prob
        self.padding_size = padding_size
        self.normalize = normalize
        self.norm_groups = norm_groups

        ch = chans * (2 ** (num_pools - 1))
        self.up_conv = torch.nn.ModuleList()
        self.up_transpose_conv = torch.nn.ModuleList()
        for _ in range(num_pools - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            torch.nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                torch.nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

    @staticmethod
    def chan_complex_to_last_dim(x):
        b, c2, h, w = x.shape
        if c2 % 2 != 0:
            raise AssertionError
        c = torch.div(c2, 2, rounding_mode="trunc")
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    @staticmethod
    def unpad(x, h_pad, w_pad, h_mult, w_mult):
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def unnorm(self, x, mean, std):
        b, c, h, w = x.shape
        input_data = x.reshape(b, self.norm_groups, -1)
        return (input_data * std + mean).reshape(b, c, h, w)

    def forward(self, x_stack, iscomplex=False, pad_sizes=None, mean=None, std=None):
        output = x_stack.pop()
        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = x_stack.pop()
            output = transpose_conv(output)

            # reflect pad on the right/bottom if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = torch.nn.functional.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        if pad_sizes is not None:
            output = self.unpad(output, *pad_sizes)
        if self.normalize and mean is not None and std is not None:
            output = self.unnorm(output, mean, std)
        if iscomplex:
            output = self.chan_complex_to_last_dim(output)

        return output


class idlsr(nn.Module):
    def __init__(
        self,
        in_chans,
        out_chans,
        seg_out_chans,
        num_iters=5,
        soft_dc=True,
        chans=32,
        num_pools=4,
        drop_prob=0.0,
        normalize=True,
        norm_groups=2,
        num_cascades=1,
        fft_type="orthogonal",
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleList(
            [
                UnetEncoder(
                    chans=chans,
                    num_pools=num_pools,
                    in_chans=in_chans,
                    drop_prob=drop_prob,
                    padding_size=15,
                    normalize=normalize,
                    norm_groups=norm_groups,
                )
                for _ in range(num_cascades)
            ]
        )
        self.decoders = nn.ModuleList(
            [
                UnetDecoder(
                    chans=chans,
                    num_pools=num_pools,
                    out_chans=out_chans,
                    drop_prob=drop_prob,
                    padding_size=15,
                    normalize=normalize,
                    norm_groups=norm_groups,
                )
                for _ in range(num_cascades)
            ]
        )
        self.seg_head = UnetDecoder(
            chans=chans,
            num_pools=num_pools,
            out_chans=seg_out_chans,
            drop_prob=drop_prob,
            padding_size=15,
            normalize=normalize,
            norm_groups=norm_groups,
        )

        self.dc = nn.ModuleList([DC(soft_dc=soft_dc) for _ in range(num_cascades)])

        self.num_iters = num_iters
        self.fft_type = fft_type

    def forward(self, masked_kpace, sensitivity_map=None, mask=None):
        preds = []
        # if sensitivity_map:
        #     masked_kpace = complex_mul(
        #         ifft2c(masked_kpace, fft_type=self.fft_type),
        #         complex_conj(sensitivity_map),
        #     ).sum(1)
        # else:
        #     masked_kpace = rss_complex(
        #         ifft2c(masked_kpace, fft_type=self.fft_type), dim=1
        #     )
        for encoder, decoder, dc in zip(self.encoders, self.decoders, self.dc):
            tmp = []
            for _ in range(self.num_iters):
                image_space = ifft2c(masked_kpace, fft_type=self.fft_type)
                output = encoder(image_space)
                stack, pad_sizes = output[0].copy(), output[2]
                pred_image = decoder(*output)
                pred_image = image_space - pred_image
                pred_kspace = fft2c(pred_image, fft_type=self.fft_type)
                masked_kpace = dc(pred_kspace, masked_kpace, mask)
                tmp.append(
                    rss_complex(ifft2c(masked_kpace, fft_type=self.fft_type), dim=1)
                )
            preds.append(tmp)
        pred_segmentation = self.seg_head(stack, False, pad_sizes)
        return preds, pred_segmentation


class IDSLRModule(pl.LightningModule):
    def __init__(
        self,
        in_chans,
        out_chans,
        seg_out_chans,
        num_iters=5,
        soft_dc=True,
        chans=32,
        num_pools=4,
        drop_prob=0.0,
        normalize=True,
        norm_groups=2,
        num_cascades=1,
        fft_type="orthogonal",
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
        self.num_pools = num_pools
        self.drop_prob = drop_prob
        self.lr = lr
        self.weight_decay = weight_decay

        self.model = idlsr(
            in_chans=self.hparams.in_chans,
            out_chans=self.hparams.out_chans,
            seg_out_chans=self.hparams.seg_out_chans,
            num_iters=self.hparams.num_iters,
            soft_dc=self.hparams.soft_dc,
            chans=self.hparams.chans,
            num_pools=self.hparams.num_pools,
            drop_prob=self.hparams.drop_prob,
            normalize=self.hparams.normalize,
            norm_groups=self.hparams.norm_groups,
            num_cascades=self.hparams.num_cascades,
            fft_type=self.hparams.fft_type,
        )
        self.example_input_array = [
            torch.rand(1, self.hparams.in_chans // 2, 200, 200, 2, device=self.device),
            torch.rand(1, 1, 200, 200, 1, device=self.device),
        ]

        self.dice_loss = DiceLoss(include_background=False)
        self.cross_entropy = MC_CrossEntropy(
            weight=torch.tensor([0.05558904, 0.29847416, 0.31283098, 0.33310577])
            if self.hparams.dataset in ["tecfidera", "techfideramri"]
            else None,
        )

    def forward(self, kspace, mask=None):
        image, seg = self.model(kspace, mask)
        return image, seg

    def step(self, batch, batch_indx=None):
        (
            y,
            sensitivity_maps,
            mask,
            init_pred,
            target,
            fname,
            slice_num,
            acc,
            segmentation,
        ) = batch
        y, mask, _ = self.process_input(y, mask)

        y = self.fold(y)[:, :31, ...]
        mask = self.fold(mask)
        target = self.fold(target)
        segmentation = self.fold(segmentation)

        pred_image, pred_seg = self.forward(y, mask=mask)

        if torch.any(torch.isnan(pred_image[-1][-1])) or torch.any(
            torch.isnan(pred_seg)
        ):
            raise ValueError

        image = torch.abs(pred_image[-1][-1]) / torch.abs(pred_image[-1][-1]).amax(
            (-1, -2), True
        )
        target = torch.abs(target) / torch.abs(target).amax((-1, -2), True)

        # print(fname, pred_seg.shape, segmentation.shape)
        loss_dict = self.calculate_metrics(
            pred_seg, segmentation, important_only=self.hparams.train_metric_only
        )
        loss_dict["l2"] = self.calculate_loss(pred_image, target)
        loss_dict["psnr"] = FM.psnr(image.unsqueeze(-3), target.unsqueeze(-3))
        loss_dict["ssim"] = FM.ssim(image.unsqueeze(-3), target.unsqueeze(-3))
        loss_dict["loss"] = (1 - 1e-5) * loss_dict["l2"] + 1e-5 * (
            loss_dict["cross_entropy"]
        )
        return loss_dict, pred_image, pred_seg

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
        loss_dict, output_recon, output_seg = self.step(batch, batch_idx)
        for metric, value in loss_dict.items():
            self.log(f"train_{metric}", value.mean().detach())
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        loss_dict, output_recon, output_seg = self.step(batch, batch_idx)
        for metric, value in loss_dict.items():
            self.log(f"val_{metric}", value.mean().detach(), sync_dist=True)
        return loss_dict, output_recon, output_seg

    def test_step(self, batch, batch_idx):
        loss_dict, output_recon, output_seg = self.step(batch, batch_idx)
        for metric, value in loss_dict.items():
            self.log(f"test_{metric}", value.mean().detach(), sync_dist=True)
        return loss_dict

    def predict_step(self, batch, batch_idx):
        fname = batch[5]
        slice_num = batch[6]
        loss_dict, output_recon, output_seg = self.step(batch, batch_idx)
        if isinstance(output_recon, list):
            output_recon = [
                i.unsqueeze(0).detach().cpu() for j in output_recon for i in j
            ]
        return (
            loss_dict,
            (fname, slice_num),
            output_recon,
            output_seg.detach().cpu(),
        )

    @staticmethod
    def process_input(y, mask):
        """Process the inputs to the network."""
        if isinstance(y, list):
            r = np.random.randint(len(y))
            y = y[r]
            mask = mask[r]
        else:
            r = 0
        return y, mask, r

    def calculate_loss(self, eta, target, _loss_fn=F.mse_loss):
        target = torch.abs(target / torch.abs(target).amax((-1, -2), True))
        cascades_loss = []
        for cascade_eta in eta:
            time_step_loss = [
                _loss_fn(
                    torch.abs(
                        time_step_eta / torch.abs(time_step_eta).amax((-1, -2), True)
                    ),
                    target,
                )
                for time_step_eta in cascade_eta
            ]
            time_step_loss_stack = torch.stack(time_step_loss)
            loss_weights = torch.logspace(
                -1, 0, steps=len(time_step_loss), device=time_step_loss_stack.device
            )
            cascades_loss.append(
                sum(time_step_loss_stack * loss_weights) / len(time_step_loss)
            )
        return sum(cascades_loss) / len(cascades_loss)

    def fold(self, tensor):
        shape = list(tensor.shape[1:])
        shape[0] = -1
        return tensor.view(shape)

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
        parser = parent_parser.add_argument_group("IDSLRModel")

        # network params
        parser.add_argument(
            "--in_chans", default=32, type=int, help="Number of U-Net input channels"
        )
        parser.add_argument(
            "--out_chans", default=32, type=int, help="Number of U-Net output chanenls"
        )
        parser.add_argument(
            "--seg_out_chans",
            default=4,
            type=int,
            help="Number of segmentation output chanenls",
        )
        parser.add_argument(
            "--chans", default=32, type=int, help="Number of top-level U-Net filters."
        )
        parser.add_argument(
            "--num_pools",
            default=4,
            type=int,
            help="Number of U-Net pooling layers.",
        )
        parser.add_argument(
            "--drop_prob", default=0.0, type=float, help="U-Net dropout probability"
        )
        parser.add_argument(
            "--num_iters",
            default=5,
            type=int,
            help="Number of times to apply the model.",
        )
        parser.add_argument(
            "--norm_groups",
            default=2,
            type=int,
            help="Number of times to apply the model.",
        )
        parser.add_argument(
            "--num_cascades",
            default=1,
            type=int,
            help="Number of models to cascade.",
        )
        parser.add_argument(
            "--normalize",
            action="store_true",
            default=True,
            help="Normalize the input.",
        )
        parser.add_argument(
            "--soft_dc",
            action="store_true",
            default=False,
            help="Add parametrs for the DC blocks.",
        )

        parser.add_argument(
            "--fft_type", type=str, default="normal", help="Type of FFT to use"
        )

        parser.add_argument(
            "--train_metric_only",
            default=True,
            type=bool,
            help="Turn on the calculation of evaluation metrics.",
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
