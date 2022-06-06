from collections import defaultdict
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics import functional as FM
from einops import rearrange

from losses import DiceLoss

from model.cirim import CIRIMModule
from model.unet import LamdaUnetModule

# temp
import wandb
import os


def retrieve_checkpoint(
    model_id, project="techfidera-recseg", epoch="best", download_dir=None
):
    api = wandb.Api()
    artifact_path = os.path.join("lysander", project, f"model-{model_id}:{epoch}")
    artifact = api.artifact(artifact_path, type="model")
    return artifact.get_path("model.ckpt").download(root=download_dir)


class RecSegModule(pl.LightningModule):
    def __init__(
        self, **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.cirim = CIRIMModule(**kwargs)
        self.lambdaunet = LamdaUnetModule(**kwargs)

        # self.cirim.load_state_dict(
        #     torch.load(retrieve_checkpoint(model_id="1pn2ol98", epoch="v39"))[
        #         "state_dict"
        #     ]
        # )
        # self.lambdaunet.load_state_dict(
        #     torch.load(retrieve_checkpoint(model_id="3s7arbgg"))["state_dict"]
        # )

        self.example_input_array = [
            torch.rand(3, 32, 320, 320, 2),  # kspace
            torch.rand(3, 32, 320, 320, 2),  # sesitivity maps
            torch.rand(3, 1, 320, 320, 1),  # mask
            torch.rand(3, 320, 320, 2),  # initial prediction
            torch.rand(3, 320, 320),  # target
        ]

    def forward(self, y, sensitivity_maps, mask, init_pred, target):
        recons = self.cirim.forward(y, sensitivity_maps, mask, init_pred, target)
        x = rearrange(torch.view_as_real(recons[-1][-1]), "b h w c -> b c h w")
        x = F.group_norm(x, num_groups=1)
        seg = self.lambdaunet.forward(x)
        return recons, seg

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
        y, mask, _ = self.cirim.model.process_inputs(y, mask)

        y = self.cirim.fold(y)
        sensitivity_maps = self.cirim.fold(sensitivity_maps)
        mask = self.cirim.fold(mask)
        target = self.cirim.fold(target)
        segmentation = self.cirim.fold(segmentation)

        preds_recon, pred_seg = self.forward(
            y, sensitivity_maps, mask, init_pred, target
        )

        # if torch.any(torch.isnan(preds[-1][-1])) or torch.any(torch.isnan(pred_seg)):
        #     raise ValueError

        loss = self.cirim.model.calculate_loss(preds_recon, target, _loss_fn=F.l1_loss)

        output = torch.abs(preds_recon[-1][-1])
        output = output / output.amax((-1, -2), True)
        target = torch.abs(target) / torch.abs(target).amax((-1, -2), True)

        loss_dict = self.lambdaunet.calculate_metrics(
            pred_seg, segmentation, important_only=False
        )
        loss_dict["l1"] = loss
        loss_dict["psnr"] = FM.psnr(output.unsqueeze(-3), target.unsqueeze(-3))
        loss_dict["ssim"] = FM.ssim(output.unsqueeze(-3), target.unsqueeze(-3))
        loss_dict["loss"] = (
            1e-3 * (loss_dict["cross_entropy"] + loss_dict["dice_loss"])
            + (1 - 1e-3) * loss_dict["l1"]
        )
        return loss_dict, preds_recon, pred_seg

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
        return loss_dict, output_recon, output_seg

    def predict_step(self, batch, batch_idx):
        fname = batch[5]
        slice_num = batch[6]
        loss_dict, output_recon, output_seg = self.step(batch, batch_idx)
        if isinstance(output_recon, list):
            output_recon = [
                i[0].unsqueeze(0).unsqueeze(0).detach().cpu()
                for j in output_recon
                for i in j
            ]
        return (
            loss_dict,
            (fname, slice_num),
            output_recon,
            output_seg[0].unsqueeze(0).detach().cpu(),
        )

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
        parser = parent_parser.add_argument_group("RecSegModel")

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
            "--fft_type", type=str, default="normal", help="Type of FFT to use"
        )

        # network params
        parser.add_argument(
            "--in_chans", default=2, type=int, help="Number of U-Net input channels"
        )
        parser.add_argument(
            "--out_chans", default=4, type=int, help="Number of U-Net output chanenls"
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
